"""Async Milvus retriever for parallel semantic and hybrid search."""
import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


def _ms(t: float) -> str:
    return f"{t*1000:.0f}ms" if t < 1 else f"{t:.2f}s"


def _tlog(label: str, t0: datetime, t1: datetime) -> str:
    fmt = "%H:%M:%S.%f"
    elapsed = _ms((t1 - t0).total_seconds())
    return f"{label}  [{t0.strftime(fmt)[:-3]} → {t1.strftime(fmt)[:-3]} | {elapsed}]"


class AsyncRetriever:
    """
    Async retriever using pymilvus AsyncMilvusClient.

    Supports:
    - Single async semantic / BM25 / hybrid search
    - Bulk vector search (N queries → one Milvus call, same collection)
    - Parallel search across multiple collections via asyncio.gather

    Typical usage:
        retriever = AsyncRetriever(uri="http://localhost:19530", embedding_function=my_embed)
        docs = await retriever.search_parallel(steps, bm25_loader=my_loader)
    """

    def __init__(
        self,
        *,
        uri: str,
        embedding_function=None,
        top_k: int = 50,
        sparse_top_k: int = 50,
        hnsw_ef_search: int = 512,
        bm25_drop_ratio: float = 0.0,
        hybrid_semantic_weight: float = 0.6,
        db_name: str = "",
    ):
        from pymilvus import AsyncMilvusClient
        self.client = AsyncMilvusClient(uri=uri, db_name=db_name)
        self.embedding_function = embedding_function
        self.top_k = top_k
        self.sparse_top_k = sparse_top_k
        self.hnsw_ef_search = hnsw_ef_search
        self.bm25_drop_ratio = bm25_drop_ratio
        self.hybrid_semantic_weight = hybrid_semantic_weight

    # ── Single-query ─────────────────────────────────────────────────────────

    async def search_semantic(
        self,
        collection_name: str,
        query: str,
        *,
        limit: Optional[int] = None,
        output_fields: Optional[List[str]] = None,
        filters: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if self.embedding_function is None:
            raise RuntimeError("embedding_function is required")

        if hasattr(self.embedding_function, "embed_query"):
            embedding = self.embedding_function.embed_query(query)
        elif callable(self.embedding_function):
            embedding = self.embedding_function(query)
        else:
            raise TypeError("embedding_function must be a callable or have an embed_query method")

        results = await self.client.search(
            collection_name=collection_name,
            data=[embedding],
            anns_field="embedding",
            search_params={"metric_type": "COSINE", "params": {"ef": self.hnsw_ef_search}},
            limit=limit or self.top_k,
            output_fields=output_fields or [],
            filter=filters or "",
        )
        return self._extract(results, output_fields or [])

    async def search_bm25(
        self,
        collection_name: str,
        sparse_vector: Dict[int, float],
        *,
        limit: Optional[int] = None,
        output_fields: Optional[List[str]] = None,
        filters: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not sparse_vector:
            return []

        results = await self.client.search(
            collection_name=collection_name,
            data=[sparse_vector],
            anns_field="sparse_vector",
            search_params={"metric_type": "IP", "params": {"drop_ratio_search": self.bm25_drop_ratio}},
            limit=limit or self.sparse_top_k,
            output_fields=output_fields or [],
            filter=filters or "",
        )
        return self._extract(results, output_fields or [])

    async def search_hybrid(
        self,
        collection_name: str,
        query: str,
        sparse_vector: Dict[int, float],
        *,
        limit: Optional[int] = None,
        output_fields: Optional[List[str]] = None,
        alpha: Optional[float] = None,
        filters: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Parallel semantic + BM25 via asyncio.gather, fused by weighted score."""
        semantic, bm25 = await asyncio.gather(
            self.search_semantic(collection_name, query, limit=limit,
                                 output_fields=output_fields, filters=filters),
            self.search_bm25(collection_name, sparse_vector, limit=limit,
                             output_fields=output_fields, filters=filters),
        )
        fused = self.fuse_results(semantic, bm25, alpha=alpha)
        return fused[:limit] if limit else fused

    # ── Bulk search (N queries → same collection, one Milvus call) ───────────

    async def search_semantic_bulk(
        self,
        collection_name: str,
        queries: List[str],
        *,
        limit: Optional[int] = None,
        output_fields: Optional[List[str]] = None,
        filters: Optional[str] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Embed all queries and send in one batched Milvus call.
        Returns one result list per query.
        """
        if self.embedding_function is None:
            raise RuntimeError("embedding_function is required")

        t0 = datetime.now()
        if hasattr(self.embedding_function, "embed_documents"):
            embeddings = self.embedding_function.embed_documents(queries)
        elif hasattr(self.embedding_function, "embed_query"):
            embeddings = [self.embedding_function.embed_query(q) for q in queries]
        elif callable(self.embedding_function):
            embeddings = [self.embedding_function(q) for q in queries]
        else:
            raise TypeError("embedding_function must be a callable, or have embed_documents/embed_query methods")
        logger.info(_tlog(f"[bulk] embed {len(queries)} quer{'y' if len(queries)==1 else 'ies'}", t0, datetime.now()))

        t1 = datetime.now()
        results = await self.client.search(
            collection_name=collection_name,
            data=embeddings,
            anns_field="embedding",
            search_params={"metric_type": "COSINE", "params": {"ef": self.hnsw_ef_search}},
            limit=limit or self.top_k,
            output_fields=output_fields or [],
            filter=filters or "",
        )
        logger.info(_tlog("[bulk] milvus search", t1, datetime.now()))

        all_docs = []
        for hits in results:
            docs = []
            for hit in hits:
                doc = {"id": hit["id"], "score": hit["distance"]}
                for f in (output_fields or []):
                    doc[f] = hit.get("entity", {}).get(f, "")
                docs.append(doc)
            all_docs.append(docs)
        return all_docs

    # ── Parallel search (retrieval plan orchestrator) ─────────────────────────

    async def search_parallel(
        self,
        steps: List[Dict[str, Any]],
        *,
        output_fields: Optional[List[str]] = None,
        bm25_loader: Optional[Callable] = None,
        filters: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute retrieval plan steps in parallel with smart batching.

        - Groups steps by collection
        - Same collection + N queries → bulk vector search (1 Milvus call)
        - Different collections → asyncio.gather across collections

        Args:
            steps: List of {query, collection, ...} dicts from query expansion
            output_fields: Fields to return from Milvus
            bm25_loader: callable(collection_name) -> sparse_gen or None
            filters: Milvus filter expression

        Returns:
            Merged, deduplicated documents sorted by score
        """
        if not steps:
            return []

        by_collection: Dict[str, List[Dict]] = defaultdict(list)
        for step in steps:
            by_collection[step.get("collection", "")].append(step)

        logger.info(
            f"Parallel search: {len(steps)} queries across "
            f"{len(by_collection)} collection(s): {list(by_collection.keys())}"
        )

        tasks = [
            self._search_collection(col, col_steps, output_fields=output_fields,
                                    bm25_loader=bm25_loader, filters=filters)
            for col, col_steps in by_collection.items()
        ]
        results_per_collection = await asyncio.gather(*tasks, return_exceptions=True)

        all_docs = []
        for col, result in zip(by_collection.keys(), results_per_collection):
            if isinstance(result, Exception):
                logger.error(f"Search failed for '{col}': {result}")
                continue
            logger.info(f"  '{col}': {len(result)} documents")
            all_docs.extend(result)

        merged = self._dedup(all_docs)
        logger.info(f"Parallel search complete: {len(merged)} unique documents")
        return merged

    async def _search_collection(
        self,
        collection_name: str,
        steps: List[Dict[str, Any]],
        *,
        output_fields: Optional[List[str]] = None,
        bm25_loader: Optional[Callable] = None,
        filters: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        queries = [s["query"] for s in steps]
        t_col = datetime.now()
        logger.info(f"[{collection_name}] {len(queries)} quer{'y' if len(queries)==1 else 'ies'}")

        bm25_gen = bm25_loader(collection_name) if bm25_loader else None

        if bm25_gen is not None:
            t0 = datetime.now()
            tasks = [
                self.search_hybrid(
                    collection_name, q, bm25_gen.generate_sparse_vector(q),
                    output_fields=output_fields, filters=filters,
                )
                for q in queries
            ]
            logger.info(_tlog("[bm25] sparse vectors", t0, datetime.now()))
            t1 = datetime.now()
            results = await asyncio.gather(*tasks)
            logger.info(_tlog(f"[hybrid] {len(tasks)} parallel", t1, datetime.now()))
            docs = [dict(doc, _collection=collection_name) for result in results for doc in result]
        elif len(queries) == 1:
            docs = await self.search_semantic(collection_name, queries[0],
                                              output_fields=output_fields, filters=filters)
            docs = [dict(doc, _collection=collection_name) for doc in docs]
        else:
            results_per_query = await self.search_semantic_bulk(
                collection_name, queries, output_fields=output_fields, filters=filters
            )
            docs = [dict(doc, _collection=collection_name)
                    for result in results_per_query for doc in result]

        logger.info(_tlog(f"[{collection_name}] done ({len(docs)} docs)", t_col, datetime.now()))
        return docs

    async def close(self) -> None:
        await self.client.close()

    # ── Score utilities ───────────────────────────────────────────────────────

    @staticmethod
    def normalize_scores(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not documents:
            return documents
        scores = [d["score"] for d in documents]
        lo, hi = min(scores), max(scores)
        for d in documents:
            d["normalized_score"] = 1.0 if hi == lo else (d["score"] - lo) / (hi - lo)
        return documents

    def fuse_results(
        self,
        semantic: List[Dict[str, Any]],
        bm25: List[Dict[str, Any]],
        *,
        alpha: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        a = alpha if alpha is not None else self.hybrid_semantic_weight
        b = 1.0 - a

        semantic = self.normalize_scores(semantic)
        bm25 = self.normalize_scores(bm25)

        fused: Dict[Any, Dict] = {}
        for doc in semantic:
            fused[doc["id"]] = {**doc, "semantic_score": doc["normalized_score"],
                                 "bm25_score": 0.0, "hybrid_score": a * doc["normalized_score"]}
        for doc in bm25:
            if doc["id"] in fused:
                fused[doc["id"]]["bm25_score"] = doc["normalized_score"]
                fused[doc["id"]]["hybrid_score"] = (
                    a * fused[doc["id"]]["semantic_score"] + b * doc["normalized_score"]
                )
            else:
                fused[doc["id"]] = {**doc, "semantic_score": 0.0,
                                     "bm25_score": doc["normalized_score"],
                                     "hybrid_score": b * doc["normalized_score"]}

        return sorted(fused.values(), key=lambda x: x["hybrid_score"], reverse=True)

    @staticmethod
    def _dedup(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep highest-scoring document per ID."""
        seen: Dict[Any, Dict] = {}
        for doc in documents:
            key = "hybrid_score" if "hybrid_score" in doc else "score"
            if doc["id"] not in seen or doc.get(key, 0) > seen[doc["id"]].get(key, 0):
                seen[doc["id"]] = doc
        return list(seen.values())

    @staticmethod
    def _extract(results, output_fields: List[str]) -> List[Dict[str, Any]]:
        docs = []
        for hits in results:
            for hit in hits:
                doc = {"id": hit["id"], "score": hit["distance"]}
                entity = hit.get("entity", {})
                for f in output_fields:
                    doc[f] = entity.get(f, "")
                docs.append(doc)
        return docs

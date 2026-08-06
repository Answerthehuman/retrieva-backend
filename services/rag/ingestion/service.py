import asyncio
import logging
from typing import Any, Dict, Optional

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from core.config.settings import get_settings
from core.providers import get_embeddings, get_llm
from services.rag.ingestion.parser.parser import NativeParser
from services.rag.ingestion.chunker.chunker import RecursiveChunker
from services.rag.ingestion.vector_store.writer import MilvusWriter

logger = logging.getLogger(__name__)

def _ensure_collection(collection_name: str, dim: int) -> Collection:
    """Create the Milvus collection if it does not exist with the standard schema."""
    if utility.has_collection(collection_name):
        return Collection(collection_name)

    logger.info("Creating Milvus collection: %s", collection_name)
    
    settings = get_settings()
    
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="document_summary", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="page", dtype=DataType.INT64),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="total_chunks", dtype=DataType.INT64),
        FieldSchema(name="chunk_size", dtype=DataType.INT64),
        FieldSchema(name="ingested_at", dtype=DataType.VARCHAR, max_length=64),
    ]
    
    if settings.hybrid_search_enabled:
        fields.append(
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR)
        )
        logger.info("Added sparse_vector field for hybrid search")
        
    schema = CollectionSchema(fields=fields, description="Retrieva Document Store", enable_dynamic_field=True)
    collection = Collection(name=collection_name, schema=schema)
    
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    logger.info("Created dense index for collection: %s", collection_name)
    
    if settings.hybrid_search_enabled:
        sparse_index_params = {
            "metric_type": "IP",
            "index_type": "SPARSE_INVERTED_INDEX",
            "params": {"drop_ratio_build": 0.0}
        }
        collection.create_index(field_name="sparse_vector", index_params=sparse_index_params)
        logger.info("Created sparse index for collection: %s", collection_name)
        
    return collection


class IngestionService:
    """Orchestrates file ingestion: parse → chunk → summarize → embed → store."""
    
    def __init__(self):
        self.settings = get_settings()
        self.llm = get_llm(self.settings)
        self.embeddings = get_embeddings(self.settings)

        # Initialize pipeline tools
        self.parser = NativeParser(vision_llm=self.llm)
        self.chunker = RecursiveChunker(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

    def _connect_milvus(self) -> None:
        """Open the synchronous pymilvus connection used for schema work.

        Idempotent, and deliberately kept out of __init__: the constructor runs
        inside an async request handler, and connections.connect() is blocking
        network I/O. Callers dispatch this via asyncio.to_thread.
        """
        if connections.has_connection("default"):
            return
        connections.connect(alias="default", uri=self.settings.milvus_uri)

    async def ingest_file(
        self, 
        file_path: str, 
        collection_name: Optional[str] = None,
        source_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Ingest a file into Milvus.
        
        Args:
            file_path: Path to the local file.
            collection_name: Target Milvus collection (creates it if missing).
            source_name: Original file name or URI for metadata.
            
        Returns:
            Dict containing ingestion statistics.
        """
        target_collection = collection_name or self.settings.milvus_default_collection

        # Schema/collection work uses the blocking pymilvus API — keep it off
        # the event loop so uploads don't stall in-flight chat streams.
        await asyncio.to_thread(self._connect_milvus)
        collection = await asyncio.to_thread(
            _ensure_collection, target_collection, self.settings.embedding_dim
        )
        await asyncio.to_thread(collection.load)

        # 1. Parse pages (returns list of strings, one per page/section)
        # CPU-bound (PyMuPDF) and potentially vision-LLM bound.
        logger.info("Parsing file: %s", file_path)
        pages = await asyncio.to_thread(self.parser.parse_pages, file_path)

        if not pages:
            logger.warning("No text extracted from %s", file_path)
            return {"inserted": 0, "collection_name": target_collection}
            
        # 2. Chunk
        logger.info("Chunking %d pages", len(pages))
        import uuid
        
        doc_id = str(uuid.uuid4())
        source = source_name or str(file_path)

        def _chunk_all_pages():
            chunks = []
            for i, page_text in enumerate(pages):
                chunks.extend(
                    self.chunker.chunk(
                        page_text,
                        document_metadata={
                            "document_id": doc_id,
                            "source": source,
                            "page": i + 1,
                        },
                    )
                )
            return chunks

        # One thread hop for the whole loop rather than one per page.
        all_chunks = await asyncio.to_thread(_chunk_all_pages)

        logger.info("Generated %d chunks total", len(all_chunks))
            
        # 3. Summarize document
        # Join a sample of the text for summarization to avoid massive context
        sample_text = "\n\n".join(pages)[:20000]
        logger.info("Generating document summary")
        
        from langchain_core.messages import HumanMessage, SystemMessage
        prompt = [
            SystemMessage(content="You are an expert at concisely summarizing documents. Provide a short 2-3 sentence summary of the following document to help with retrieval and cataloging. Focus on the core topic and key findings."),
            HumanMessage(content=sample_text)
        ]
        response = await self.llm.ainvoke(prompt)
        doc_summary = str(response.content)
        logger.info("Document summary: %.100s...", doc_summary)
        
        # Generate BM25 sparse vectors if hybrid search is enabled.
        # Vocabulary building is CPU-bound and stats persistence touches disk and
        # Redis, so the whole block runs off the event loop.
        if self.settings.hybrid_search_enabled:
            logger.info("Generating BM25 sparse vectors for ingestion...")

            def _build_sparse_vectors():
                from core.utils.bm25 import SparseVectorGenerator, save_bm25_stats, load_bm25_stats

                # Load existing stats or initialize fresh generator
                sparse_generator = SparseVectorGenerator()
                existing_stats = load_bm25_stats(target_collection)
                if existing_stats:
                    sparse_generator.vocab = existing_stats["vocab"]
                    sparse_generator.idf_scores = {int(k): v for k, v in existing_stats["idf_scores"].items()}
                    sparse_generator._doc_frequencies = existing_stats.get("doc_frequencies", {})
                    sparse_generator.doc_count = existing_stats["doc_count"]
                    sparse_generator._total_doc_length = existing_stats.get("total_doc_length", 0)
                    sparse_generator.avg_doc_length = existing_stats["avg_doc_length"]
                    sparse_generator.k1 = existing_stats.get("k1", 1.2)
                    sparse_generator.b = existing_stats.get("b", 0.75)
                    if sparse_generator._total_doc_length == 0 and sparse_generator.doc_count > 0:
                        sparse_generator._total_doc_length = int(sparse_generator.avg_doc_length * sparse_generator.doc_count)

                # Build vocabulary with new chunk content
                chunk_texts = [chunk["content"] for chunk in all_chunks]
                sparse_generator.build_vocabulary(chunk_texts, incremental=True)

                # Generate sparse vector for each chunk
                for chunk in all_chunks:
                    chunk["sparse_vector"] = sparse_generator.generate_sparse_vector(chunk["content"])

                # Save stats back to disk/cache
                redis_client = None
                try:
                    import redis
                    redis_client = redis.Redis(host=self.settings.redis_host, port=self.settings.redis_port, decode_responses=True)
                except Exception:
                    pass

                stats_dict = {
                    "vocab": sparse_generator.vocab,
                    "idf_scores": {str(k): v for k, v in sparse_generator.idf_scores.items()},
                    "doc_frequencies": sparse_generator._doc_frequencies,
                    "doc_count": sparse_generator.doc_count,
                    "total_doc_length": sparse_generator._total_doc_length,
                    "avg_doc_length": sparse_generator.avg_doc_length,
                    "k1": sparse_generator.k1,
                    "b": sparse_generator.b
                }
                save_bm25_stats(target_collection, stats_dict, redis_client=redis_client)

            await asyncio.to_thread(_build_sparse_vectors)

        # 4. Embed and Store
        logger.info("Writing to Milvus collection: %s", target_collection)
        writer = MilvusWriter(
            collection=collection,
            embedding_model=self.embeddings,
            batch_size=100
        )

        stats = await writer.upsert(all_chunks, document_summary=doc_summary)
        stats["collection_name"] = target_collection
        stats["document_summary"] = doc_summary
        stats["file_name"] = source
        
        return stats

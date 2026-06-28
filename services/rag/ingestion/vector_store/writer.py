"""Milvus batch writer — embeds chunks and upserts into a collection."""
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Chunk dict keys that map to non-obvious Milvus field names
_FIELD_ALIASES: Dict[str, List[str]] = {
    "text": ["content", "text"],
    "embedding": ["dense_vector", "embedding"],
}

# Integer fields — return 0 instead of "" when missing
_INT_FIELDS = {"page", "chunk_index", "total_chunks", "section_level"}


def _extract_field(chunk: Dict[str, Any], field_name: str) -> Any:
    """Pull a field value from a chunk dict, trying aliases and returning safe defaults."""
    keys = _FIELD_ALIASES.get(field_name, [field_name])
    for key in keys:
        if key in chunk:
            val = chunk[key]
            return val if val is not None else (0 if field_name in _INT_FIELDS else "")
    return 0 if field_name in _INT_FIELDS else ""


class MilvusWriter:
    """
    Embeds chunks (if not already embedded) and upserts them into a Milvus collection.

    Args:
        collection: A loaded pymilvus Collection instance (caller creates and loads it).
        embedding_model: LangChain-compatible embeddings. Used only for chunks that
            don't already have a `dense_vector` field.
        batch_size: Number of chunks per Milvus insert call.
    """

    def __init__(self, *, collection, embedding_model=None, batch_size: int = 100, embed_batch_size: int = 50):
        self._collection = collection
        self._embedding_model = embedding_model
        self._batch_size = batch_size
        self._embed_batch_size = embed_batch_size

        # Inspect schema once — get field names and which is the primary key
        schema = collection.schema
        self._field_names = [f.name for f in schema.fields]
        self._primary_key = next(
            (f.name for f in schema.fields if getattr(f, "is_primary", False)), "id"
        )
        logger.info(
            f"MilvusWriter ready: collection='{collection.name}', "
            f"fields={self._field_names}, primary='{self._primary_key}'"
        )

    def upsert(
        self,
        chunks: List[Dict[str, Any]],
        *,
        document_summary: str = "",
    ) -> Dict[str, int]:
        """
        Embed (if needed) and upsert chunks into the collection.

        Args:
            chunks: List of chunk dicts. Each must have at least `id` and `content`.
                Chunks that already have `dense_vector` skip embedding.
            document_summary: Document-level summary written into every chunk row
                if the collection has a `document_summary` field.

        Returns:
            Dict with `inserted` count.
        """
        if not chunks:
            return {"inserted": 0}

        chunks = self._ensure_embeddings(chunks)
        ingested_at = datetime.now(timezone.utc).isoformat()

        total_inserted = 0
        for i in range(0, len(chunks), self._batch_size):
            batch = chunks[i:i + self._batch_size]
            data = self._build_insert_data(batch, document_summary=document_summary, ingested_at=ingested_at)
            self._collection.upsert(data)
            total_inserted += len(batch)
            logger.info(
                f"Upserted batch {i // self._batch_size + 1}: "
                f"{len(batch)} chunks → '{self._collection.name}'"
            )

        self._collection.flush()
        logger.info(f"Flush complete. Total upserted: {total_inserted}")
        return {"inserted": total_inserted}

    def _ensure_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate dense_vector for chunks that don't already have one."""
        needs_embedding = [c for c in chunks if not c.get("dense_vector")]
        if not needs_embedding:
            return chunks

        if self._embedding_model is None:
            raise ValueError(
                f"{len(needs_embedding)} chunks are missing `dense_vector` "
                "and no embedding_model was provided."
            )

        texts = [c["content"] for c in needs_embedding]
        logger.info(f"Generating embeddings for {len(texts)} chunks in batches of {self._embed_batch_size}")

        all_vectors: List = []
        for i in range(0, len(texts), self._embed_batch_size):
            batch = texts[i:i + self._embed_batch_size]
            vectors = self._embedding_model.embed_documents(batch)
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"Embedding batch {i // self._embed_batch_size + 1} returned "
                    f"{len(vectors)} vectors for {len(batch)} texts. "
                    "Check that the embedding model is accessible and the texts are valid."
                )
            all_vectors.extend(vectors)
            logger.info(f"  Embedded batch {i // self._embed_batch_size + 1}/{-(-len(texts) // self._embed_batch_size)}")

        idx = 0
        result = []
        for chunk in chunks:
            if not chunk.get("dense_vector"):
                result.append({**chunk, "dense_vector": all_vectors[idx]})
                idx += 1
            else:
                result.append(chunk)
        return result

    def _build_insert_data(
        self,
        batch: List[Dict[str, Any]],
        *,
        document_summary: str,
        ingested_at: str,
    ) -> List[Dict[str, Any]]:
        """Map chunk dicts to the collection's field schema."""
        rows = []
        for chunk in batch:
            row: Dict[str, Any] = {}
            for field_name in self._field_names:
                if field_name == "document_summary":
                    row[field_name] = chunk.get("document_summary") or document_summary
                elif field_name == "ingested_at":
                    row[field_name] = ingested_at
                elif field_name == "sparse_vector":
                    # sparse_vector must be omitted entirely if not present (pymilvus requirement)
                    sv = chunk.get("sparse_vector")
                    if sv:
                        row[field_name] = sv
                else:
                    row[field_name] = _extract_field(chunk, field_name)
            rows.append(row)
        return rows

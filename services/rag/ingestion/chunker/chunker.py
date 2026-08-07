"""Recursive character text splitter wrapper."""
import logging
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]


class RecursiveChunker:
    """
    Splits plain text into chunks using LangChain's RecursiveCharacterTextSplitter.

    Args:
        chunk_size: Target chunk size in characters.
        chunk_overlap: Character overlap between consecutive chunks.
        separators: Ordered list of separator strings to split on.
        min_chunk_length: Chunks shorter than this are dropped.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 600,
        chunk_overlap: int = 120,
        separators: Optional[List[str]] = None,
        min_chunk_length: int = 50,
    ):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self._min_length = min_chunk_length
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators or _DEFAULT_SEPARATORS,
            keep_separator=True,
            is_separator_regex=False,
        )

    def chunk(self, text: str, *, document_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Split text and return a list of chunk dicts.

        Each chunk dict contains: id, content, chunk_index, total_chunks,
        chunk_size, chunker_used, plus all keys from document_metadata.
        """
        raw_chunks = self._splitter.split_text(text)
        base_metadata = document_metadata or {}
        doc_id = base_metadata.get("document_id") or str(uuid.uuid4())

        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text or len(chunk_text) < self._min_length:
                continue
            chunks.append({
                **base_metadata,
                "id": f"{doc_id}_chunk_{i}",
                "content": chunk_text,
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
                "chunk_size": len(chunk_text),
                "chunker_used": "recursive",
            })

        logger.info(f"Created {len(chunks)} chunks from document '{doc_id}'")
        return chunks

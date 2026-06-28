"""Cross-encoder reranker for document relevance scoring."""
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


class Reranker:
    """
    Cross-encoder reranker using sentence-transformers.

    Lazy-loads the model on first use.

    Usage:
        reranker = Reranker(model_name="BAAI/bge-reranker-v2-m3", top_k=10)
        reranker.load()   # optional — called automatically on first rerank()
        docs = reranker.rerank(query, documents)
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        top_k: int = 10,
        batch_size: int = 20,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.batch_size = batch_size
        self._model = None

    def load(self) -> None:
        """Load the cross-encoder model. Safe to call multiple times."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading reranker: {self.model_name}")
            self._model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        *,
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents by relevance to query.

        Args:
            query: Search query
            documents: List of dicts with a 'text' field
            top_k: Override the instance top_k for this call
            batch_size: Override batch size for this call

        Returns:
            Top-k documents sorted by rerank_score descending
        """
        if not documents:
            return []

        k = top_k or self.top_k
        bs = batch_size or self.batch_size

        if self._model is None:
            try:
                self.load()
            except ImportError:
                logger.warning("sentence-transformers not installed — returning original order")
                return documents[:k]

        try:
            pairs = [[query, doc["text"]] for doc in documents]
            scores = self._model.predict(pairs, batch_size=bs)
            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)
            reranked = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)
            logger.info(f"Reranked {len(documents)} → top {k}")
            return reranked[:k]
        except Exception as e:
            logger.warning(f"Reranking failed: {e} — returning original order")
            return documents[:k]

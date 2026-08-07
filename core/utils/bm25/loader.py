"""Factory for a BM25 stats loader, keyed by collection name."""
from typing import Callable, Optional

from .bm25_cache import load_bm25_stats
from .generator import SparseVectorGenerator


def build_bm25_loader(settings) -> Callable[[str], Optional[SparseVectorGenerator]]:
    """
    Return a callable(collection_name) -> SparseVectorGenerator | None.

    Loads cached BM25 stats for a collection and rehydrates a SparseVectorGenerator
    from them. Returns None if hybrid search is disabled or no stats are cached yet.
    """

    def bm25_loader(col_name: str) -> Optional[SparseVectorGenerator]:
        if not settings.hybrid_search_enabled:
            return None
        stats = load_bm25_stats(col_name)
        if not stats:
            return None
        gen = SparseVectorGenerator()
        gen.vocab = stats["vocab"]
        gen.idf_scores = {int(k): v for k, v in stats["idf_scores"].items()}
        gen._doc_frequencies = stats.get("doc_frequencies", {})
        gen.doc_count = stats["doc_count"]
        gen._total_doc_length = stats.get("total_doc_length", 0)
        gen.avg_doc_length = stats["avg_doc_length"]
        gen.k1 = stats.get("k1", 1.2)
        gen.b = stats.get("b", 0.75)
        if gen._total_doc_length == 0 and gen.doc_count > 0:
            gen._total_doc_length = int(gen.avg_doc_length * gen.doc_count)
        return gen

    return bm25_loader

"""BM25 sparse vector generation."""
import logging
import math
from typing import Dict, List, Any
from collections import Counter
from .tokenizer import BM25Tokenizer

logger = logging.getLogger(__name__)


class SparseVectorGenerator:
    """Generate BM25-based sparse vectors for hybrid retrieval."""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.tokenizer = BM25Tokenizer()
        self.k1 = k1
        self.b = b
        self.vocab: Dict[str, int] = {}
        self.idf_scores: Dict[int, float] = {}
        self.doc_count: int = 0
        self.avg_doc_length: float = 0.0
        self._doc_frequencies: Dict[str, int] = {}
        self._total_doc_length: int = 0  # Track total for proper avg calculation
    
    def build_vocabulary(self, documents: List[str], incremental: bool = True) -> None:
        """
        Build vocabulary and compute IDF scores from document corpus.
        
        Args:
            documents: List of document texts to process
            incremental: If True, merge with existing vocabulary. If False, start fresh.
        """
        if not documents:
            logger.warning("Empty document list")
            return
        
        existing_vocab_size = len(self.vocab)
        existing_doc_count = self.doc_count
        
        if not incremental:
            # Fresh start - clear everything
            self.vocab = {}
            self._doc_frequencies = {}
            self.doc_count = 0
            self._total_doc_length = 0
            existing_vocab_size = 0
            existing_doc_count = 0
        
        logger.info(f"Building vocabulary from {len(documents)} documents (incremental={incremental})...")
        
        new_doc_count = len(documents)
        new_total_length = 0
        
        for doc_idx, doc_text in enumerate(documents):
            tokens = self.tokenizer.tokenize(doc_text)
            new_total_length += len(tokens)
            
            # Update doc frequencies (merge with existing)
            for token in set(tokens):
                self._doc_frequencies[token] = self._doc_frequencies.get(token, 0) + 1
        
        # Update cumulative stats
        self.doc_count += new_doc_count
        self._total_doc_length += new_total_length
        
        # Rebuild vocab with token IDs (consistent alphabetical order)
        self.vocab = {}
        token_id = 0
        for token in sorted(self._doc_frequencies.keys()):
            self.vocab[token] = token_id
            token_id += 1
        
        # Calculate average doc length across ALL documents
        self.avg_doc_length = self._total_doc_length / self.doc_count if self.doc_count > 0 else 0.0
        
        # Recompute IDF scores with updated doc_count
        self._compute_idf_scores()
        
        new_tokens = len(self.vocab) - existing_vocab_size
        logger.info(
            f"Vocabulary: {existing_vocab_size} → {len(self.vocab)} tokens (+{new_tokens} new), "
            f"Docs: {existing_doc_count} → {self.doc_count}, "
            f"Avg length: {self.avg_doc_length:.2f}"
        )
    
    def _compute_idf_scores(self) -> None:
        """Compute IDF scores for all tokens."""
        for token, token_id in self.vocab.items():
            doc_freq = self._doc_frequencies.get(token, 0)
            idf = math.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
            self.idf_scores[token_id] = idf
    
    def generate_sparse_vector(self, text: str) -> Dict[int, float]:
        """Generate BM25 sparse vector for a document."""
        if not self.vocab:
            return {}
        
        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            return {}
        
        term_freq = Counter(tokens)
        doc_length = len(tokens)
        sparse_vector = {}
        
        for token, freq in term_freq.items():
            if token not in self.vocab:
                continue
            
            token_id = self.vocab[token]
            idf = self.idf_scores.get(token_id, 0.0)
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            bm25_weight = idf * (numerator / denominator)
            
            # Milvus expects integer keys as indices and float weights
            if bm25_weight > 0:
                sparse_vector[token_id] = bm25_weight
        
        return sparse_vector
    
    def generate_vectors_batch(self, texts: List[str]) -> List[Dict[int, float]]:
        """Generate sparse vectors for multiple texts."""
        return [self.generate_sparse_vector(text) for text in texts]
    
    def save_stats(self, filepath: str) -> None:
        """Save BM25 vocabulary and IDF scores to a JSON file."""
        import json
        from pathlib import Path
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        stats = {
            "vocab": self.vocab,
            "idf_scores": {str(k): v for k, v in self.idf_scores.items()},
            "doc_frequencies": self._doc_frequencies,
            "doc_count": self.doc_count,
            "total_doc_length": self._total_doc_length,
            "avg_doc_length": self.avg_doc_length,
            "k1": self.k1,
            "b": self.b
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved BM25 stats to {filepath}: {len(self.vocab)} tokens, {self.doc_count} docs")
    
    def load_stats(self, filepath: str) -> None:
        """Load BM25 vocabulary and IDF scores from a JSON file."""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        self.vocab = stats["vocab"]
        self.idf_scores = {int(k): v for k, v in stats["idf_scores"].items()}
        self._doc_frequencies = stats.get("doc_frequencies", {})
        self.doc_count = stats["doc_count"]
        self._total_doc_length = stats.get("total_doc_length", 0)
        self.avg_doc_length = stats["avg_doc_length"]
        self.k1 = stats.get("k1", 1.2)
        self.b = stats.get("b", 0.75)
        
        # If total_doc_length wasn't saved, estimate it
        if self._total_doc_length == 0 and self.doc_count > 0:
            self._total_doc_length = int(self.avg_doc_length * self.doc_count)
        
        logger.info(f"Loaded BM25 stats from {filepath}: {len(self.vocab)} tokens, {self.doc_count} docs")

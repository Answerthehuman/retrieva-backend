"""BM25 tokenizer for text preprocessing."""
import re
from typing import List, Dict
from collections import Counter


class BM25Tokenizer:
    """Tokenizer for BM25 sparse vector generation."""
    
    def __init__(self):
        self.stop_words = self._load_stop_words()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        if not text or not text.strip():
            return []
        return self._tokenize_simple(text)
    
    def _tokenize_simple(self, text: str) -> List[str]:
        """Simple regex-based tokenization."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        # Keep only tokens with length > 2 and not in stop_words
        tokens = [t for t in tokens if t and len(t) > 2 and t not in self.stop_words]
        return tokens
    
    def _load_stop_words(self) -> set:
        """Load common English stop words."""
        return {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was', 'were',
            'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'may', 'might', 'must', 'can', 'of', 'for', 'to', 'in',
            'by', 'with', 'from', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'and', 'but',
            'or', 'if', 'because', 'until', 'while', 'this', 'that', 'these', 'those'
        }
    
    def compute_token_statistics(self, tokens: List[str]) -> Dict:
        """Compute statistics for a tokenized document."""
        if not tokens:
            return {
                "term_frequencies": Counter(),
                "doc_length": 0,
                "unique_tokens": 0,
                "avg_token_length": 0.0
            }
        
        term_frequencies = Counter(tokens)
        doc_length = len(tokens)
        unique_tokens = len(term_frequencies)
        avg_token_length = sum(len(t) for t in tokens) / doc_length
        
        return {
            "term_frequencies": term_frequencies,
            "doc_length": doc_length,
            "unique_tokens": unique_tokens,
            "avg_token_length": avg_token_length
        }
    
    def batch_tokenize(self, texts: List[str]) -> List[List[str]]:
        """Tokenize multiple texts."""
        return [self.tokenize(text) for text in texts]


def tokenize_text(text: str) -> List[str]:
    """Convenience function for quick tokenization."""
    tokenizer = BM25Tokenizer()
    return tokenizer.tokenize(text)

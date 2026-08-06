"""Ollama embeddings factory — open-source, self-hosted embedding models.

Anthropic does not serve an embeddings endpoint, so retrieval vectors come from
a local Ollama server instead.

⚠️ The vector dimension is baked into the Milvus collection schema at creation
time. Switching embedding model means changing ``EMBEDDING_DIM`` to match *and*
re-ingesting into a fresh collection — an existing collection cannot be reused
across models of different width.

    | Model              | Dim  | Notes                                        |
    |--------------------|------|----------------------------------------------|
    | nomic-embed-text   |  768 | Default. Small, fast, strong general recall. |
    | bge-m3             | 1024 | Larger; better multilingual / long-passage.  |
    | mxbai-embed-large  | 1024 | Alternative English-centric option.          |

Pull the model before first use:  ``ollama pull nomic-embed-text``
"""
from typing import Optional

DEFAULT_MODEL = "nomic-embed-text"

#: Vector width per model, used to validate settings against the Milvus schema.
MODEL_DIMENSIONS = {
    "nomic-embed-text": 768,
    "bge-m3": 1024,
    "mxbai-embed-large": 1024,
}


def expected_dimension(model: str) -> Optional[int]:
    """Known vector width for a model, or None if unrecognized."""
    # Tolerate an explicit tag such as "bge-m3:latest".
    return MODEL_DIMENSIONS.get(model.split(":", 1)[0])


def get_ollama_embeddings(
    *,
    model: str = DEFAULT_MODEL,
    base_url: Optional[str] = None,
):
    """Return a LangChain-compatible Ollama embeddings model.

    Args:
        model: Ollama embedding model name (must already be pulled).
        base_url: Ollama server URL; defaults to the library's localhost:11434.
    """
    from langchain_ollama import OllamaEmbeddings

    kwargs: dict = {"model": model}
    if base_url:
        kwargs["base_url"] = base_url

    return OllamaEmbeddings(**kwargs)

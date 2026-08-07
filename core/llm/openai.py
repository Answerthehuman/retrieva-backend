"""OpenAI LLM and Embeddings factory."""
from typing import Optional


def get_openai(
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
):
    """Return a LangChain-compatible OpenAI chat model."""
    from langchain_openai import ChatOpenAI

    kwargs: dict = dict(model=model, temperature=temperature)
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if api_key is not None:
        kwargs["api_key"] = api_key

    return ChatOpenAI(**kwargs)


def get_openai_embeddings(
    *,
    model: str = "text-embedding-3-large",
):
    """Return a LangChain-compatible OpenAI embeddings model."""
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=model)

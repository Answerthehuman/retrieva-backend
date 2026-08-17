"""OpenAI LLM and Embeddings factory."""


def get_openai(
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int | None = None,
    api_key: str | None = None,
):
    """Return a LangChain-compatible OpenAI chat model."""
    from langchain_openai import ChatOpenAI

    kwargs: dict = {"model": model, "temperature": temperature}
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

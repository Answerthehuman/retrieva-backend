"""Gemini LLM and Embeddings factory."""
from typing import Optional


def get_gemini(
    *,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    thinking: bool = False,
    thinking_budget: Optional[int] = None,
    google_api_key: Optional[str] = None,
    additional_headers: Optional[dict] = None,
):
    """Return a LangChain-compatible Gemini chat model."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    if thinking_budget is None:
        thinking_budget = 8000 if thinking else 0

    kwargs: dict = dict(
        model=model,
        temperature=1 if thinking else temperature,
        thinking_budget=thinking_budget,
    )
    if max_tokens is not None:
        kwargs["max_output_tokens"] = max_tokens
    if google_api_key is not None:
        kwargs["google_api_key"] = google_api_key
    if additional_headers is not None:
        kwargs["additional_headers"] = additional_headers

    return ChatGoogleGenerativeAI(**kwargs)


def get_gemini_embeddings(
    *,
    model: str = "models/gemini-embedding-2",
    task_type: str = "retrieval_document",
    output_dimensionality: int = 3072,
    additional_headers: Optional[dict] = None,
):
    """Return a LangChain-compatible Gemini embeddings model."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    kwargs: dict = dict(
        model=model,
        task_type=task_type,
        output_dimensionality=output_dimensionality,
    )
    if additional_headers is not None:
        kwargs["additional_headers"] = additional_headers
        
    return GoogleGenerativeAIEmbeddings(**kwargs)

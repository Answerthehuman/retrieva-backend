"""Gemini LLM and Embeddings factory."""


def get_gemini(
    *,
    model: str = "gemini-3.5-flash-lite",
    temperature: float = 0.0,
    max_tokens: int | None = None,
    thinking: bool = False,
    thinking_budget: int | None = None,
    google_api_key: str | None = None,
    additional_headers: dict | None = None,
):
    """Return a LangChain-compatible Gemini chat model.

    Gemini 3.x replaced the integer `thinking_budget` (2.x/1.5-era) with a
    `thinking_level` enum (minimal/low/medium/high) and cannot disable thinking
    entirely — sending `thinking_budget` to a 3.x model is a 400
    invalid-argument error. The two parameter styles are not interchangeable,
    so branch on the model family rather than always sending one shape.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    kwargs: dict = {"model": model, "temperature": temperature}
    if max_tokens is not None:
        kwargs["max_output_tokens"] = max_tokens
    if google_api_key is not None:
        kwargs["google_api_key"] = google_api_key
    if additional_headers is not None:
        kwargs["additional_headers"] = additional_headers

    if model.startswith("gemini-3"):
        # Gemini 3.x models use fixed sampling — passing `temperature` doesn't
        # error, but logs a UserWarning on every single call ("uses fixed
        # sampling defaults; the sampling parameter(s) temperature will be
        # ignored"). Drop it rather than spam the logs with a no-op.
        kwargs.pop("temperature", None)
        # gemini-3.5-flash-lite already defaults to its cheapest level
        # ("minimal") with the param omitted, so only set it when the caller
        # explicitly wants heavier reasoning.
        if thinking:
            kwargs["thinking_level"] = "high"
    else:
        # Pre-3.x models (gemini-2.5-*, gemini-1.5-*): thinking_budget is the
        # only supported knob, and the API requires temperature=1 whenever
        # thinking is enabled.
        if thinking_budget is None:
            thinking_budget = 8000 if thinking else 0
        kwargs["temperature"] = 1 if thinking else temperature
        kwargs["thinking_budget"] = thinking_budget

    return ChatGoogleGenerativeAI(**kwargs)


def get_gemini_embeddings(
    *,
    model: str = "models/gemini-embedding-2",
    task_type: str = "retrieval_document",
    output_dimensionality: int = 3072,
    additional_headers: dict | None = None,
):
    """Return a LangChain-compatible Gemini embeddings model."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    kwargs: dict = {
        "model": model,
        "task_type": task_type,
        "output_dimensionality": output_dimensionality,
    }
    if additional_headers is not None:
        kwargs["additional_headers"] = additional_headers

    return GoogleGenerativeAIEmbeddings(**kwargs)

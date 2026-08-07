"""Translate provider/infrastructure exceptions into actionable messages.

Vendor SDKs raise multi-line stack chatter (Google auth in particular). Surfacing
that verbatim in a chat bubble or a toast tells the user nothing they can act on.
"""


def friendly_error(exc: Exception) -> str:
    text = str(exc)
    lowered = text.lower()
    name = type(exc).__name__.lower()

    if "defaultcredentialserror" in name or "default credentials" in lowered:
        return (
            "The backend has no LLM credentials configured. "
            "Set GOOGLE_API_KEY (or OPENAI_API_KEY with LLM_PROVIDER=openai) and restart."
        )
    if "api key" in lowered or "unauthenticated" in lowered or "permission denied" in lowered:
        return f"The AI provider rejected the request: {text.splitlines()[0][:200]}"
    if "quota" in lowered or "rate limit" in lowered or "429" in lowered:
        return "The AI provider is rate-limiting or out of quota. Try again shortly."
    if "collection" in lowered and "exist" in lowered:
        return "That collection doesn't exist yet — upload a document first."
    if "milvus" in lowered or "19530" in lowered:
        return "Could not reach the vector store (Milvus)."

    return text.splitlines()[0][:300] if text else "Unknown error."

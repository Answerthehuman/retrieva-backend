"""LLM factory exports.

Defaults: Claude for every generation call, Ollama for embeddings.
Gemini and OpenAI remain available as opt-in fallbacks.
"""
from .anthropic import get_anthropic
from .gemini import get_gemini, get_gemini_embeddings
from .ollama import get_ollama_embeddings
from .openai import get_openai, get_openai_embeddings

__all__ = [
    "get_anthropic",
    "get_ollama_embeddings",
    "get_gemini",
    "get_gemini_embeddings",
    "get_openai",
    "get_openai_embeddings",
]

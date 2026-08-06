"""Claude (Anthropic) chat model factory.

Every LLM call in Retrieva — agent generation, metadata-filter extraction,
document summarization, and PDF vision OCR — runs through this factory.

Two model-specific rules are enforced here rather than left to callers:

* ``temperature`` / ``top_p`` / ``top_k`` are **rejected with a 400** on the
  current Claude models (Opus 5, Opus 4.8/4.7, Sonnet 5, Fable 5). The
  parameter is therefore dropped for those models instead of being forwarded.
  Steer behaviour with prompting, not sampling parameters.
* Extended thinking with a fixed ``budget_tokens`` is removed on those models.
  Adaptive thinking (``{"type": "adaptive"}``) is the supported replacement,
  with depth controlled by ``effort``.
"""
from typing import Optional

# Models that reject sampling parameters and fixed thinking budgets.
# Everything in the Claude 5 / Opus 4.7+ generation behaves this way.
_NO_SAMPLING_PARAMS_PREFIXES = (
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
)

DEFAULT_MODEL = "claude-opus-5"


def _rejects_sampling_params(model: str) -> bool:
    return model.startswith(_NO_SAMPLING_PARAMS_PREFIXES)


def get_anthropic(
    *,
    model: str = DEFAULT_MODEL,
    temperature: Optional[float] = None,
    max_tokens: int = 16000,
    thinking: bool = False,
    effort: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """Return a LangChain-compatible Claude chat model.

    Args:
        model: Claude model id (e.g. ``claude-opus-5``).
        temperature: Silently ignored on models that reject sampling params.
        max_tokens: Output cap. Note this bounds thinking *plus* response text,
            so leave headroom when thinking is enabled.
        thinking: Enable adaptive thinking. Off by default — the agent loop
            makes many small tool-routing calls where thinking is wasted spend.
        effort: ``low`` | ``medium`` | ``high`` | ``xhigh`` | ``max``.
        api_key: Falls back to ANTHROPIC_API_KEY in the environment.
    """
    from langchain_anthropic import ChatAnthropic

    kwargs: dict = {"model": model, "max_tokens": max_tokens}

    # Passing temperature to Opus 5 / Sonnet 5 / Fable 5 is a 400, so only
    # forward it to older models that still accept it.
    if temperature is not None and not _rejects_sampling_params(model):
        kwargs["temperature"] = temperature

    if thinking:
        # `budget_tokens` is removed on these models; adaptive is the only mode.
        kwargs["thinking"] = {"type": "adaptive"}

    if effort is not None:
        kwargs["output_config"] = {"effort": effort}

    if api_key is not None:
        kwargs["api_key"] = api_key

    return ChatAnthropic(**kwargs)

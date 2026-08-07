"""Vision-based image description helper."""
import base64
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@lru_cache(maxsize=None)
def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


class ImageDescriber:
    """
    Describes images extracted from documents using a vision LLM.

    Args:
        llm: Vision-capable LangChain LLM (e.g. Gemini, GPT-4o).
        prompt: Custom prompt template. Variable: {context}.
    """

    def __init__(self, *, llm, prompt: Optional[str] = None):
        self._llm = llm
        self._custom_prompt = prompt

    def describe(self, image_bytes: bytes, *, context: str = "") -> str:
        """Return a text description of an image given surrounding context."""
        try:
            template = self._custom_prompt or _load_prompt("image_description.md")
            text = template.replace("{context}", context)
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            message = HumanMessage(content=[
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ])
            response = self._llm.invoke([message])
            # .text, not .content — some providers return a list of content
            # blocks rather than a plain string; .text normalizes either shape.
            return f"[Image Description: {response.text}]"
        except Exception as e:
            logger.error(f"Image description failed: {e}")
            return "[Image: Undescribed]"

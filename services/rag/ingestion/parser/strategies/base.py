"""Base strategy and shared vision prompt loader."""
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Optional

from ..image_describer import ImageDescriber

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@lru_cache(maxsize=None)
def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


VISION_PROMPT = _load_prompt("vision_ocr.md")


class ParserStrategy(ABC):
    """
    Abstract base for file-type parsing strategies.

    Args:
        vision_llm: Vision-capable LangChain LLM used for OCR and image description.
        image_describer: ImageDescriber instance for describing embedded images.
        vision_prompt: Custom vision OCR prompt. Defaults to vision_ocr.md.
    """

    def __init__(
        self,
        *,
        vision_llm,
        image_describer: ImageDescriber,
        vision_prompt: Optional[str] = None,
    ):
        self.vision_llm = vision_llm
        self.image_describer = image_describer
        self.vision_prompt = vision_prompt or _load_prompt("vision_ocr.md")

    @abstractmethod
    def parse(self, file_path: Path):
        """Parse file and return extracted content."""

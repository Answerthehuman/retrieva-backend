"""NativeParser — orchestrates file-type strategies."""
import logging
from pathlib import Path
from typing import List, Optional, Union

from .image_describer import ImageDescriber
from .strategies import (
    PdfParserStrategy,
    DocxParserStrategy,
    PptxParserStrategy,
    ExcelParserStrategy,
    TextParserStrategy,
)

logger = logging.getLogger(__name__)

_EXTENSION_MAP = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",
    ".pptx": "pptx",
    ".ppt": "pptx",
    ".xlsx": "excel",
    ".xls": "excel",
    ".csv": "excel",
    ".txt": "text",
    ".md": "text",
    ".markdown": "text",
}


class NativeParser:
    """
    Parses documents using file-type-specific strategies.

    Args:
        vision_llm: Vision-capable LangChain LLM (required for PDF, PPTX, DOCX images).
        libreoffice_path: Path to the soffice executable for PPTX→PDF conversion.
        vision_prompt: Custom OCR prompt for vision-based strategies.
        image_description_prompt: Custom prompt for embedded image descriptions.
    """

    def __init__(
        self,
        *,
        vision_llm=None,
        libreoffice_path: str = "soffice",
        vision_prompt: Optional[str] = None,
        image_description_prompt: Optional[str] = None,
    ):
        self._vision_llm = vision_llm
        image_describer = ImageDescriber(llm=vision_llm, prompt=image_description_prompt) if vision_llm else None

        strategy_kwargs = dict(
            vision_llm=vision_llm,
            image_describer=image_describer,
            vision_prompt=vision_prompt,
        )

        self._strategies = {
            "pdf": PdfParserStrategy(**strategy_kwargs),
            "docx": DocxParserStrategy(**strategy_kwargs),
            "pptx": PptxParserStrategy(libreoffice_path=libreoffice_path, **strategy_kwargs),
            "excel": ExcelParserStrategy(**strategy_kwargs),
            "text": TextParserStrategy(**strategy_kwargs),
        }

    def parse(self, file_path: Union[str, Path]) -> str:
        """
        Parse a file and return its full text content as a single string.

        Args:
            file_path: Path to the file to parse.

        Returns:
            Extracted text content. Returns empty string on failure.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return ""

        file_type = _EXTENSION_MAP.get(file_path.suffix.lower())
        if file_type is None:
            logger.warning(f"Unsupported file type '{file_path.suffix}', falling back to text")
            file_type = "text"

        strategy = self._strategies[file_type]
        result = strategy.parse(file_path)

        if isinstance(result, list):
            return "\n\n".join(item.get("text", "") for item in result if isinstance(item, dict))
        return result or ""

    def parse_pages(self, file_path: Union[str, Path]) -> List[str]:
        """
        Parse a file and return per-page text as a list.

        For PDFs this returns one string per page (enables windowed LLM chunking).
        For all other formats it returns a single-element list with the full text.

        Args:
            file_path: Path to the file to parse.

        Returns:
            List of page/section strings. Empty list on failure.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []

        file_type = _EXTENSION_MAP.get(file_path.suffix.lower())
        if file_type is None:
            file_type = "text"

        strategy = self._strategies[file_type]

        if file_type == "pdf" and hasattr(strategy, "parse_pages"):
            pages = strategy.parse_pages(file_path)
            return pages if pages else []

        text = strategy.parse(file_path)
        if isinstance(text, list):
            text = "\n\n".join(item.get("text", "") for item in text if isinstance(item, dict))
        return [text] if text else []

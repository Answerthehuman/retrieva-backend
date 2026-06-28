"""Plain text / Markdown parsing strategy."""
import logging
from pathlib import Path

from .base import ParserStrategy

logger = logging.getLogger(__name__)


class TextParserStrategy(ParserStrategy):
    """Reads plain text, Markdown, and similar files as-is."""

    def parse(self, file_path: Path) -> str:
        logger.info(f"Parsing text file: {file_path}")
        try:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.error(f"Text parsing failed: {e}")
            return ""

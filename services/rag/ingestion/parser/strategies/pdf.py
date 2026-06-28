"""PDF parsing strategy — high-fidelity vision OCR via PyMuPDF."""
import base64
import logging
from pathlib import Path
from typing import List

from langchain_core.messages import HumanMessage

from .base import ParserStrategy

logger = logging.getLogger(__name__)


class PdfParserStrategy(ParserStrategy):
    """Renders each PDF page to a high-res image and extracts content via vision LLM."""

    def parse_pages(self, file_path: Path) -> List[str]:
        """Return per-page text as a list."""
        import fitz

        logger.info(f"Parsing PDF (vision OCR): {file_path}")
        try:
            doc = fitz.open(str(file_path))
            pages = []
            for i, page in enumerate(doc):
                logger.info(f"  Page {i + 1}/{len(doc)}")
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
                message = HumanMessage(content=[
                    {"type": "text", "text": self.vision_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ])
                response = self.vision_llm.invoke([message])
                pages.append(response.content)
            doc.close()
            return pages
        except Exception as e:
            logger.error(f"PDF parsing failed: {e}")
            return []

    def parse(self, file_path: Path) -> str:
        return "\n\n".join(self.parse_pages(file_path))

"""PDF parsing strategy — native text extraction with Vision LLM OCR fallback."""

import base64
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage

from .base import ParserStrategy

logger = logging.getLogger(__name__)

# Heuristic thresholds
_MIN_TEXT_LENGTH = 50  # Below this, the page is likely scanned
_MAX_DRAWINGS_NATIVE = 30  # Above this many vector drawings, treat as complex layout


def _page_needs_vision(page, extracted_text: str) -> bool:
    """Decide whether a page needs Vision LLM OCR instead of native text.

    Returns True when:
    - The page has very little native text AND contains images (scanned page).
    - The page contains tables detected by PyMuPDF.
    - The page has a very high density of vector drawings (complex diagrams/charts).
    """
    text_len = len(extracted_text.strip())

    # Check for scanned pages: little text + embedded images
    image_list = page.get_images(full=True)
    if text_len < _MIN_TEXT_LENGTH and len(image_list) > 0:
        return True

    # Check for tables (PyMuPDF >= 1.23 supports find_tables)
    try:
        tables = page.find_tables()
        if tables and len(tables.tables) > 0:
            return True
    except (AttributeError, Exception):
        # find_tables may not exist in older PyMuPDF versions — skip
        pass

    # Check for complex vector drawings
    try:
        drawings = page.get_drawings()
        if len(drawings) > _MAX_DRAWINGS_NATIVE:
            return True
    except (AttributeError, Exception):
        pass

    return False


class PdfParserStrategy(ParserStrategy):
    """Extracts text natively via PyMuPDF; falls back to Vision LLM OCR for
    scanned or complex pages (tables, diagrams, charts)."""

    def _extract_via_vision(self, page) -> str:
        """Render page to image and call the Vision LLM."""
        import fitz

        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type": "text", "text": self.vision_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ]
        )
        response = self.vision_llm.invoke([message])
        # .text, not .content — some providers return content as a list of
        # blocks rather than a plain string. This return value flows straight
        # into RecursiveCharacterTextSplitter.split_text(), which requires a
        # str; a list here would raise deep inside the chunker on every
        # scanned page, far from this line.
        return response.text

    def parse_pages(self, file_path: Path) -> list[str]:
        """Return per-page text as a list.

        For each page the strategy first tries native text extraction.
        If the heuristic detects a scanned page or complex layout AND a
        Vision LLM is configured, it falls back to high-fidelity OCR.
        """
        import fitz

        logger.info(f"Parsing PDF: {file_path}")
        try:
            doc = fitz.open(str(file_path))
            pages: list[str] = []
            vision_count = 0

            for i, page in enumerate(doc):
                native_text = page.get_text("text") or ""

                if _page_needs_vision(page, native_text) and self.vision_llm is not None:
                    logger.info(f"  Page {i + 1}/{len(doc)} — Vision LLM OCR (scanned/complex)")
                    text = self._extract_via_vision(page)
                    vision_count += 1
                else:
                    logger.info(f"  Page {i + 1}/{len(doc)} — native text extraction")
                    text = native_text

                pages.append(text)

            doc.close()

            if vision_count:
                logger.info(
                    f"PDF complete: {len(pages)} pages ({vision_count} used Vision OCR, "
                    f"{len(pages) - vision_count} native)"
                )
            else:
                logger.info(f"PDF complete: {len(pages)} pages (all native text extraction)")

            return pages
        except Exception as e:
            logger.error(f"PDF parsing failed: {e}")
            return []

    def parse(self, file_path: Path) -> str:
        return "\n\n".join(self.parse_pages(file_path))

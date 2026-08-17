"""PPTX parsing strategy — LibreOffice → PDF → vision OCR per slide."""

import base64
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from .base import ParserStrategy

logger = logging.getLogger(__name__)

# Common Windows install locations for LibreOffice
_WINDOWS_SOFFICE_PATHS = [
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
]


def _resolve_soffice(hint: str = "soffice") -> str:
    """Find the soffice executable.

    Priority:
    1. The caller-supplied *hint* (may be a full path or just ``soffice``).
    2. ``soffice`` on the system PATH (``shutil.which``).
    3. Well-known Windows install directories.

    Raises ``FileNotFoundError`` with an actionable message if nothing is found.
    """
    # If the hint is already an absolute path that exists, use it directly
    if os.path.isabs(hint) and os.path.isfile(hint):
        return hint

    # Try resolving via PATH
    found = shutil.which(hint)
    if found:
        return found

    # Windows fallback: check default install locations
    if sys.platform == "win32":
        for candidate in _WINDOWS_SOFFICE_PATHS:
            if os.path.isfile(candidate):
                logger.info("Found LibreOffice at %s", candidate)
                return candidate

    raise FileNotFoundError(
        f"LibreOffice (soffice) not found. Tried: PATH lookup for '{hint}'"
        + (f", {_WINDOWS_SOFFICE_PATHS}" if sys.platform == "win32" else "")
        + ". Install LibreOffice: winget install TheDocumentFoundation.LibreOffice"
        if sys.platform == "win32"
        else ". Install LibreOffice via your package manager."
    )


class PptxParserStrategy(ParserStrategy):
    """
    Converts PPTX → PDF via LibreOffice, then renders each slide via vision LLM.

    Args:
        libreoffice_path: Path to the soffice executable.
            Defaults to ``soffice`` (auto-resolved via PATH and Windows defaults).
    """

    def __init__(self, *, libreoffice_path: str = "soffice", **kwargs):
        super().__init__(**kwargs)
        self._soffice_hint = libreoffice_path

    def parse(self, file_path: Path) -> list[dict[str, Any]]:
        import fitz

        logger.info(f"Parsing PPTX (vision OCR): {file_path}")

        soffice = _resolve_soffice(self._soffice_hint)

        try:
            slides = []
            with tempfile.TemporaryDirectory() as tmp:
                pdf_path = os.path.join(tmp, file_path.stem + ".pdf")

                logger.info("Converting PPTX → PDF via LibreOffice (%s)", soffice)
                subprocess.run(
                    [soffice, "--headless", "--convert-to", "pdf", "--outdir", tmp, str(file_path)],
                    check=True,
                    capture_output=True,
                )
                if not os.path.exists(pdf_path):
                    raise RuntimeError(f"LibreOffice did not produce PDF at {pdf_path}")

                doc = fitz.open(pdf_path)
                for i, page in enumerate(doc):
                    logger.info(f"  Slide {i + 1}/{len(doc)}")
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    image_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
                    message = HumanMessage(
                        content=[
                            {"type": "text", "text": self.vision_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                            },
                        ]
                    )
                    response = self.vision_llm.invoke([message])
                    # .text, not .content — some providers return content as
                    # a list of blocks; .text normalizes either shape.
                    slides.append({"text": response.text, "slide_number": i + 1})
                doc.close()

            return slides
        except Exception as e:
            logger.error(f"PPTX parsing failed: {e}")
            return []

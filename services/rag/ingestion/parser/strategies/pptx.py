"""PPTX parsing strategy — LibreOffice → PDF → vision OCR per slide."""
import base64
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage

from .base import ParserStrategy

logger = logging.getLogger(__name__)


class PptxParserStrategy(ParserStrategy):
    """
    Converts PPTX → PDF via LibreOffice, then renders each slide via vision LLM.

    Args:
        libreoffice_path: Path to the soffice executable.
            Defaults to "soffice" (must be on PATH).
    """

    def __init__(self, *, libreoffice_path: str = "soffice", **kwargs):
        super().__init__(**kwargs)
        self.libreoffice_path = libreoffice_path

    def parse(self, file_path: Path) -> List[Dict[str, Any]]:
        import fitz

        logger.info(f"Parsing PPTX (vision OCR): {file_path}")
        try:
            slides = []
            with tempfile.TemporaryDirectory() as tmp:
                pdf_path = os.path.join(tmp, file_path.stem + ".pdf")

                logger.info("Converting PPTX → PDF via LibreOffice")
                subprocess.run(
                    [self.libreoffice_path, "--headless", "--convert-to", "pdf",
                     "--outdir", tmp, str(file_path)],
                    check=True, capture_output=True,
                )
                if not os.path.exists(pdf_path):
                    raise RuntimeError(f"LibreOffice did not produce PDF at {pdf_path}")

                doc = fitz.open(pdf_path)
                for i, page in enumerate(doc):
                    logger.info(f"  Slide {i + 1}/{len(doc)}")
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    image_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
                    message = HumanMessage(content=[
                        {"type": "text", "text": self.vision_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    ])
                    response = self.vision_llm.invoke([message])
                    slides.append({"text": response.content, "slide_number": i + 1})
                doc.close()

            return slides
        except Exception as e:
            logger.error(f"PPTX parsing failed: {e}")
            return []

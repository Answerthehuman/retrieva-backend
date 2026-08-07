"""Excel / CSV parsing strategy — converts sheets to Markdown tables."""
import logging
from pathlib import Path

from .base import ParserStrategy

logger = logging.getLogger(__name__)


class ExcelParserStrategy(ParserStrategy):
    """Reads all sheets from Excel or CSV files and renders them as Markdown tables."""

    def parse(self, file_path: Path) -> str:
        import pandas as pd

        logger.info(f"Parsing spreadsheet: {file_path}")
        try:
            if file_path.suffix.lower() == ".csv":
                sheets = {"Sheet1": pd.read_csv(file_path)}
            else:
                sheets = pd.read_excel(file_path, sheet_name=None)

            content = []
            for name, df in sheets.items():
                content.append(f"--- Sheet: {name} ---")
                content.append(df.fillna("").to_markdown(index=False))
                content.append("")

            return "\n".join(content)
        except Exception as e:
            logger.error(f"Spreadsheet parsing failed: {e}")
            return ""

"""Fetch and cache Milvus filter syntax documentation for use in LLM prompts."""

from pathlib import Path

DOCS_URLS = [
    "https://milvus.io/docs/boolean.md",
    "https://milvus.io/docs/basic-operators.md",
    "https://milvus.io/docs/filtering-templating.md",
]

OUTPUT_PATH = Path(__file__).parent / "milvus_filter_docs.md"


def save_content() -> None:
    """Fetch Milvus docs and write them to milvus_filter_docs.md."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        raise ImportError(
            "docling is required to refresh Milvus filter docs. "
            "Install it with: pip install docling\n"
            "This is a dev-time operation — the pre-built milvus_filter_docs.md "
            "ships with the package and should not need regenerating at runtime."
        ) from None

    converter = DocumentConverter()
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for url in DOCS_URLS:
            result = converter.convert(url)
            f.write(f"# Source: {url}\n\n")
            f.write(result.document.export_to_markdown())
            f.write("\n\n---\n\n")


if __name__ == "__main__":
    save_content()
    print(f"Saved to: {OUTPUT_PATH}")

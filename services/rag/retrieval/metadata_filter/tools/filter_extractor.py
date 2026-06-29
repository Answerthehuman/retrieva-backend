"""Extract Milvus filter expressions from natural language using an LLM."""
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Optional

from .fetch_milvus_docs import save_content

_MILVUS_DOCS_PATH = Path(__file__).parent / "milvus_filter_docs.md"
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_milvus_docs() -> str:
    if not _MILVUS_DOCS_PATH.exists():
        save_content()
    return _load_prompt(_MILVUS_DOCS_PATH)


def extract_filters_from_query(
    query: str,
    available_fields: List[Dict[str, Any]],
    *,
    llm=None,
) -> Dict[str, Any]:
    """
    Extract a Milvus filter expression from a natural language query.

    Args:
        query: User's natural language query
        available_fields: Filterable field definitions from schema_loader
        llm: LangChain-compatible LLM with an invoke() method

    Returns:
        {
            "milvus_expression": str | None,
            "cleaned_query": str,
            "confidence": float,
        }
    """
    if not llm:
        logger.warning("No LLM provided — returning empty filters")
        return {"milvus_expression": None, "cleaned_query": query, "confidence": 0.0}

    try:
        fields_desc = _format_fields_for_prompt(available_fields)
        milvus_docs = _load_milvus_docs()

        system_prompt = _load_prompt(_PROMPTS_DIR / "system_prompt.md").format(
            milvus_docs=milvus_docs
        )
        user_prompt = _load_prompt(_PROMPTS_DIR / "user_prompt.md").format(
            fields_desc=fields_desc,
            query=query,
        )

        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        parsed = _parse_llm_response(response.content.strip())
        logger.info(f"Extracted filter: {parsed.get('milvus_expression')}")
        return {
            "milvus_expression": parsed.get("milvus_expression"),
            "cleaned_query": parsed.get("cleaned_query", query),
            "confidence": parsed.get("confidence", 0.5),
        }

    except Exception as e:
        logger.error(f"Filter extraction failed: {e}")
        return {"milvus_expression": None, "cleaned_query": query, "confidence": 0.0, "error": str(e)}


def _format_fields_for_prompt(fields: List[Dict[str, Any]]) -> str:
    lines = []
    for f in fields:
        examples = f.get("examples", [])
        examples_str = f" (examples: {', '.join(examples[:3])})" if examples else ""
        dtype = f.get("dtype") or f.get("type", "")
        lines.append(f"- {f['name']} ({dtype}): {f['description']}{examples_str}")
    return "\n".join(lines)


def _parse_llm_response(content: str) -> Dict[str, Any]:
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    return json.loads(content.strip())

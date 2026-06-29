"""Build and validate Milvus filter expressions from structured filter dicts."""
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def build_milvus_expression(filters: List[Dict[str, Any]]) -> Optional[str]:
    """
    Build a Milvus filter expression from a list of filter dicts.

    Args:
        filters: [{"field": str, "operator": str, "value": Any}, ...]

    Returns:
        Milvus filter expression string, or None if no valid conditions.

    Example:
        [{"field": "project_name", "operator": "like", "value": "Nashik"},
         {"field": "document_type", "operator": "==", "value": "proposal"}]
        → 'project_name like "%Nashik%" and document_type == "proposal"'
    """
    conditions = [c for f in filters if (c := _build_condition(f["field"], f["operator"], f["value"]))]
    if not conditions:
        return None
    expression = " and ".join(conditions)
    logger.info(f"Built Milvus expression: {expression}")
    return expression


def validate_expression(expression: str) -> bool:
    """Basic sanity check on a Milvus filter expression."""
    if not expression:
        return False
    if expression.count('"') % 2 != 0:
        return False
    valid_ops = ["==", "!=", ">", ">=", "<", "<=", "like", "in", "not in", "and", "or", "&&", "||"]
    return any(op in expression for op in valid_ops)


def _build_condition(field: str, operator: str, value: Any) -> Optional[str]:
    if operator == "like":
        return f'{field} like "%{_escape(str(value))}%"'
    if operator == "in":
        if isinstance(value, list):
            return f'{field} in [{", ".join(_fmt(v) for v in value)}]'
        return f'{field} == {_fmt(value)}'
    if operator == "not in":
        if isinstance(value, list):
            return f'{field} not in [{", ".join(_fmt(v) for v in value)}]'
        return f'{field} != {_fmt(value)}'
    if operator in ("==", "!=", ">", ">=", "<", "<="):
        return f'{field} {operator} {_fmt(value)}'
    logger.warning(f"Unknown operator: {operator}")
    return None


def _fmt(value: Any) -> str:
    if isinstance(value, str):
        return f'"{_escape(value)}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')

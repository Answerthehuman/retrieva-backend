"""Field value cache backed by an injected Redis client."""
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FieldCache:
    """
    Thin Redis cache for filterable field definitions.

    The caller creates and owns the Redis client. This class reads/writes
    a single JSON key that stores the full field list so the node can skip
    re-querying Milvus on every call.

    Args:
        redis_client: Any redis.Redis-compatible client.
        key: Redis key to store the serialised field list under.

    Usage (ingestion pipeline — add new example values after upsert):
        cache = FieldCache(redis_client, "myapp:fields:compass_docs")
        cache.update("file_name", ["new_doc.pdf"])
    """

    def __init__(self, redis_client, key: str):
        self._r = redis_client
        self._key = key

    def get(self) -> Optional[List[Dict[str, Any]]]:
        """Return the cached field list, or None on miss/error."""
        try:
            raw = self._r.get(self._key)
            if raw:
                return json.loads(raw)
        except Exception as e:
            logger.warning(f"FieldCache.get failed: {e}")
        return None

    def set(self, fields: List[Dict[str, Any]]) -> None:
        """Write the full field list to cache."""
        try:
            self._r.set(self._key, json.dumps(fields))
        except Exception as e:
            logger.warning(f"FieldCache.set failed: {e}")

    def update(self, field_name: str, new_values: List[str]) -> None:
        """
        Merge new example values into the cached entry for one field.
        Safe to call from ingestion pipelines after upserting documents.
        """
        try:
            fields = self.get() or []
            for field in fields:
                if field["name"] == field_name:
                    existing = set(field.get("examples", []))
                    existing.update(new_values)
                    field["examples"] = sorted(existing)
                    break
            self.set(fields)
        except Exception as e:
            logger.warning(f"FieldCache.update failed: {e}")

    def clear(self) -> None:
        """Delete the cached entry."""
        try:
            self._r.delete(self._key)
        except Exception as e:
            logger.warning(f"FieldCache.clear failed: {e}")

"""Local file-based and Redis-cached BM25 vocabulary store."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Config
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "bm25_stats"
REDIS_KEY_PREFIX = "retrieva:bm25_stats"
CACHE_TTL = 3600  # 1 hour


def save_bm25_stats(collection_name: str, stats: dict[str, Any], redis_client=None) -> bool:
    """
    Save BM25 stats to local JSON file (persistent) and optionally cache in Redis.
    """
    try:
        # 1. Save to local filesystem (source of truth)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        file_path = DATA_DIR / f"bm25_stats_{collection_name}.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        logger.info(
            f"Saved BM25 stats locally to {file_path} ({len(stats.get('vocab', {}))} tokens)"
        )

        # 2. Optionally update Redis cache
        if redis_client is not None:
            try:
                redis_client.setex(
                    f"{REDIS_KEY_PREFIX}:{collection_name}", CACHE_TTL, json.dumps(stats)
                )
                logger.info(f"Cached BM25 stats in Redis for {collection_name}")
            except Exception as e:
                logger.warning(f"Failed to cache BM25 stats in Redis: {e}")

        return True
    except Exception as e:
        logger.error(f"Failed to save BM25 stats: {e}")
        return False


def load_bm25_stats(collection_name: str, redis_client=None) -> dict[str, Any] | None:
    """
    Load BM25 stats from Redis cache, fallback to local JSON file.
    """
    # 1. Try Redis cache first if client is provided
    if redis_client is not None:
        try:
            cached = redis_client.get(f"{REDIS_KEY_PREFIX}:{collection_name}")
            if cached:
                logger.debug(f"BM25 cache hit for {collection_name}")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis cache query failed: {e}")

    # 2. Cache miss or no Redis client -> load from local filesystem
    stats = load_bm25_stats_from_disk(collection_name)

    # 3. Populate Redis cache if we got stats and redis client is active
    if stats and redis_client is not None:
        try:
            redis_client.setex(
                f"{REDIS_KEY_PREFIX}:{collection_name}", CACHE_TTL, json.dumps(stats)
            )
        except Exception:
            pass

    return stats


def load_bm25_stats_from_disk(collection_name: str) -> dict[str, Any] | None:
    """
    Load BM25 stats directly from local filesystem (source of truth).
    """
    try:
        # Check if comma-separated (fallback for multi-collections)
        collections = [c.strip() for c in collection_name.split(",") if c.strip()]
        if not collections:
            return None

        # Try the first valid stats file
        for col in collections:
            file_path = DATA_DIR / f"bm25_stats_{col}.json"
            if file_path.exists():
                with open(file_path, encoding="utf-8") as f:
                    stats = json.load(f)
                logger.info(
                    f"Loaded BM25 stats from disk: {file_path} ({len(stats.get('vocab', {}))} tokens)"
                )
                return stats
        return None
    except Exception as e:
        logger.error(f"Failed to load BM25 stats from disk: {e}")
        return None

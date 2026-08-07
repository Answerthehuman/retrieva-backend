"""CallableBackend — wraps save_fn / load_fn callables into the MemoryBackend protocol."""
import asyncio
import inspect
import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class CallableBackend:
    """
    Wraps a pair of callables into a standard save/load interface.

    Use this when you want full control over storage — pass any save/load
    functions and the system will call them.
    Supports both sync and async callables.

    Args:
        save_fn: Callable ``(turn: Dict) -> None``. Sync or async.
        load_fn: Callable ``(session_id: str, max_turns: int) -> List[Dict]``.
            Sync or async. Return [] on the first turn.
    """

    def __init__(self, *, save_fn: Callable, load_fn: Callable):
        self._save_fn = save_fn
        self._load_fn = load_fn

    @staticmethod
    async def _call(fn: Callable, *args, **kwargs) -> Any:
        if inspect.iscoroutinefunction(fn):
            return await fn(*args, **kwargs)
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def save(self, turn: Dict[str, Any]) -> None:
        await self._call(self._save_fn, turn)

    async def load(self, session_id: str, max_turns: int) -> List[Dict[str, Any]]:
        result = await self._call(self._load_fn, session_id, max_turns)
        return result or []

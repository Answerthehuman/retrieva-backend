import logging
from typing import Any, Dict, List

from .backend import CallableBackend

from core.db.database import SessionLocal
from core.db.models.chat_message import ChatMessage

logger = logging.getLogger(__name__)


async def _save_turn(turn: Dict[str, Any]) -> None:
    """Save a memory turn to the database as a ChatMessage."""
    db = SessionLocal()
    try:
        msg = ChatMessage(
            session_id=turn["session_id"],
            message_type="memory_turn",
            content=turn.get("summary", ""),
            # We can use metadata to store the raw exchange if we add a metadata JSON column,
            # but for now we just store the summary in the content field.
        )
        db.add(msg)
        db.commit()
    except Exception as e:
        logger.error(f"Failed to save memory turn: {e}")
        db.rollback()
    finally:
        db.close()


async def _load_turns(session_id: str, max_turns: int) -> List[Dict[str, Any]]:
    """Load recent memory turns from the database."""
    db = SessionLocal()
    try:
        # Load memory_turn messages ordered by created_at desc, limit max_turns, then reverse
        msgs = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id, ChatMessage.message_type == "memory_turn")
            .order_by(ChatMessage.created_at.desc())
            .limit(max_turns)
            .all()
        )
        
        turns = []
        # Re-reverse to chronological order
        for i, msg in enumerate(reversed(msgs)):
            turns.append({
                "session_id": session_id,
                "turn_index": i,
                "summary": msg.content,
                "timestamp": msg.created_at.isoformat() if msg.created_at else "",
            })
        return turns
    except Exception as e:
        logger.error(f"Failed to load memory turns: {e}")
        return []
    finally:
        db.close()


def get_memory_backend() -> CallableBackend:
    """Return a CallableBackend configured with SQLAlchemy save/load functions."""
    return CallableBackend(save_fn=_save_turn, load_fn=_load_turns)

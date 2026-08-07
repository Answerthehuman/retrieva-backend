import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from core.db.database import SessionLocal, get_db
from core.utils.errors import friendly_error
from core.db.models.chat_message import ChatMessage
from core.db.models.session import Session as SessionModel
from services.rag.pipeline.rag_pipeline import RAGPipeline
from shared.schemas.chat import (
    ChatMessageResponse,
    CreateSessionRequest,
    SendMessageRequest,
    SessionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat")

# How many prior messages are replayed into the LLM context.
HISTORY_LIMIT = 10


def _to_message_response(m: ChatMessage) -> ChatMessageResponse:
    return ChatMessageResponse(
        id=m.id,
        role=m.message_type,
        content=m.content,
        sources=m.sources,
        created_at=m.created_at.isoformat(),
    )


@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(request: CreateSessionRequest, db: Session = Depends(get_db)):
    session_id = str(uuid.uuid4())
    session = SessionModel(
        session_id=session_id,
        user_id=request.user_id,
        email=request.email,
        status="created",
        first_question=request.first_question,
        created_at=datetime.utcnow(),
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    return SessionResponse(
        session_id=session.session_id,
        user_id=session.user_id,
        email=session.email,
        status=session.status,
        created_at=session.created_at.isoformat(),
        first_question=session.first_question,
    )


@router.get("/sessions/{session_id}/messages", response_model=List[ChatMessageResponse])
def list_messages(session_id: str, db: Session = Depends(get_db)):
    """Full transcript for a session, including the sources behind each answer."""
    session = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
        .all()
    )
    return [_to_message_response(m) for m in messages]


@router.post("/sessions/{session_id}/messages")
async def stream_message(session_id: str, request: SendMessageRequest):
    """Stream an agent answer as SSE, persisting both turns.

    All database work runs through asyncio.to_thread: SQLAlchemy's session API
    is synchronous, and blocking it here would stall every other in-flight
    stream on the same event loop.
    """

    def _load_context():
        """Fetch history, then persist the incoming user turn.

        History is read *before* the new message is written so the current turn
        is not replayed twice — once as history and again as the query.
        """
        db = SessionLocal()
        try:
            session = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
            if not session:
                return None

            # Newest-first with a LIMIT, then reversed — avoids loading an
            # entire (unbounded) transcript just to keep the last few turns.
            recent = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
                .limit(HISTORY_LIMIT)
                .all()
            )
            history = [
                {"role": m.message_type, "content": m.content} for m in reversed(recent)
            ]

            db.add(
                ChatMessage(
                    session_id=session_id,
                    message_type="user",
                    content=request.message,
                )
            )
            db.commit()
            return history
        finally:
            db.close()

    chat_history = await asyncio.to_thread(_load_context)
    if chat_history is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    def _save_answer(content: str, sources: list):
        db = SessionLocal()
        try:
            db.add(
                ChatMessage(
                    session_id=session_id,
                    message_type="assistant",
                    content=content,
                    sources=sources or None,
                )
            )
            db.commit()
        except Exception as e:
            logger.error("Failed to save assistant message: %s", e)
        finally:
            db.close()

    async def event_generator():
        chunks: List[str] = []
        sources: List[dict] = []

        try:
            # Constructed inside the try: building the pipeline resolves the LLM
            # and embedding providers, which raises when credentials are absent.
            # Outside, that exception would kill the generator before its first
            # yield and hand the client an empty 200 with no explanation.
            pipeline = RAGPipeline()

            async for sse_string in pipeline.query(
                query_text=request.message,
                collection_name=request.collection_name,
                chat_history=chat_history,
                filters=request.filters,
                mode=request.mode,
                # Groups every turn of this conversation under one Langfuse
                # session instead of scattering them as unrelated traces.
                session_id=session_id,
                user_id=request.email,
            ):
                yield sse_string

                # Mirror the stream into memory so the finished answer and the
                # documents it was grounded in can be persisted together.
                if not sse_string.startswith("data: "):
                    continue
                try:
                    payload = json.loads(sse_string[6:].strip())
                except json.JSONDecodeError:
                    continue

                if payload.get("type") == "token" and "content" in payload:
                    chunks.append(payload["content"])
                elif payload.get("format") == "retrieval_complete":
                    for doc in payload.get("documents", []):
                        if doc not in sources:
                            sources.append(doc)

        except Exception as e:
            logger.error("Agent stream failed: %s", e, exc_info=True)
            yield f'data: {json.dumps({"type": "event", "format": "error", "message": friendly_error(e)})}\n\n'

        if chunks:
            await asyncio.to_thread(_save_answer, "".join(chunks), sources)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )

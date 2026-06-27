from datetime import datetime
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from core.db.database import get_db
from core.db.models.chat_message import ChatMessage
from core.db.models.session import Session as SessionModel
from core.utils.sse import format_sse_event
from services.rag.pipeline.rag_pipeline import RAGPipeline
from shared.schemas.chat import (
    CreateSessionRequest,
    SessionResponse,
    SendMessageRequest,
)

router = APIRouter(prefix="/chat")


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


@router.post("/sessions/{session_id}/messages")
async def stream_message(session_id: str, request: SendMessageRequest, db: Session = Depends(get_db)):
    session = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    user_message = ChatMessage(
        session_id=session_id,
        message_type="user",
        content=request.message,
    )
    db.add(user_message)
    db.commit()

    pipeline = RAGPipeline()
    
    # Load recent chat history (e.g. last 10 messages)
    history_msgs = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    chat_history = [{"role": m.message_type, "content": m.content} for m in history_msgs[-10:]]

    async def event_generator():
        full_response = []
        
        async for sse_string in pipeline.query(
            query_text=request.message,
            chat_history=chat_history,
            filters=request.filters,
        ):
            yield sse_string
            
            # If this is a token event, we want to accumulate it for the DB save
            import json
            try:
                if sse_string.startswith("data: "):
                    payload_str = sse_string[6:].strip()
                    payload = json.loads(payload_str)
                    if payload.get("type") == "token" and "content" in payload:
                        full_response.append(payload["content"])
            except Exception as e:
                # Log but don't fail the stream
                import logging
                logging.getLogger(__name__).warning("Error accumulating token for DB save: %s", e)

        # Save assistant's response to database
        if full_response:
            assistant_message = ChatMessage(
                session_id=session_id,
                message_type="assistant",
                content="".join(full_response),
            )
            # Use a fresh session for the background save to avoid concurrency issues with generator
            db_save = next(get_db())
            try:
                db_save.add(assistant_message)
                db_save.commit()
            except Exception as e:
                import logging
                logging.getLogger(__name__).error("Failed to save assistant message: %s", e)
            finally:
                db_save.close()

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

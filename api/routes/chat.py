from datetime import datetime
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ...core.db.database import get_db
from ...core.db.models.chat_message import ChatMessage
from ...core.db.models.session import Session as SessionModel
from ...core.utils.sse import format_sse_event
from ...services.rag.pipeline.rag_pipeline import RAGPipeline
from ...shared.schemas.chat import (
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

    async def event_generator():
        yield format_sse_event("event", {"format": "retrieval_start"})
        result = pipeline.query(request.message)
        yield format_sse_event("event", {"format": "retrieval_complete", "documents": result["documents"]})
        yield format_sse_event("event", {"format": "generation_start"})

        for chunk in result["response"]:
            yield format_sse_event("token", {"format": "markdown", "content": chunk})

        yield format_sse_event("event", {"format": "generation_complete"})

        assistant_message = ChatMessage(
            session_id=session_id,
            message_type="assistant",
            content="".join(result["response"]),
        )
        db.add(assistant_message)
        db.commit()

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

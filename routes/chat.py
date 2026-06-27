import uuid
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from ..core.db import SessionLocal
from ..core.models import ChatMessage, ChatSession
from ..libs.sse import format_sse_event
from ..services.rag.pipeline import RAGPipeline

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/sessions")
def create_session():
    session_id = str(uuid.uuid4())
    with SessionLocal() as db:
        chat_session = ChatSession(id=session_id)
        db.add(chat_session)
        db.commit()
    return {"session_id": session_id}


@router.post("/sessions/{session_id}/messages")
async def stream_message(session_id: str, request: Request):
    with SessionLocal() as db:
        session = db.get(ChatSession, session_id)
        if session is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

        payload = await request.json()
        user_text = payload.get("message")
        if not user_text:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="message is required")

        user_message = ChatMessage(session_id=session_id, role="user", content=user_text)
        db.add(user_message)
        db.commit()

    pipeline = RAGPipeline()

    async def event_generator():
        yield format_sse_event("event", {"format": "retrieval_start"})
        result = pipeline.query(user_text)
        yield format_sse_event("event", {"format": "retrieval_complete", "documents": result["documents"]})
        yield format_sse_event("event", {"format": "generation_start"})

        for chunk in result["response"]:
            yield format_sse_event("token", {"format": "markdown", "content": chunk})

        yield format_sse_event("event", {"format": "generation_complete"})

        with SessionLocal() as db_session:
            assistant_message = ChatMessage(session_id=session_id, role="assistant", content="".join(result["response"]))
            db_session.add(assistant_message)
            db_session.commit()

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

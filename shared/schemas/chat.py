from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    email: Optional[str] = Field(None, description="Optional email for metadata filtering")
    first_question: Optional[str] = Field(None, description="Optional first question to initialize session")


class SessionResponse(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    email: Optional[str] = None
    status: str
    created_at: str
    first_question: Optional[str] = None

    class Config:
        from_attributes = True


class SendMessageRequest(BaseModel):
    message: str = Field(..., description="User message/question", min_length=1)
    email: Optional[str] = Field(None, description="Optional email for metadata filtering")
    filters: Optional[str] = Field(None, description="Optional filter expression")
    collection_name: Optional[str] = Field(
        None, description="Milvus collection to search; falls back to the server default"
    )


class ChatMessageResponse(BaseModel):
    id: int
    role: str = Field(..., description='"user" or "assistant"')
    content: str
    sources: Optional[List[Dict[str, Any]]] = None
    created_at: str

    class Config:
        from_attributes = True

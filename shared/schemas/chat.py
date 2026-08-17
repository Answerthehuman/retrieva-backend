from typing import Any

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    user_id: str | None = Field(None, description="Optional user identifier")
    email: str | None = Field(None, description="Optional email for metadata filtering")
    first_question: str | None = Field(
        None, description="Optional first question to initialize session"
    )


class SessionResponse(BaseModel):
    session_id: str
    user_id: str | None = None
    email: str | None = None
    status: str
    created_at: str
    first_question: str | None = None

    class Config:
        from_attributes = True


class SendMessageRequest(BaseModel):
    message: str = Field(..., description="User message/question", min_length=1)
    email: str | None = Field(None, description="Optional email for metadata filtering")
    filters: str | None = Field(None, description="Optional filter expression")
    collection_name: str | None = Field(
        None, description="Milvus collection to search; falls back to the server default"
    )
    mode: str | None = Field(
        None,
        description=(
            "Action mode shaping how the agent searches and structures its answer: "
            "summarise | insights | analyse | explain. Omit for normal chat. "
            "An unrecognised value degrades to normal chat rather than erroring."
        ),
    )


class ChatMessageResponse(BaseModel):
    id: int
    role: str = Field(..., description='"user" or "assistant"')
    content: str
    sources: list[dict[str, Any]] | None = None
    created_at: str

    class Config:
        from_attributes = True

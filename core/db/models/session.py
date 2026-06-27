from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text
from sqlalchemy.orm import relationship

from ..database import Base


class Session(Base):
    __tablename__ = "sessions"

    session_id = Column(String(255), primary_key=True, index=True)
    user_id = Column(String(255), nullable=True, index=True)
    email = Column(String(255), nullable=True, index=True)
    status = Column(String(50), default="created", nullable=False)
    first_question = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    chat_messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

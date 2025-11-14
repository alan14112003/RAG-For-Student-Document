from datetime import datetime
from typing import Optional
from sqlalchemy import String, Text, DateTime, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.app.models.base import Base


class ChatSession(Base):
    """
    Phiên chat của người dùng.
    Mỗi user có thể có nhiều sessions, nhưng hiện tại thiết kế cho 1 session chính.
    """
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Session ID"
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User sở hữu session"
    )
    session_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Tên phiên chat (optional)"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        comment="Thời gian tạo session"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
        comment="Thời gian cập nhật cuối"
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="chat_sessions")
    messages: Mapped[list["ChatMessage"]] = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at"
    )

    def __repr__(self) -> str:
        return f"<ChatSession(id={self.id}, user_id={self.user_id})>"


class ChatMessage(Base):
    """
    Tin nhắn trong phiên chat.
    """
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Message ID"
    )
    session_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Session chứa message"
    )
    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="Role: 'user' hoặc 'assistant'"
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Nội dung tin nhắn"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        comment="Thời gian tạo message"
    )

    # Additional metadata (optional)
    model_used: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Model AI được sử dụng (cho assistant messages)"
    )
    tokens_used: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Số tokens sử dụng"
    )

    # Relationships
    session: Mapped["ChatSession"] = relationship("ChatSession", back_populates="messages")

    def __repr__(self) -> str:
        return f"<ChatMessage(id={self.id}, session_id={self.session_id}, role={self.role})>"

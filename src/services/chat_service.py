from __future__ import annotations

import logging
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.app.models.chat import ChatSession, ChatMessage
from src.app.schemas.chat import (
    ChatMessageCreate,
    ChatMessageResponse,
    ChatSessionCreate,
    ChatSessionResponse,
    ChatHistoryResponse
)

logger = logging.getLogger(__name__)


class ChatService:
    """Service để quản lý lịch sử chat của người dùng"""

    def __init__(self) -> None:
        logger.info("Chat service initialized")

    async def get_or_create_session(self, db: AsyncSession, user_id: str, session_name: Optional[str] = None) -> ChatSession:
        """Lấy hoặc tạo session chat cho user. Hiện tại mỗi user có 1 session."""
        # Tìm session hiện tại của user
        stmt = select(ChatSession).where(ChatSession.user_id == user_id)
        result = await db.execute(stmt)
        session = result.scalar_one_or_none()

        if session is None:
            # Tạo session mới
            session = ChatSession(
                user_id=user_id,
                session_name=session_name or f"Chat của {user_id}"
            )
            db.add(session)
            await db.commit()
            await db.refresh(session)
            logger.info(f"Created new chat session for user {user_id}")

        return session

    async def add_message(
        self,
        db: AsyncSession,
        session_id: int,
        role: str,
        content: str,
        model_used: Optional[str] = None,
        tokens_used: Optional[int] = None
    ) -> ChatMessage:
        """Thêm message vào session"""
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            model_used=model_used,
            tokens_used=tokens_used
        )
        db.add(message)
        await db.commit()
        await db.refresh(message)
        logger.debug(f"Added {role} message to session {session_id}")
        return message

    async def save_user_message(self, db: AsyncSession, user_id: str, question: str) -> ChatMessage:
        """Lưu câu hỏi của user"""
        session = await self.get_or_create_session(db, user_id)
        return await self.add_message(db, session.id, "user", question)

    async def save_assistant_response(
        self,
        db: AsyncSession,
        user_id: str,
        response: str,
        model_used: Optional[str] = None,
        tokens_used: Optional[int] = None
    ) -> ChatMessage:
        """Lưu câu trả lời của assistant"""
        session = await self.get_or_create_session(db, user_id)
        return await self.add_message(db, session.id, "assistant", response, model_used, tokens_used)

    async def get_chat_history(self, db: AsyncSession, user_id: str) -> Optional[ChatHistoryResponse]:
        """Lấy lịch sử chat của user"""
        stmt = select(ChatSession).where(ChatSession.user_id == user_id)
        result = await db.execute(stmt)
        session = result.scalar_one_or_none()

        if session is None:
            return None

        # Lấy messages của session
        stmt = select(ChatMessage).where(ChatMessage.session_id == session.id).order_by(ChatMessage.created_at)
        result = await db.execute(stmt)
        messages = result.scalars().all()

        return ChatHistoryResponse(
            session=ChatSessionResponse.from_orm(session),
            messages=[ChatMessageResponse.from_orm(msg) for msg in messages]
        )

    async def get_recent_messages(self, db: AsyncSession, user_id: str, limit: int = 50) -> List[ChatMessageResponse]:
        """Lấy các messages gần đây của user"""
        session = await self.get_or_create_session(db, user_id)

        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session.id)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
        )
        result = await db.execute(stmt)
        messages = result.scalars().all()

        # Reverse to get chronological order
        return [ChatMessageResponse.from_orm(msg) for msg in reversed(messages)]

    async def delete_chat_history(self, db: AsyncSession, user_id: str) -> bool:
        """Xóa lịch sử chat của user"""
        stmt = select(ChatSession).where(ChatSession.user_id == user_id)
        result = await db.execute(stmt)
        session = result.scalar_one_or_none()

        if session is None:
            return False

        # Delete session (cascade will delete messages)
        await db.delete(session)
        await db.commit()
        logger.info(f"Deleted chat history for user {user_id}")
        return True

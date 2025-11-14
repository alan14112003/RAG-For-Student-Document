# src/app/api/chat.py
import logging
from typing import Optional, Callable
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.app.models.base import get_db
from src.app.models import User
from src.app.schemas.chat import ChatHistoryResponse
from src.app.utils.auth import get_current_active_user
from src.services.chat_service import ChatService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])

# Dependency providers that can be configured during app startup
_chat_service_provider: Optional[Callable[[], ChatService]] = None


def set_chat_service_provider(provider: Callable[[], ChatService]) -> None:
    """Register the callable that supplies the Chat service instance."""
    global _chat_service_provider
    _chat_service_provider = provider


def get_chat_service() -> ChatService:
    """Dependency để lấy Chat service instance."""
    if _chat_service_provider is None:
        raise RuntimeError("Chat service provider has not been configured")
    service = _chat_service_provider()
    if service is None:
        raise RuntimeError("Configured Chat service provider returned None")
    return service


@router.get(
    "/history",
    response_model=ChatHistoryResponse,
    summary="Get chat history",
    description="Lấy lịch sử chat của user hiện tại"
)
async def get_chat_history(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    chat_service: ChatService = Depends(get_chat_service),
):
    """Lấy lịch sử chat của user"""

    logger.info(f"User {current_user.id} requesting chat history")

    try:
        history = await chat_service.get_chat_history(db, current_user.id)

        if history is None:
            # Trả về empty history nếu chưa có
            return ChatHistoryResponse(
                session=None,
                messages=[]
            )

        logger.info(f"Found {len(history.messages)} messages in chat history for user {current_user.id}")

        return history

    except Exception as e:
        logger.error(f"Failed to get chat history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get chat history: {str(e)}"
        )


@router.delete(
    "/history",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete chat history",
    description="Xóa toàn bộ lịch sử chat của user hiện tại"
)
async def delete_chat_history(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    chat_service: ChatService = Depends(get_chat_service),
):
    """Xóa lịch sử chat của user"""

    logger.info(f"User {current_user.id} deleting chat history")

    try:
        success = await chat_service.delete_chat_history(db, current_user.id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat history not found"
            )

        logger.info(f"Successfully deleted chat history for user {current_user.id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete chat history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete chat history: {str(e)}"
        )

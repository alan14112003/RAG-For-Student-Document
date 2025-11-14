# src/app/schemas/chat.py
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class ChatMessageBase(BaseModel):
    role: str = Field(..., description="Role of the message: 'user' or 'assistant'")
    content: str = Field(..., description="Content of the message")


class ChatMessageCreate(ChatMessageBase):
    pass


class ChatMessageResponse(ChatMessageBase):
    id: int
    session_id: int
    created_at: datetime
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None

    class Config:
        from_attributes = True


class ChatSessionBase(BaseModel):
    session_name: Optional[str] = Field(None, description="Optional name for the chat session")


class ChatSessionCreate(ChatSessionBase):
    pass


class ChatSessionResponse(ChatSessionBase):
    id: int
    user_id: str
    created_at: datetime
    updated_at: datetime
    messages: List[ChatMessageResponse] = []

    class Config:
        from_attributes = True


class ChatHistoryResponse(BaseModel):
    """Response for getting chat history of a user"""
    session: ChatSessionResponse
    messages: List[ChatMessageResponse]

    class Config:
        from_attributes = True


class ChatQueryRequest(BaseModel):
    """Request for querying documents with chat history saving"""
    query: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(5, ge=1, le=20, description="Number of similar chunks to retrieve")
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(1000, ge=1, le=4000)


class ChatQueryResponse(BaseModel):
    """Response for document query with chat history"""
    query: str
    answer: dict  # Same as existing QueryWithLLMResult.answer
    sources: List[dict]  # Same as existing sources
    context_used: str
    model: str
    retrieved_chunks: int
    message_id: Optional[int] = Field(None, description="ID of saved user message")
    response_message_id: Optional[int] = Field(None, description="ID of saved assistant response")

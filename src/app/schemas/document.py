# src/app/schemas/document.py
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ChunkResponse(BaseModel):
    chunk_index: int
    content: str
    start_char: int
    end_char: int
    metadata: Optional[Dict[str, Any]] = Field(default=None, alias="additional_metadata")

    class Config:
        from_attributes = True


class DocumentUploadResponse(BaseModel):
    document_id: str
    file_name: str
    file_size: int
    content_length: int
    chunk_count: int
    collection_name: str
    created_at: datetime


class DocumentResponse(BaseModel):
    id: str
    user_id: str
    file_name: str
    file_size: int
    mime_type: Optional[str] = None
    content_length: int
    chunk_count: int
    collection_name: str
    metadata: Optional[Dict[str, Any]] = Field(default=None, alias="additional_metadata")
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DocumentDetailResponse(DocumentResponse):
    full_content: str
    chunks: List[ChunkResponse] = []


class DocumentListResponse(BaseModel):
    total: int
    documents: List[DocumentResponse]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    k: int = Field(5, ge=1, le=20)
    metadata_filter: Optional[Dict[str, Any]] = None


class SearchResultChunk(BaseModel):
    chunk_index: int
    start_char: int
    end_char: int
    content: str
    score: float
    file_name: str


class SearchResultDocument(BaseModel):
    document_id: str
    chunks: List[SearchResultChunk]


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultDocument]
    total_chunks: int


class HighlightRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    document_id: str
    k: int = Field(5, ge=1, le=20)


class HighlightResponse(BaseModel):
    document_id: str
    document_name: str
    full_content: str
    highlights: List[SearchResultChunk]


class SourceChunk(BaseModel):
    source_id: str
    document_id: str
    file_name: str
    chunk_index: int
    content: str
    score: float
    start_char: int
    end_char: int
    source_path: Optional[str] = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    k: int = Field(5, ge=1, le=20)
    metadata_filter: Optional[Dict[str, Any]] = None
    temperature: float = Field(0.1, ge=0.0, le=1.0)
    max_tokens: int = Field(10000, ge=1, le=20000)


class AnswerReference(BaseModel):
    source_id: str
    document_id: str
    file_name: str
    chunk_index: int
    score: float
    snippet: str
    start_char: int
    end_char: int
    source_path: Optional[str] = None
    explanation: Optional[str] = None


class AnswerValue(BaseModel):
    content: str
    references: List[AnswerReference] = Field(default_factory=list)


class QueryResponse(BaseModel):
    query: str
    answer: AnswerValue
    sources: List[SourceChunk] = Field(default_factory=list)
    context_used: str
    model: str

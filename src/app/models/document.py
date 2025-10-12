from datetime import datetime
from typing import Optional
from sqlalchemy import String, Text, Integer, DateTime, ForeignKey, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.app.models.base import Base


class Document(Base):
    """
    Lưu trữ thông tin tài liệu đã được xử lý.
    Mỗi document có nhiều chunks với vị trí được track để highlight.
    """
    __tablename__ = "documents"

    # Primary info
    id: Mapped[str] = mapped_column(
        String(36), 
        primary_key=True, 
        comment="Document ID (UUIDv4) từ RAG service"
    )
    user_id: Mapped[str] = mapped_column(
        String(36), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False, 
        index=True,
        comment="Owner của document"
    )
    
    # File information
    file_name: Mapped[str] = mapped_column(String(255), nullable=False, comment="Tên file gốc")
    file_path: Mapped[str] = mapped_column(String(500), nullable=False, comment="Đường dẫn lưu file")
    file_size: Mapped[int] = mapped_column(Integer, nullable=False, comment="Kích thước file (bytes)")
    mime_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, comment="MIME type")
    
    # Content information
    full_content: Mapped[str] = mapped_column(
        Text, 
        nullable=False, 
        comment="Nội dung đầy đủ của tài liệu (để hiển thị và highlight)"
    )
    content_length: Mapped[int] = mapped_column(
        Integer, 
        nullable=False, 
        comment="Độ dài nội dung (characters)"
    )
    chunk_count: Mapped[int] = mapped_column(
        Integer, 
        nullable=False, 
        comment="Số lượng chunks đã cắt"
    )
    
    # Vector store information
    collection_name: Mapped[str] = mapped_column(
        String(255), 
        nullable=False, 
        index=True,
        comment="Tên collection trong Qdrant"
    )
    
    # Additional metadata
    additional_metadata: Mapped[Optional[dict]] = mapped_column(
        "metadata",
        JSON, 
        nullable=True,
        comment="Metadata bổ sung (tags, category, etc.)"
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow, 
        nullable=False,
        comment="Thời gian upload"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow, 
        onupdate=datetime.utcnow, 
        nullable=False,
        comment="Thời gian cập nhật"
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="documents")
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk", 
        back_populates="document", 
        cascade="all, delete-orphan",
        order_by="DocumentChunk.chunk_index"
    )

    # Indexes
    __table_args__ = (
        Index("ix_documents_user_created", "user_id", "created_at"),
        Index("ix_documents_collection", "collection_name"),
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, file_name={self.file_name}, user_id={self.user_id})>"


class DocumentChunk(Base):
    """
    Lưu trữ từng chunk của document với vị trí chính xác.
    Dùng start_char và end_char để highlight text trong frontend.
    """
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(
        Integer, 
        primary_key=True, 
        autoincrement=True,
        comment="Auto-increment ID"
    )
    document_id: Mapped[str] = mapped_column(
        String(36), 
        ForeignKey("documents.id", ondelete="CASCADE"), 
        nullable=False, 
        index=True,
        comment="Document ID mà chunk này thuộc về"
    )
    
    # Chunk information
    chunk_index: Mapped[int] = mapped_column(
        Integer, 
        nullable=False,
        comment="Thứ tự chunk trong document (0, 1, 2, ...)"
    )
    content: Mapped[str] = mapped_column(
        Text, 
        nullable=False,
        comment="Nội dung của chunk"
    )
    
    # Position tracking for highlighting
    start_char: Mapped[int] = mapped_column(
        Integer, 
        nullable=False,
        comment="Vị trí bắt đầu trong full_content (character index)"
    )
    end_char: Mapped[int] = mapped_column(
        Integer, 
        nullable=False,
        comment="Vị trí kết thúc trong full_content (character index)"
    )
    
    # Additional metadata
    additional_metadata: Mapped[Optional[dict]] = mapped_column(
        "metadata",
        JSON, 
        nullable=True,
        comment="Metadata của chunk (từ RAG service)"
    )
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow, 
        nullable=False,
        comment="Thời gian tạo chunk"
    )

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    # Indexes
    __table_args__ = (
        Index("ix_chunks_document_index", "document_id", "chunk_index"),
        Index("ix_chunks_position", "document_id", "start_char", "end_char"),
    )

    def __repr__(self) -> str:
        return f"<DocumentChunk(id={self.id}, document_id={self.document_id}, chunk_index={self.chunk_index})>"

    @property
    def length(self) -> int:
        """Độ dài của chunk"""
        return self.end_char - self.start_char


# src/app/api/documents.py
import logging
import io
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from src.app.models.base import get_db
from src.app.models import User, Document, DocumentChunk
from src.app.schemas.document import (
    DocumentUploadResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
    SearchResultDocument,
    SearchResultChunk,
    HighlightRequest,
    HighlightResponse,
    SourceChunk,
)
from src.app.utils.auth import get_current_active_user
from src.services.minio_service import MinIOService
from src.services.rag_service.rag_service import RagService
from src.services.llm_service import LLMService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])

# Configuration
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".doc", ".docx", ".csv"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Dependency providers that can be configured during app startup
_rag_service_provider: Optional[Callable[[], RagService]] = None
_minio_service_provider: Optional[Callable[[], MinIOService]] = None
# Thêm dependency cho LLM service
_llm_service_provider: Optional[Callable[[], LLMService]] = None


def set_rag_service_provider(provider: Callable[[], RagService]) -> None:
    """Register the callable that supplies the RAG service instance."""
    global _rag_service_provider
    _rag_service_provider = provider


def set_minio_service_provider(provider: Callable[[], MinIOService]) -> None:
    """Register the callable that supplies the MinIO service instance."""
    global _minio_service_provider
    _minio_service_provider = provider

def set_llm_service_provider(provider: Callable[[], LLMService]) -> None:
    """Register the callable that supplies the LLM service instance."""
    global _llm_service_provider
    _llm_service_provider = provider


def get_llm_service() -> LLMService:
    """Dependency để lấy LLM service instance."""
    if _llm_service_provider is None:
        raise RuntimeError("LLM service provider has not been configured")
    service = _llm_service_provider()
    if service is None:
        raise RuntimeError("Configured LLM service provider returned None")
    return service



def get_rag_service() -> RagService:
    """Dependency để lấy RAG service instance."""
    if _rag_service_provider is None:
        raise RuntimeError("RAG service provider has not been configured")
    service = _rag_service_provider()
    if service is None:
        raise RuntimeError("Configured RAG service provider returned None")
    return service


def get_minio_service() -> MinIOService:
    """Dependency để lấy MinIO service instance."""
    if _minio_service_provider is None:
        raise RuntimeError("MinIO service provider has not been configured")
    service = _minio_service_provider()
    if service is None:
        raise RuntimeError("Configured MinIO service provider returned None")
    return service


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload document",
    description="Upload và xử lý tài liệu. File sẽ được lưu vào MinIO, cắt thành chunks và lưu vào vector store."
)
async def upload_document(
    file: UploadFile = File(..., description="File to upload"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    rag_service: RagService = Depends(get_rag_service),
    minio_service: MinIOService = Depends(get_minio_service),
):
    """Upload và xử lý tài liệu mới với MinIO storage"""
    
    logger.info(f"User {current_user.id} uploading file: {file.filename}")
    
    # 1. Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type '{file_ext}' not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # 2. Check if file already exists for this user
    existing_doc = await db.execute(
        select(Document)
        .where(
            Document.user_id == current_user.id,
            Document.file_name == file.filename
        )
    )
    existing = existing_doc.scalar_one_or_none()
    
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"File '{file.filename}' already exists."
        )
    
    # 3. Read and validate file
    content = await file.read()
    file_size = len(content)
    
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty"
        )
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large ({file_size / 1024 / 1024:.2f} MB). Maximum size: {MAX_FILE_SIZE / 1024 / 1024} MB"
        )
    
    # 4. Generate unique object name for MinIO
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    original_stem = Path(file.filename).stem
    safe_filename = "".join(c if c.isalnum() or c in "._- " else "_" for c in original_stem)
    
    # Object path: user_id/timestamp_uniqueid_filename.ext
    minio_object_name = f"{current_user.id}/{timestamp}_{unique_id}_{safe_filename}{file_ext}"
    
    # 5. Create temporary file for RAG processing
    temp_dir = Path("data/temp") / current_user.id
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / f"{unique_id}_{file.filename}"
    
    try:
        # 6. Write to temp file
        with open(temp_file, "wb") as f:
            f.write(content)
        
        logger.info(f"Temp file created: {temp_file}")
        
        # 7. Upload to MinIO
        minio_service.upload_fileobj(
            file_data=io.BytesIO(content),
            object_name=minio_object_name,
            length=file_size,
            content_type=file.content_type or "application/octet-stream",
        )
        
        logger.info(f"File uploaded to MinIO: {minio_object_name}")
        
        # 8. Process with RAG service
        summary = await rag_service.ingest_file(
            user_id=current_user.id,
            file_path=temp_file,
            metadata={
                "mime_type": file.content_type,
                "original_filename": file.filename,
                "minio_object": minio_object_name,
                "minio_bucket": minio_service.bucket_name,
            }
        )
        
        logger.info(
            f"Document processed: {summary.document_info.document_id}, "
            f"chunks: {summary.chunk_count}"
        )
        
        # 9. Save to database
        document = Document(
            id=summary.document_info.document_id,
            user_id=current_user.id,
            file_name=file.filename,
            file_path=minio_object_name,  # Store MinIO object path
            file_size=file_size,
            mime_type=file.content_type,
            full_content=summary.document_info.full_content,
            content_length=summary.document_info.content_length,
            chunk_count=summary.chunk_count,
            collection_name=summary.collection_name,
            additional_metadata={
                **summary.document_info.metadata,
                "storage": "minio",
                "bucket": minio_service.bucket_name,
                "object_name": minio_object_name,
            },
        )
        
        db.add(document)
        
        # 10. Save chunks to database
        for chunk_info in summary.document_info.chunks:
            chunk = DocumentChunk(
                document_id=document.id,
                chunk_index=chunk_info.chunk_index,
                content=chunk_info.content,
                start_char=chunk_info.start_char,
                end_char=chunk_info.end_char,
                additional_metadata=chunk_info.metadata,
            )
            db.add(chunk)
        
        await db.commit()
        await db.refresh(document)
        
        logger.info(f"Document saved to database: {document.id}")
        
        return DocumentUploadResponse(
            document_id=document.id,
            file_name=document.file_name,
            file_size=document.file_size,
            content_length=document.content_length,
            chunk_count=document.chunk_count,
            collection_name=document.collection_name,
            created_at=document.created_at,
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions (validation errors, conflicts)
        raise
    
    except Exception as e:
        logger.error(f"Failed to process document: {str(e)}", exc_info=True)
        
        # Rollback database
        await db.rollback()
        
        # Cleanup: delete from MinIO if uploaded
        try:
            if minio_service.file_exists(minio_object_name):
                minio_service.delete_file(minio_object_name)
                logger.info(f"Cleaned up MinIO object: {minio_object_name}")
        except Exception as cleanup_error:
            logger.error(f"Failed to cleanup MinIO: {cleanup_error}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}"
        )
    
    finally:
        # Always cleanup temp file
        if temp_file.exists():
            try:
                temp_file.unlink()
                logger.debug(f"Cleaned up temp file: {temp_file}")
            except Exception as cleanup_error:
                logger.error(f"Failed to cleanup temp file: {cleanup_error}")


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List documents",
    description="Lấy danh sách tài liệu của user với pagination"
)
async def list_documents(
    skip: int = Query(0, ge=0, description="Number of documents to skip"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of documents to return"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Lấy danh sách tài liệu của user"""
    
    logger.info(f"User {current_user.id} listing documents (skip={skip}, limit={limit})")
    
    # Get total count
    count_result = await db.execute(
        select(func.count(Document.id)).where(Document.user_id == current_user.id)
    )
    total = count_result.scalar() or 0
    
    # Get documents with pagination
    result = await db.execute(
        select(Document)
        .where(Document.user_id == current_user.id)
        .order_by(Document.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    documents = result.scalars().all()
    
    logger.info(f"Found {len(documents)} documents (total: {total})")
    
    return DocumentListResponse(
        total=total,
        documents=documents
    )


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Get document details",
    description="Lấy chi tiết tài liệu bao gồm full content và tất cả chunks"
)
async def get_document(
    document_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Lấy chi tiết tài liệu với tất cả chunks"""
    
    logger.info(f"User {current_user.id} getting document: {document_id}")
    
    # Get document
    result = await db.execute(
        select(Document)
        .where(Document.id == document_id, Document.user_id == current_user.id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Get chunks ordered by chunk_index
    chunks_result = await db.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
    )
    chunks = chunks_result.scalars().all()
    
    logger.info(f"Document {document_id} has {len(chunks)} chunks")
    
    return DocumentDetailResponse(
        id=document.id,
        user_id=document.user_id,
        file_name=document.file_name,
        file_size=document.file_size,
        mime_type=document.mime_type,
        content_length=document.content_length,
        chunk_count=document.chunk_count,
        collection_name=document.collection_name,
        metadata=document.additional_metadata,
        created_at=document.created_at,
        updated_at=document.updated_at,
        full_content=document.full_content,
        chunks=chunks,
    )


@router.get(
    "/{document_id}/download",
    summary="Get document download URL",
    description="Tạo presigned URL để download file từ MinIO"
)
async def get_download_url(
    document_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    minio_service: MinIOService = Depends(get_minio_service),
):
    """Tạo presigned URL để download file"""
    
    logger.info(f"User {current_user.id} requesting download URL for: {document_id}")
    
    # Get document
    result = await db.execute(
        select(Document)
        .where(Document.id == document_id, Document.user_id == current_user.id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    try:
        # Generate presigned URL (expires in 1 hour)
        from datetime import timedelta
        
        download_url = minio_service.get_file_url(
            object_name=document.file_path,
            expires=timedelta(hours=1)
        )
        
        logger.info(f"Generated download URL for document: {document_id}")
        
        return {
            "document_id": document.id,
            "file_name": document.file_name,
            "download_url": download_url,
            "expires_in_seconds": 3600,
        }
    
    except Exception as e:
        logger.error(f"Failed to generate download URL: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate download URL: {str(e)}"
        )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete document",
    description="Xóa tài liệu khỏi database, vector store và MinIO storage"
)
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    rag_service: RagService = Depends(get_rag_service),
    minio_service: MinIOService = Depends(get_minio_service),
):
    """Xóa tài liệu"""
    
    logger.info(f"User {current_user.id} deleting document: {document_id}")
    
    # Get document
    result = await db.execute(
        select(Document)
        .where(Document.id == document_id, Document.user_id == current_user.id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    try:
        # 1. Delete from vector store
        deleted_chunks = await rag_service.delete_document(current_user.id, document_id)
        logger.info(f"Deleted {deleted_chunks} chunks from vector store")
        
        # 2. Delete from MinIO
        minio_object = document.file_path
        if minio_service.file_exists(minio_object):
            minio_service.delete_file(minio_object)
            logger.info(f"Deleted file from MinIO: {minio_object}")
        else:
            logger.warning(f"File not found in MinIO: {minio_object}")
        
        # 3. Delete from database (cascade will delete chunks)
        await db.delete(document)
        await db.commit()
        
        logger.info(f"Document {document_id} deleted successfully")
    
    except Exception as e:
        logger.error(f"Failed to delete document: {str(e)}", exc_info=True)
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search documents",
    description="Tìm kiếm semantic trong tất cả tài liệu của user"
)
async def search_documents(
    search_data: SearchRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    rag_service: RagService = Depends(get_rag_service),
):
    """Tìm kiếm trong tất cả tài liệu"""
    
    logger.info(f"User {current_user.id} searching: '{search_data.query}' (k={search_data.k})")
    
    try:
        # Search in vector store
        search_results = await rag_service.search_with_highlights(
            user_id=current_user.id,
            query=search_data.query,
            k=search_data.k,
            metadata_filter=search_data.metadata_filter,
        )
        
        # Convert to response format
        results = []
        total_chunks = 0
        
        for document_id, chunks in search_results.items():
            result_chunks = [
                SearchResultChunk(
                    chunk_index=chunk["chunk_index"],
                    start_char=chunk["start_char"],
                    end_char=chunk["end_char"],
                    content=chunk["content"],
                    score=chunk["score"],
                    file_name=chunk["file_name"],
                    content_format=chunk.get("content_format", "markdown"),
                )
                for chunk in chunks
            ]
            
            results.append(
                SearchResultDocument(
                    document_id=document_id,
                    chunks=result_chunks
                )
            )
            
            total_chunks += len(result_chunks)
        
        logger.info(f"Search returned {total_chunks} chunks from {len(results)} documents")
        
        return SearchResponse(
            query=search_data.query,
            results=results,
            total_chunks=total_chunks,
        )
    
    except Exception as e:
        logger.error(f"Search failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query documents with LLM",
    description="Đặt câu hỏi và nhận câu trả lời từ LLM dựa trên tài liệu của user"
)
async def query_documents(
    query_data: QueryRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    rag_service: RagService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service),
):
    """
    Query với RAG + LLM để trả lời câu hỏi
    
    Flow:
    1. Retrieve relevant chunks từ vector store
    2. Tạo context từ các chunks
    3. Gọi LLM để generate answer
    4. Trả về answer kèm sources
    """
    
    logger.info(
        f"User {current_user.id} querying: '{query_data.query}' "
        f"(k={query_data.k}, temp={query_data.temperature})"
    )
    
    try:
        # Gọi RAG service với LLM
        result = await rag_service.query_with_llm(
            user_id=current_user.id,
            question=query_data.query,
            llm_service=llm_service,
            k=query_data.k,
            metadata_filter=query_data.metadata_filter,
            temperature=query_data.temperature,
            max_tokens=query_data.max_tokens,
        )
        
        # Convert sources sang SourceChunk schema
        sources = [
            SourceChunk(
                source_id=source["source_id"],
                document_id=source["document_id"],
                file_name=source["file_name"],
                chunk_index=source["chunk_index"],
                content=source["content"],
                score=source["score"],
                start_char=source["start_char"],
                end_char=source["end_char"],
                source_path=source.get("source_path"),
                content_format=source.get("content_format", "markdown"),
            )
            for source in result.sources
        ]
        
        logger.info(
            f"Query completed: {result.retrieved_chunks} chunks retrieved, "
            f"answer length: {len(result.answer)} chars"
        )
        
        return QueryResponse(
            query=result.query,
            answer=result.answer,
            sources=sources,
            context_used=result.context_used,
            model=result.model,
        )
    
    except Exception as e:
        logger.error(f"Query failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )


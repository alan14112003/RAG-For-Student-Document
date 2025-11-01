from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from src.services.llm_service import LLMService
from src.services.rag_service.converter import ConverterFactory
from src.services.rag_service.qdrant_storage.qdrant_storage import QdrantStorage
from src.services.text_utils import ensure_markdown

logger = logging.getLogger(__name__)


@dataclass
class ChunkInfo:
    """Thông tin về mỗi chunk được cắt ra"""
    chunk_index: int
    content: str
    start_char: int  # Vị trí bắt đầu trong tài liệu gốc
    end_char: int    # Vị trí kết thúc trong tài liệu gốc
    metadata: Dict[str, Any]


@dataclass
class DocumentInfo:
    """Thông tin tài liệu để lưu vào database"""
    document_id: str  # UUID hoặc ID duy nhất
    user_id: str
    file_name: str
    file_path: str
    full_content: str  # Nội dung đầy đủ của tài liệu
    content_length: int
    chunks: List[ChunkInfo]  # Danh sách các chunks
    metadata: Dict[str, Any]
    created_at: datetime

@dataclass
class QueryWithLLMResult:
    """Kết quả query với LLM answer"""
    query: str
    answer: Dict[str, Any]
    sources: List[Dict[str, Any]]
    context_used: str
    model: str
    retrieved_chunks: int

@dataclass
class IngestionSummary:
    """Kết quả sau khi ingest tài liệu"""
    user_id: str
    collection_name: str
    document_info: DocumentInfo  # Thông tin tài liệu để lưu vào DB
    chunk_count: int


class RagService:
    """
    Orchestrates ingestion and retrieval with per-user isolation and document tracking.
    Mỗi user có 1 collection riêng trong Qdrant.
    """

    def __init__(
        self,
        collection_prefix: str = "user_documents",
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        embedding_model: str = "mxbai-embed-large",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        recreate_collections: bool = False,
    ) -> None:
        self.collection_prefix = collection_prefix
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self._recreate_collections = recreate_collections

        # Initialize embeddings
        self._embedding = OllamaEmbeddings(model=self.embedding_model)
        self._vector_size = len(self._embedding.embed_query("__dimension_probe__"))

        # Initialize Qdrant client
        self._qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self._qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self._client = QdrantClient(url=self._qdrant_url, api_key=self._qdrant_api_key)

        # Initialize text splitter
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

        # Cache storage cho từng user
        self._storage_cache: Dict[str, QdrantStorage] = {}

        logger.info(
            "RagService initialized (prefix=%s, chunk_size=%d, overlap=%d)",
            self.collection_prefix,
            self.chunk_size,
            self.chunk_overlap,
        )

    def _get_collection_name(self, user_id: str) -> str:
        """Tạo collection name cho user"""
        return f"{self.collection_prefix}_{user_id}"

    def _get_storage(self, user_id: str) -> QdrantStorage:
        """
        Lấy hoặc tạo storage cho user.
        Mỗi user có 1 collection riêng.
        """
        if not user_id:
            raise ValueError("user_id must be provided")
        
        # Check cache
        if user_id in self._storage_cache:
            return self._storage_cache[user_id]
        
        # Tạo storage mới
        collection_name = self._get_collection_name(user_id)
        storage = QdrantStorage(
            collection_name=collection_name,
            embedding=self._embedding,
            vector_size=self._vector_size,
            client=self._client,
        )
        
        # Tạo collection nếu chưa tồn tại
        storage.create_collection(force_recreate=self._recreate_collections)
        
        # Cache storage
        self._storage_cache[user_id] = storage
        
        logger.info("Created storage for user=%s, collection=%s", user_id, collection_name)
        
        return storage

    def _generate_document_id(self) -> str:
        """Tạo document_id duy nhất bằng UUID4"""
        return str(uuid.uuid4())

    def _calculate_chunk_positions(
        self, 
        full_content: str, 
        chunks: List[Document]
    ) -> List[Tuple[int, int]]:
        """
        Tính toán vị trí start_char và end_char của mỗi chunk trong tài liệu gốc.
        Trả về list các tuple (start_char, end_char)
        """
        positions = []
        search_start = 0
        
        for chunk in chunks:
            chunk_content = chunk.page_content
            # Tìm vị trí của chunk trong full_content
            start_pos = full_content.find(chunk_content, search_start)
            
            if start_pos == -1:
                # Nếu không tìm thấy chính xác, ước lượng vị trí
                start_pos = search_start
                end_pos = start_pos + len(chunk_content)
            else:
                end_pos = start_pos + len(chunk_content)
                search_start = start_pos + 1
            
            positions.append((start_pos, end_pos))
        
        return positions

    def _sanitize_metadata_value(self, value: Any) -> Any:
        """Chuyển đổi các giá trị metadata về dạng có thể serialize"""
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): self._sanitize_metadata_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._sanitize_metadata_value(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    def _sanitize_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Sanitize toàn bộ metadata dict"""
        sanitized: Dict[str, Any] = {}
        if not metadata:
            return sanitized
        for key, value in metadata.items():
            sanitized[str(key)] = self._sanitize_metadata_value(value)
        return sanitized

    async def ingest_file(
        self,
        user_id: str,
        file_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestionSummary:
        """
        Convert, split, and persist a document into user's vector store.
        Trả về DocumentInfo để lưu vào database.
        """
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        if not user_id:
            raise ValueError("user_id must be provided")

        storage = self._get_storage(user_id)
        
        # Tạo document_id duy nhất bằng UUIDv7
        document_id = self._generate_document_id()

        logger.info("Starting ingestion: user=%s, file=%s, document_id=%s", 
                   user_id, file_path, document_id)
        
        # Convert tài liệu
        converter = ConverterFactory.create("file")
        documents = converter.convert(str(file_path), metadata=metadata)
        if not documents:
            raise ValueError(f"No content extracted from {file_path}")

        raw_metadata: Dict[str, Any] = dict(metadata or {})
        raw_metadata.setdefault("content_format", "markdown")

        for idx, doc in enumerate(documents):
            doc.page_content = ensure_markdown(
                doc.page_content,
                source_name=file_path.name,
                prefer_title=(idx == 0),
            )
            doc.metadata = doc.metadata or {}
            doc.metadata.update(raw_metadata)

        sanitized_metadata = self._sanitize_metadata(raw_metadata)

        # Ghép nội dung đầy đủ
        full_content = "\n\n".join([doc.page_content for doc in documents])
        logger.debug("Extracted %d documents, total length: %d chars", 
                    len(documents), len(full_content))

        # Cắt thành chunks
        chunks = self._text_splitter.split_documents(documents)
        if not chunks:
            raise ValueError(f"No chunks generated from {file_path}")

        # Tính toán vị trí chunks
        chunk_positions = self._calculate_chunk_positions(full_content, chunks)

        # Chuẩn bị metadata và lưu chunks
        chunk_infos: List[ChunkInfo] = []
        
        for idx, (chunk, (start_char, end_char)) in enumerate(zip(chunks, chunk_positions)):
            chunk_meta = {
                "user_id": user_id,
                "document_id": document_id,
                "source": str(file_path),
                "file_name": file_path.name,
                "chunk_index": idx,
                "start_char": start_char,
                "end_char": end_char,
            }
            
            if sanitized_metadata:
                chunk_meta.update(sanitized_metadata)
            
            chunk_meta.setdefault("content_format", "markdown")

            chunk.metadata = chunk_meta

            chunk_info = ChunkInfo(
                chunk_index=idx,
                content=chunk.page_content,
                start_char=start_char,
                end_char=end_char,
                metadata=chunk_meta
            )
            chunk_infos.append(chunk_info)

        # Lưu vào Qdrant
        logger.info("Persisting %d chunks for document_id=%s to collection=%s", 
                   len(chunks), document_id, storage.collection_name)
        storage.create_collection()
        await storage.add_documents(chunks)

        # Tạo DocumentInfo
        document_info = DocumentInfo(
            document_id=document_id,
            user_id=user_id,
            file_name=file_path.name,
            file_path=str(file_path),
            full_content=full_content,
            content_length=len(full_content),
            chunks=chunk_infos,
            metadata=sanitized_metadata,
            created_at=datetime.now()
        )

        return IngestionSummary(
            user_id=user_id,
            collection_name=storage.collection_name,
            document_info=document_info,
            chunk_count=len(chunks),
        )

    async def ingest_files(
        self,
        user_id: str,
        files: Sequence[Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[IngestionSummary]:
        """Ingest nhiều files cùng lúc cho 1 user"""
        summaries: List[IngestionSummary] = []
        for path in files:
            summaries.append(await self.ingest_file(user_id, path, metadata=metadata))
        return summaries

    async def search(
        self,
        user_id: str,
        query: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Tìm kiếm chunks trong collection của user"""
        if not user_id:
            raise ValueError("user_id must be provided")
            
        storage = self._get_storage(user_id)
        
        logger.info("Searching in user=%s collection (k=%d)", user_id, k)
        
        return await storage.search(query, k=k, filter=metadata_filter)

    async def search_with_scores(
        self,
        user_id: str,
        query: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Tìm kiếm chunks kèm similarity score"""
        if not user_id:
            raise ValueError("user_id must be provided")
            
        storage = self._get_storage(user_id)
        
        logger.info("Searching with scores in user=%s collection (k=%d)", user_id, k)
        
        return await storage.search_with_score(query=query, k=k, filter=metadata_filter)

    async def search_with_highlights(
        self,
        user_id: str,
        query: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Tìm kiếm và nhóm kết quả theo document_id để highlight.
        Trả về: {document_id: [{chunk_index, start_char, end_char, content, score}]}
        """
        results = await self.search_with_scores(user_id, query, k, metadata_filter)
        
        grouped_results: Dict[str, List[Dict[str, Any]]] = {}
        
        for doc, score in results:
            document_id = doc.metadata.get("document_id", "unknown")
            
            if document_id not in grouped_results:
                grouped_results[document_id] = []
            
            grouped_results[document_id].append({
                "chunk_index": doc.metadata.get("chunk_index"),
                "start_char": doc.metadata.get("start_char"),
                "end_char": doc.metadata.get("end_char"),
                "content": doc.page_content,
                "score": score,
                "file_name": doc.metadata.get("file_name"),
                "content_format": "markdown",
            })
        
        # Sắp xếp theo score
        for document_id in grouped_results:
            grouped_results[document_id].sort(key=lambda x: x["score"], reverse=True)
        
        return grouped_results

    async def delete_document(
        self, 
        user_id: str, 
        document_id: str
    ) -> int:
        """
        Xóa tất cả chunks của một document khỏi vector store.
        Trả về số chunks đã xóa.
        """
        if not user_id:
            raise ValueError("user_id must be provided")
            
        storage = self._get_storage(user_id)
        
        deleted_count = await storage.delete_by_filter(document_id)
        logger.info("Deleted %d chunks for document_id=%s", deleted_count, document_id)
        return deleted_count

    def get_collection_info(self, user_id: str) -> Dict[str, Any]:
        """Lấy thông tin collection của user"""
        if not user_id:
            raise ValueError("user_id must be provided")
            
        storage = self._get_storage(user_id)
        return storage.get_collection_info()

    def delete_collection(self, user_id: str) -> None:
        """Xóa toàn bộ collection của user"""
        if not user_id:
            raise ValueError("user_id must be provided")
            
        storage = self._get_storage(user_id)
        storage.delete_collection()
        
        # Remove from cache
        self._storage_cache.pop(user_id, None)
        
        logger.info("Deleted collection for user=%s", user_id)

    def list_user_collections(self) -> List[str]:
        """Liệt kê tất cả collections của users"""
        try:
            collections = self._client.get_collections().collections
            user_collections = [
                col.name for col in collections 
                if col.name.startswith(f"{self.collection_prefix}_")
            ]
            return user_collections
        except Exception as e:
            logger.error("Failed to list collections: %s", e)
            return []


    async def query_with_llm(
        self,
        user_id: str,
        question: str,
        llm_service: LLMService,  # LLMService instance
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ) -> QueryWithLLMResult:
        """
        Query với RAG và trả lời bằng LLM
        
        Args:
            user_id: ID của user
            question: Câu hỏi
            llm_service: Instance của LLMService
            k: Số lượng chunks để retrieve
            metadata_filter: Filter metadata
            temperature: Temperature cho LLM
            max_tokens: Max tokens cho response
            
        Returns:
            QueryWithLLMResult với answer và sources
        """
        
        logger.info(
            f"Query with LLM for user={user_id}, question='{question[:100]}...', k={k}"
        )
        
        # 1. Retrieve relevant chunks từ vector store
        results = await self.search_with_scores(
            user_id=user_id,
            query=question,
            k=k,
            metadata_filter=metadata_filter,
        )

        logger.info(f"Retrieved {len(results)} relevant chunks for question: '{question}'")
        
        if not results:
            logger.warning(f"No relevant chunks found for question: '{question}'")
            fallback_message = (
                "### Xin loi\n\n"
                "Toi chua tim thay thong tin lien quan trong tai lieu de tra loi cau hoi nay."
            )
            answer_payload = llm_service.build_answer_payload(fallback_message, [])
            return QueryWithLLMResult(
                query=question,
                answer=answer_payload,
                sources=[],
                context_used="",
                model=llm_service.model,
                retrieved_chunks=0,
            )
        
        # 2. Chuẩn bị sources và context
        sources = []
        context_parts = []
        
        for idx, (doc, score) in enumerate(results, 1):
            metadata = doc.metadata or {}
            source_id = f"S{idx}"

            source_info = {
                "source_id": source_id,
                "chunk_index": metadata.get("chunk_index", 0),
                "document_id": metadata.get("document_id", "unknown"),
                "file_name": metadata.get("file_name", "unknown"),
                "content": doc.page_content,
                "score": float(score),
                "start_char": metadata.get("start_char", 0),
                "end_char": metadata.get("end_char", 0),
                "source_path": metadata.get("source"),
                "content_format": "markdown",
            }
            sources.append(source_info)
            
            # Format cho context
            context_parts.append(
                f"### Nguon {source_id} - {source_info['file_name']} "
                f"(Do lien quan: {score:.2f})\n{doc.page_content}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        
        logger.info(f"Retrieved {len(sources)} chunks, total context length: {len(context)}")
        
        # 3. Gọi LLM để generate answer
        try:
            llm_response = llm_service.answer_with_context(
                question=question,
                context=context,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            logger.info("LLM answer generated successfully")

            answer_payload = llm_service.build_answer_payload(
                llm_response.answer,
                sources,
            )
            
            return QueryWithLLMResult(
                query=question,
                answer=answer_payload,
                sources=sources,
                context_used=context,
                model=llm_response.model,
                retrieved_chunks=len(sources),
            )
            
        except Exception as e:
            logger.error(f"Failed to generate LLM answer: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to generate answer: {str(e)}")



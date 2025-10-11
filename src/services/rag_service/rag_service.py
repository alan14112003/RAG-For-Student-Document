from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from src.rag_service.converter import ConverterFactory
from src.rag_service.qdrant_storage.qdrant_storage import QdrantStorage

logger = logging.getLogger(__name__)

USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,63}$")


@dataclass
class IngestionSummary:
    user_id: str
    collection_name: str
    source_path: Path
    document_count: int
    chunk_count: int
    metadata: Dict[str, Any]
    chunk_metadata: List[Dict[str, Any]]


class RagService:
    """Orchestrates ingestion and retrieval with per-user isolation."""

    def __init__(
        self,
        collection_prefix: str = "student_documents",
        data_dir: Path | str = "data/uploads",
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        embedding_model: str = "mxbai-embed-large",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        recreate_collections: bool = False,
    ) -> None:
        self.collection_prefix = collection_prefix
        self._base_data_dir = Path(data_dir)
        self._base_data_dir.mkdir(parents=True, exist_ok=True)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embedding_model = embedding_model
        self._embedding = OllamaEmbeddings(model=self.embedding_model)
        self._vector_size = len(self._embedding.embed_query("__dimension_probe__"))

        self._qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self._qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self._client = QdrantClient(url=self._qdrant_url, api_key=self._qdrant_api_key)

        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

        self._recreate_collections = recreate_collections
        self._storage_cache: Dict[str, QdrantStorage] = {}

        logger.info(
            "RagService ready with per-user collections (prefix=%s, chunk_size=%d, overlap=%d)",
            self.collection_prefix,
            self.chunk_size,
            self.chunk_overlap,
        )

    @property
    def base_data_dir(self) -> Path:
        return self._base_data_dir

    @property
    def text_splitter(self) -> RecursiveCharacterTextSplitter:
        return self._text_splitter

    def _normalize_user_id(self, user_id: str) -> str:
        if not user_id:
            raise ValueError("user_id must be provided.")
        identifier = user_id.strip()
        if not USER_ID_PATTERN.match(identifier):
            raise ValueError(
                "user_id must be 3-64 characters long and contain only letters, numbers, dots, underscores, or hyphens."
            )
        return identifier

    def _collection_name(self, user_id: str) -> str:
        return f"{self.collection_prefix}_{user_id}"

    def collection_name_for_user(self, user_id: str) -> str:
        """Expose the resolved collection name for external modules."""
        identifier = self._normalize_user_id(user_id)
        return self._collection_name(identifier)

    def user_data_dir(self, user_id: str) -> Path:
        identifier = self._normalize_user_id(user_id)
        user_dir = self.base_data_dir / identifier
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    def _get_or_create_storage(self, user_id: str) -> QdrantStorage:
        identifier = self._normalize_user_id(user_id)
        if identifier in self._storage_cache:
            return self._storage_cache[identifier]

        collection_name = self._collection_name(identifier)
        storage = QdrantStorage(
            collection_name=collection_name,
            embedding=self._embedding,
            vector_size=self._vector_size,
            client=self._client,
        )
        storage.create_collection(force_recreate=self._recreate_collections)
        self._storage_cache[identifier] = storage
        return storage

    def _prepare_metadata(
        self,
        user_id: str,
        source_path: Path,
        metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        base_metadata = {
            "source": str(source_path),
            "file_name": source_path.name,
            "user_id": user_id,
            "collection": self._collection_name(user_id),
        }
        if metadata:
            base_metadata.update(metadata)
        return base_metadata

    def _sanitize_metadata_value(self, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): self._sanitize_metadata_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._sanitize_metadata_value(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _sanitize_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        if not metadata:
            return sanitized
        for key, value in metadata.items():
            sanitized[str(key)] = self._sanitize_metadata_value(value)
        return sanitized

    def _format_metadata_block(self, metadata: Dict[str, Any]) -> str:
        if not metadata:
            return "[]"
        return json.dumps(metadata, ensure_ascii=False, sort_keys=True)

    def _decorate_chunk_content(self, content: str, metadata_text: str) -> str:
        if not metadata_text:
            return content
        return f"{content}\n\n[metadata] {metadata_text}"

    async def ingest_file(
        self,
        user_id: str,
        file_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestionSummary:
        """Convert, split, and persist a document into the user's dedicated vector store."""
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        identifier = self._normalize_user_id(user_id)
        storage = self._get_or_create_storage(identifier)
        converter = ConverterFactory.create("file")
        prepared_metadata = self._prepare_metadata(identifier, file_path, metadata)

        logger.info("Starting conversion for user=%s file=%s", identifier, file_path)
        documents = converter.convert(str(file_path), metadata=prepared_metadata)
        if not documents:
            raise ValueError(f"No content extracted from {file_path}")

        logger.debug("Converted %s into %d raw documents", file_path, len(documents))

        chunks = self.text_splitter.split_documents(documents)
        if not chunks:
            raise ValueError(f"No chunks generated from {file_path}")

        sanitized_document_metadata = self._sanitize_metadata(prepared_metadata)

        chunk_metadata: List[Dict[str, Any]] = []
        combined_contents: List[str] = []

        for idx, chunk in enumerate(chunks):
            chunk_meta = self._sanitize_metadata(chunk.metadata or {})
            chunk_meta.setdefault("user_id", identifier)
            chunk_meta.setdefault("source", str(file_path))
            chunk_meta["chunk_index"] = idx

            metadata_text = self._format_metadata_block(chunk_meta)
            enriched_content = self._decorate_chunk_content(chunk.page_content, metadata_text)

            chunk.page_content = enriched_content
            chunk.metadata = chunk_meta

            combined_contents.append(enriched_content)
            chunk_metadata.append(
                {
                    "chunk_index": idx,
                    "content_length": len(enriched_content),
                    "content_preview": enriched_content[:200],
                    "metadata": chunk_meta,
                    "metadata_text": metadata_text,
                    "content": enriched_content,
                }
            )

        logger.info("Persisting %d chunks for user=%s file=%s", len(chunks), identifier, file_path)
        await storage.add_documents(chunks)

        sanitized_document_metadata["processed_text"] = "\n\n".join(combined_contents)
        sanitized_document_metadata["total_chunks"] = len(chunk_metadata)

        return IngestionSummary(
            user_id=identifier,
            collection_name=storage.collection_name,
            source_path=file_path,
            document_count=len(documents),
            chunk_count=len(chunks),
            metadata=sanitized_document_metadata,
            chunk_metadata=chunk_metadata,
        )

    async def ingest_files(
        self,
        user_id: str,
        files: Sequence[Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[IngestionSummary]:
        summaries: List[IngestionSummary] = []
        for path in files:
            summaries.append(await self.ingest_file(user_id, path, metadata=metadata))
        return summaries

    async def query(
        self,
        user_id: str,
        question: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Retrieve the most similar chunks within the user's dedicated vector store."""
        identifier = self._normalize_user_id(user_id)
        storage = self._get_or_create_storage(identifier)
        logger.info("Executing semantic search for user=%s (k=%d)", identifier, k)
        qdrant_filter = self._build_filter(identifier, metadata_filter)
        return await storage.search(question, k=k, filter=qdrant_filter)

    async def query_with_scores(
        self,
        user_id: str,
        question: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        identifier = self._normalize_user_id(user_id)
        storage = self._get_or_create_storage(identifier)
        logger.info("Executing semantic search with scores for user=%s (k=%d)", identifier, k)
        qdrant_filter = self._build_filter(identifier, metadata_filter)
        results = await storage.search_with_score(query=question, k=k, filter=qdrant_filter)
        return results

    def get_collection_info(self, user_id: str) -> Dict[str, Any]:
        identifier = self._normalize_user_id(user_id)
        storage = self._get_or_create_storage(identifier)
        return storage.get_collection_info()

    def delete_collection(self, user_id: str) -> None:
        identifier = self._normalize_user_id(user_id)
        storage = self._get_or_create_storage(identifier)
        storage.delete_collection()
        self._storage_cache.pop(identifier, None)

    def _build_filter(
        self,
        user_id: str,
        metadata_filter: Optional[Dict[str, Any]],
    ) -> qdrant_models.Filter:
        conditions: List[qdrant_models.FieldCondition] = [
            qdrant_models.FieldCondition(
                key="user_id",
                match=qdrant_models.MatchValue(value=user_id),
            )
        ]

        if metadata_filter:
            for key, value in metadata_filter.items():
                if key == "user_id" or value in (None, "", []):
                    continue

                if isinstance(value, (list, tuple, set)):
                    conditions.append(
                        qdrant_models.FieldCondition(
                            key=key,
                            match=qdrant_models.MatchAny(any=list(value)),
                        )
                    )
                else:
                    conditions.append(
                        qdrant_models.FieldCondition(
                            key=key,
                            match=qdrant_models.MatchValue(value=value),
                        )
                    )

        return qdrant_models.Filter(must=conditions)

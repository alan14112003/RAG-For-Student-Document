from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams

logger = logging.getLogger(__name__)


class QdrantStorage:
    """Wrapper around Qdrant that exposes a minimal async interface for LangChain."""

    def __init__(
        self,
        collection_name: str,
        embedding: Any,
        client: QdrantClient,
        vector_size: int,
        distance_metric: Distance = Distance.COSINE,
    ) -> None:
        if not collection_name:
            raise ValueError("Collection name must be provided.")

        self.collection_name = collection_name
        self.embeddings = embedding
        self.client = client
        self.vector_size = vector_size
        self.distance_metric = distance_metric

        logger.debug("Initialized QdrantStorage for collection=%s", self.collection_name)

    # -------------------------------------------------------------------------
    # Collection lifecycle
    # -------------------------------------------------------------------------
    def collection_exists(self) -> bool:
        try:
            collections = self.client.get_collections().collections
        except Exception as exc:
            logger.error("Failed to list Qdrant collections: %s", exc)
            return False
        return any(col.name == self.collection_name for col in collections)

    def _create_collection(self) -> None:
        logger.info(
            "Creating Qdrant collection '%s' (size=%d, distance=%s)",
            self.collection_name,
            self.vector_size,
            self.distance_metric.name,
        )
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=self.distance_metric),
            )
        except UnexpectedResponse as exc:
            if "already exists" in str(exc).lower():
                logger.info("Collection '%s' already exists.", self.collection_name)
            else:
                raise
        except Exception as exc:
            logger.exception("Unable to create collection '%s'", self.collection_name)
            raise

    def create_collection(self, force_recreate: bool = False) -> None:
        if force_recreate and self.collection_exists():
            self.delete_collection()
        if not self.collection_exists():
            self._create_collection()

    def delete_collection(self) -> None:
        if not self.collection_exists():
            logger.warning("Collection '%s' does not exist. Skip deletion.", self.collection_name)
            return
        logger.info("Deleting Qdrant collection '%s'", self.collection_name)
        self.client.delete_collection(self.collection_name)

    # -------------------------------------------------------------------------
    # Vector store helpers
    # -------------------------------------------------------------------------
    def _load_vectorstore(self) -> QdrantVectorStore:
        if not self.collection_exists():
            raise ValueError(
                f"Collection '{self.collection_name}' does not exist. Call create_collection() first."
            )
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    async def add_documents(self, documents: Sequence[Document]) -> None:
        if not documents:
            logger.warning("Empty document list received. Skip ingestion.")
            return
        vectorstore = self._load_vectorstore()
        await vectorstore.aadd_documents(list(documents))
        logger.info("Persisted %d documents into collection '%s'", len(documents), self.collection_name)

    async def search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        if not query:
            raise ValueError("Query must not be empty.")

        vectorstore = self._load_vectorstore()
        if filter:
            results = await vectorstore.asimilarity_search(query, k=k, filter=filter)
        else:
            results = await vectorstore.asimilarity_search(query, k=k)
        logger.debug("Search returned %d documents for query='%s'", len(results), query)
        return results

    async def search_with_score(
        self,
        query: str,
        k: int = 5,
    ) -> List[Tuple[Document, float]]:
        if not query:
            raise ValueError("Query must not be empty.")

        vectorstore = self._load_vectorstore()
        results = await vectorstore.asimilarity_search_with_score(query, k=k)
        logger.debug("Search with score returned %d results for query='%s'", len(results), query)
        return results

    def get_collection_info(self) -> Dict[str, Any]:
        if not self.collection_exists():
            return {"exists": False, "name": self.collection_name}

        info = self.client.get_collection(self.collection_name)
        return {
            "exists": True,
            "name": self.collection_name,
            "points_count": info.points_count,
            "vector_size": info.config.params.vectors.size,
            "distance": info.config.params.vectors.distance.name,
        }


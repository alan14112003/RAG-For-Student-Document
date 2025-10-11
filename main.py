from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.app.api.api import create_api_router
from src.app.database import dispose_engine, engine
from src.app.model import Base
from src.app.storage import MinioStorage
from src.rag_service.rag_service import RagService


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    collection_prefix: str = os.getenv(
        "QDRANT_COLLECTION_PREFIX",
        os.getenv("QDRANT_COLLECTION", "student_documents"),
    )
    data_dir: Path = Path(os.getenv("DATA_DIR", "data/uploads"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
    qdrant_url: Optional[str] = os.getenv("QDRANT_URL")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    recreate_collection: bool = os.getenv("QDRANT_RECREATE_COLLECTION", "false").lower() == "true"
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))  # 50 MB default
    minio_endpoint: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    minio_access_key: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    minio_secret_key: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    minio_bucket: str = os.getenv("MINIO_BUCKET", "rag-documents")
    minio_secure: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"
    minio_region: Optional[str] = os.getenv("MINIO_REGION") or None
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "3000"))
    api_reload: bool = os.getenv("API_RELOAD", "false").lower() == "true"


config = AppConfig()

rag_service = RagService(
    collection_prefix=config.collection_prefix,
    data_dir=config.data_dir,
    chunk_size=config.chunk_size,
    chunk_overlap=config.chunk_overlap,
    embedding_model=config.embedding_model,
    qdrant_url=config.qdrant_url,
    qdrant_api_key=config.qdrant_api_key,
    recreate_collections=config.recreate_collection,
)

object_storage = MinioStorage(
    endpoint=config.minio_endpoint,
    access_key=config.minio_access_key,
    secret_key=config.minio_secret_key,
    bucket_name=config.minio_bucket,
    secure=config.minio_secure,
    region=config.minio_region,
)

app = FastAPI(
    title="Student Knowledge RAG API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


def _configure_cors(application: FastAPI) -> None:
    raw_origins = os.getenv("CORS_ALLOW_ORIGINS")
    if raw_origins:
        origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    else:
        origins = ["*"]

    allow_credentials = "*" not in origins

    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=allow_credentials,
    )


_configure_cors(app)


@app.on_event("startup")
async def on_startup() -> None:
    object_storage.ensure_bucket()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schema ensured.")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await dispose_engine()


app.include_router(create_api_router(rag_service, object_storage, config.max_upload_bytes))


__all__ = ["app"]


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.api_reload,
    )

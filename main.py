# main.py
import logging
import os
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.services.llm_service import LLMService
from src.app.models.base import engine
from src.app.models.base import Base
from src.app.api.auth import router as auth_router
from src.app.api.documents import router as documents_router
from src.services.rag_service.rag_service import RagService
from src.services.minio_service import MinIOService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Singleton service instances
rag_service: Optional[RagService] = None
minio_service: Optional[MinIOService] = None
llm_service: Optional[LLMService] = None

def get_rag_service_instance() -> RagService:
    """Get the singleton RAG service instance"""
    global rag_service
    if rag_service is None:
        rag_service = RagService(
            collection_prefix=os.getenv("QDRANT_COLLECTION_PREFIX", "user_documents"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "mxbai-embed-large"),
            qdrant_url=os.getenv("QDRANT_URL"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            recreate_collections=os.getenv("QDRANT_RECREATE_COLLECTION", "false").lower() == "true",
        )
    return rag_service


def get_minio_service_instance() -> MinIOService:
    """Get the singleton MinIO service instance"""
    global minio_service
    if minio_service is None:
        minio_service = MinIOService(
            endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            bucket_name=os.getenv("MINIO_BUCKET", "rag-documents"),
            secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
        )
    return minio_service


def get_llm_service_instance() -> LLMService:
    """Get the singleton LLM service instance"""
    global llm_service
    if llm_service is None:
        llm_service = LLMService(
            model=os.getenv("LLM_MODEL", "llama3"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            timeout=int(os.getenv("LLM_TIMEOUT", "120")),
        )
    return llm_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for the application"""
    # Startup
    logger.info("Starting up application...")
    
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created")
    
    # Initialize RAG service
    get_rag_service_instance()
    logger.info("RAG service initialized")

    # Initialize LLM service
    llm = get_llm_service_instance()
    logger.info(f"LLM service initialized (model={llm.model}, base_url={llm.base_url})")
    
    # Log available models
    try:
        available_models = llm.get_available_models()
        if available_models:
            logger.info(f"Available Ollama models: {', '.join(available_models)}")
        else:
            logger.warning("Could not fetch available Ollama models")
    except Exception as e:
        logger.warning(f"Failed to fetch Ollama models: {e}")

    # Initialize MinIO service
    get_minio_service_instance()
    logger.info("MinIO service initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    await engine.dispose()


# Create FastAPI app
app = FastAPI(
    title="RAG Document Management API",
    description="API for document upload, search, and retrieval with RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# Configure CORS
def configure_cors(application: FastAPI) -> None:
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


configure_cors(app)


# Override the dependency in documents router to use singleton
from src.app.api import documents
documents.set_rag_service_provider(get_rag_service_instance)
documents.set_llm_service_provider(get_llm_service_instance)
documents.set_minio_service_provider(get_minio_service_instance)


# Include routers
app.include_router(auth_router)
app.include_router(documents_router)


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "RAG Document Management API is running",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "database": "connected",
        "rag_service": "initialized",
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("API_RELOAD", "false").lower() == "true",
    )

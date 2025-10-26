from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.app.api import documents as documents_module
from src.app.api.documents import router as documents_router
from src.app.models.base import get_db
from src.app.utils.auth import get_current_active_user
from src.services.rag_service.rag_service import QueryWithLLMResult


class FakeUser:
    def __init__(self, user_id: str = "user-test") -> None:
        self.id = user_id
        self.is_active = True


class DummyLLMService:
    """Placeholder object to satisfy dependency injection."""

    def __init__(self) -> None:
        self.model = "dummy-gemini"


class DummyRagService:
    """Fake RAG service that records invocations and returns canned data."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    async def query_with_llm(
        self,
        user_id: str,
        question: str,
        llm_service: DummyLLMService,
        k: int = 5,
        metadata_filter: Dict[str, Any] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ) -> QueryWithLLMResult:
        self.calls.append(
            {
                "user_id": user_id,
                "question": question,
                "k": k,
                "metadata_filter": metadata_filter,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "model": llm_service.model,
            }
        )

        dummy_sources = [
            {
                "source_id": "S1",
                "document_id": "doc-1",
                "file_name": "policy.pdf",
                "chunk_index": 0,
                "content": "Chinh sach bao mat duoc mo ta tai day.",
                "score": 0.98,
                "start_char": 0,
                "end_char": 64,
                "source_path": "s3://bucket/policy.pdf",
            }
        ]

        references = [
            {
                "source_id": "S1",
                "document_id": "doc-1",
                "file_name": "policy.pdf",
                "chunk_index": 0,
                "score": 0.98,
                "snippet": "Chinh sach bao mat duoc mo ta tai day.",
                "start_char": 0,
                "end_char": 64,
                "source_path": "s3://bucket/policy.pdf",
                "explanation": "Trich tu policy.pdf, chunk #0 (ky tu 0-64)",
            }
        ]

        answer_payload = {
            "content": "Theo [Nguon S1], chinh sach bao mat nam 2024 quy "
            "dinh ro vai tro va trach nhiem cua cac phong ban.",
            "references": references,
        }

        return QueryWithLLMResult(
            query=question,
            answer=answer_payload,
            sources=dummy_sources,
            context_used="Dummy context",
            model=llm_service.model,
            retrieved_chunks=len(dummy_sources),
        )


@pytest.fixture(scope="session")
def dummy_rag_service() -> DummyRagService:
    return DummyRagService()


@pytest.fixture(scope="session")
def api_client(dummy_rag_service: DummyRagService) -> TestClient:
    """
    Spin up a lightweight FastAPI app that only mounts the documents router and
    overrides heavy dependencies with lightweight fakes for deterministic testing.
    """
    documents_module.set_rag_service_provider(lambda: dummy_rag_service)
    documents_module.set_llm_service_provider(DummyLLMService)

    app = FastAPI()
    app.include_router(documents_router)

    fake_user = FakeUser()

    async def override_get_db():
        yield None

    app.dependency_overrides[get_current_active_user] = lambda: fake_user
    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as client:
        yield client

from __future__ import annotations

from typing import Dict


def test_query_documents_returns_structured_answer(api_client):
    payload: Dict[str, object] = {
        "query": "Tom tat chinh sach bao mat",
        "k": 3,
        "temperature": 0.2,
        "max_tokens": 256,
    }

    response = api_client.post("/documents/query", json=payload)

    assert response.status_code == 200
    body = response.json()

    assert body["query"] == payload["query"]
    assert body["answer"]["content"].startswith("Theo [Nguon S1]")
    assert body["answer"]["references"][0]["source_id"] == "S1"
    assert body["sources"][0]["source_id"] == "S1"
    assert body["model"] == "dummy-gemini"


def test_query_documents_validation_error(api_client):
    response = api_client.post("/documents/query", json={"query": ""})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(err["loc"][-1] == "query" for err in detail)

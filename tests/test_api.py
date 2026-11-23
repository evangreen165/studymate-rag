from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from studymate.api import main
from studymate.rag import RAGResponse
from studymate.retrieval import Source


class FakeRAG:
    def __init__(self) -> None:
        self.retriever = SimpleNamespace(
            index=SimpleNamespace(
                count=lambda: 1,
                read_meta=lambda: {"document_chunks": {"d1": 3}},
                stats=lambda: {"chunks": 1},
            )
        )

    def index_count(self) -> int:
        return 1

    def answer(self, question: str, **kwargs) -> RAGResponse:
        return RAGResponse(
            request_id="req1",
            question=question,
            answer="Ответ [1]",
            sources=[
                Source(
                    chunk_id="c1",
                    text="Текст",
                    score=0.5,
                    doc_id="d1",
                    title="Документ",
                    topic="Git",
                    topic_slug="git",
                    page=1,
                    rel_path="git/a.pdf",
                )
            ],
            provider="mock",
            model="mock",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            tokens_per_second=50.0,
            latency_ms=100.0,
            retrieval_ms=10.0,
            generation_ms=80.0,
            top_k=6,
        )

    def status(self) -> dict:
        return {
            "index": {"chunks": 1, "meta": {}},
            "corpus": {"unique_documents": 1},
            "embedding_model": "test",
            "retrieval": {"chunks": 1},
            "providers": {},
            "chunk_size": 512,
            "chunk_overlap": 64,
        }


def test_health() -> None:
    client = TestClient(main.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_validation() -> None:
    client = TestClient(main.app)
    response = client.post("/chat", json={"question": ""})
    assert response.status_code == 422


def test_chat_success(monkeypatch) -> None:
    monkeypatch.setattr(main, "rag", FakeRAG())
    client = TestClient(main.app)
    response = client.post(
        "/chat", json={"question": "Что такое Git?", "provider": "mock", "top_k": 3}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Ответ [1]"
    assert payload["sources"][0]["page"] == 1
    assert payload["usage"]["total_tokens"] == 15
    assert payload["timings"]["total_ms"] == 100.0


def test_ready_empty(monkeypatch) -> None:
    class EmptyRAG(FakeRAG):
        def index_count(self) -> int:
            return 0

    monkeypatch.setattr(main, "rag", EmptyRAG())
    client = TestClient(main.app)
    response = client.get("/ready")
    assert response.status_code == 503


def test_metrics_endpoint() -> None:
    client = TestClient(main.app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "studymate_requests_total" in response.text

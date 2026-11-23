from __future__ import annotations

from types import SimpleNamespace

import pytest

from studymate.config import Settings
from studymate.llm import MockLLM
from studymate.rag import IndexNotReady, RAGService
from studymate.retrieval import Source


class FakeRetriever:
    def __init__(self, sources: list[Source], count: int = 1) -> None:
        self.sources = sources
        self.index = SimpleNamespace(count=lambda: count, read_meta=lambda: {})

    def search(self, question: str, **kwargs) -> list[Source]:
        return self.sources

    def stats(self) -> dict:
        return {"chunks": self.index.count()}


def make_source(score: float = 0.8) -> Source:
    return Source(
        chunk_id="c1",
        text="Git хранит историю изменений. Коммит фиксирует снимок проекта.",
        score=score,
        dense_score=score,
        doc_id="d1",
        title="Основы Git",
        topic="Git",
        topic_slug="git",
        page=10,
        rel_path="git/a.pdf",
    )


def test_answer_with_context(monkeypatch) -> None:
    settings = Settings(INDEX_DIR="tmp", MANIFEST_PATH="tmp/manifest.json")
    service = RAGService(
        settings, retriever=FakeRetriever([make_source()]), llm=MockLLM()
    )
    response = service.answer("Что такое коммит?", top_k=3)
    assert response.answer
    assert "[1]" in response.answer
    assert response.sources[0].page == 10
    assert response.total_tokens > 0
    assert response.tokens_per_second >= 0


def test_refusal_without_sources() -> None:
    settings = Settings(INDEX_DIR="tmp", MANIFEST_PATH="tmp/manifest.json")
    service = RAGService(settings, retriever=FakeRetriever([]), llm=MockLLM())
    response = service.answer("Что такое Git?")
    assert response.refusal is True
    assert "нет" in response.answer.lower()


def test_empty_index() -> None:
    settings = Settings(INDEX_DIR="tmp", MANIFEST_PATH="tmp/manifest.json")
    service = RAGService(settings, retriever=FakeRetriever([], count=0), llm=MockLLM())
    with pytest.raises(IndexNotReady):
        service.answer("Вопрос")


def test_no_context_baseline() -> None:
    settings = Settings(INDEX_DIR="tmp", MANIFEST_PATH="tmp/manifest.json")
    service = RAGService(settings, retriever=FakeRetriever([]), llm=MockLLM())
    result = service.answer_no_context("Что такое Git?")
    assert result["contexts"] == []
    assert result["answer"]

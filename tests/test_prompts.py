from __future__ import annotations

from studymate.chunking import _split_sentences
from studymate.prompts import (
    build_context,
    build_messages,
    cited_sources,
    parse_citations,
)
from studymate.retrieval import Source


def make_source(number: int, text: str = "Текст фрагмента.") -> Source:
    return Source(
        chunk_id=f"c{number}",
        text=text,
        score=1.0 / number,
        doc_id="doc",
        title=f"Документ {number}",
        topic="Git",
        topic_slug="git",
        page=number,
        rel_path="git/doc.pdf",
    )


def test_split_sentences() -> None:
    assert _split_sentences("Первое. Второе! Третье?") == [
        "Первое.",
        "Второе!",
        "Третье?",
    ]


def test_context_and_messages() -> None:
    sources = [make_source(1), make_source(2)]
    context = build_context(sources)
    assert "[1]" in context and "[2]" in context
    assert "Документ 1" in context
    messages = build_messages("Что такое Git?", sources)
    assert messages[0]["role"] == "system"
    assert "Что такое Git?" in messages[1]["content"]


def test_citations() -> None:
    sources = [make_source(1), make_source(2)]
    answer = "Первое утверждение [1], второе [2], ошибочная ссылка [7]."
    assert parse_citations(answer, 2) == [1, 2]
    assert [source.chunk_id for source in cited_sources(answer, sources)] == ["c1", "c2"]
    assert cited_sources("Без ссылок.", sources) == sources

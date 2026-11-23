from __future__ import annotations

import pytest

from studymate.chunking import Chunk
from studymate.config import Settings
from studymate.index import HashEmbedder, VectorIndex
from studymate.retrieval import Retriever


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        INDEX_DIR=tmp_path / "chroma",
        MANIFEST_PATH=tmp_path / "manifest.json",
        INDEX_META_PATH=tmp_path / "meta.json",
        EMBEDDING_MODEL="test",
        use_hybrid=False,
    )


def test_index_and_retrieve(settings: Settings, tmp_path) -> None:
    index = VectorIndex(settings, embedder=HashEmbedder(), path=tmp_path / "chroma")
    chunks = [
        Chunk(
            chunk_id="a1",
            text="Git хранит историю изменений и коммиты.",
            metadata={
                "doc_id": "doc1",
                "title": "Git",
                "topic": "Git",
                "topic_slug": "git",
                "page": 1,
                "rel_path": "git/a.pdf",
            },
        ),
        Chunk(
            chunk_id="a2",
            text="SQL SELECT выбирает строки из таблицы.",
            metadata={
                "doc_id": "doc2",
                "title": "SQL",
                "topic": "SQL",
                "topic_slug": "sql",
                "page": 2,
                "rel_path": "sql/b.pdf",
            },
        ),
    ]
    index.build(chunks)
    assert index.count() == 2

    retriever = Retriever(settings, index=index, use_hybrid=False)
    results = retriever.search("git коммиты", top_k=1)
    assert results
    assert results[0].doc_id == "doc1"
    assert results[0].page == 1

    filtered = retriever.search("git коммиты", top_k=2, topic_slug="sql")
    assert filtered
    assert all(source.topic_slug == "sql" for source in filtered)


def test_delete_document(settings: Settings, tmp_path) -> None:
    index = VectorIndex(settings, embedder=HashEmbedder(), path=tmp_path / "chroma")
    index.build(
        [
            Chunk(
                chunk_id="x1",
                text="Текст",
                metadata={
                    "doc_id": "d1",
                    "title": "T",
                    "topic": "T",
                    "topic_slug": "other",
                    "page": 1,
                    "rel_path": "t.pdf",
                },
            )
        ]
    )
    assert index.delete_document("d1") == 1
    assert index.count() == 0

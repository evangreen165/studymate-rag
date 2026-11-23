from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import read_chunks_jsonl
from .config import Settings, get_settings
from .index import VectorIndex

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_]{2,}")
_STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все",
    "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по",
    "только", "ее", "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из",
    "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или",
    "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять", "уж", "вам",
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "for", "on",
    "with", "as", "by", "that", "this", "it", "be", "from", "at", "which",
}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text) if token.lower() not in _STOPWORDS]


@dataclass
class Source:
    chunk_id: str
    text: str
    score: float
    doc_id: str
    title: str
    topic: str
    topic_slug: str
    page: int
    rel_path: str
    dense_score: float = 0.0
    bm25_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "topic": self.topic,
            "topic_slug": self.topic_slug,
            "page": self.page,
            "score": round(self.score, 4),
            "dense_score": round(self.dense_score, 4),
            "bm25_score": round(self.bm25_score, 4),
            "snippet": self.text[:700],
            "rel_path": self.rel_path,
        }


class BM25Store:
    def __init__(self, index: VectorIndex, path: Path | None = None) -> None:
        if path and path.exists():
            chunks = read_chunks_jsonl(path)
            self.ids = [chunk.chunk_id for chunk in chunks]
            self.documents = [chunk.text for chunk in chunks]
            self.metadatas = [chunk.metadata for chunk in chunks]
        else:
            payload = index.get_all(include_documents=True)
            self.ids = list(payload.get("ids", []))
            self.documents = list(payload.get("documents", []) or [])
            self.metadatas = list(payload.get("metadatas", []) or [])
        corpus = [tokenize(document) for document in self.documents]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in ranked if scores[i] > 0]


class Retriever:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        index: VectorIndex | None = None,
        use_hybrid: bool | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.index = index or VectorIndex(self.settings)
        self.use_hybrid = self.settings.use_hybrid if use_hybrid is None else use_hybrid
        self._bm25: BM25Store | None = None

    @property
    def bm25(self) -> BM25Store:
        if self._bm25 is None:
            self._bm25 = BM25Store(self.index, self.settings.bm25_path)
        return self._bm25

    def _where(self, topic_slug: str | None) -> dict | None:
        if not topic_slug or topic_slug == "all":
            return None
        return {"topic_slug": topic_slug}

    @staticmethod
    def _mmr(
        query_embedding: np.ndarray,
        embeddings: np.ndarray,
        candidates: list[int],
        top_k: int,
        lambda_mult: float,
    ) -> list[int]:
        if not candidates:
            return []
        selected: list[int] = []
        remaining = candidates.copy()
        query_sim = embeddings @ query_embedding
        while remaining and len(selected) < top_k:
            if not selected:
                best = max(remaining, key=lambda index: query_sim[index])
            else:
                selected_matrix = embeddings[selected]
                redundancy = np.max(embeddings[remaining] @ selected_matrix.T, axis=1)
                mmr_scores = lambda_mult * query_sim[remaining] - (1 - lambda_mult) * redundancy
                best = remaining[int(np.argmax(mmr_scores))]
            selected.append(best)
            remaining.remove(best)
        return selected

    def search(
        self,
        question: str,
        *,
        top_k: int | None = None,
        topic_slug: str | None = None,
    ) -> list[Source]:
        settings = self.settings
        top_k = top_k or settings.top_k
        fetch_k = max(top_k, settings.fetch_k)
        query_embedding = np.asarray(
            self.index.embedder.get_query_embedding(question), dtype=np.float32
        )
        payload = self.index.query(
            query_embedding.tolist(),
            top_k=fetch_k,
            where=self._where(topic_slug),
            include_embeddings=True,
        )
        ids = (payload.get("ids") or [[]])[0]
        documents = (payload.get("documents") or [[]])[0]
        metadatas = (payload.get("metadatas") or [[]])[0]
        distances = (payload.get("distances") or [[]])[0]
        embeddings = np.asarray((payload.get("embeddings") or [[]])[0], dtype=np.float32)

        dense_scores = [1.0 - float(distance) for distance in distances]
        dense_order = list(range(len(ids)))
        if len(embeddings) == len(ids) and embeddings.size:
            mmr_order = self._mmr(
                query_embedding, embeddings, dense_order, min(fetch_k, top_k * 3), settings.mmr_lambda
            )
        else:
            mmr_order = dense_order

        sources: dict[str, Source] = {}
        dense_rank: dict[str, int] = {}
        for rank, position in enumerate(mmr_order):
            metadata = metadatas[position] or {}
            source = Source(
                chunk_id=str(ids[position]),
                text=str(documents[position]),
                score=float(dense_scores[position]),
                dense_score=float(dense_scores[position]),
                doc_id=str(metadata.get("doc_id", "")),
                title=str(metadata.get("title", "")),
                topic=str(metadata.get("topic", "")),
                topic_slug=str(metadata.get("topic_slug", "")),
                page=int(metadata.get("page", 0) or 0),
                rel_path=str(metadata.get("rel_path", "")),
            )
            sources[source.chunk_id] = source
            dense_rank[source.chunk_id] = rank

        bm25_rank: dict[str, int] = {}
        if self.use_hybrid:
            for rank, (position, score) in enumerate(self.bm25.search(question, fetch_k)):
                if position >= len(self.bm25.ids):
                    continue
                chunk_id = self.bm25.ids[position]
                metadata = self.bm25.metadatas[position] or {}
                if chunk_id not in sources:
                    sources[chunk_id] = Source(
                        chunk_id=chunk_id,
                        text=self.bm25.documents[position],
                        score=0.0,
                        doc_id=str(metadata.get("doc_id", "")),
                        title=str(metadata.get("title", "")),
                        topic=str(metadata.get("topic", "")),
                        topic_slug=str(metadata.get("topic_slug", "")),
                        page=int(metadata.get("page", 0) or 0),
                        rel_path=str(metadata.get("rel_path", "")),
                    )
                sources[chunk_id].bm25_score = score
                bm25_rank[chunk_id] = rank

        k = settings.rrf_k
        fused: list[Source] = []
        for chunk_id, source in sources.items():
            score = 0.0
            if chunk_id in dense_rank:
                score += 1.0 / (k + dense_rank[chunk_id] + 1)
            if chunk_id in bm25_rank:
                score += 1.0 / (k + bm25_rank[chunk_id] + 1)
            source.score = score
            fused.append(source)
        fused.sort(key=lambda item: item.score, reverse=True)
        return fused[:top_k]

    def stats(self) -> dict:
        bm25_chunks = 0
        if self.use_hybrid:
            try:
                bm25_chunks = len(self.bm25.ids)
            except Exception:
                bm25_chunks = 0
        return {
            "chunks": self.index.count(),
            "bm25_chunks": bm25_chunks,
            "embedding_model": self.settings.embedding_model,
            "hybrid": self.use_hybrid,
            "top_k": self.settings.top_k,
            "dense_max_per_doc": self.settings.dense_max_per_doc,
        }

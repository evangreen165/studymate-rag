from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

import numpy as np
from nltk.stem.snowball import RussianStemmer
from rank_bm25 import BM25Okapi

from .chunking import read_chunks_jsonl
from .config import Settings, get_settings
from .index import NumpyIndex, VectorIndex
from .topics import infer_topic_with_score

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_]{2,}")
_CYR_RE = re.compile(r"[А-Яа-яЁё]")
_DOTS_RE = re.compile(r"\.{5,}")
_TOC_RE = re.compile(r"оглавление|содержание", re.IGNORECASE)
_DEFINITION_RE = re.compile(
    r"(?:—|-|–|:)?\s*(?:это|представляет собой|означает|называется|понимается)\b"
)
_STEMMER = RussianStemmer()
_STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все",
    "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по",
    "только", "ее", "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из",
    "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или",
    "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять", "уж", "вам",
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "for", "on",
    "with", "as", "by", "that", "this", "it", "be", "from", "at", "which",
}

# Расширение запроса англоязычными терминами: корпус смешанный, а шпаргалки по Git — на английском.
_EXPANSIONS: tuple[tuple[str, str], ...] = (
    ("отмен", "reset revert checkout restore"),
    ("коммит", "commit"),
    ("ветк", "branch"),
    ("слиян", "merge"),
    ("перебазир", "rebase"),
    ("рабоч", "working directory"),
    ("файл", "file"),
    ("удал", "remove rm"),
    ("истори", "log history"),
    ("изменен", "changes diff"),
    ("индекс", "index btree"),
    ("соединен", "join"),
    ("таблиц", "table"),
    ("транзакц", "transaction acid"),
    ("класс", "class"),
    ("наследован", "inheritance"),
    ("полиморф", "polymorphism"),
    ("паттерн", "pattern"),
    ("спутник", "satellite orbit"),
    ("маршрутизатор", "router routing"),
    ("протокол", "protocol"),
    ("трехсторонн", "тройное handshake"),
    ("рукопожат", "handshake"),
    ("сложност", "complexity"),
    ("сортиров", "sorting"),
    ("граф", "graph"),
)


@lru_cache(maxsize=200_000)
def _stem_token(token: str) -> str:
    return _STEMMER.stem(token) if _CYR_RE.search(token) else token


def tokenize(text: str) -> list[str]:
    normalized = text.replace("ё", "е").replace("Ё", "Е")
    tokens = [
        token.lower()
        for token in _TOKEN_RE.findall(normalized)
        if token.lower() not in _STOPWORDS
    ]
    return [_stem_token(token) for token in tokens]


def expand_query(question: str) -> str:
    lower = question.lower()
    additions: list[str] = []
    for marker, expansion in _EXPANSIONS:
        if marker in lower:
            additions.extend(expansion.split())
    if not additions:
        return question
    return question + " " + " ".join(dict.fromkeys(additions))


def is_noise_chunk(text: str) -> bool:
    """Отбрасывает оглавления и предметные указатели.

    Такие страницы содержат термины вопроса, но не объясняют их, поэтому
    часто занимают верхние места в BM25.
    """
    if not text:
        return True
    if len(_DOTS_RE.findall(text)) >= 2:
        return True
    head = text[:300].lower()
    if ("оглавление" in head or "содержание" in head) and _DOTS_RE.search(text):
        return True
    if "предметный указатель" in head or "алфавитный указатель" in head:
        return True
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) >= 20:
        short_numeric = sum(
            1
            for line in lines
            if len(line.strip()) < 32 and re.search(r"\d{1,4}\s*$", line)
        )
        if short_numeric / len(lines) > 0.7:
            return True
    return False


def _query_terms(question: str) -> list[str]:
    return [term for term in dict.fromkeys(tokenize(question)) if len(term) >= 3]


def _key_phrases(question: str) -> list[tuple[str, str]]:
    """Пары основ соседних слов вопроса — грубый аналог ключевой фразы."""
    terms = [term for term in _query_terms(question) if len(term) >= 5]
    phrases = list(pairwise(terms))
    if phrases:
        return [(first[:6], second[:6]) for first, second in phrases]
    longest = max(_query_terms(question), key=len, default="")
    return [(longest[:6], "")] if len(longest) >= 5 else []


def _phrase_pattern(first: str, second: str) -> str:
    if second:
        return rf"{re.escape(first)}[а-яa-z]*\s+{re.escape(second)}"
    return rf"{re.escape(first)}"


def _definition_boost(question: str, lowered: str) -> float:
    asks_definition = any(
        marker in question.lower()
        for marker in ("что такое", "определение", "что значит", "что понимают")
    )
    if not asks_definition:
        return 0.0
    for first, second in _key_phrases(question):
        for match in re.finditer(_phrase_pattern(first, second), lowered):
            after = lowered[match.start() : match.start() + 260]
            before = lowered[max(0, match.start() - 120) : match.end()]
            if _DEFINITION_RE.search(after) or _DEFINITION_RE.search(before):
                return 0.35
    return 0.0


def _phrase_boost(question: str, lowered: str) -> float:
    for first, second in _key_phrases(question):
        if second and re.search(_phrase_pattern(first, second), lowered):
            return 0.14
    return 0.0


def _heuristic_boost(question: str, source: Source, topic: str | None) -> float:
    text = source.text
    lowered = text.lower().replace("ё", "е")
    head = lowered[:400]
    terms = _query_terms(question)
    if not terms:
        return 0.0
    boost = 0.0
    if topic and source.topic_slug == topic:
        boost += 0.06
    heading_hits = sum(1 for term in terms if term[:6] in head)
    boost += min(0.10, 0.035 * heading_hits)
    title = source.title.lower()
    if any(term[:6] in title for term in terms):
        boost += 0.04
    boost += _phrase_boost(question, lowered)
    boost += _definition_boost(question, lowered)
    return boost


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
        self.topic_slugs = [str(metadata.get("topic_slug", "")) for metadata in self.metadatas]
        corpus = [tokenize(document) for document in self.documents]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def search(
        self, query: str, top_k: int, topic_slug: str | None = None
    ) -> list[tuple[int, float]]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        if topic_slug and topic_slug != "all":
            mask = np.asarray([slug == topic_slug for slug in self.topic_slugs])
            if mask.any():
                scores = np.where(mask, scores, -np.inf)
        ranked = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in ranked if np.isfinite(scores[i]) and scores[i] > 0]


class Retriever:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        index: VectorIndex | None = None,
        use_hybrid: bool | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if index is None:
            try:
                candidate = VectorIndex(self.settings)
                if candidate.count() == 0:
                    raise RuntimeError("ChromaDB пуст")
                index = candidate
            except Exception:
                index = NumpyIndex(self.settings)
        self.index = index
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

    def _search_once(
        self,
        question: str,
        top_k: int,
        topic_slug: str | None,
        fetch_k: int,
        *,
        hard_filter: bool = False,
    ) -> list[Source]:
        settings = self.settings
        query_embedding = np.asarray(
            self.index.embedder.get_query_embedding(question), dtype=np.float32
        )
        where = self._where(topic_slug) if hard_filter else None
        try:
            payload = self.index.query(
                query_embedding.tolist(),
                top_k=fetch_k,
                where=where,
                include_embeddings=True,
            )
        except Exception:
            if isinstance(self.index, NumpyIndex):
                raise
            self.index = NumpyIndex(self.settings)
            payload = self.index.query(
                query_embedding.tolist(),
                top_k=fetch_k,
                where=where,
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
                query_embedding,
                embeddings,
                dense_order,
                min(fetch_k, max(top_k * 4, 24)),
                settings.mmr_lambda,
            )
        else:
            mmr_order = dense_order

        sources: dict[str, Source] = {}
        dense_rank: dict[str, int] = {}
        for rank, position in enumerate(mmr_order):
            text = str(documents[position])
            if is_noise_chunk(text):
                continue
            metadata = metadatas[position] or {}
            source = Source(
                chunk_id=str(ids[position]),
                text=text,
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
            for rank, (position, score) in enumerate(
                self.bm25.search(
                    expand_query(question),
                    fetch_k * 3,
                    topic_slug if hard_filter else None,
                )
            ):
                if position >= len(self.bm25.ids):
                    continue
                chunk_id = self.bm25.ids[position]
                text = self.bm25.documents[position]
                if is_noise_chunk(text):
                    continue
                metadata = self.bm25.metadatas[position] or {}
                if chunk_id not in sources:
                    sources[chunk_id] = Source(
                        chunk_id=chunk_id,
                        text=text,
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

        if not sources:
            return []

        k = settings.rrf_k
        max_bm25 = max((source.bm25_score for source in sources.values()), default=0.0)
        dense_values = [source.dense_score for source in sources.values() if source.dense_score > 0]
        min_dense = min(dense_values) if dense_values else 0.0
        max_dense = max(dense_values) if dense_values else 0.0
        dense_range = max_dense - min_dense

        for chunk_id, source in sources.items():
            rrf = 0.0
            if chunk_id in dense_rank:
                rrf += 1.0 / (k + dense_rank[chunk_id] + 1)
            if chunk_id in bm25_rank:
                rrf += 1.0 / (k + bm25_rank[chunk_id] + 1)
            rrf_norm = min(1.0, rrf * (k + 1))
            bm25_norm = source.bm25_score / max_bm25 if max_bm25 > 0 else 0.0
            if source.dense_score <= 0:
                dense_norm = 0.0
            elif dense_range > 1e-6:
                dense_norm = (source.dense_score - min_dense) / dense_range
            else:
                dense_norm = 1.0
            lowered = source.text.lower().replace("ё", "е")
            terms = _query_terms(question)
            coverage = (
                sum(1 for term in terms if term[:6] in lowered) / len(terms)
                if terms
                else 0.0
            )
            source.score = (
                0.28 * dense_norm
                + 0.34 * bm25_norm
                + 0.24 * coverage
                + 0.14 * rrf_norm
                + _heuristic_boost(question, source, topic_slug)
            )

        fused = sorted(sources.values(), key=lambda item: (item.score, item.bm25_score), reverse=True)
        selected: list[Source] = []
        per_page: dict[tuple[str, int], int] = {}
        for source in fused:
            key = (source.doc_id, source.page)
            if per_page.get(key, 0) >= 2:
                continue
            per_page[key] = per_page.get(key, 0) + 1
            selected.append(source)
            if len(selected) >= top_k:
                break
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
        explicit_topic = topic_slug if topic_slug and topic_slug != "all" else None
        routed_topic, confidence = infer_topic_with_score(question)
        if explicit_topic:
            routed_topic = explicit_topic
            hard_filter = True
        else:
            hard_filter = confidence >= 4
        fetch_k = max(80, settings.fetch_k, top_k * 10)
        return self._search_once(
            question,
            top_k,
            routed_topic,
            fetch_k,
            hard_filter=hard_filter,
        )

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


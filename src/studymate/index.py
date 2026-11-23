from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from .chunking import Chunk, collect_chunks, read_chunks_jsonl
from .config import Settings, get_settings
from .utils import read_json, sha256_text, utc_now_iso, write_json


class HashEmbedder:
    """Детерминированный эмбеддер для тестов: не требует скачивания модели."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().split():
            vector[hash(token) % self.dim] += 1.0
        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
        return vector.tolist()

    def get_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)

    def get_query_embedding(self, text: str) -> list[float]:
        return self._embed(text)

    def get_text_embedding_batch(self, texts: Sequence[str], **_: object) -> list[list[float]]:
        return [self._embed(text) for text in texts]


def build_embedder(settings: Settings | None = None) -> HuggingFaceEmbedding:
    settings = settings or get_settings()
    return HuggingFaceEmbedding(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        embed_batch_size=settings.embedding_batch_size,
        max_length=settings.embedding_max_length,
        normalize=True,
        show_progress_bar=False,
    )


@dataclass
class IndexMeta:
    built_at: str
    mode: str
    embedding_model: str
    chunks: int
    documents: int
    build_seconds: float
    chunk_size: int
    chunk_overlap: int
    corpus_hash: str


class VectorIndex:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder: object | None = None,
        path: Path | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.path = path or self.settings.index_dir
        self.collection_name = collection_name or self.settings.collection_name
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.path),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = embedder
        self.last_ids: list[str] = []
        self.last_embeddings: list[list[float]] = []

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = build_embedder(self.settings)
        return self._embedder

    def count(self) -> int:
        return int(self.collection.count())

    def reset(self) -> None:
        with contextlib.suppress(Exception):
            self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: Iterable[Chunk],
        *,
        batch_size: int = 32,
        progress: Callable[[int, int], None] | None = None,
        upsert: bool = True,
    ) -> int:
        chunk_list = list(chunks)
        total = len(chunk_list)
        added = 0
        self.last_ids = []
        self.last_embeddings = []
        for start in range(0, total, batch_size):
            batch = chunk_list[start : start + batch_size]
            texts = [chunk.text for chunk in batch]
            embeddings = self.embedder.get_text_embedding_batch(texts)
            metadatas = []
            for chunk in batch:
                metadata = {
                    key: value
                    for key, value in chunk.metadata.items()
                    if isinstance(value, (str, int, float, bool))
                }
                metadata["chunk_id"] = chunk.chunk_id
                metadatas.append(metadata)
            method = self.collection.upsert if upsert else self.collection.add
            method(
                ids=[chunk.chunk_id for chunk in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            self.last_ids.extend(chunk.chunk_id for chunk in batch)
            self.last_embeddings.extend(embeddings)
            added += len(batch)
            if progress:
                progress(added, total)
        return added

    def verify(self) -> bool:
        """Проверяет, что HNSW-индекс читается, и заставляет его сохраниться на диск."""
        payload = self.collection.get(limit=1, include=["embeddings"])
        embeddings = payload.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return False
        self.collection.query(query_embeddings=[list(embeddings[0])], n_results=1)
        return True

    def close(self) -> None:
        """Закрывает клиент Chroma, чтобы HNSW-файлы были сброшены на диск."""
        import gc

        with contextlib.suppress(AttributeError):
            del self.collection
        with contextlib.suppress(Exception):
            self.client.clear_system_cache()
        del self.client
        gc.collect()

    def build(
        self,
        chunks: Iterable[Chunk],
        *,
        rebuild: bool = True,
        progress: Callable[[int, int], None] | None = None,
    ) -> int:
        if rebuild:
            self.reset()
        return self.add_chunks(chunks, progress=progress, upsert=not rebuild)

    def query(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        where: dict | None = None,
        include_embeddings: bool = False,
    ) -> dict:
        include = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=include,
        )

    def get_all(self, *, include_documents: bool = True) -> dict:
        include = ["metadatas"]
        if include_documents:
            include.append("documents")
        return self.collection.get(include=include)

    def delete_document(self, doc_id: str) -> int:
        existing = self.collection.get(where={"doc_id": doc_id}, include=[])
        ids = existing.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def write_meta(self, meta: IndexMeta | dict, path: Path | None = None) -> None:
        payload = meta.__dict__ if isinstance(meta, IndexMeta) else meta
        write_json(path or self.settings.index_meta_path, payload)

    def read_meta(self, path: Path | None = None) -> dict:
        return read_json(path or self.settings.index_meta_path, default={})


class NumpyIndex:
    """Резервный плотный индекс на NumPy: используется, если HNSW-сегмент Chroma не читается."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder: object | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._embedder = embedder
        self.cache_dir = self.settings.cache_dir
        npz_path = self.cache_dir / "dense_embeddings.npz"
        chunks_path = self.cache_dir / "dense_chunks.jsonl"
        if not npz_path.exists() or not chunks_path.exists():
            raise FileNotFoundError("Нет кэша плотных эмбеддингов для NumPy-индекса")
        payload = np.load(npz_path, allow_pickle=False)
        self.ids = [str(value) for value in payload["ids"].tolist()]
        self.vectors = np.asarray(payload["embeddings"], dtype=np.float32)
        chunks = read_chunks_jsonl(chunks_path)
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.documents = [by_id[chunk_id].text if chunk_id in by_id else "" for chunk_id in self.ids]
        self.metadatas = [by_id[chunk_id].metadata if chunk_id in by_id else {} for chunk_id in self.ids]
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / np.maximum(norms, 1e-9)

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = build_embedder(self.settings)
        return self._embedder

    def count(self) -> int:
        return len(self.ids)

    def query(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        where: dict | None = None,
        include_embeddings: bool = False,
    ) -> dict:
        query = np.asarray(query_embedding, dtype=np.float32)
        query = query / max(float(np.linalg.norm(query)), 1e-9)
        scores = self.vectors @ query
        candidates = list(range(len(self.ids)))
        if where:
            candidates = [
                index
                for index in candidates
                if all(self.metadatas[index].get(key) == value for key, value in where.items())
            ]
        candidates.sort(key=lambda index: scores[index], reverse=True)
        selected = candidates[:top_k]
        result = {
            "ids": [[self.ids[index] for index in selected]],
            "documents": [[self.documents[index] for index in selected]],
            "metadatas": [[self.metadatas[index] for index in selected]],
            "distances": [[1.0 - float(scores[index]) for index in selected]],
        }
        if include_embeddings:
            result["embeddings"] = [[self.vectors[index].tolist() for index in selected]]
        return result

    def get_all(self, *, include_documents: bool = True) -> dict:
        payload = {"ids": self.ids, "metadatas": self.metadatas}
        if include_documents:
            payload["documents"] = self.documents
        return payload

    def delete_document(self, doc_id: str) -> int:
        keep = [index for index, meta in enumerate(self.metadatas) if meta.get("doc_id") != doc_id]
        removed = len(self.ids) - len(keep)
        self.ids = [self.ids[index] for index in keep]
        self.documents = [self.documents[index] for index in keep]
        self.metadatas = [self.metadatas[index] for index in keep]
        self.vectors = self.vectors[keep]
        return removed

    def read_meta(self, path: Path | None = None) -> dict:
        return read_json(path or self.settings.index_meta_path, default={})

    def write_meta(self, meta: dict, path: Path | None = None) -> None:
        write_json(path or self.settings.index_meta_path, meta)

    def close(self) -> None:
        return None


def build_index(
    settings: Settings | None = None,
    *,
    demo: bool = False,
    max_docs: int | None = None,
    rebuild: bool = True,
    embedder: object | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    settings = settings or get_settings()
    started = time.perf_counter()
    chunks = collect_chunks(settings, max_docs=max_docs)
    index = VectorIndex(settings, embedder=embedder, path=settings.index_dir)
    added = index.build(chunks, rebuild=rebuild, progress=progress)
    elapsed = time.perf_counter() - started
    from .corpus import load_manifest

    manifest = load_manifest(settings)
    corpus_hash = sha256_text("|".join(sorted(chunk.chunk_id for chunk in chunks)))
    meta = IndexMeta(
        built_at=utc_now_iso(),
        mode="demo" if demo else "full",
        embedding_model=settings.embedding_model,
        chunks=added,
        documents=int(manifest.get("summary", {}).get("unique_documents", 0)),
        build_seconds=round(elapsed, 2),
        chunk_size=settings.chunk_size_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
        corpus_hash=corpus_hash,
    )
    index.write_meta(meta)
    return {
        "chunks": added,
        "documents": meta.documents,
        "build_seconds": meta.build_seconds,
        "mode": meta.mode,
        "embedding_model": meta.embedding_model,
    }


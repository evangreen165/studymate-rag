from __future__ import annotations

import threading
import time
import traceback
from collections import Counter
from collections.abc import Iterable

from ..chunking import (
    collect_chunks,
    read_chunks_jsonl,
    sample_chunks,
    write_chunks_jsonl,
)
from ..config import Settings, get_settings
from ..corpus import load_manifest
from ..index import VectorIndex
from ..utils import sha256_text, utc_now_iso


class IndexCancelled(RuntimeError):
    pass


class IndexJobManager:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._state: dict = {
            "status": "idle",
            "progress": 0,
            "embedded": 0,
            "total": 0,
            "message": "Индекс готов к построению",
            "error": None,
            "started_at": None,
            "finished_at": None,
            "rebuild": False,
            "mode": None,
            "chunks": 0,
        }

    def status(self) -> dict:
        with self._lock:
            return dict(self._state)

    def start(
        self,
        *,
        rebuild: bool = False,
        doc_ids: Iterable[str] | None = None,
    ) -> dict:
        with self._lock:
            if self._state["status"] == "running":
                raise RuntimeError("Индексация уже выполняется")
            self._cancel.clear()
            self._state.update(
                {
                    "status": "running",
                    "progress": 0,
                    "embedded": 0,
                    "total": 0,
                    "message": "Подготовка чанков…",
                    "error": None,
                    "started_at": utc_now_iso(),
                    "finished_at": None,
                    "rebuild": rebuild,
                    "chunks": 0,
                }
            )
            self._thread = threading.Thread(
                target=self._run,
                kwargs={"rebuild": rebuild, "doc_ids": set(doc_ids) if doc_ids else None},
                daemon=True,
                name="studymate-indexer",
            )
            self._thread.start()
            return dict(self._state)

    def cancel(self) -> dict:
        self._cancel.set()
        with self._lock:
            if self._state["status"] == "running":
                self._state["message"] = "Остановка после текущего батча…"
            return dict(self._state)

    def _update(self, **fields: object) -> None:
        with self._lock:
            self._state.update(fields)

    def _progress(self, embedded: int, total: int) -> None:
        if self._cancel.is_set():
            raise IndexCancelled("Индексация отменена пользователем")
        percent = int(embedded / total * 100) if total else 0
        self._update(
            embedded=embedded,
            total=total,
            progress=percent,
            message=f"Эмбеддинги: {embedded} из {total} чанков",
        )

    def _run(self, *, rebuild: bool, doc_ids: set[str] | None) -> None:
        started = time.perf_counter()
        try:
            chunks = collect_chunks(self.settings, doc_ids=doc_ids)
            total = len(chunks)
            if total == 0:
                raise RuntimeError(
                    "Не найдено ни одного чанка. Проверьте, что PDF содержат текстовый слой."
                )
            dense_chunks = sample_chunks(chunks, self.settings.dense_max_per_doc)
            self._update(
                total=len(dense_chunks),
                message=f"Найдено чанков: {total}; плотных: {len(dense_chunks)}",
            )
            index = VectorIndex(self.settings)
            if rebuild:
                self._update(message="Очистка старого индекса…")
                index.reset()
            added = index.add_chunks(dense_chunks, progress=self._progress, upsert=not rebuild)
            index.verify()

            bm25_chunks = chunks
            if not rebuild and doc_ids and self.settings.bm25_path.exists():
                existing = [
                    chunk
                    for chunk in read_chunks_jsonl(self.settings.bm25_path)
                    if chunk.doc_id not in doc_ids
                ]
                bm25_chunks = existing + chunks
            write_chunks_jsonl(bm25_chunks, self.settings.bm25_path)

            counts = Counter(chunk.doc_id for chunk in dense_chunks)
            bm25_counts = Counter(chunk.doc_id for chunk in bm25_chunks)
            manifest = load_manifest(self.settings)
            meta = {
                "built_at": utc_now_iso(),
                "mode": manifest.get("mode", "full"),
                "embedding_model": self.settings.embedding_model,
                "chunks": added,
                "dense_chunks": added,
                "bm25_chunks": len(bm25_chunks),
                "documents": len(bm25_counts),
                "build_seconds": round(time.perf_counter() - started, 2),
                "chunk_size": self.settings.chunk_size_tokens,
                "chunk_overlap": self.settings.chunk_overlap_tokens,
                "dense_max_per_doc": self.settings.dense_max_per_doc,
                "corpus_hash": sha256_text("|".join(sorted(chunk.chunk_id for chunk in bm25_chunks))),
                "document_chunks": dict(counts),
                "bm25_document_chunks": dict(bm25_counts),
                "rebuild": rebuild,
            }
            index.write_meta(meta)
            index.close()
            self._update(
                status="done",
                progress=100,
                chunks=added,
                message=f"Готово: {added} плотных векторов, {len(bm25_chunks)} чанков BM25",
                finished_at=utc_now_iso(),
                mode=meta["mode"],
            )
        except IndexCancelled as exc:
            self._update(
                status="cancelled",
                message=str(exc),
                finished_at=utc_now_iso(),
            )
        except Exception as exc:
            self._update(
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                message="Ошибка индексации",
                finished_at=utc_now_iso(),
            )
            traceback.print_exc()

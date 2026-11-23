from __future__ import annotations

import argparse
import time
from collections import Counter

from studymate.chunking import (
    chunk_stats,
    collect_chunks,
    sample_chunks,
    write_chunks_jsonl,
)
from studymate.config import get_settings
from studymate.corpus import build_manifest
from studymate.index import VectorIndex
from studymate.utils import sha256_text, utc_now_iso, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Построение гибридного индекса StudyMate")
    parser.add_argument("--demo", action="store_true", help="Индексировать demo-корпус")
    parser.add_argument("--max-docs", type=int, default=None, help="Ограничить число PDF")
    parser.add_argument("--limit-chunks", type=int, default=None, help="Ограничить число чанков")
    parser.add_argument("--dense-max-per-doc", type=int, default=None, help="Плотных чанков на документ")
    parser.add_argument("--no-rebuild", action="store_true", help="Добавить в существующий индекс")
    parser.add_argument("--skip-manifest", action="store_true", help="Не пересобирать манифест")
    parser.add_argument("--no-bm25", action="store_true", help="Не сохранять BM25-корпус")
    args = parser.parse_args()

    settings = get_settings()
    if not args.skip_manifest:
        print("Сборка манифеста…")
        build_manifest(settings, demo=args.demo, max_docs=args.max_docs)

    print("Чанкинг полного корпуса…")
    all_chunks = collect_chunks(settings, max_docs=args.max_docs)
    if args.limit_chunks:
        all_chunks = all_chunks[: args.limit_chunks]
    dense_limit = args.dense_max_per_doc or settings.dense_max_per_doc
    dense_chunks = sample_chunks(all_chunks, dense_limit)
    print(f"Всего чанков: {len(all_chunks)}; плотный индекс: {len(dense_chunks)}")
    print("Статистика плотного индекса:", chunk_stats(dense_chunks))

    if not args.no_bm25:
        write_chunks_jsonl(all_chunks, settings.bm25_path)
        print(f"BM25-корпус: {settings.bm25_path} ({len(all_chunks)} чанков)")

    index = VectorIndex(settings)
    if not args.no_rebuild:
        index.reset()
    started = time.perf_counter()

    def progress(done: int, total: int) -> None:
        percent = int(done / total * 100) if total else 0
        print(f"  {done}/{total} ({percent}%)", flush=True)

    added = index.add_chunks(dense_chunks, progress=progress, upsert=args.no_rebuild)
    index.verify()
    if index.last_embeddings:
        import numpy as np

        write_chunks_jsonl(dense_chunks, settings.cache_dir / "dense_chunks.jsonl")
        np.savez_compressed(
            settings.cache_dir / "dense_embeddings.npz",
            ids=np.asarray(index.last_ids),
            embeddings=np.asarray(index.last_embeddings, dtype=np.float32),
        )
    index.close()
    elapsed = time.perf_counter() - started
    dense_counts = Counter(chunk.doc_id for chunk in dense_chunks)
    bm25_counts = Counter(chunk.doc_id for chunk in all_chunks)
    meta = {
        "built_at": utc_now_iso(),
        "mode": "demo" if args.demo else "full",
        "embedding_model": settings.embedding_model,
        "chunks": added,
        "dense_chunks": added,
        "bm25_chunks": len(all_chunks),
        "documents": len({chunk.doc_id for chunk in all_chunks}),
        "build_seconds": round(elapsed, 2),
        "chunk_size": settings.chunk_size_tokens,
        "chunk_overlap": settings.chunk_overlap_tokens,
        "dense_max_per_doc": dense_limit,
        "corpus_hash": sha256_text("|".join(sorted(chunk.chunk_id for chunk in all_chunks))),
        "document_chunks": dict(dense_counts),
        "bm25_document_chunks": dict(bm25_counts),
    }
    write_json(settings.index_meta_path, meta)
    print(f"Готово: {added} плотных векторов за {elapsed:.1f} с")
    print(f"Индекс: {settings.index_dir}")
    print(f"Мета: {settings.index_meta_path}")


if __name__ == "__main__":
    main()

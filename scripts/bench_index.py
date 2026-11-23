from __future__ import annotations

import time
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from studymate.chunking import read_chunks_jsonl
from studymate.config import get_settings
from studymate.index import build_embedder

settings = get_settings()
chunks = read_chunks_jsonl(Path("data/bm25_corpus.json"))
subset = chunks[:: max(1, len(chunks) // 64)][:64]
texts = [chunk.text for chunk in subset]
print("chunks", len(texts))

started = time.perf_counter()
embedder = build_embedder(settings)
print("load", round(time.perf_counter() - started, 2))
started = time.perf_counter()
vectors = embedder.get_text_embedding_batch(texts)
print("embed 64", round(time.perf_counter() - started, 2))

path = Path("data/cache/bench_chroma")
client = chromadb.PersistentClient(
    path=str(path), settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True)
)
collection = client.get_or_create_collection("bench", metadata={"hnsw:space": "cosine"})
started = time.perf_counter()
for start in range(0, len(texts), 32):
    batch = subset[start : start + 32]
    collection.upsert(
        ids=[chunk.chunk_id for chunk in batch],
        embeddings=vectors[start : start + 32],
        documents=[chunk.text for chunk in batch],
        metadatas=[chunk.metadata for chunk in batch],
    )
print("chroma 64", round(time.perf_counter() - started, 2))
print("count", collection.count())

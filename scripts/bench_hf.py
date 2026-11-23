from __future__ import annotations

import json
import time
from pathlib import Path

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import SentenceTransformer

rows = []
with Path("data/bm25_corpus.json").open("r", encoding="utf-8") as fh:
    for line in fh:
        if line.strip():
            rows.append(json.loads(line))
texts = [row["text"] for row in rows[:: max(1, len(rows) // 48)]][:48]
print("texts", len(texts), "avg chars", sum(map(len, texts)) / len(texts))

for max_length in (512, 1024):
    started = time.perf_counter()
    model = HuggingFaceEmbedding(
        model_name="deepvk/USER2-base",
        device="cpu",
        embed_batch_size=16,
        max_length=max_length,
        normalize=True,
        show_progress_bar=False,
    )
    load = time.perf_counter() - started
    started = time.perf_counter()
    model.get_text_embedding_batch(texts)
    elapsed = time.perf_counter() - started
    print(
        f"HF max_length={max_length} load={load:.1f}s encode={elapsed:.1f}s "
        f"per_chunk={elapsed / len(texts):.3f}s"
    )

started = time.perf_counter()
st = SentenceTransformer("deepvk/USER2-base", device="cpu")
st.max_seq_length = 512
print("ST load", round(time.perf_counter() - started, 1))
started = time.perf_counter()
st.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
elapsed = time.perf_counter() - started
print(f"ST max_length=512 encode={elapsed:.1f}s per_chunk={elapsed / len(texts):.3f}s")

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from sentence_transformers import SentenceTransformer


def load_texts(count: int = 256) -> list[str]:
    texts: list[str] = []
    for path in list(Path("data/cache/extracted").glob("*.json"))[:10]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for page in payload["pages"][:10]:
            if len(page["text"]) > 1200:
                texts.append(page["text"][:2600])
    return (texts * 5)[:count]


if __name__ == "__main__":
    torch.set_num_threads(2)
    texts = load_texts(256)
    started = time.perf_counter()
    model = SentenceTransformer("deepvk/USER2-base", device="cpu")
    print("load", round(time.perf_counter() - started, 2), flush=True)

    for workers in (2, 4):
        torch.set_num_threads(2)
        started = time.perf_counter()
        pool = model.start_multi_process_pool(target_devices=["cpu"] * workers)
        started_encode = time.perf_counter()
        embeddings = model.encode_multi_process(
            texts, pool, batch_size=32, normalize_embeddings=True
        )
        elapsed = time.perf_counter() - started_encode
        model.stop_multi_process_pool(pool)
        print(
            f"workers={workers} chunks={len(texts)} encode={elapsed:.2f}s "
            f"per_chunk={elapsed / len(texts):.3f}s shape={embeddings.shape}",
            flush=True,
        )

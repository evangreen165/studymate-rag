from __future__ import annotations

import json
from pathlib import Path

from studymate.config import get_settings
from studymate.evaluation import load_eval_items
from studymate.retrieval import Retriever, Source


def is_gold(source: Source, item) -> bool:
    return source.doc_id == item.doc_id and abs(int(source.page) - int(item.page)) <= 1


def hit_ratio(sources: list[Source], item) -> float:
    return 1.0 if any(is_gold(source, item) for source in sources) else 0.0


def main() -> None:
    settings = get_settings()
    items = load_eval_items(Path("data/eval/test.jsonl"))
    hybrid = Retriever(settings, use_hybrid=True)
    dense = Retriever(settings, use_hybrid=False)

    result: dict = {}
    for k in (3, 4, 6, 8):
        hits = [
            hit_ratio(hybrid.search(item.question, top_k=k), item)
            for item in items
        ]
        result[f"top_k_{k}"] = {"gold_hit": round(sum(hits) / len(hits), 3)}

    dense_hits = [hit_ratio(dense.search(item.question, top_k=6), item) for item in items]
    result["dense_only"] = {"gold_hit": round(sum(dense_hits) / len(dense_hits), 3)}

    bm25_hits = []
    bm25 = hybrid.bm25
    for item in items:
        ranked = bm25.search(item.question, 6)
        sources = []
        for position, score in ranked:
            metadata = bm25.metadatas[position]
            sources.append(
                Source(
                    chunk_id=bm25.ids[position],
                    text=bm25.documents[position],
                    score=score,
                    doc_id=str(metadata.get("doc_id", "")),
                    title=str(metadata.get("title", "")),
                    topic=str(metadata.get("topic", "")),
                    topic_slug=str(metadata.get("topic_slug", "")),
                    page=int(metadata.get("page", 0) or 0),
                    rel_path=str(metadata.get("rel_path", "")),
                )
            )
        bm25_hits.append(hit_ratio(sources, item))
    result["bm25_only"] = {"gold_hit": round(sum(bm25_hits) / len(bm25_hits), 3)}

    hybrid_hits = [hit_ratio(hybrid.search(item.question, top_k=6), item) for item in items]
    result["hybrid"] = {"gold_hit": round(sum(hybrid_hits) / len(hybrid_hits), 3)}
    result["n"] = len(items)
    Path("results").mkdir(exist_ok=True)
    Path("results/retrieval_experiments.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

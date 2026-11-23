from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

import numpy as np

from .config import Settings, get_settings
from .rag import RAGService


@dataclass
class EvalItem:
    id: str
    question: str
    reference: str
    topic: str = ""
    topic_slug: str = ""
    doc_id: str = ""
    title: str = ""
    page: int = 0

    @classmethod
    def from_dict(cls, payload: dict) -> "EvalItem":
        return cls(
            id=str(payload.get("id", "")),
            question=str(payload.get("question", "")),
            reference=str(payload.get("answer") or payload.get("reference") or ""),
            topic=str(payload.get("topic", "")),
            topic_slug=str(payload.get("topic_slug", "")),
            doc_id=str(payload.get("doc_id", "")),
            title=str(payload.get("title", "")),
            page=int(payload.get("page", 0) or 0),
        )


def load_eval_items(path: Path, *, limit: int | None = None) -> list[EvalItem]:
    items: list[EvalItem] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            items.append(EvalItem.from_dict(json.loads(line)))
            if limit and len(items) >= limit:
                break
    return items


def run_rag(
    service: RAGService,
    items: Iterable[EvalItem],
    *,
    provider: str | None = None,
    model: str | None = None,
    top_k: int | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for index, item in enumerate(items, start=1):
        print(f"  RAG {index}: {item.question[:70]}", flush=True)
        try:
            result = service.answer(
                item.question, top_k=top_k, provider=provider, model=model
            )
            payload = result.to_dict()
            payload["reference"] = item.reference
            payload["topic"] = item.topic
            payload["topic_slug"] = item.topic_slug
            payload["gold_doc_id"] = item.doc_id
            payload["gold_page"] = item.page
            payload["gold_title"] = item.title
            payload["id"] = item.id
            rows.append(payload)
        except Exception as exc:  # noqa: BLE001 - один сбой не должен ломать прогон
            rows.append(
                {
                    "id": item.id,
                    "question": item.question,
                    "answer": "",
                    "reference": item.reference,
                    "contexts": [],
                    "sources": [],
                    "topic": item.topic,
                    "topic_slug": item.topic_slug,
                    "gold_doc_id": item.doc_id,
                    "gold_page": item.page,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows


def run_no_context(
    service: RAGService,
    items: Iterable[EvalItem],
    *,
    provider: str | None = None,
    model: str | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for index, item in enumerate(items, start=1):
        print(f"  baseline {index}: {item.question[:70]}", flush=True)
        try:
            result = service.answer_no_context(
                item.question, provider=provider, model=model
            )
            rows.append(
                {
                    "id": item.id,
                    "question": item.question,
                    "answer": result["answer"],
                    "reference": item.reference,
                    "contexts": [],
                    "sources": [],
                    "topic": item.topic,
                    "topic_slug": item.topic_slug,
                    "gold_doc_id": item.doc_id,
                    "gold_page": item.page,
                    "usage": result.get("usage", {}),
                    "latency_ms": result.get("latency_ms", 0),
                    "provider": result.get("provider"),
                    "model": result.get("model"),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "id": item.id,
                    "question": item.question,
                    "answer": "",
                    "reference": item.reference,
                    "contexts": [],
                    "sources": [],
                    "topic": item.topic,
                    "topic_slug": item.topic_slug,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows


def build_ragas_components(settings: Settings, judge_model: str | None = None):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY не задан — RAGAS-судья недоступен")
    judge = ChatOpenAI(
        model=judge_model or settings.llm_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=0.0,
        timeout=settings.llm_timeout_s,
        max_retries=3,
        default_headers={
            "HTTP-Referer": "https://github.com/evangreen165/studymate-rag",
            "X-Title": "StudyMate RAGAS judge",
        },
    )
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainLLMWrapper(judge), LangchainEmbeddingsWrapper(embeddings)


def ragas_evaluate(
    rows: list[dict],
    settings: Settings,
    *,
    judge_model: str | None = None,
    metrics: list | None = None,
    max_workers: int = 4,
):
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.run_config import RunConfig

    llm, embeddings = build_ragas_components(settings, judge_model)
    dataset_rows = [
        {
            "user_input": row["question"],
            "retrieved_contexts": row.get("contexts", []),
            "response": row.get("answer", ""),
            "reference": row.get("reference", ""),
        }
        for row in rows
    ]
    dataset = EvaluationDataset.from_list(dataset_rows)
    chosen = metrics or [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ]
    return evaluate(
        dataset,
        metrics=chosen,
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=max_workers, timeout=240),
        raise_exceptions=False,
        show_progress=True,
    )


def bootstrap_ci(values: list[float], *, n: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    clean = [value for value in values if value is not None and not math.isnan(value)]
    if not clean:
        return (float("nan"), float("nan"))
    rng = random.Random(42)
    means = []
    for _ in range(n):
        sample = [rng.choice(clean) for _ in clean]
        means.append(mean(sample))
    means.sort()
    low = means[int((alpha / 2) * n)]
    high = means[int((1 - alpha / 2) * n) - 1]
    return (round(low, 4), round(high, 4))


def summarize_scores(frame) -> dict:
    summary: dict[str, dict] = {}
    for column in frame.columns:
        if column in {"user_input", "retrieved_contexts", "response", "reference"}:
            continue
        values = []
        for value in frame[column].tolist():
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isnan(number):
                values.append(number)
        if not values:
            continue
        low, high = bootstrap_ci(values)
        summary[str(column)] = {
            "mean": round(float(np.mean(values)), 4),
            "median": round(float(np.median(values)), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "n": len(values),
            "ci95": [low, high],
        }
    return summary


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def error_examples(frame, rows: list[dict], *, limit: int = 10) -> list[dict]:
    if "faithfulness" in frame.columns:
        order = frame["faithfulness"].fillna(1.0).sort_values().index.tolist()
    elif "answer_correctness" in frame.columns:
        order = frame["answer_correctness"].fillna(1.0).sort_values().index.tolist()
    else:
        order = list(range(len(rows)))
    examples = []
    for position in order[:limit]:
        if position >= len(rows):
            continue
        row = rows[position]
        record = frame.iloc[position].to_dict()
        examples.append(
            {
                "question": row.get("question"),
                "answer": row.get("answer"),
                "reference": row.get("reference"),
                "sources": [
                    {
                        "title": source.get("title"),
                        "page": source.get("page"),
                        "score": source.get("score"),
                    }
                    for source in (row.get("sources") or [])[:5]
                ],
                "scores": {
                    key: record.get(key)
                    for key in (
                        "faithfulness",
                        "answer_relevancy",
                        "context_precision",
                        "context_recall",
                        "answer_correctness",
                    )
                    if key in record
                },
            }
        )
    return examples

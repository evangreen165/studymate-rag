from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from studymate.config import get_settings
from studymate.evaluation import (
    error_examples,
    load_eval_items,
    ragas_evaluate,
    run_no_context,
    run_rag,
    save_json,
    summarize_scores,
)
from studymate.rag import RAGService

RESULTS = Path("results")
EVAL = Path("data/eval")


def gold_hit(row: dict) -> bool:
    gold_doc = row.get("gold_doc_id")
    gold_page = row.get("gold_page")
    if not gold_doc:
        return False
    for source in row.get("sources") or []:
        if source.get("doc_id") == gold_doc and abs(int(source.get("page", 0)) - int(gold_page or 0)) <= 1:
            return True
    return False


def to_frame(result, rows: list[dict]) -> pd.DataFrame:
    frame = result.to_pandas()
    frame["topic"] = [row.get("topic", "") for row in rows[: len(frame)]]
    frame["topic_slug"] = [row.get("topic_slug", "") for row in rows[: len(frame)]]
    frame["gold_hit"] = [gold_hit(row) for row in rows[: len(frame)]]
    return frame


def per_topic(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in frame.columns:
        return pd.DataFrame()
    return (
        frame.groupby("topic")[metric]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
        .round(4)
    )


def write_report(
    split: str,
    frame: pd.DataFrame,
    summary: dict,
    baseline_frame: pd.DataFrame | None,
    baseline_summary: dict | None,
    errors: list[dict],
    raw_rows: list[dict],
    meta: dict,
) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"ragas_report_{split}.md"
    lines = [
        f"# Оценка качества RAG — {split}",
        "",
        f"Вопросов: {len(frame)}. Модель: `{meta.get('model')}`. "
        f"Судья: `{meta.get('judge')}`. Top-k: {meta.get('top_k')}.",
        "",
        "## Основные метрики",
        "",
        "| Метрика | Среднее | 95% CI | Медиана | Мин | Макс | N |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in summary.items():
        lines.append(
            f"| {name} | {values['mean']:.4f} | "
            f"{values['ci95'][0]:.4f}–{values['ci95'][1]:.4f} | "
            f"{values['median']:.4f} | {values['min']:.4f} | {values['max']:.4f} | {values['n']} |"
        )
    if baseline_summary:
        lines += [
            "",
            "## Baseline: LLM без поиска",
            "",
            "| Метрика | RAG | Baseline |",
            "| --- | ---: | ---: |",
        ]
        for name, values in baseline_summary.items():
            rag_value = summary.get(name, {}).get("mean")
            rag_text = f"{rag_value:.4f}" if rag_value is not None else "—"
            lines.append(f"| {name} | {rag_text} | {values['mean']:.4f} |")
    lines += ["", "## По темам (Faithfulness)", ""]
    topic_table = per_topic(frame, "faithfulness")
    if not topic_table.empty:
        lines.append("| Тема | Faithfulness | Вопросов |")
        lines.append("| --- | ---: | ---: |")
        for topic, row in topic_table.iterrows():
            lines.append(f"| {topic} | {row['mean']:.4f} | {int(row['count'])} |")
    lines += ["", "## Точность попадания в gold-страницу", ""]
    lines.append(f"Доля вопросов, где среди источников есть gold-документ и страница ±1: **{frame['gold_hit'].mean():.3f}**")
    lines += ["", "## 10 характерных ошибок", ""]
    for index, example in enumerate(errors, start=1):
        lines.append(f"### {index}. {example['question']}")
        lines.append("")
        lines.append(f"**Ответ:** {example['answer']}")
        lines.append("")
        lines.append(f"**Эталон:** {example['reference']}")
        lines.append("")
        scores = ", ".join(f"{key}={value}" for key, value in example["scores"].items())
        lines.append(f"**Оценки:** {scores}")
        lines.append("")
        sources = "; ".join(
            f"{source['title']} стр. {source['page']}" for source in example["sources"]
        )
        lines.append(f"**Источники:** {sources}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS-оценка StudyMate")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--skip-ragas", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    items = load_eval_items(EVAL / f"{args.split}.jsonl", limit=args.limit)
    print(f"Вопросов: {len(items)}")
    service = RAGService(settings)

    rows = run_rag(service, items, top_k=args.top_k)
    RESULTS.mkdir(parents=True, exist_ok=True)
    save_json(RESULTS / f"rag_{args.split}.json", rows)

    baseline_rows = []
    baseline_frame = None
    baseline_summary = None
    if not args.no_baseline:
        baseline_rows = run_no_context(service, items)
        save_json(RESULTS / f"baseline_{args.split}.json", baseline_rows)

    if args.skip_ragas:
        print("RAGAS пропущен")
        return

    result = ragas_evaluate(rows, settings, judge_model=args.judge_model)
    frame = to_frame(result, rows)
    frame.to_csv(RESULTS / f"ragas_{args.split}.csv", index=False)
    summary = summarize_scores(frame)
    save_json(RESULTS / f"ragas_{args.split}.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    baseline_frame = None
    if baseline_rows:
        from ragas.metrics import answer_correctness, answer_relevancy

        baseline_result = ragas_evaluate(
            baseline_rows,
            settings,
            judge_model=args.judge_model,
            metrics=[answer_relevancy, answer_correctness],
        )
        baseline_frame = baseline_result.to_pandas()
        baseline_frame.to_csv(RESULTS / f"baseline_ragas_{args.split}.csv", index=False)
        baseline_summary = summarize_scores(baseline_frame)
        save_json(RESULTS / f"baseline_ragas_{args.split}.json", baseline_summary)

    errors = error_examples(frame, rows, limit=10)
    save_json(RESULTS / f"errors_{args.split}.json", errors)
    report = write_report(
        args.split,
        frame,
        summary,
        baseline_frame,
        baseline_summary,
        errors,
        rows,
        {
            "model": settings.llm_model,
            "judge": args.judge_model or settings.llm_model,
            "top_k": args.top_k or settings.top_k,
        },
    )
    print(f"Отчёт: {report}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "images"

plt.rcParams["font.family"] = "DejaVu Sans"


def mean(summary: dict, key: str) -> float:
    return float(summary.get(key, {}).get("mean", 0) or 0)


def main() -> None:
    rag = json.loads((RESULTS / "ragas_test.json").read_text(encoding="utf-8")) if (RESULTS / "ragas_test.json").exists() else {}
    base = json.loads((RESULTS / "baseline_ragas_test.json").read_text(encoding="utf-8")) if (RESULTS / "baseline_ragas_test.json").exists() else {}
    metrics = ["Faithfulness", "Answer Relevancy", "Answer Correctness"]
    keys = ["faithfulness", "answer_relevancy", "answer_correctness"]
    rag_values = [mean(rag, key) for key in keys]
    base_values = [mean(base, key) for key in keys]

    x = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
    ax.bar(x - width / 2, rag_values, width, label="RAG", color="#4f8cff")
    ax.bar(x + width / 2, base_values, width, label="LLM без поиска", color="#f0a52b")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Значение метрики")
    ax.set_title("Качество RAG и baseline на тестовой выборке")
    ax.legend(frameon=False)
    for index, value in enumerate(rag_values):
        ax.text(index - width / 2, value + 0.02, f"{value:.2f}", ha="center", fontsize=9)
    for index, value in enumerate(base_values):
        ax.text(index + width / 2, value + 0.02, f"{value:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "ragas-metrics.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", OUT / "ragas-metrics.png")


if __name__ == "__main__":
    main()

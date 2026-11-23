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
    rag = (
        json.loads((RESULTS / "ragas_test.json").read_text(encoding="utf-8"))
        if (RESULTS / "ragas_test.json").exists()
        else {}
    )
    metrics = ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"]
    keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    values = [mean(rag, key) for key in keys]

    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
    ax.bar(x, values, 0.55, color=["#4f8cff", "#4f8cff", "#a78bfa", "#19b5a5"])
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Значение метрики")
    ax.set_title("Метрики RAG на тестовой выборке (30 вопросов)")
    for index, value in enumerate(values):
        ax.text(index, value + 0.02, f"{value:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0.80, color="#e05d5d", linewidth=1, linestyle="--")
    ax.axhline(0.75, color="#f0a52b", linewidth=1, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "ragas-metrics.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", OUT / "ragas-metrics.png")


if __name__ == "__main__":
    main()

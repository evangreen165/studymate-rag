from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from studymate.chunking import read_chunks_jsonl, sample_chunks

OUT = Path("docs/images")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
BLUE = "#4f8cff"
PURPLE = "#7c5cff"
TEAL = "#19b5a5"
ORANGE = "#f0a52b"
DARK = "#1b2433"


def main() -> None:
    manifest = json.loads(Path("data/manifest.json").read_text(encoding="utf-8"))
    docs = [doc for doc in manifest["documents"] if not doc.get("duplicate_of")]
    summary = manifest["summary"]

    topics = summary["topics"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=200)
    names = list(topics.keys())
    counts = [topics[name] for name in names]
    axes[0].barh(names[::-1], counts[::-1], color=BLUE)
    axes[0].set_title("Документов по темам")
    axes[0].set_xlabel("Документов")
    for index, value in enumerate(counts[::-1]):
        axes[0].text(value + 0.2, index, str(value), va="center", fontsize=9)

    pages = Counter()
    for doc in docs:
        pages[doc["topic"]] += int(doc.get("text_pages", 0))
    page_values = [pages.get(name, 0) for name in names]
    axes[1].barh(names[::-1], page_values[::-1], color=PURPLE)
    axes[1].set_title("Страниц с текстовым слоем по темам")
    axes[1].set_xlabel("Страниц")
    for index, value in enumerate(page_values[::-1]):
        axes[1].text(value + 50, index, str(value), va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "eda-topics.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    top = sorted(docs, key=lambda doc: int(doc.get("text_pages", 0)), reverse=True)[:15]
    fig, ax = plt.subplots(figsize=(13, 6), dpi=200)
    labels = [doc["title"][:58] for doc in top][::-1]
    values = [int(doc.get("text_pages", 0)) for doc in top][::-1]
    ax.barh(labels, values, color=TEAL)
    ax.set_title("Топ-15 документов по числу страниц с текстом")
    ax.set_xlabel("Страниц с текстом")
    for index, value in enumerate(values):
        ax.text(value + 15, index, str(value), va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "eda-documents.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    chunks = read_chunks_jsonl(Path("data/bm25_corpus.json"))
    dense = sample_chunks(chunks, 70)
    lengths = [len(chunk.text) for chunk in chunks]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=200)
    axes[0].hist(lengths, bins=40, color=ORANGE, edgecolor="white")
    axes[0].set_title("Распределение длины чанков")
    axes[0].set_xlabel("Символов в чанке")
    axes[0].set_ylabel("Чанков")
    axes[0].axvline(sum(lengths) / len(lengths), color=DARK, linestyle="--", linewidth=1.2)
    axes[0].text(
        sum(lengths) / len(lengths) + 80,
        axes[0].get_ylim()[1] * 0.85,
        f"среднее {sum(lengths) / len(lengths):.0f}",
        fontsize=9,
    )
    axes[1].bar(["BM25\n(все чанки)", "Плотный\nиндекс"], [len(chunks), len(dense)], color=[ORANGE, PURPLE])
    axes[1].set_title("Размер индексов")
    axes[1].set_ylabel("Чанков")
    for index, value in enumerate([len(chunks), len(dense)]):
        axes[1].text(index, value + 350, str(value), ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "eda-chunks.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    low = sorted(docs, key=lambda doc: int(doc.get("low_text_count", 0)), reverse=True)[:12]
    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=200)
    labels = [doc["title"][:55] for doc in low][::-1]
    values = [int(doc.get("low_text_count", 0)) for doc in low][::-1]
    ax.barh(labels, values, color="#e05d5d")
    ax.set_title("Документы с наибольшим числом страниц без текста")
    ax.set_xlabel("Страниц с менее чем 200 символами")
    fig.tight_layout()
    fig.savefig(OUT / "eda-lowtext.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("EDA figures written")


if __name__ == "__main__":
    main()

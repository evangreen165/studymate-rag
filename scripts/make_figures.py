from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

OUT = Path("docs/images")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#4f8cff"
PURPLE = "#7c5cff"
TEAL = "#19b5a5"
ORANGE = "#f0a52b"
DARK = "#1b2433"
LIGHT = "#eef3ff"


def box(ax, x, y, w, h, text, color=BLUE, fontsize=10, text_color="white"):
    patch = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=0,
        facecolor=color,
        alpha=0.95,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        wrap=True,
    )


def arrow(ax, start, end, text="", color="#5b6b82", style="-|>", rad=0.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=13,
        linewidth=1.5,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)
    if text:
        ax.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.16,
            text,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#4a586b",
        )


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(14, 8), dpi=200)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Архитектура StudyMate: RAG-ассистент по учебным материалам", fontsize=16, pad=14)

    box(ax, 0.3, 3.7, 1.5, 0.9, "Студент", color=DARK, fontsize=11)
    box(ax, 2.3, 3.6, 2.3, 1.1, "Streamlit UI\nчат, материалы,\nметрики", color=BLUE)
    box(ax, 5.2, 3.6, 2.1, 1.1, "FastAPI\nREST API\n/chat /documents", color=PURPLE)
    box(ax, 7.9, 3.5, 2.4, 1.3, "RAG-ядро\nпоиск + промпт\nцитаты и отказ", color=TEAL)
    box(ax, 11.0, 5.4, 2.5, 1.0, "ChromaDB\nплотный индекс\n(USER2-base)", color=BLUE, fontsize=9)
    box(ax, 11.0, 3.7, 2.5, 1.0, "BM25\nполный корпус\n19 235 чанков", color=ORANGE, fontsize=9)
    box(ax, 11.0, 2.0, 2.5, 1.0, "LLM\nLlama-3.1-70B\n(OpenRouter)", color=PURPLE, fontsize=9)

    box(ax, 0.8, 0.6, 2.3, 0.9, "Корпус PDF\n41 документ", color=DARK, fontsize=10)
    box(ax, 3.7, 0.6, 2.8, 0.9, "Ingestion\nPyMuPDF → очистка\nчанкинг 800/100", color=TEAL, fontsize=9)
    box(ax, 7.1, 0.6, 2.4, 0.9, "USER2-base\nэмбеддинги", color=BLUE, fontsize=9)

    arrow(ax, (1.8, 4.15), (2.3, 4.15), "HTTP")
    arrow(ax, (4.6, 4.15), (5.2, 4.15), "REST")
    arrow(ax, (7.3, 4.15), (7.9, 4.15))
    arrow(ax, (10.3, 4.4), (11.0, 5.6), "dense")
    arrow(ax, (10.3, 4.0), (11.0, 4.2), "BM25")
    arrow(ax, (10.3, 3.7), (11.0, 2.7), "LLM")
    arrow(ax, (3.1, 1.5), (3.7, 1.5))
    arrow(ax, (6.5, 1.5), (7.1, 1.5))
    arrow(ax, (9.5, 1.5), (11.0, 5.3), "плотные\nвекторы")
    elbow = FancyArrowPatch(
        (4.6, 0.53),
        (11.0, 3.65),
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.5,
        color="#5b6b82",
        connectionstyle="angle,angleA=0,angleB=90,rad=0",
    )
    ax.add_patch(elbow)
    ax.text(8.4, 0.34, "все чанки", ha="center", va="bottom", fontsize=7.5, color="#4a586b")

    ax.text(
        7.0,
        7.3,
        "Плотный индекс отвечает за смысловое сходство, BM25 — за точные термины и покрытие всего корпуса.\n"
        "Результаты объединяются RRF, затем LLM формирует ответ только по найденным фрагментам.",
        ha="center",
        va="top",
        fontsize=9,
        color="#4a586b",
    )
    fig.tight_layout()
    fig.savefig(OUT / "architecture.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "architecture.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pipeline() -> None:
    fig, ax = plt.subplots(figsize=(14, 8), dpi=200)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Пайплайн данных и обработки запроса", fontsize=16, pad=14)

    ax.text(0.2, 7.3, "Офлайн: подготовка индекса", fontsize=12, fontweight="bold", color=PURPLE)
    offline = [
        (0.3, "PDF\n41 документ", DARK),
        (2.2, "PyMuPDF\nизвлечение\nстраниц", TEAL),
        (4.1, "очистка\nколонтитулы\nдубликаты", TEAL),
        (6.0, "чанкинг\n800 токенов\noverlap 100", BLUE),
        (7.9, "стратификация\n2 189 чанков", ORANGE),
        (9.8, "USER2-base\n768d", BLUE),
        (11.7, "ChromaDB\nплотный индекс", PURPLE),
    ]
    for x, text, color in offline:
        box(ax, x, 5.7, 1.55, 1.05, text, color=color, fontsize=8.5)
    for index in range(len(offline) - 1):
        arrow(ax, (offline[index][0] + 1.55, 6.22), (offline[index + 1][0], 6.22))

    box(ax, 9.8, 4.25, 1.55, 0.8, "BM25-корпус\n19 235 чанков", color=ORANGE, fontsize=8.5)
    arrow(ax, (7.9 + 1.55, 5.7), (10.2, 5.05), "все чанки")

    ax.text(0.2, 4.0, "Онлайн: ответ на вопрос", fontsize=12, fontweight="bold", color=PURPLE)
    online = [
        (0.3, "вопрос\nстудента", DARK),
        (2.2, "эмбеддинг\nUSER2-base", BLUE),
        (4.1, "dense top-k\n+ MMR", PURPLE),
        (6.0, "RRF\nслияние", TEAL),
        (7.9, "промпт\nс цитатами", ORANGE),
        (9.8, "LLM\nLlama-3.1-70B", PURPLE),
        (11.7, "ответ +\nисточники", TEAL),
    ]
    for x, text, color in online:
        box(ax, x, 2.3, 1.55, 1.0, text, color=color, fontsize=8.5)
    for index in range(len(online) - 1):
        arrow(ax, (online[index][0] + 1.55, 2.8), (online[index + 1][0], 2.8))

    arrow(ax, (10.2, 4.25), (7.0, 3.3), "BM25 top-k")
    arrow(ax, (12.5, 4.25), (12.5, 3.3), "контекст")

    ax.text(
        7.0,
        1.2,
        "Плотный поиск и BM25 дополняют друг друга: первый находит смысловые совпадения, второй — точные термины,\n"
        "команды и формулы. В ответ передаются только фрагменты, прошедшие порог релевантности.",
        ha="center",
        va="center",
        fontsize=9,
        color="#4a586b",
    )
    fig.tight_layout()
    fig.savefig(OUT / "pipeline.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "pipeline.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def use_case() -> None:
    fig, ax = plt.subplots(figsize=(14, 8), dpi=200)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("UML-диаграмма вариантов использования", fontsize=16, pad=14)

    ax.add_patch(mpatches.Rectangle((3.1, 0.5), 7.8, 7.0, fill=False, linewidth=1.2, color="#8090a8"))
    ax.text(7.0, 7.2, "StudyMate", ha="center", fontsize=11, color="#4a586b")

    cases = [
        (4.0, 6.2, "Задать вопрос"),
        (4.0, 5.2, "Получить источники\nи страницы"),
        (4.0, 4.2, "Фильтровать\nпо теме"),
        (4.0, 3.2, "Смотреть историю\nчатов"),
        (4.0, 2.2, "Оценить ответ"),
        (7.6, 6.0, "Загрузить PDF"),
        (7.6, 5.0, "Запустить\nиндексацию"),
        (7.6, 4.0, "Удалить документ"),
        (7.6, 3.0, "Смотреть метрики\nи токены"),
        (7.6, 2.0, "Экспортировать чат"),
    ]
    for x, y, text in cases:
        ellipse = mpatches.Ellipse((x, y), 3.2, 0.75, facecolor=LIGHT, edgecolor=BLUE, linewidth=1.2)
        ax.add_patch(ellipse)
        ax.text(x, y, text, ha="center", va="center", fontsize=8.5, color=DARK)

    # акторы
    for x, label in [(1.3, "Студент"), (12.7, "Администратор\n/ преподаватель")]:
        ax.add_patch(mpatches.Circle((x, 5.9), 0.28, facecolor="white", edgecolor=DARK, linewidth=1.4))
        ax.plot([x, x], [5.62, 4.6], color=DARK, linewidth=1.4)
        ax.plot([x - 0.55, x + 0.55], [5.2, 5.2], color=DARK, linewidth=1.4)
        ax.plot([x, x - 0.5], [4.6, 3.8], color=DARK, linewidth=1.4)
        ax.plot([x, x + 0.5], [4.6, 3.8], color=DARK, linewidth=1.4)
        ax.text(x, 3.5, label, ha="center", va="top", fontsize=9)

    for y in (6.2, 5.2, 4.2, 3.2, 2.2):
        ax.plot([1.3, 2.4], [5.2, y], color="#8a97aa", linewidth=0.9)
    for y in (6.0, 5.0, 4.0, 3.0, 2.0):
        ax.plot([12.7, 9.2], [5.2, y], color="#8a97aa", linewidth=0.9)

    fig.tight_layout()
    fig.savefig(OUT / "use-case.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "use-case.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    pipeline()
    use_case()
    print("figures written to", OUT)

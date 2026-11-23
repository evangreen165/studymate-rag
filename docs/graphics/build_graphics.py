from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "docs" / "images"
OUT = ROOT / "docs" / "graphics"
OUT.mkdir(parents=True, exist_ok=True)

FONT_PATH = Path(r"C:\Windows\Fonts\arial.ttf")
BOLD_PATH = Path(r"C:\Windows\Fonts\arialbd.ttf")
if not FONT_PATH.exists():
    import matplotlib

    FONT_PATH = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    BOLD_PATH = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"
pdfmetrics.registerFont(TTFont("GOST", str(FONT_PATH)))
pdfmetrics.registerFont(TTFont("GOST-Bold", str(BOLD_PATH)))

W, H = landscape(A4)
LEFT = 20 * mm
OTHER = 5 * mm


def text(c, x, y, value, size=9, bold=False, align="left"):
    c.setFont("GOST-Bold" if bold else "GOST", size)
    if align == "center":
        c.drawCentredString(x, y, value)
    elif align == "right":
        c.drawRightString(x, y, value)
    else:
        c.drawString(x, y, value)


def title_block(c, *, code: str, sheet: int, title: str):
    width = 185 * mm
    height = 55 * mm
    x0 = W - OTHER - width
    y0 = OTHER
    c.setLineWidth(1.0)
    c.rect(x0, y0, width, height)

    # верхняя часть: название
    c.line(x0, y0 + 35 * mm, x0 + width, y0 + 35 * mm)
    text(c, x0 + 4 * mm, y0 + 44 * mm, title, 11, True)
    text(c, x0 + 4 * mm, y0 + 39 * mm, "Ассистент по учебным материалам курса с ответами на вопросы студентов", 9)

    # левая часть: разработчик / проверяющий
    c.line(x0 + 65 * mm, y0, x0 + 65 * mm, y0 + 35 * mm)
    rows = [
        ("Разраб.", "Кузьма Е.А."),
        ("Пров.", "Савенко А.Г."),
        ("Н. контр.", ""),
        ("Утв.", ""),
    ]
    row_h = 8.75 * mm
    for index, (role, name) in enumerate(rows):
        y = y0 + 35 * mm - (index + 1) * row_h
        if index < 3:
            c.line(x0, y, x0 + 65 * mm, y)
        text(c, x0 + 2 * mm, y + 2.2 * mm, role, 8)
        text(c, x0 + 20 * mm, y + 2.2 * mm, name, 8)

    # правая часть: код, лист, листов
    c.line(x0 + 65 * mm, y0 + 17.5 * mm, x0 + width, y0 + 17.5 * mm)
    c.line(x0 + 65 * mm, y0 + 8.75 * mm, x0 + width, y0 + 8.75 * mm)
    c.line(x0 + 125 * mm, y0, x0 + 125 * mm, y0 + 35 * mm)
    text(c, x0 + 95 * mm, y0 + 26 * mm, code, 11, True, "center")
    text(c, x0 + 95 * mm, y0 + 12 * mm, "Лист", 8, False, "center")
    text(c, x0 + 95 * mm, y0 + 3.5 * mm, str(sheet), 10, True, "center")
    text(c, x0 + 155 * mm, y0 + 12 * mm, "Листов", 8, False, "center")
    text(c, x0 + 155 * mm, y0 + 3.5 * mm, "4", 10, True, "center")


def frame(c, sheet: int, title: str, code: str):
    c.setLineWidth(1.2)
    c.rect(LEFT, OTHER, W - LEFT - OTHER, H - 2 * OTHER)
    text(c, W / 2, H - 16 * mm, title, 15, True, "center")
    title_block(c, code=code, sheet=sheet, title=title)
    text(c, LEFT + 4 * mm, H - 25 * mm, "Формат A4", 8)
    text(c, W - OTHER - 4 * mm, H - 25 * mm, "ГОСТ 19.701-90", 8, align="right")


def image_page(path: Path, *, name: str, title: str, code: str, sheet: int, max_w: float, max_h: float):
    c = canvas.Canvas(str(path), pagesize=landscape(A4))
    frame(c, sheet, title, code)
    image_path = IMAGES / name
    if image_path.exists():
        from PIL import Image

        with Image.open(image_path) as image:
            iw, ih = image.size
        scale = min(max_w / iw, max_h / ih)
        width, height = iw * scale, ih * scale
        x = LEFT + (W - LEFT - OTHER - width) / 2
        y = OTHER + 55 * mm + (H - OTHER - (OTHER + 55 * mm) - height) / 2 - 6 * mm
        c.drawImage(str(image_path), x, y, width=width, height=height, preserveAspectRatio=True)
    c.save()


def results_page():
    c = canvas.Canvas(str(OUT / "Лист_4_Результаты_оценки.pdf"), pagesize=landscape(A4))
    frame(c, 4, "Результаты оценки качества", "ГУИР 381571.14.004")
    chart = IMAGES / "ragas-metrics.png"
    if chart.exists():
        from PIL import Image

        with Image.open(chart) as image:
            iw, ih = image.size
        max_w, max_h = 150 * mm, 100 * mm
        scale = min(max_w / iw, max_h / ih)
        width, height = iw * scale, ih * scale
        c.drawImage(str(chart), LEFT + 8 * mm, OTHER + 62 * mm, width=width, height=height, preserveAspectRatio=True)

    metrics_path = ROOT / "results" / "ragas_test.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    def mean(key: str) -> str:
        value = metrics.get(key, {}).get("mean")
        return f"{value:.3f}" if value is not None else "—"

    rows = [
        ("Faithfulness", mean("faithfulness"), "≥ 0,80"),
        ("Answer Relevancy", mean("answer_relevancy"), "≥ 0,75"),
        ("Context Precision", mean("context_precision"), "—"),
        ("Context Recall", mean("context_recall"), "—"),
    ]
    x0 = LEFT + 175 * mm
    y0 = OTHER + 112 * mm
    text(c, x0, y0 + 8 * mm, "Итоговые метрики", 11, True)
    text(c, x0, y0 + 2 * mm, "Метрика", 8, True)
    text(c, x0 + 48 * mm, y0 + 2 * mm, "Значение", 8, True)
    text(c, x0 + 78 * mm, y0 + 2 * mm, "Цель", 8, True)
    for index, (name, value, target) in enumerate(rows, start=1):
        y = y0 - index * 7 * mm
        c.line(x0, y + 2.5 * mm, x0 + 100 * mm, y + 2.5 * mm)
        text(c, x0, y, name, 9)
        text(c, x0 + 50 * mm, y, value, 9, True)
        text(c, x0 + 78 * mm, y, target, 8)

    text(
        c,
        LEFT + 8 * mm,
        OTHER + 60 * mm,
        "Оценка RAGAS на 30 отложенных вопросах; судья — Llama-3.1-70B; baseline — LLM без поиска.",
        8,
    )
    c.save()


def main() -> None:
    image_page(
        OUT / "Лист_1_Архитектура.pdf",
        name="architecture.png",
        title="Схема архитектуры интеллектуальной системы",
        code="ГУИР 381571.14.001",
        sheet=1,
        max_w=250 * mm,
        max_h=120 * mm,
    )
    image_page(
        OUT / "Лист_2_Пайплайн.pdf",
        name="pipeline.png",
        title="Диаграмма данных и пайплайна",
        code="ГУИР 381571.14.002",
        sheet=2,
        max_w=250 * mm,
        max_h=120 * mm,
    )
    image_page(
        OUT / "Лист_3_Варианты_использования.pdf",
        name="use-case.png",
        title="Диаграмма вариантов использования",
        code="ГУИР 381571.14.003",
        sheet=3,
        max_w=250 * mm,
        max_h=120 * mm,
    )
    results_page()
    print("graphics written to", OUT)


if __name__ == "__main__":
    main()

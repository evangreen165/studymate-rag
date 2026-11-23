from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "presentation"
IMAGES = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

BG = RGBColor(8, 12, 23)
PANEL = RGBColor(17, 26, 44)
TEXT = RGBColor(233, 238, 251)
MUTED = RGBColor(147, 164, 195)
BLUE = RGBColor(79, 140, 255)
PURPLE = RGBColor(167, 139, 250)
TEAL = RGBColor(25, 181, 165)
ORANGE = RGBColor(240, 165, 43)
RED = RGBColor(224, 93, 93)
WHITE = RGBColor(255, 255, 255)


def results() -> dict:
    path = ROOT / "results" / "ragas_test.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def baseline() -> dict:
    path = ROOT / "results" / "baseline_ragas_test.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def value(summary: dict, key: str) -> str:
    number = summary.get(key, {}).get("mean")
    return f"{number:.3f}" if number is not None else "—"


def new_deck() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    return prs


def background(slide, color=BG) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_bar(slide, color=BLUE) -> None:
    shape = slide.shapes.add_shape(1, 0, 0, Inches(0.12), prs.slide_height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def slide_title(prs, title: str, subtitle: str | None = None, number: int = 1):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background(slide)
    add_bar(slide)
    box = slide.shapes.add_textbox(Inches(0.55), Inches(0.35), Inches(12.2), Inches(0.75))
    frame = box.text_frame
    frame.text = title
    frame.paragraphs[0].font.size = Pt(28)
    frame.paragraphs[0].font.bold = True
    frame.paragraphs[0].font.color.rgb = TEXT
    frame.paragraphs[0].font.name = "Segoe UI"
    if subtitle:
        sub = slide.shapes.add_textbox(Inches(0.58), Inches(1.05), Inches(12.0), Inches(0.5))
        p = sub.text_frame.paragraphs[0]
        p.text = subtitle
        p.font.size = Pt(13)
        p.font.color.rgb = MUTED
        p.font.name = "Segoe UI"
    foot = slide.shapes.add_textbox(Inches(0.55), Inches(7.05), Inches(9.0), Inches(0.3))
    p = foot.text_frame.paragraphs[0]
    p.text = "StudyMate · БГУИР · 2026/2027"
    p.font.size = Pt(9)
    p.font.color.rgb = MUTED
    num = slide.shapes.add_textbox(Inches(12.35), Inches(7.02), Inches(0.5), Inches(0.3))
    p = num.text_frame.paragraphs[0]
    p.text = str(number)
    p.alignment = PP_ALIGN.RIGHT
    p.font.size = Pt(9)
    p.font.color.rgb = MUTED
    return slide


def textbox(slide, x, y, w, h, text, size=16, color=TEXT, bold=False, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = 0
    frame.margin_right = 0
    frame.margin_top = 0
    frame.margin_bottom = 0
    p = frame.paragraphs[0]
    p.text = text
    p.alignment = align
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = "Segoe UI"
    return box


def bullets(slide, x, y, w, h, items: list[str], size=15, color=TEXT, spacing=8):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = 0
    for index, item in enumerate(items):
        p = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        p.text = "•  " + item
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.font.name = "Segoe UI"
        p.space_after = Pt(spacing)
    return box


def panel(slide, x, y, w, h, fill=PANEL, line=None):
    shape = slide.shapes.add_shape(5, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line:
        shape.line.color.rgb = line
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def picture(slide, path: Path, x, y, w=None, h=None):
    if not path.exists():
        return
    if w and h:
        slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    elif w:
        slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))
    elif h:
        slide.shapes.add_picture(str(path), Inches(x), Inches(y), height=Inches(h))


def metric_cards(slide, values: list[tuple[str, str, str]]):
    widths = 2.75
    gap = 0.22
    x = 0.6
    for title, value_text, target in values:
        panel(slide, x, 1.7, widths, 1.55, fill=PANEL, line=RGBColor(44, 69, 112))
        textbox(slide, x + 0.2, 1.85, widths - 0.4, 0.35, title, 12, MUTED)
        textbox(slide, x + 0.2, 2.2, widths - 0.4, 0.65, value_text, 30, WHITE, True)
        textbox(slide, x + 0.2, 2.9, widths - 0.4, 0.25, target, 10, MUTED)
        x += widths + gap


def simple_table(slide, data: list[list[str]], x, y, w, h, col_widths=None, font_size=11):
    rows, cols = len(data), len(data[0])
    table = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(h)).table
    if col_widths:
        for index, width in enumerate(col_widths):
            table.columns[index].width = Inches(width)
    for row in range(rows):
        for col in range(cols):
            cell = table.cell(row, col)
            cell.text = str(data[row][col])
            cell.fill.solid()
            cell.fill.fore_color.rgb = PANEL if row else RGBColor(28, 43, 72)
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(font_size)
                paragraph.font.color.rgb = TEXT
                paragraph.font.name = "Segoe UI"
                paragraph.font.bold = row == 0
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    return table


def build() -> None:
    global prs
    prs = new_deck()
    rag = results()
    base = baseline()

    # 1 title
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background(slide)
    add_bar(slide, BLUE)
    textbox(slide, 0.8, 1.05, 11.8, 0.4, "КУРСОВАЯ РАБОТА · ПРИКЛАДНЫЕ ИНТЕЛЛЕКТУАЛЬНЫЕ СИСТЕМЫ", 12, BLUE, True)
    textbox(slide, 0.8, 1.55, 11.8, 1.5, "StudyMate", 54, WHITE, True)
    textbox(slide, 0.8, 3.0, 11.8, 1.0, "Ассистент по учебным материалам курса\nс ответами на вопросы студентов", 24, TEXT, True)
    textbox(slide, 0.8, 4.45, 11.8, 0.4, "RAG · гибридный поиск · цитаты · оценка качества RAGAS", 15, MUTED)
    textbox(slide, 0.8, 6.15, 6.0, 0.8, "Студент: гр. 381571 Кузьма Е.А.\nРуководитель: ст. преподаватель Савенко А.Г.", 11, MUTED)
    textbox(slide, 9.4, 6.15, 3.2, 0.8, "Минск, 2027", 11, MUTED, align=PP_ALIGN.RIGHT)

    # 2 problem
    slide = slide_title(prs, "Проблема", "Десятки PDF, тысячи страниц и вопросы, сформулированные обычным языком", 2)
    bullets(slide, 0.65, 1.75, 6.4, 4.8, [
        "Учебные материалы курса — 41 PDF и 15 784 страницы по семи темам.",
        "Полнотекстовый поиск не понимает синонимы и перефразирования.",
        "Универсальный чат-бот отвечает «по памяти» и не может подтвердить факты.",
        "Студенту нужен короткий ответ с указанием документа и страницы.",
        "Преподаватель тратит время на повторяющиеся объяснения.",
    ], 16)
    panel(slide, 7.3, 1.75, 5.3, 4.4, fill=PANEL, line=RGBColor(44, 69, 112))
    textbox(slide, 7.6, 1.95, 4.7, 0.35, "Что должно быть в ответе", 13, BLUE, True)
    bullets(slide, 7.6, 2.4, 4.7, 3.4, [
        "Опора только на материалы курса",
        "Цитаты [1], [2] и номера страниц",
        "Честный отказ при отсутствии контекста",
        "Метрики токенов и задержек",
    ], 13)
    textbox(slide, 7.6, 5.35, 4.7, 0.5, "→ RAG: retrieval + generation", 15, TEAL, True)

    # 3 goal/tasks
    slide = slide_title(prs, "Цель и задачи", "Разработать прототип RAG-ассистента и доказать качество количественно", 3)
    panel(slide, 0.65, 1.65, 5.7, 1.15, fill=RGBColor(20, 36, 64), line=BLUE)
    textbox(slide, 0.9, 1.8, 5.2, 0.8, "Цель: отвечать на вопросы студентов по корпусу PDF с опорой на источники и оценкой качества.", 14, TEXT, True)
    bullets(slide, 0.7, 3.0, 5.6, 3.4, [
        "Проанализировать предметную область",
        "Подготовить и паспортизировать корпус",
        "Обосновать архитектуру и модели",
        "Построить гибридный индекс",
        "Реализовать API, UI, Docker и CI",
        "Оценить качество и разобрать ошибки",
    ], 13)
    bullets(slide, 7.0, 1.65, 5.8, 4.9, [
        "Корпус: 41 PDF, 15 784 страницы, 7 тем",
        "Индекс: ChromaDB + BM25, USER2-base",
        "LLM: Llama-3.1-70B через OpenRouter",
        "API: FastAPI + OpenAPI",
        "UI: Streamlit, история чатов, метрики",
        "Оценка: RAGAS, 30 тестовых вопросов",
        "Baseline: та же LLM без поиска",
    ], 16)

    # 4 dataset
    slide = slide_title(prs, "Данные", "41 PDF, 15 784 страницы, 13 918 страниц с текстовым слоем", 4)
    picture(slide, IMAGES / "eda-topics.png", 0.55, 1.55, w=8.0)
    panel(slide, 8.75, 1.65, 3.95, 4.6, fill=PANEL, line=RGBColor(44, 69, 112))
    textbox(slide, 9.0, 1.8, 3.5, 0.35, "Паспорт корпуса", 13, BLUE, True)
    bullets(slide, 9.0, 2.25, 3.5, 3.8, [
        "7 тем, 41 документ",
        "490 МБ, 29,2 млн символов",
        "Алгоритмы — 17 документов",
        "Git — 4 шпаргалки",
        "5 сканов без текста",
        "Две версии Олифера",
        "Индекс: 18 693 BM25 + 794",
    ], 11)
    textbox(slide, 0.6, 5.9, 8.0, 0.6, "Вывод: корпус пригоден для RAG, но требует дедупликации, фильтрации сканов и отдельной проверки по темам.", 12, TEAL, True)

    # 5 data quality
    slide = slide_title(prs, "Качество данных", "EDA: что ломает поиск и как это учтено", 5)
    picture(slide, IMAGES / "eda-lowtext.png", 0.55, 1.55, w=6.5)
    bullets(slide, 7.3, 1.7, 5.4, 4.9, [
        "Сканы и страницы с рисунками → статус scanned",
        "Формулы и таблицы теряют структуру",
        "Смешанный язык: русский + английские команды",
        "Мало данных по Git и ООП",
        "Чанкинг с токенизатором USER2: 800/100",
        "Средний чанк 1 570 символов, 18 693 чанков",
    ], 14)

    # 6 approach comparison
    slide = slide_title(prs, "Выбор подхода", "Почему RAG, а не классический ML или чат-бот без поиска", 6)
    simple_table(slide, [
        ["Критерий", "Classical ML", "Fine-tuning", "LLM без поиска", "RAG"],
        ["Ссылка на источник", "Нет", "Нет", "Нет", "Да"],
        ["Обновление знаний", "Переобучение", "Переобучение", "Промпт", "Переиндексация"],
        ["Галлюцинации", "Нет", "Нет", "Высокий риск", "Низкий"],
        ["Нужны метки", "Да", "Да", "Нет", "Нет"],
        ["Сложность", "Низкая", "Высокая", "Низкая", "Средняя"],
    ], 0.55, 1.55, 12.2, 4.4, [2.2, 2.4, 2.4, 2.5, 2.5], 12)
    textbox(slide, 0.65, 6.2, 12.0, 0.5, "RAG сохраняет естественность ответа и делает его проверяемым: каждое утверждение сопоставимо с источником.", 13, TEAL, True)

    # 7 architecture
    slide = slide_title(prs, "Архитектура", "UI → API → RAG-ядро → ChromaDB / BM25 → LLM", 7)
    picture(slide, IMAGES / "architecture.png", 0.5, 1.45, w=8.3)
    bullets(slide, 9.0, 1.65, 3.8, 4.8, [
        "Streamlit UI: чат, материалы, метрики",
        "FastAPI: валидация, OpenAPI, логи",
        "RAG: MMR + RRF + цитаты",
        "ChromaDB: плотный индекс",
        "BM25: все 18 693 чанков",
        "OpenRouter / локальная Qwen",
    ], 12)

    # 8 pipeline
    slide = slide_title(prs, "Пайплайн", "Офлайн-индексация и онлайн-ответ", 8)
    picture(slide, IMAGES / "pipeline.png", 0.55, 1.45, w=10.8)
    textbox(slide, 0.7, 6.55, 12.0, 0.4, "PDF → PyMuPDF → очистка → чанкинг → USER2-base → ChromaDB; все чанки → BM25; запрос → dense + BM25 → RRF → LLM → ответ + источники.", 11, MUTED)

    # 9 stack
    slide = slide_title(prs, "Реализация", "Стек, API и интерфейс", 9)
    bullets(slide, 0.7, 1.6, 6.1, 4.9, [
        "Python 3.12, uv, LlamaIndex, ChromaDB",
        "USER2-base (768d) + BM25 + RRF",
        "FastAPI + Pydantic + OpenAPI",
        "Streamlit: чат, история, загрузка PDF",
        "Docker Compose: api + ui",
        "GitHub Actions: ruff, pytest, docker build",
        "RAGAS 0.3.9, судья Llama-3.1-70B",
    ], 14)
    picture(slide, IMAGES / "api-swagger.png", 7.0, 1.75, w=5.8)
    textbox(slide, 7.0, 5.6, 5.8, 0.8, "/health · /ready · /chat · /documents\n/corpus/upload · /index · /metrics", 12, MUTED)

    # 10 UI
    slide = slide_title(prs, "Интерфейс", "Ответ с источниками, управление корпусом и метрики агента", 10)
    picture(slide, IMAGES / "ui-chat.png", 0.5, 1.55, w=6.0)
    picture(slide, IMAGES / "ui-materials.png", 6.75, 1.55, w=6.1)
    textbox(slide, 0.55, 6.55, 12.2, 0.4, "Токены, токены/с, задержки поиска и генерации, статусы индексации, история чатов и экспорт.", 11, MUTED)

    # 11 corpus management
    slide = slide_title(prs, "Работа с корпусом", "Загрузка, индексация, удаление и прозрачные статусы", 11)
    picture(slide, IMAGES / "ui-documents.png", 0.55, 1.5, w=7.0)
    bullets(slide, 7.85, 1.65, 4.9, 4.9, [
        "Список документов и тем",
        "Статусы: indexed, sparse_only, scanned, error",
        "Загрузка PDF и выбор темы",
        "Фоновая индексация с прогрессом",
        "Отмена и полная переиндексация",
        "Удаление документа из манифеста и индекса",
    ], 13)

    # 12 evaluation methodology
    slide = slide_title(prs, "Методика оценки", "30 отложенных вопросов, RAGAS, baseline без поиска", 12)
    bullets(slide, 0.7, 1.65, 6.2, 4.8, [
        "10 валидационных вопросов — для настройки",
        "30 тестовых — не используются при подборе",
        "Эталонные ответы и gold-страницы",
        "Faithfulness ≥ 0,80 — основная метрика",
        "Answer Relevancy ≥ 0,75",
        "Context Precision / Recall, Answer Correctness",
        "95 % бутстрэп-доверительные интервалы",
    ], 14)
    panel(slide, 7.0, 1.7, 5.7, 4.5, fill=PANEL, line=RGBColor(44, 69, 112))
    textbox(slide, 7.25, 1.9, 5.2, 0.4, "Baseline", 14, BLUE, True)
    textbox(slide, 7.25, 2.35, 5.2, 2.6, "Та же LLM (Llama-3.1-70B), но без поиска по документам. Отвечает по общим знаниям; ссылки и подтверждение фактов отсутствуют.", 13, TEXT)
    textbox(slide, 7.25, 4.6, 5.2, 1.2, "Сравнение показывает, что даёт именно RAG, а не только сильная модель.", 12, MUTED)

    # 13 results
    faith = value(rag, "faithfulness")
    rel = value(rag, "answer_relevancy")
    prec = value(rag, "context_precision")
    rec = value(rag, "context_recall")
    base_rel = value(base, "answer_relevancy")
    slide = slide_title(prs, "Результаты", "RAG на отложенной тестовой выборке", 13)
    metric_cards(slide, [
        ("Faithfulness", faith, "цель ≥ 0,80"),
        ("Answer Relevancy", rel, "цель ≥ 0,75"),
        ("Context Precision", prec, "дополнительная"),
        ("Context Recall", rec, "дополнительная"),
    ])
    simple_table(slide, [
        ["Метрика", "RAG", "Baseline"],
        ["Faithfulness", faith, "—"],
        ["Answer Relevancy", rel, base_rel],
        ["Context Precision", prec, "—"],
        ["Context Recall", rec, "—"],
    ], 0.7, 3.6, 5.6, 2.5, [2.4, 1.6, 1.6], 11)
    picture(slide, IMAGES / "ragas-metrics.png", 6.7, 3.35, w=6.0)
    textbox(slide, 0.7, 6.45, 12.0, 0.5, "Faithfulness и Answer Relevancy выше целей; baseline хуже по Answer Correctness (0,45 против 0,52) и не даёт источников.", 10, MUTED)

    # 14 errors/limitations
    slide = slide_title(prs, "Ошибки и ограничения", "Что мешает качеству и куда развиваться", 14)
    bullets(slide, 0.65, 1.65, 6.0, 5.0, [
        "Промах поиска: редкий чанк вне плотной подвыборки",
        "Граница чанка: ответ распределён между фрагментами",
        "Сканы и формулы: нет текстового слоя",
        "Многошаговые вопросы: нужен второй документ",
        "Языковой разрыв: русский вопрос — английская шпаргалка",
        "Избыточность: модель добавляет детали вне контекста",
    ], 13)
    panel(slide, 6.9, 1.7, 5.8, 4.7, fill=PANEL, line=RGBColor(44, 69, 112))
    textbox(slide, 7.15, 1.9, 5.3, 0.35, "Направления улучшения", 14, TEAL, True)
    bullets(slide, 7.15, 2.35, 5.3, 3.7, [
        "OCR для сканов",
        "Полный плотный индекс (GPU/ONNX)",
        "Реранкер для Context Precision",
        "Перефразирование запроса / HyDE",
        "Расширение корпуса Git и ООП",
        "Мониторинг и обратная связь",
    ], 12)

    # 15 conclusions
    slide = slide_title(prs, "Выводы", "Прототип работает, качество измерено, ограничения зафиксированы", 15)
    bullets(slide, 0.65, 1.65, 6.2, 4.9, [
        "Собран и паспортизирован корпус: 41 PDF, 15 784 страницы",
        "Построен гибридный индекс: 18 693 BM25 + 794 плотных",
        "Реализованы API, UI, Docker и CI",
        "RAG превосходит baseline по обоснованности",
        "Цели Faithfulness ≥ 0,80 и Answer Relevancy ≥ 0,75",
        "Определены направления развития",
    ], 13)
    panel(slide, 7.15, 1.65, 5.5, 4.5, fill=RGBColor(20, 36, 64), line=BLUE)
    textbox(slide, 7.4, 1.9, 5.0, 0.4, "Ссылки", 14, BLUE, True)
    textbox(slide, 7.4, 2.4, 5.0, 2.2, "GitHub:\n github.com/evangreen165/studymate-rag\n\nПЗ, графика и презентация:\n docs/", 12, TEXT)
    textbox(slide, 7.4, 5.0, 5.0, 0.7, "Спасибо за внимание!", 18, WHITE, True)
    picture(slide, IMAGES / "qr-github.png", 11.35, 5.0, w=1.15)
    textbox(slide, 10.9, 6.3, 1.9, 0.3, "репозиторий", 9, MUTED)

    out = OUT / "Презентация_защита_StudyMate.pptx"
    prs.save(out)
    print(f"saved {out}, slides={len(prs.slides)}")


if __name__ == "__main__":
    build()

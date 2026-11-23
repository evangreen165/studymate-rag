from __future__ import annotations

import json
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
PZ_DIR = ROOT / "docs" / "pz"
CONTENT_DIR = PZ_DIR / "content"
OUT_DOCX = PZ_DIR / "ПЗ_Ассистент_по_учебным_материалам_Кузьма.docx"

FONT = "Times New Roman"


def _read_json(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        return default or {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default or {}


def collect_vars() -> dict:
    manifest = _read_json(ROOT / "data" / "manifest.json")
    meta = _read_json(ROOT / "data" / "index_meta.json")
    ragas = _read_json(ROOT / "results" / "ragas_test.json")
    baseline = _read_json(ROOT / "results" / "baseline_ragas_test.json")
    report = ROOT / "results" / "ragas_report_test.md"
    gold_hit = "—"
    if report.exists():
        match = re.search(r"\*\*([0-9.]+)\*\*", report.read_text(encoding="utf-8"))
        if match:
            gold_hit = match.group(1)
    perf = _read_json(ROOT / "results" / "performance.json")
    retrieval = _read_json(ROOT / "results" / "retrieval_experiments.json")
    summary = manifest.get("summary", {})
    faith = ragas.get("faithfulness", {}).get("mean")
    rel = ragas.get("answer_relevancy", {}).get("mean")

    def topic_table() -> str:
        csv_path = ROOT / "results" / "ragas_test.csv"
        if not csv_path.exists():
            return "| Тема | Faithfulness | Вопросов |\n| --- | ---: | ---: |\n| — | — | — |"
        try:
            import pandas as pd

            frame = pd.read_csv(csv_path)
            if "topic" not in frame or "faithfulness" not in frame:
                raise ValueError
            grouped = (
                frame.groupby("topic")["faithfulness"]
                .agg(["mean", "count"])
                .sort_values("mean", ascending=False)
            )
            rows = ["| Тема | Faithfulness | Вопросов |", "| --- | ---: | ---: |"]
            for topic, row in grouped.iterrows():
                rows.append(f"| {topic} | {row['mean']:.3f} | {int(row['count'])} |")
            return "\n".join(rows)
        except Exception:
            return "| Тема | Faithfulness | Вопросов |\n| --- | ---: | ---: |\n| — | — | — |"

    def errors_table() -> str:
        path = ROOT / "results" / "errors_test.json"
        if not path.exists():
            return "| Вопрос | Faithfulness | Причина |\n| --- | ---: | --- |\n| — | — | — |"
        try:
            errors = json.loads(path.read_text(encoding="utf-8"))
            rows = ["| Вопрос | Faithfulness | Причина |", "| --- | ---: | --- |"]
            for item in errors[:10]:
                question = str(item.get("question", ""))[:90].replace("|", "/")
                score = item.get("scores", {}).get("faithfulness")
                score_text = f"{float(score):.3f}" if score is not None else "—"
                rows.append(f"| {question} | {score_text} | требует разбора |")
            return "\n".join(rows)
        except Exception:
            return "| Вопрос | Faithfulness | Причина |\n| --- | ---: | --- |\n| — | — | — |"

    return {
        "DOCS": str(summary.get("documents", "—")),
        "UNIQUE_DOCS": str(summary.get("unique_documents", "—")),
        "PAGES": str(summary.get("pages", "—")),
        "TEXT_PAGES": str(summary.get("text_pages", "—")),
        "CHARS": f"{summary.get('chars', 0):,}".replace(",", " "),
        "SIZE_MB": f"{summary.get('size_bytes', 0) / 1024 / 1024:.1f}",
        "BM25_CHUNKS": str(meta.get("bm25_chunks", "—")),
        "DENSE_CHUNKS": str(meta.get("dense_chunks", meta.get("chunks", "—"))),
        "BUILD_MIN": f"{meta.get('build_seconds', 0) / 60:.1f}" if meta.get("build_seconds") else "—",
        "FAITH": f"{faith:.3f}" if faith is not None else "—",
        "REL": f"{rel:.3f}" if rel is not None else "—",
        "PREC": f"{ragas.get('context_precision', {}).get('mean', float('nan')):.3f}" if ragas.get("context_precision") else "—",
        "REC": f"{ragas.get('context_recall', {}).get('mean', float('nan')):.3f}" if ragas.get("context_recall") else "—",
        "CORR": f"{ragas.get('answer_correctness', {}).get('mean', float('nan')):.3f}" if ragas.get("answer_correctness") else "—",
        "BASE_REL": f"{baseline.get('answer_relevancy', {}).get('mean', float('nan')):.3f}" if baseline.get("answer_relevancy") else "—",
        "BASE_CORR": f"{baseline.get('answer_correctness', {}).get('mean', float('nan')):.3f}" if baseline.get("answer_correctness") else "—",
        "GOLD_HIT": gold_hit,
        "P50": f"{perf.get('p50_ms', 0) / 1000:.2f}" if perf.get("p50_ms") else "—",
        "P95": f"{perf.get('p95_ms', 0) / 1000:.2f}" if perf.get("p95_ms") else "—",
        "RPS": f"{perf.get('rps', 0):.2f}" if perf.get("rps") else "—",
        "RAM": f"{perf.get('ram_mb', 0) / 1024:.1f}" if perf.get("ram_mb") else "—",
        "TOPK3_HIT": str(retrieval.get("top_k_3", {}).get("gold_hit", "—")),
        "TOPK6_HIT": str(retrieval.get("top_k_6", {}).get("gold_hit", "—")),
        "TOPK8_HIT": str(retrieval.get("top_k_8", {}).get("gold_hit", "—")),
        "DENSE_HIT": str(retrieval.get("dense_only", {}).get("gold_hit", "—")),
        "BM25_HIT": str(retrieval.get("bm25_only", {}).get("gold_hit", "—")),
        "HYBRID_HIT": str(retrieval.get("hybrid", {}).get("gold_hit", "—")),
        "FAITH_STATUS": "достигнута" if faith is not None and faith >= 0.8 else "не достигнута",
        "REL_STATUS": "достигнута" if rel is not None and rel >= 0.75 else "не достигнута",
        "UNIQUENESS": "___",
        "TOPIC_TABLE": topic_table(),
        "ERRORS_TABLE": errors_table(),
    }


def set_cell_text(cell, text: str, bold: bool = False, size: int = 12) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    paragraph.paragraph_format.first_line_indent = Cm(0)
    paragraph.paragraph_format.left_indent = Cm(0)
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing = 1.0
    run = paragraph.add_run(text)
    run.font.name = FONT
    run.font.size = Pt(size)
    run.bold = bold
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)


def set_page_numbering(section, start: int) -> None:
    sect_pr = section._sectPr
    pg_num = sect_pr.find(qn("w:pgNumType"))
    if pg_num is None:
        pg_num = OxmlElement("w:pgNumType")
        sect_pr.append(pg_num)
    pg_num.set(qn("w:start"), str(start))


def add_page_field(paragraph) -> None:
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)
    run.font.name = FONT
    run.font.size = Pt(12)


def setup_styles(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(14)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.first_line_indent = Cm(1.25)
    normal.paragraph_format.line_spacing = Pt(18)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(0)

    for name, size, space_before, space_after in (
        ("Heading 1", 16, 0, 8),
        ("Heading 2", 14, 10, 4),
        ("Heading 3", 14, 8, 3),
    ):
        style = doc.styles[name]
        style.font.name = FONT
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        style.paragraph_format.first_line_indent = Cm(0)
        style.paragraph_format.line_spacing = 1.0
        style.paragraph_format.space_before = Pt(space_before)
        style.paragraph_format.space_after = Pt(space_after)
        style.paragraph_format.keep_with_next = True

    code = doc.styles.add_style("CodeBlock", 1)
    code.font.name = "Consolas"
    code._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
    code.font.size = Pt(9)
    code.paragraph_format.first_line_indent = Cm(0)
    code.paragraph_format.line_spacing = 1.0
    code.paragraph_format.space_before = Pt(0)
    code.paragraph_format.space_after = Pt(0)


def add_runs(paragraph, text: str) -> None:
    pattern = re.compile(r"(\*\*.+?\*\*|\*[^*]+?\*|`[^`]+?`)")
    position = 0
    for match in pattern.finditer(text):
        if match.start() > position:
            run = paragraph.add_run(text[position : match.start()])
            run.font.name = FONT
            run.font.size = Pt(14)
        token = match.group(0)
        run = paragraph.add_run(token[2:-2] if token.startswith("**") else token[1:-1])
        run.font.name = "Consolas" if token.startswith("`") else FONT
        run.font.size = Pt(12) if token.startswith("`") else Pt(14)
        run.bold = token.startswith("**")
        run.italic = token.startswith("*") and not token.startswith("**")
        run._element.rPr.rFonts.set(qn("w:eastAsia"), run.font.name)
        position = match.end()
    if position < len(text):
        run = paragraph.add_run(text[position:])
        run.font.name = FONT
        run.font.size = Pt(14)


class MarkdownBuilder:
    def __init__(self, doc: Document, variables: dict) -> None:
        self.doc = doc
        self.vars = variables
        self.figures = 0
        self.tables = 0
        self.first_h1 = True

    def replace(self, text: str) -> str:
        for key, value in self.vars.items():
            text = text.replace("{{" + key + "}}", str(value))
        return text

    def heading(self, text: str, level: int) -> None:
        paragraph = self.doc.add_paragraph(style=f"Heading {level}")
        run = paragraph.add_run(self.replace(text))
        run.font.name = FONT
        run.font.size = Pt(16 if level == 1 else 14)
        run.bold = True
        if level == 1:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            paragraph.paragraph_format.page_break_before = not self.first_h1
            self.first_h1 = False

    def paragraph(self, text: str) -> None:
        paragraph = self.doc.add_paragraph()
        add_runs(paragraph, self.replace(text))

    def bullet(self, text: str) -> None:
        paragraph = self.doc.add_paragraph(style="List Bullet")
        paragraph.paragraph_format.first_line_indent = Cm(0)
        paragraph.paragraph_format.line_spacing = Pt(18)
        paragraph.paragraph_format.space_after = Pt(0)
        add_runs(paragraph, self.replace(text))

    def numbered(self, text: str) -> None:
        paragraph = self.doc.add_paragraph(style="List Number")
        paragraph.paragraph_format.first_line_indent = Cm(0)
        paragraph.paragraph_format.line_spacing = Pt(18)
        paragraph.paragraph_format.space_after = Pt(0)
        add_runs(paragraph, self.replace(text))

    def figure(self, path: str, caption: str) -> None:
        image_path = (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path)
        paragraph = self.doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.first_line_indent = Cm(0)
        run = paragraph.add_run()
        run.add_picture(str(image_path), width=Cm(16.2))
        self.figures += 1
        cap = self.doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.first_line_indent = Cm(0)
        cap.paragraph_format.line_spacing = 1.0
        cap.paragraph_format.space_after = Pt(8)
        add_runs(cap, f"Рисунок {self.figures} – {self.replace(caption)}")

    def table(self, rows: list[list[str]], caption: str | None = None) -> None:
        if not rows:
            return
        self.tables += 1
        if caption:
            cap = self.doc.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
            cap.paragraph_format.first_line_indent = Cm(0)
            cap.paragraph_format.line_spacing = 1.0
            cap.paragraph_format.space_before = Pt(6)
            cap.paragraph_format.space_after = Pt(3)
            add_runs(cap, f"Таблица {self.tables} – {self.replace(caption)}")
        columns = max(len(row) for row in rows)
        table = self.doc.add_table(rows=len(rows), cols=columns)
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        if columns == 2:
            widths = [3.5, 12.7]
        elif columns == 3:
            widths = [4.2, 5.5, 6.5]
        elif columns == 4:
            widths = [1.6, 6.0, 4.3, 4.3]
        else:
            widths = [16.2 / columns] * columns
        table.autofit = False
        layout = OxmlElement("w:tblLayout")
        layout.set(qn("w:type"), "fixed")
        table._tbl.tblPr.append(layout)
        for column_index, width in enumerate(widths):
            table.columns[column_index].width = Cm(width)
        for row_index, row in enumerate(rows):
            for column_index in range(columns):
                value = row[column_index] if column_index < len(row) else ""
                cell = table.cell(row_index, column_index)
                cell.width = Cm(widths[column_index])
                set_cell_text(
                    cell,
                    self.replace(value),
                    bold=row_index == 0,
                    size=12,
                )
        tr_pr = table.rows[0]._tr.get_or_add_trPr()
        header = OxmlElement("w:tblHeader")
        header.set(qn("w:val"), "true")
        tr_pr.append(header)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(0)

    def code(self, lines: list[str]) -> None:
        for line in lines:
            paragraph = self.doc.add_paragraph(style="CodeBlock")
            paragraph.paragraph_format.first_line_indent = Cm(0)
            paragraph.paragraph_format.line_spacing = 1.0
            run = paragraph.add_run(line.rstrip("\n"))
            run.font.name = "Consolas"
            run.font.size = Pt(9)

    def render(self, text: str) -> None:
        lines = self.replace(text).splitlines()
        index = 0
        while index < len(lines):
            line = lines[index]
            stripped = line.strip()
            if not stripped:
                index += 1
                continue
            if stripped == "<!-- pagebreak -->":
                self.doc.add_page_break()
                index += 1
                continue
            if stripped.startswith("```"):
                code_lines: list[str] = []
                index += 1
                while index < len(lines) and not lines[index].strip().startswith("```"):
                    code_lines.append(lines[index])
                    index += 1
                index += 1
                self.code(code_lines)
                continue
            if stripped.startswith("### "):
                self.heading(stripped[4:], 3)
                index += 1
                continue
            if stripped.startswith("## "):
                self.heading(stripped[3:], 2)
                index += 1
                continue
            if stripped.startswith("# "):
                self.heading(stripped[2:], 1)
                index += 1
                continue
            if stripped.startswith("!["):
                match = re.match(r"!\[(.*?)\]\((.+?)\)", stripped)
                if match:
                    self.figure(match.group(2), match.group(1))
                index += 1
                continue
            if stripped.startswith("|"):
                rows: list[list[str]] = []
                while index < len(lines) and lines[index].strip().startswith("|"):
                    row_line = lines[index].strip().strip("|")
                    cells = [cell.strip() for cell in row_line.split("|")]
                    if not all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells):
                        rows.append(cells)
                    index += 1
                caption = None
                if rows and rows[0] and rows[0][0].startswith("Таблица:"):
                    caption = rows[0][0].split(":", 1)[1].strip()
                    rows = rows[1:]
                self.table(rows, caption)
                continue
            if stripped.startswith("- "):
                self.bullet(stripped[2:])
                index += 1
                continue
            if re.match(r"^\d+\.\s", stripped):
                self.numbered(re.sub(r"^\d+\.\s", "", stripped))
                index += 1
                continue
            if stripped.startswith(">"):
                paragraph = self.doc.add_paragraph()
                paragraph.paragraph_format.left_indent = Cm(1.25)
                paragraph.paragraph_format.first_line_indent = Cm(0)
                paragraph.paragraph_format.line_spacing = 1.0
                run = paragraph.add_run(stripped.lstrip("> ").strip())
                run.italic = True
                run.font.name = FONT
                run.font.size = Pt(12)
                index += 1
                continue
            self.paragraph(stripped)
            index += 1


def title_page(doc: Document) -> None:
    def center(text: str, size: int = 12, bold: bool = False, space_after: int = 0) -> None:
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.first_line_indent = Cm(0)
        paragraph.paragraph_format.line_spacing = 1.0
        paragraph.paragraph_format.space_after = Pt(space_after)
        run = paragraph.add_run(text)
        run.font.name = FONT
        run.font.size = Pt(size)
        run.bold = bold

    center("Министерство образования Республики Беларусь", 12)
    center("Учреждение образования «Белорусский государственный университет", 12)
    center("информатики и радиоэлектроники»", 12, space_after=18)
    center("Институт информационных технологий", 12)
    center("Факультет компьютерных технологий", 12)
    center("Кафедра информационных систем и технологий", 12, space_after=18)
    center("Дисциплина: Прикладные интеллектуальные системы (ПИС)", 12, space_after=60)
    center("ПОЯСНИТЕЛЬНАЯ ЗАПИСКА", 18, True, 6)
    center("к курсовой работе", 14, space_after=6)
    center("на тему", 14, space_after=10)
    center("«АССИСТЕНТ ПО УЧЕБНЫМ МАТЕРИАЛАМ КУРСА", 15, True)
    center("С ОТВЕТАМИ НА ВОПРОСЫ СТУДЕНТОВ»", 15, True, space_after=60)

    for label, value in (
        ("Студент:", "гр. 381571 Кузьма Е.А."),
        ("Руководитель:", "ст. преподаватель Савенко А.Г."),
    ):
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        paragraph.paragraph_format.first_line_indent = Cm(0)
        paragraph.paragraph_format.line_spacing = 1.0
        run = paragraph.add_run(f"{label} {value}")
        run.font.name = FONT
        run.font.size = Pt(13)

    for _ in range(4):
        doc.add_paragraph()
    center("Защищено с оценкой: ______________________", 12, space_after=2)
    center("(подпись)                              (расшифровка)", 9, space_after=8)
    center("«____» ________________ 2027 г.", 12, space_after=40)
    center("Минск 2027", 13)


def task_sheet(doc: Document) -> None:
    def line(text: str, size: int = 11, bold: bool = False, align=WD_ALIGN_PARAGRAPH.LEFT) -> None:
        paragraph = doc.add_paragraph()
        paragraph.alignment = align
        paragraph.paragraph_format.first_line_indent = Cm(0)
        paragraph.paragraph_format.line_spacing = 1.0
        paragraph.paragraph_format.space_after = Pt(0)
        run = paragraph.add_run(text)
        run.font.name = FONT
        run.font.size = Pt(size)
        run.bold = bold

    line("Учреждение образования «Белорусский государственный университет", 11, align=WD_ALIGN_PARAGRAPH.CENTER)
    line("информатики и радиоэлектроники»", 11, align=WD_ALIGN_PARAGRAPH.CENTER)
    line("Институт информационных технологий", 11, align=WD_ALIGN_PARAGRAPH.CENTER)
    line("Факультет компьютерных технологий", 11, align=WD_ALIGN_PARAGRAPH.CENTER, )
    line("", 6)
    line("УТВЕРЖДАЮ", 11)
    line("Заведующий кафедрой ИСиТ", 11)
    line("–––––––––––– А.И. Парамонов", 11)
    line("(подпись)", 9)
    line("«19» сентября 2026 г.", 11)
    line("", 6)
    line("ЗАДАНИЕ", 14, True, WD_ALIGN_PARAGRAPH.CENTER)
    line("на курсовую работу", 12, False, WD_ALIGN_PARAGRAPH.CENTER)
    line("", 4)
    line("студенту          Кузьма Евгению Александровичу", 11)
    line("1. Тема работы   «Ассистент по учебным материалам курса с ответами на вопросы студентов»", 11)
    line("2. Срок сдачи студентом законченной работы – 04.01.2027 г.", 11)
    line("3. Исходные данные к работе", 11)
    line(
        "Язык программирования – Python; дополнительные средства реализации: RAG-фреймворк LlamaIndex; "
        "векторная база данных ChromaDB; REST API на FastAPI; веб-интерфейс на Streamlit; контейнеризация Docker; "
        "библиотека для оценки качества RAGAS; внешние LLM через API OpenRouter (с возможностью резервного "
        "использования локальных моделей); предметная область – образовательные материалы (IT-дисциплины: "
        "Computer Science, Git, Linux, SQL, алгоритмы и структуры данных, компьютерные сети, ООП); прикладная "
        "задача – чат-бот для ответов на вопросы студентов по учебным материалам курса; архитектурный подход – "
        "RAG (Retrieval-Augmented Generation); данные – открытые учебные материалы в формате PDF, 41 файл, "
        "около 14 500 страниц; основная модель – LLM через API OpenRouter (Llama-3.1-70B-Instruct); модель "
        "эмбеддингов USER2-base (deepvk); baseline – та же LLM без поиска по документам; метрики – "
        "Faithfulness (RAGAS) ≥ 0,80, Answer Relevancy ≥ 0,75.",
        10,
    )
    line("Работа выполняется индивидуально; роль – инженер машинного обучения.", 10)
    line("4. Содержание пояснительной записки", 11, True)
    for item in (
        "Введение",
        "1. Анализ предметной области и общая постановка задачи",
        "2. Предварительная обработка и анализ данных",
        "3. Архитектурное решение",
        "4. Разработка и обучение модели",
        "5. Особенности реализации",
        "6. Анализ качества результатов",
        "Выводы",
        "Список используемых источников",
        "Приложение А. Развёртывание проекта",
    ):
        line(item, 10)
    line("5. Перечень графического материала", 11, True)
    for item in (
        "1. Схема архитектуры интеллектуальной системы (UML-диаграмма развёртывания) – формат А4, лист 1.",
        "2. Диаграмма данных / пайплайна – формат А4, лист 1.",
        "3. Диаграмма вариантов использования – формат А4, лист 1.",
        "4. Результаты оценки качества – формат А4, лист 1.",
    ):
        line(item, 10)
    line("6. Консультант по курсовой работе – Савенко А.Г.", 11)
    line("7. Дата выдачи задания – 21.09.2026 г.", 11)
    line("8. Календарный график работы над курсовой работой", 11, True)
    rows = [
        ["№", "Содержание работ", "Срок", "%"],
        ["1", "Раздел 1", "05.10.2026", "10"],
        ["2", "Раздел 2", "19.10.2026", "20"],
        ["3", "Раздел 3", "02.11.2026", "35"],
        ["4", "Раздел 4", "16.11.2026", "60"],
        ["5", "Раздел 5", "07.12.2026", "75"],
        ["6", "Раздел 6", "28.12.2026", "90"],
        ["7", "Оформление ПЗ и графического материала", "04.01.2027", "100"],
    ]
    table = doc.add_table(rows=len(rows), cols=4)
    table.style = "Table Grid"
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            set_cell_text(table.cell(row_index, column_index), value, bold=row_index == 0, size=9)
    line("", 4)
    line("Руководитель    ––––––            А.Г. Савенко", 11)
    line("Задание принял к исполнению    –––____––            Е.А. Кузьма", 11)


def add_toc(doc: Document) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.first_line_indent = Cm(0)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("СОДЕРЖАНИЕ")
    run.bold = True
    run.font.name = FONT
    run.font.size = Pt(16)
    doc.add_paragraph()
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.first_line_indent = Cm(0)
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = ' TOC \\o "1-2" \\h \\z \\u '
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_sep)
    text = OxmlElement("w:t")
    text.text = "Оглавление обновится при открытии документа."
    run._r.append(text)
    run._r.append(fld_end)
    run.font.name = FONT
    run.font.size = Pt(12)


def build() -> None:
    variables = collect_vars()
    doc = Document()
    setup_styles(doc)
    section = doc.sections[0]
    section.page_height = Cm(29.7)
    section.page_width = Cm(21)
    section.left_margin = Cm(3)
    section.right_margin = Cm(1.5)
    section.top_margin = Cm(2)
    section.bottom_margin = Cm(2)
    section.footer_distance = Cm(1)

    title_page(doc)
    task_section = doc.add_section(WD_SECTION.NEW_PAGE)
    task_section.top_margin = Cm(1.5)
    task_section.bottom_margin = Cm(1.5)
    task_sheet(doc)
    body_section = doc.add_section(WD_SECTION.NEW_PAGE)
    body_section.left_margin = Cm(3)
    body_section.right_margin = Cm(1.5)
    body_section.top_margin = Cm(2)
    body_section.bottom_margin = Cm(2)
    body_section.footer.is_linked_to_previous = False
    footer = body_section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.paragraph_format.first_line_indent = Cm(0)
    add_page_field(footer)
    set_page_numbering(body_section, 3)

    abstract = (CONTENT_DIR / "00_referat.md").read_text(encoding="utf-8")
    builder = MarkdownBuilder(doc, variables)
    builder.first_h1 = False
    builder.render(abstract)
    doc.add_page_break()
    add_toc(doc)
    doc.add_page_break()

    for path in sorted(CONTENT_DIR.glob("*.md")):
        if path.name == "00_referat.md":
            continue
        builder.render(path.read_text(encoding="utf-8"))

    doc.save(OUT_DOCX)
    print(f"saved {OUT_DOCX}")
    print(f"figures: {builder.figures}, tables: {builder.tables}")


if __name__ == "__main__":
    build()

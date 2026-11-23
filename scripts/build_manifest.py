from __future__ import annotations

import argparse
from pathlib import Path

from studymate.config import get_settings
from studymate.corpus import build_manifest
from studymate.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Построение манифеста и паспорта корпуса")
    parser.add_argument("--demo", action="store_true", help="Собрать манифест demo-корпуса")
    parser.add_argument("--max-docs", type=int, default=None, help="Ограничить число PDF")
    parser.add_argument("--output", type=Path, default=None, help="Путь к manifest.json")
    parser.add_argument("--no-uploads", action="store_true", help="Не включать data/uploads")
    args = parser.parse_args()

    settings = get_settings()
    manifest = build_manifest(
        settings,
        demo=args.demo,
        max_docs=args.max_docs,
        output_path=args.output,
        include_uploads=not args.no_uploads,
    )
    summary = manifest["summary"]
    print("\n=== Манифест готов ===")
    print(f"Документов: {summary['documents']} (уникальных: {summary['unique_documents']})")
    print(f"Страниц: {summary['pages']} (с текстом: {summary['text_pages']})")
    print(f"Символов: {summary['chars']:,}".replace(",", " "))
    print(f"Темы: {summary['topics']}")
    print(f"Ошибки: {len(summary['errors'])}, дубликаты: {len(summary['duplicates'])}, сканы: {len(summary['scanned_documents'])}")
    print(f"Манифест: {args.output or settings.manifest_path}")

    # Краткий паспорт для документации
    passport_path = (args.output or settings.manifest_path).with_name(
        "passport_demo.md" if args.demo else "passport.md"
    )
    lines = [
        "# Паспорт корпуса учебных материалов",
        "",
        f"Режим: **{manifest['mode']}**. Сформирован: {manifest['generated_at']}.",
        "",
        "## Сводные показатели",
        "",
        "| Показатель | Значение |",
        "| --- | ---: |",
        f"| PDF-файлов | {summary['documents']} |",
        f"| Уникальных документов | {summary['unique_documents']} |",
        f"| Всего страниц | {summary['pages']} |",
        f"| Страниц с текстовым слоем | {summary['text_pages']} |",
        f"| Символов текста | {summary['chars']} |",
        f"| Объём файлов | {summary['size_bytes'] / 1024 / 1024:.1f} МБ |",
        "",
        "## Распределение по темам",
        "",
        "| Тема | Документов |",
        "| --- | ---: |",
    ]
    for topic, count in summary["topics"].items():
        lines.append(f"| {topic} | {count} |")
    lines += [
        "",
        "## Документы",
        "",
        "| № | Тема | Документ | Страниц | С текстом | Символов | Язык (доля кириллицы) |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for number, doc in enumerate(manifest["documents"], start=1):
        lines.append(
            f"| {number} | {doc['topic']} | {doc['title']} | {doc['pages']} | "
            f"{doc['text_pages']} | {doc['chars']} | {doc['language_ratio']:.2f} |"
        )
    if summary["errors"]:
        lines += ["", "## Ошибки извлечения", ""]
        for item in summary["errors"]:
            lines.append(f"- `{item['path']}` — {item['error']}")
    if summary["scanned_documents"]:
        lines += ["", "## Сканы без текстового слоя", ""]
        for item in summary["scanned_documents"]:
            lines.append(f"- `{item['path']}` — {item['pages']} стр.")
    passport_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Паспорт: {passport_path}")


if __name__ == "__main__":
    main()

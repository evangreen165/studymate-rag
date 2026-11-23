from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

from studymate.config import get_settings
from studymate.corpus import load_extracted, manifest_documents
from studymate.llm import build_llm
from studymate.topics import TOPICS

PROMPT = """Ниже фрагмент учебного материала по теме «{topic}».

Фрагмент:
\"\"\"
{fragment}
\"\"\"

Сформулируй один содержательный вопрос, который студент мог бы задать по этому материалу.
Требования к вопросу:
- проверяй понимание или применение, а не простое определение;
- не ссылайся на рисунки, приложения, номера глав и страниц;
- если фрагмент содержит команды или код, сформулируй практическую задачу
  («как сделать …?»), а не проси пересказать список;
- вопрос должен быть понятен без чтения фрагмента.

Дай эталонный ответ на 3–6 предложений, используя только этот фрагмент, без внешних знаний.
Верни строго JSON без markdown:
{{"question": "...", "answer": "..."}}
"""

BAD_PAGE_MARKERS = (
    "оглавление",
    "содержание",
    "предметный указатель",
    "библиография",
    "список литературы",
    "алфавитный указатель",
)


def parse_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not payload.get("question") or not payload.get("answer"):
        return None
    return payload


def pick_pages(per_topic: int, seed: int = 42) -> list[dict]:
    settings = get_settings()
    documents = manifest_documents(settings)
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for doc in documents:
        cached = load_extracted(doc["doc_id"], settings)
        if not cached:
            continue
        for page in cached.get("pages", []):
            text = str(page.get("text", "")).strip()
            lowered = text[:400].lower()
            if not (900 <= len(text) <= 6000):
                continue
            if any(marker in lowered for marker in BAD_PAGE_MARKERS):
                continue
            by_topic[doc["topic_slug"]].append(
                {
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "topic": doc["topic"],
                    "topic_slug": doc["topic_slug"],
                    "page": int(page["page"]),
                    "text": text,
                }
            )

    rng = random.Random(seed)
    selected: list[dict] = []
    for topic in TOPICS:
        pages = by_topic.get(topic.slug, [])
        if not pages:
            continue
        rng.shuffle(pages)
        seen_docs: set[str] = set()
        chosen: list[dict] = []
        for page in pages:
            if page["doc_id"] in seen_docs:
                continue
            chosen.append(page)
            seen_docs.add(page["doc_id"])
            if len(chosen) >= per_topic:
                break
        if len(chosen) < per_topic:
            for page in pages:
                if page not in chosen:
                    chosen.append(page)
                if len(chosen) >= per_topic:
                    break
        selected.extend(chosen)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация кандидатов вопросов для оценки RAG")
    parser.add_argument("--per-topic", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("data/eval/candidates.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    settings = get_settings()
    pages = pick_pages(args.per_topic, args.seed)
    print(f"выбрано страниц: {len(pages)}")
    llm = build_llm(settings, provider="openrouter")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with args.output.open("w", encoding="utf-8") as fh:
        for index, page in enumerate(pages, start=1):
            if index % 5 == 0:
                print(f"  {index}/{len(pages)}", flush=True)
            prompt = PROMPT.format(
                topic=page["topic"],
                fragment=page["text"][:2600],
            )
            try:
                response = llm.complete(
                    [
                        {"role": "system", "content": "Ты помогаешь составить тестовый набор для RAG-системы. Отвечай только JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=450,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  ошибка на {page['title']} стр. {page['page']}: {exc}")
                continue
            payload = parse_json(response.text)
            if not payload:
                print(f"  не разобран JSON: {page['title']} стр. {page['page']}")
                continue
            row = {
                "topic": page["topic"],
                "topic_slug": page["topic_slug"],
                "doc_id": page["doc_id"],
                "title": page["title"],
                "page": page["page"],
                "question": payload["question"].strip(),
                "answer": payload["answer"].strip(),
                "fragment": page["text"][:1800],
                "model": response.model,
            }
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"готово: {len(rows)} кандидатов -> {args.output}")
    by_topic: dict[str, int] = defaultdict(int)
    for row in rows:
        by_topic[row["topic_slug"]] += 1
    print("по темам:", dict(by_topic))


if __name__ == "__main__":
    main()

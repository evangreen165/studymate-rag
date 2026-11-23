from __future__ import annotations

import json
from pathlib import Path

CANDIDATES = Path("data/eval/candidates2.jsonl")
OUT = Path("data/eval")

TEST_INDEXES = [
    # CS
    2, 4, 5, 8,
    # Git
    9, 10, 11, 12, 14,
    # Linux
    15, 17, 19, 20,
    # SQL
    23, 25, 26, 27, 30,
    # Algorithms
    32, 33, 36, 37,
    # Networks
    39, 40, 41, 46,
    # OOP
    47, 48, 49, 51,
]
VAL_INDEXES = [1, 3, 13, 18, 21, 29, 31, 35, 42, 50]


def main() -> None:
    rows = [
        json.loads(line)
        for line in CANDIDATES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 54, len(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, indexes in (("test", TEST_INDEXES), ("val", VAL_INDEXES)):
        selected = []
        for number, index in enumerate(indexes, start=1):
            row = rows[index - 1]
            selected.append(
                {
                    "id": f"{name}_{number:02d}",
                    "topic": row["topic"],
                    "topic_slug": row["topic_slug"],
                    "doc_id": row["doc_id"],
                    "title": row["title"],
                    "page": row["page"],
                    "question": row["question"].strip(),
                    "answer": row["answer"].strip(),
                    "source": "candidates2",
                    "candidate_index": index,
                }
            )
        path = OUT / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for item in selected:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"{name}: {len(selected)} -> {path}")


if __name__ == "__main__":
    main()

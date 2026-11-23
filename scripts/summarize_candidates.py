from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/eval/candidates.jsonl")
    parser.add_argument("--output", default="data/eval/candidates_summary.md")
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    lines: list[str] = []
    for index, row in enumerate(rows, start=1):
        lines.append(f"## {index}. [{row['topic_slug']}] {row['title'][:70]} — стр. {row['page']}")
        lines.append(f"**Q:** {row['question']}")
        lines.append("")
        lines.append("**A:** " + row["answer"].replace("\n", " "))
        lines.append("")
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print(f"rows: {len(rows)} -> {args.output}")


if __name__ == "__main__":
    main()

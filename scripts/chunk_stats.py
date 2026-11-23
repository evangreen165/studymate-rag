from __future__ import annotations

import argparse
from collections import Counter

from studymate.chunking import chunk_stats, collect_chunks
from studymate.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=None)
    args = parser.parse_args()
    settings = get_settings()
    chunks = collect_chunks(settings, max_docs=args.max_docs)
    stats = chunk_stats(chunks)
    topics = Counter(chunk.metadata.get("topic", "?") for chunk in chunks)
    docs = Counter(chunk.doc_id for chunk in chunks)
    print("chunks:", stats)
    print("topics:", dict(topics))
    print("documents:", len(docs))
    print("top documents:")
    for doc_id, count in docs.most_common(15):
        title = next(
            (chunk.metadata.get("title") for chunk in chunks if chunk.doc_id == doc_id), doc_id
        )
        print(f"  {count:5d}  {title}")


if __name__ == "__main__":
    main()

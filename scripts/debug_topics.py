from __future__ import annotations

import os
import unicodedata
from pathlib import Path

from studymate.corpus import TOPIC_MAP, topic_for_path

BASE = Path(r"C:\Users\user\Documents\курсовая\data")

print("TOPIC_MAP:")
for key in TOPIC_MAP:
    print(" ", repr(key), [hex(ord(c)) for c in key][:10])

print("\nDIRS:")
for name in os.listdir(BASE):
    path = BASE / name
    if not path.is_dir() or ord(name[0]) < 128:
        continue
    lowered = name.strip().lower()
    print(" ", repr(name), "->", repr(lowered), "NFKC=", repr(unicodedata.normalize("NFKC", lowered)))
    print("    in map:", lowered in TOPIC_MAP, "topic:", topic_for_path(next(path.glob("*.pdf")), BASE) if list(path.glob("*.pdf")) else None)
    for key in TOPIC_MAP:
        if key == lowered:
            print("    exact match")
        elif unicodedata.normalize("NFKC", key) == unicodedata.normalize("NFKC", lowered):
            print("    NFKC match")

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

_WS_RE = re.compile(r"[ \t\u00a0]+")
_MANY_NL_RE = re.compile(r"\n{3,}")
_PAGE_NUM_RE = re.compile(r"^\s*(?:[-–—]\s*)?\d{1,4}\s*(?:[-–—])?\s*$")
_CYR_RE = re.compile(r"[А-Яа-яЁё]")
_LAT_RE = re.compile(r"[A-Za-z]")


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def stable_id(*parts: Any) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()[:24]


def clean_text(text: str) -> str:
    """Мягкая очистка текста страницы: пробелы, пустые строки, колонтитулы-цифры."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if _PAGE_NUM_RE.match(stripped):
            continue
        lines.append(stripped)
    text = "\n".join(lines)
    text = _MANY_NL_RE.sub("\n\n", text)
    return text.strip()


def language_ratio(text: str) -> float:
    cyr = len(_CYR_RE.findall(text))
    lat = len(_LAT_RE.findall(text))
    total = cyr + lat
    return round(cyr / total, 4) if total else 0.0


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def human_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if size < 1024 or unit == "ГБ":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} ГБ"

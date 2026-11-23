from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pymupdf

from .config import Settings, get_settings
from .utils import (
    clean_text,
    language_ratio,
    read_json,
    sha256_file,
    sha256_text,
    utc_now_iso,
    write_json,
)

TOPIC_MAP: dict[str, tuple[str, str]] = {
    "computer science": ("cs", "Computer Science"),
    "cs": ("cs", "Computer Science"),
    "git": ("git", "Git"),
    "linux": ("linux", "Linux"),
    "sql базы данных": ("sql", "SQL и базы данных"),
    "sql": ("sql", "SQL и базы данных"),
    "алгоритмы, структуры данных, математика, логика": (
        "algorithms",
        "Алгоритмы и структуры данных",
    ),
    "algorithms": ("algorithms", "Алгоритмы и структуры данных"),
    "компьютерные сети": ("networks", "Компьютерные сети"),
    "networks": ("networks", "Компьютерные сети"),
    "ооп": ("oop", "ООП"),
    "oop": ("oop", "ООП"),
}
DEFAULT_TOPIC = ("other", "Другое")
SKIP_DIRS = {"demo", "uploads", "cache", "eval", "chat_history", ".ipynb_checkpoints"}

_HEADER_MIN_PAGES = 3
_HEADER_RATIO = 0.25
_HEADER_MAX_LEN = 120
_JUNK_TITLE_RE = re.compile(r"^(microsoft word|untitled|document|без названия)", re.I)


@dataclass
class PageText:
    page: int
    text: str

    @property
    def chars(self) -> int:
        return len(self.text)


@dataclass
class ExtractedDocument:
    doc_id: str
    path: Path
    rel_path: str
    title: str
    topic: str
    topic_slug: str
    pages: list[PageText] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def text_pages(self) -> int:
        return sum(1 for page in self.pages if page.chars >= 200)

    @property
    def chars(self) -> int:
        return sum(page.chars for page in self.pages)

    @property
    def low_text_pages(self) -> list[int]:
        return [page.page for page in self.pages if page.chars < 200]


def _normalize_part(part: str) -> str:
    return unicodedata.normalize("NFC", part).strip().lower()


def topic_for_path(path: Path, root: Path) -> tuple[str, str]:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    for part in relative.parts[:-1]:
        key = _normalize_part(part)
        if key in TOPIC_MAP:
            return TOPIC_MAP[key]
    return DEFAULT_TOPIC


def _clean_title(raw: str | None, path: Path) -> str:
    title = (raw or "").strip()
    title = re.sub(r"\s+", " ", title)
    lowered = title.lower()
    filename_is_cyrillic = bool(re.search(r"[А-Яа-яЁё]", path.stem))
    title_is_cyrillic = bool(re.search(r"[А-Яа-яЁё]", title))
    junk = (
        len(title) < 5
        or _JUNK_TITLE_RE.search(title)
        or "pdf document" in lowered
        or lowered.endswith(".indd")
        or lowered.endswith(".doc")
        or (filename_is_cyrillic and not title_is_cyrillic)
    )
    if junk:
        title = path.stem
    title = title.replace("_", " ").strip()
    return title[:180]


def _strip_repeated_lines(pages: list[str]) -> list[str]:
    """Убирает повторяющиеся колонтитулы в начале и конце страниц."""
    if len(pages) < _HEADER_MIN_PAGES:
        return pages
    counter: Counter[str] = Counter()
    for text in pages:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in lines[:2] + lines[-2:]:
            if 2 <= len(line) <= _HEADER_MAX_LEN:
                counter[line] += 1
    threshold = max(_HEADER_MIN_PAGES, int(len(pages) * _HEADER_RATIO))
    repeated = {line for line, count in counter.items() if count >= threshold}

    cleaned: list[str] = []
    for text in pages:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not lines:
            cleaned.append("")
            continue
        while lines and lines[0] in repeated:
            lines.pop(0)
        while lines and lines[-1] in repeated:
            lines.pop()
        cleaned.append("\n".join(lines))
    return cleaned


def extract_document(path: Path, root: Path, doc_id: str | None = None) -> ExtractedDocument:
    rel_path = str(path.relative_to(root)) if path.is_relative_to(root) else path.name
    topic_slug, topic = topic_for_path(path, root)
    doc_id = doc_id or sha256_text(rel_path)[:16]
    doc = ExtractedDocument(
        doc_id=doc_id,
        path=path,
        rel_path=rel_path,
        title=path.stem,
        topic=topic,
        topic_slug=topic_slug,
    )
    try:
        with pymupdf.open(path) as pdf:
            if pdf.needs_pass and not pdf.authenticate(""):
                raise ValueError("PDF защищён паролем")
            doc.metadata = {key: value for key, value in (pdf.metadata or {}).items() if value}
            doc.title = _clean_title(doc.metadata.get("title"), path)
            raw_pages = [page.get_text("text") or "" for page in pdf]
    except Exception as exc:  # noqa: BLE001 - ошибка конкретного PDF не должна ронять весь корпус
        doc.error = f"{type(exc).__name__}: {exc}"
        return doc

    raw_pages = _strip_repeated_lines(raw_pages)
    doc.pages = [
        PageText(page=index + 1, text=clean_text(text))
        for index, text in enumerate(raw_pages)
    ]
    return doc


def document_entry(doc: ExtractedDocument, path: Path, root: Path) -> dict:
    stat = path.stat()
    return {
        "doc_id": doc.doc_id,
        "path": doc.rel_path,
        "rel_path": doc.rel_path,
        "title": doc.title,
        "topic": doc.topic,
        "topic_slug": doc.topic_slug,
        "pages": doc.page_count,
        "text_pages": doc.text_pages,
        "chars": doc.chars,
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
        "language_ratio": language_ratio("\n".join(page.text for page in doc.pages)),
        "low_text_pages": doc.low_text_pages[:50],
        "low_text_count": len(doc.low_text_pages),
        "error": doc.error,
        "is_demo": "demo" in Path(doc.rel_path).parts,
        "duplicate_of": None,
    }


def scan_pdfs(
    roots: Iterable[Path],
    *,
    include_demo: bool = False,
    include_uploads: bool = True,
) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() != ".pdf":
                continue
            parts = {part.lower() for part in path.relative_to(root).parts[:-1]}
            if not include_demo and "demo" in parts:
                continue
            if not include_uploads and "uploads" in parts:
                continue
            if (parts & SKIP_DIRS) - {"demo", "uploads"}:
                continue
            paths.append(path)
    return paths


def refresh_summary(manifest: dict) -> dict:
    documents = manifest.get("documents", [])
    unique = [doc for doc in documents if not doc.get("duplicate_of")]
    topics: Counter[str] = Counter(doc.get("topic", "Другое") for doc in unique)
    manifest["summary"] = {
        "documents": len(documents),
        "unique_documents": len(unique),
        "indexed_candidates": len(
            [doc for doc in unique if not doc.get("error") and doc.get("text_pages", 0) > 0]
        ),
        "pages": sum(int(doc.get("pages", 0)) for doc in documents),
        "unique_pages": sum(int(doc.get("pages", 0)) for doc in unique),
        "text_pages": sum(int(doc.get("text_pages", 0)) for doc in unique),
        "chars": sum(int(doc.get("chars", 0)) for doc in unique),
        "size_bytes": sum(int(doc.get("size_bytes", 0)) for doc in documents),
        "topics": dict(sorted(topics.items(), key=lambda item: (-item[1], item[0]))),
        "errors": [
            {"doc_id": doc.get("doc_id"), "path": doc.get("rel_path"), "error": doc.get("error")}
            for doc in documents
            if doc.get("error")
        ],
        "duplicates": [
            {
                "doc_id": doc.get("doc_id"),
                "path": doc.get("rel_path"),
                "duplicate_of": doc.get("duplicate_of"),
            }
            for doc in documents
            if doc.get("duplicate_of")
        ],
        "scanned_documents": [
            {"doc_id": doc.get("doc_id"), "path": doc.get("rel_path"), "pages": doc.get("pages")}
            for doc in unique
            if doc.get("text_pages", 0) == 0 and not doc.get("error")
        ],
    }
    return manifest


def build_manifest(
    settings: Settings | None = None,
    *,
    demo: bool = False,
    include_uploads: bool = True,
    max_docs: int | None = None,
    output_path: Path | None = None,
    cache_text: bool = True,
) -> dict:
    settings = settings or get_settings()
    base_root = settings.corpus_dir
    roots = [settings.demo_corpus_dir] if demo else [settings.corpus_dir]
    if include_uploads:
        roots.append(settings.uploads_dir)

    paths = scan_pdfs(roots, include_demo=demo, include_uploads=include_uploads)
    if max_docs is not None:
        paths = paths[:max_docs]

    documents: list[dict] = []
    seen_hashes: dict[str, str] = {}

    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path.name}", flush=True)
        doc = extract_document(path, base_root)
        entry = document_entry(doc, path, base_root)
        if entry["sha256"] in seen_hashes:
            entry["duplicate_of"] = seen_hashes[entry["sha256"]]
        else:
            seen_hashes[entry["sha256"]] = entry["doc_id"]
        documents.append(entry)

        if cache_text and not doc.error:
            cache_path = settings.cache_dir / "extracted" / f"{doc.doc_id}.json"
            write_json(
                cache_path,
                {
                    "doc_id": doc.doc_id,
                    "rel_path": doc.rel_path,
                    "title": doc.title,
                    "topic": doc.topic,
                    "topic_slug": doc.topic_slug,
                    "pages": [
                        {"page": page.page, "text": page.text}
                        for page in doc.pages
                        if page.chars >= 50
                    ],
                },
            )

    manifest = {
        "generated_at": utc_now_iso(),
        "mode": "demo" if demo else "full",
        "root": str(base_root),
        "documents": documents,
    }
    refresh_summary(manifest)
    output_path = output_path or settings.manifest_path
    write_json(output_path, manifest)
    return manifest


def load_manifest(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    return read_json(settings.manifest_path, default={"documents": [], "summary": {}})


def save_manifest(manifest: dict, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    refresh_summary(manifest)
    write_json(settings.manifest_path, manifest)


def load_extracted(doc_id: str, settings: Settings | None = None) -> dict | None:
    settings = settings or get_settings()
    return read_json(settings.cache_dir / "extracted" / f"{doc_id}.json", default=None)


def manifest_documents(settings: Settings | None = None) -> list[dict]:
    manifest = load_manifest(settings)
    return [
        doc
        for doc in manifest.get("documents", [])
        if not doc.get("duplicate_of") and not doc.get("error") and doc.get("text_pages", 0) > 0
    ]

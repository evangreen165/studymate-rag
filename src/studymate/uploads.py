from __future__ import annotations

import re
import shutil
import time
import unicodedata
from pathlib import Path

from .config import Settings, get_settings
from .corpus import (
    document_entry,
    extract_document,
    load_manifest,
    save_manifest,
)
from .topics import TOPIC_BY_SLUG
from .utils import sha256_text, write_json

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
_SAFE_RE = re.compile(r"[^\w\-.() ]+", re.UNICODE)


class UploadError(ValueError):
    pass


def safe_filename(filename: str) -> str:
    name = unicodedata.normalize("NFC", Path(filename).name)
    name = _SAFE_RE.sub("_", name).strip(" ._")
    name = re.sub(r"\s+", "_", name)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name[:120]


def save_upload(
    data: bytes,
    filename: str,
    *,
    topic_slug: str,
    settings: Settings | None = None,
) -> Path:
    settings = settings or get_settings()
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError(f"Файл больше {MAX_UPLOAD_BYTES // 1024 // 1024} МБ")
    if not data.startswith(b"%PDF"):
        raise UploadError("Файл не похож на PDF")
    if topic_slug not in TOPIC_BY_SLUG:
        topic_slug = "other"

    safe_name = safe_filename(filename)
    folder = settings.uploads_dir / time.strftime("%Y%m%d")
    folder.mkdir(parents=True, exist_ok=True)
    digest = sha256_text(safe_name + str(len(data)))[:8]
    path = folder / f"{digest}_{safe_name}"
    path.write_bytes(data)
    return path


def register_uploaded_document(
    path: Path,
    *,
    topic_slug: str,
    settings: Settings | None = None,
) -> dict:
    settings = settings or get_settings()
    topic = TOPIC_BY_SLUG.get(topic_slug, TOPIC_BY_SLUG["other"])
    doc = extract_document(path, settings.corpus_dir)
    doc.topic_slug = topic.slug
    doc.topic = topic.label

    if not doc.error:
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

    entry = document_entry(doc, path, settings.corpus_dir)
    entry["topic"] = topic.label
    entry["topic_slug"] = topic.slug
    entry["is_demo"] = False
    entry["duplicate_of"] = None

    manifest = load_manifest(settings)
    documents = manifest.setdefault("documents", [])
    documents = [item for item in documents if item.get("doc_id") != entry["doc_id"]]
    documents.append(entry)
    manifest["documents"] = documents
    save_manifest(manifest, settings)
    return entry


def remove_document(
    doc_id: str,
    *,
    delete_file: bool = True,
    settings: Settings | None = None,
) -> dict:
    settings = settings or get_settings()
    manifest = load_manifest(settings)
    documents = manifest.get("documents", [])
    target = next((doc for doc in documents if doc.get("doc_id") == doc_id), None)
    if not target:
        return {"removed": False, "doc_id": doc_id}
    manifest["documents"] = [doc for doc in documents if doc.get("doc_id") != doc_id]
    save_manifest(manifest, settings)

    cache_path = settings.cache_dir / "extracted" / f"{doc_id}.json"
    if cache_path.exists():
        cache_path.unlink()

    deleted_file = False
    path = Path(target.get("path", ""))
    if not path.is_absolute():
        path = settings.corpus_dir / path
    if delete_file and path.exists() and path.is_relative_to(settings.uploads_dir):
        path.unlink()
        deleted_file = True
        parent = path.parent
        if parent != settings.uploads_dir and not any(parent.iterdir()):
            shutil.rmtree(parent, ignore_errors=True)
    return {"removed": True, "doc_id": doc_id, "deleted_file": deleted_file, "entry": target}

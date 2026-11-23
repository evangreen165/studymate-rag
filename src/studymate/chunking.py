from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from transformers import AutoTokenizer

from .config import Settings, get_settings
from .corpus import extract_document, load_extracted, manifest_documents
from .utils import sha256_text, stable_id

MIN_PAGE_CHARS = 50
MIN_CHUNK_CHARS = 80


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return str(self.metadata.get("doc_id", ""))

    @property
    def page(self) -> int:
        return int(self.metadata.get("page", 0))


def _split_sentences(text: str) -> list[str]:
    return [part for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


@lru_cache(maxsize=2)
def _get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def _tokenizer(model_name: str):
    tokenizer = _get_tokenizer(model_name)

    def count(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    return count


def _make_splitter(settings: Settings) -> SentenceSplitter:
    return SentenceSplitter(
        chunk_size=settings.chunk_size_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
        paragraph_separator="\n\n",
        secondary_chunking_regex=r"(?<=[.!?])\s+",
        chunking_tokenizer_fn=_split_sentences,
        tokenizer=_tokenizer(settings.embedding_model),
    )


def _page_texts(doc: dict, settings: Settings) -> list[tuple[int, str]]:
    cached = load_extracted(doc["doc_id"], settings)
    if cached:
        return [(int(page["page"]), str(page["text"])) for page in cached.get("pages", [])]

    path = Path(doc["path"])
    if not path.is_absolute():
        path = settings.corpus_dir / path
    if not path.exists():
        return []
    extracted = extract_document(path, settings.corpus_dir, doc_id=doc["doc_id"])
    return [(page.page, page.text) for page in extracted.pages if page.chars >= MIN_PAGE_CHARS]


def iter_chunks(
    settings: Settings | None = None,
    *,
    max_docs: int | None = None,
    doc_ids: set[str] | None = None,
) -> Iterator[Chunk]:
    settings = settings or get_settings()
    splitter = _make_splitter(settings)
    documents = manifest_documents(settings)
    if doc_ids is not None:
        documents = [doc for doc in documents if doc["doc_id"] in doc_ids]
    if max_docs is not None:
        documents = documents[:max_docs]

    seen_text: set[str] = set()
    for doc in documents:
        pages = _page_texts(doc, settings)
        for page_number, text in pages:
            if len(text) < MIN_PAGE_CHARS:
                continue
            base_metadata = {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "topic": doc["topic"],
                "topic_slug": doc["topic_slug"],
                "page": page_number,
                "rel_path": doc["rel_path"],
                "source": doc["path"],
            }
            node_doc = Document(text=text, metadata=base_metadata)
            nodes = splitter.get_nodes_from_documents([node_doc])
            for index, node in enumerate(nodes):
                chunk_text = str(node.get_content()).strip()
                if len(chunk_text) < MIN_CHUNK_CHARS:
                    continue
                normalized = " ".join(chunk_text.lower().split())
                text_hash = sha256_text(normalized)
                if text_hash in seen_text:
                    continue
                seen_text.add(text_hash)
                metadata = dict(base_metadata)
                metadata["page"] = page_number
                metadata["chunk_index"] = index
                metadata["text_hash"] = text_hash
                yield Chunk(
                    chunk_id=stable_id(doc["doc_id"], page_number, index, text_hash),
                    text=chunk_text,
                    metadata=metadata,
                )


def collect_chunks(
    settings: Settings | None = None,
    *,
    max_docs: int | None = None,
    doc_ids: set[str] | None = None,
) -> list[Chunk]:
    return list(iter_chunks(settings, max_docs=max_docs, doc_ids=doc_ids))


def sample_chunks(chunks: list[Chunk], max_per_doc: int = 70) -> list[Chunk]:
    """Стратифицированная подвыборка: равномерно по всему документу, не только начало."""
    grouped: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.doc_id, []).append(chunk)
    selected: list[Chunk] = []
    for doc_chunks in grouped.values():
        if len(doc_chunks) <= max_per_doc:
            selected.extend(doc_chunks)
            continue
        step = (len(doc_chunks) - 1) / (max_per_doc - 1)
        indices = sorted({int(round(index * step)) for index in range(max_per_doc)})
        selected.extend(doc_chunks[index] for index in indices)
    return selected


def write_chunks_jsonl(chunks: list[Chunk], path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(
                json.dumps(
                    {"chunk_id": chunk.chunk_id, "text": chunk.text, "metadata": chunk.metadata},
                    ensure_ascii=False,
                )
                + "\n"
            )


def read_chunks_jsonl(path: Path) -> list[Chunk]:
    import json

    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            payload = json.loads(line)
            chunks.append(
                Chunk(
                    chunk_id=payload["chunk_id"],
                    text=payload["text"],
                    metadata=payload.get("metadata", {}),
                )
            )
    return chunks


def chunk_stats(chunks: list[Chunk]) -> dict:
    lengths = [len(chunk.text) for chunk in chunks]
    if not lengths:
        return {"chunks": 0, "avg_chars": 0, "min_chars": 0, "max_chars": 0}
    return {
        "chunks": len(lengths),
        "avg_chars": round(sum(lengths) / len(lengths), 1),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "total_chars": sum(lengths),
    }

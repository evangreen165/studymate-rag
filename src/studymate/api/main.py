from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import orjson
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from .. import __version__
from ..config import get_settings
from ..corpus import load_manifest
from ..index import VectorIndex
from ..llm import LLMError, provider_status
from ..logging import request_id_ctx, setup_logging
from ..rag import IndexNotReady, RAGService
from ..topics import TOPICS
from ..uploads import UploadError, register_uploaded_document, remove_document, save_upload
from .jobs import IndexJobManager
from .schemas import (
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    DocumentOut,
    DocumentsResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    IndexRequest,
    IndexStatus,
    ReadyResponse,
    TopicOut,
    UploadResponse,
)

settings = get_settings()
setup_logging(settings)
logger = logging.getLogger("studymate.api")
rag = RAGService(settings)
jobs = IndexJobManager(settings)
STARTED_AT = time.time()

REQUESTS = Counter(
    "studymate_requests_total", "HTTP requests", ["method", "path", "status"]
)
LATENCY = Histogram(
    "studymate_request_latency_seconds", "HTTP request latency", ["method", "path"]
)
TOKENS = Counter("studymate_llm_tokens_total", "LLM tokens", ["kind"])
CHAT_REQUESTS = Counter("studymate_chat_requests_total", "Chat requests", ["status"])
FEEDBACK = Counter("studymate_feedback_total", "User feedback", ["rating"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "startup",
        extra={
            "model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "index_dir": str(settings.index_dir),
        },
    )
    yield
    logger.info("shutdown")


app = FastAPI(
    title="StudyMate RAG API",
    version=__version__,
    description=(
        "REST API ассистента по учебным материалам курса. "
        "Отвечает на вопросы студентов с опорой на найденные фрагменты PDF и возвращает источники."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    token = request_id_ctx.set(request_id)
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration_ms = (time.perf_counter() - started) * 1000
        REQUESTS.labels(request.method, request.url.path, str(status_code)).inc()
        LATENCY.labels(request.method, request.url.path).observe(duration_ms / 1000)
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )
        request_id_ctx.reset(token)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok", version=__version__, uptime_s=round(time.time() - STARTED_AT, 1)
    )


@app.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse:
    manifest = load_manifest(settings)
    chunks = rag.index_count()
    meta = rag.retriever.index.read_meta()
    bm25_chunks = int(meta.get("bm25_chunks", 0)) if isinstance(meta, dict) else 0
    if chunks == 0:
        raise HTTPException(
            status_code=503,
            detail="Индекс пуст. Выполните POST /index или scripts/build_index.py.",
        )
    return ReadyResponse(
        status="ready",
        chunks=chunks,
        bm25_chunks=bm25_chunks,
        documents=int(manifest.get("summary", {}).get("unique_documents", 0)),
        embedding_model=settings.embedding_model,
        llm_provider=settings.llm_provider,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> dict:
    try:
        result = rag.answer(
            payload.question,
            top_k=payload.top_k,
            topic=payload.topic,
            provider=payload.provider,
            model=payload.model,
            max_tokens=payload.max_tokens,
        )
    except IndexNotReady as exc:
        CHAT_REQUESTS.labels("not_ready").inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        CHAT_REQUESTS.labels("bad_request").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMError as exc:
        CHAT_REQUESTS.labels("llm_error").inc()
        raise HTTPException(status_code=502, detail=f"Ошибка LLM: {exc}") from exc
    CHAT_REQUESTS.labels("ok" if not result.refusal else "refusal").inc()
    TOKENS.labels("prompt").inc(result.prompt_tokens)
    TOKENS.labels("completion").inc(result.completion_tokens)
    logger.info(
        "chat",
        extra={
            "tokens": result.total_tokens,
            "model": result.model,
            "duration_ms": round(result.latency_ms, 2),
        },
    )
    return result.to_dict()


@app.get("/documents", response_model=DocumentsResponse)
def documents() -> dict:
    manifest = load_manifest(settings)
    meta = rag.retriever.index.read_meta()
    chunk_counts = meta.get("document_chunks", {}) if isinstance(meta, dict) else {}
    bm25_counts = meta.get("bm25_document_chunks", {}) if isinstance(meta, dict) else {}
    indexed = set(chunk_counts)
    sparse_only = set(bm25_counts)
    items: list[DocumentOut] = []
    for doc in manifest.get("documents", []):
        doc_id = doc.get("doc_id", "")
        if doc.get("duplicate_of"):
            status = "duplicate"
        elif doc.get("error"):
            status = "error"
        elif int(doc.get("text_pages", 0)) == 0:
            status = "scanned"
        elif doc_id in indexed:
            status = "indexed"
        elif doc_id in sparse_only:
            status = "sparse_only"
        else:
            status = "not_indexed"
        items.append(
            DocumentOut(
                doc_id=doc_id,
                title=doc.get("title", ""),
                topic=doc.get("topic", ""),
                topic_slug=doc.get("topic_slug", ""),
                pages=int(doc.get("pages", 0)),
                text_pages=int(doc.get("text_pages", 0)),
                chars=int(doc.get("chars", 0)),
                size_bytes=int(doc.get("size_bytes", 0)),
                language_ratio=float(doc.get("language_ratio", 0)),
                is_demo=bool(doc.get("is_demo", False)),
                chunks=int(chunk_counts.get(doc_id, 0)),
                status=status,
                rel_path=doc.get("rel_path", ""),
            )
        )
    return {
        "documents": items,
        "summary": manifest.get("summary", {}),
        "index": {
            "chunks": rag.index_count(),
            "meta": meta,
        },
    }


@app.get("/topics", response_model=list[TopicOut])
def topics() -> list[dict]:
    manifest = load_manifest(settings)
    counts: dict[str, int] = {}
    for doc in manifest.get("documents", []):
        if doc.get("duplicate_of"):
            continue
        slug = doc.get("topic_slug", "other")
        counts[slug] = counts.get(slug, 0) + 1
    return [
        TopicOut(slug=topic.slug, label=topic.label, short=topic.short, documents=counts.get(topic.slug, 0))
        for topic in TOPICS
    ]


@app.get("/stats")
def stats() -> dict:
    return {**rag.status(), "index_job": jobs.status()}


@app.get("/providers")
def providers() -> dict:
    return provider_status(settings)


@app.post("/index", response_model=IndexStatus)
def start_index(payload: IndexRequest | None = None) -> dict:
    payload = payload or IndexRequest()
    try:
        return jobs.start(rebuild=payload.rebuild, doc_ids=payload.doc_ids)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/index/status", response_model=IndexStatus)
def index_status() -> dict:
    return jobs.status()


@app.post("/index/cancel", response_model=IndexStatus)
def cancel_index() -> dict:
    return jobs.cancel()


@app.post("/corpus/upload", response_model=UploadResponse)
async def upload_corpus(
    files: list[UploadFile] = File(...),
    topic: str = Form("other"),
) -> dict:
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Не более 50 файлов за один раз")
    uploaded: list[DocumentOut] = []
    errors: list[str] = []
    for file in files:
        filename = file.filename or "document.pdf"
        try:
            data = await file.read()
            path = save_upload(data, filename, topic_slug=topic, settings=settings)
            entry = register_uploaded_document(path, topic_slug=topic, settings=settings)
            if entry.get("error"):
                errors.append(f"{filename}: {entry['error']}")
            uploaded.append(
                DocumentOut(
                    doc_id=entry["doc_id"],
                    title=entry["title"],
                    topic=entry["topic"],
                    topic_slug=entry["topic_slug"],
                    pages=entry["pages"],
                    text_pages=entry["text_pages"],
                    chars=entry["chars"],
                    size_bytes=entry["size_bytes"],
                    language_ratio=entry["language_ratio"],
                    is_demo=False,
                    chunks=0,
                    status="not_indexed" if not entry.get("error") else "error",
                    rel_path=entry["rel_path"],
                )
            )
        except UploadError as exc:
            errors.append(f"{filename}: {exc}")
        except Exception as exc:  # noqa: BLE001 - ошибка одного файла не должна ронять пакет
            errors.append(f"{filename}: {type(exc).__name__}: {exc}")
    return {"uploaded": uploaded, "errors": errors}


@app.delete("/corpus/documents/{doc_id}", response_model=DeleteResponse)
def delete_document(doc_id: str) -> dict:
    result = remove_document(doc_id, settings=settings)
    if not result.get("removed"):
        raise HTTPException(status_code=404, detail="Документ не найден")
    try:
        VectorIndex(settings).delete_document(doc_id)
    except Exception as exc:  # noqa: BLE001 - документ уже удалён из манифеста
        logger.warning("delete_index_failed", extra={"error": str(exc)})
    return DeleteResponse(
        removed=True, doc_id=doc_id, deleted_file=bool(result.get("deleted_file"))
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(payload: FeedbackRequest) -> FeedbackResponse:
    path = Path(settings.corpus_dir) / "feedback.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "request_id": payload.request_id,
        "rating": payload.rating,
        "comment": payload.comment,
    }
    with path.open("ab") as fh:
        fh.write(orjson.dumps(record) + b"\n")
    FEEDBACK.labels(str(payload.rating)).inc()
    return FeedbackResponse()


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def run() -> None:
    import uvicorn

    uvicorn.run(
        "studymate.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    run()

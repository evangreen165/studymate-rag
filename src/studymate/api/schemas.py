from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SourceOut(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    topic: str
    topic_slug: str
    page: int
    score: float
    dense_score: float = 0.0
    bm25_score: float = 0.0
    snippet: str
    rel_path: str = ""


class UsageOut(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float = 0.0


class TimingsOut(BaseModel):
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=6, ge=1, le=12)
    topic: str | None = Field(default=None, max_length=40)
    provider: str | None = Field(default=None, max_length=30)
    model: str | None = Field(default=None, max_length=120)
    max_tokens: int | None = Field(default=None, ge=64, le=4096)


class ChatResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    sources: list[SourceOut] = []
    used_sources: list[SourceOut] = []
    provider: str = ""
    model: str = ""
    usage: UsageOut
    timings: TimingsOut
    finish_reason: str | None = None
    refusal: bool = False
    topic: str | None = None
    top_k: int = 0
    error: str | None = None


class DocumentOut(BaseModel):
    doc_id: str
    title: str
    topic: str
    topic_slug: str
    pages: int = 0
    text_pages: int = 0
    chars: int = 0
    size_bytes: int = 0
    language_ratio: float = 0.0
    is_demo: bool = False
    chunks: int = 0
    status: str = "indexed"
    rel_path: str = ""


class DocumentsResponse(BaseModel):
    documents: list[DocumentOut] = []
    summary: dict = {}
    index: dict = {}


class TopicOut(BaseModel):
    slug: str
    label: str
    short: str
    documents: int = 0


class IndexRequest(BaseModel):
    rebuild: bool = False
    doc_ids: list[str] | None = None


class IndexStatus(BaseModel):
    status: str
    progress: int = 0
    embedded: int = 0
    total: int = 0
    message: str = ""
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    rebuild: bool = False
    mode: str | None = None
    chunks: int = 0


class UploadResponse(BaseModel):
    uploaded: list[DocumentOut] = []
    errors: list[str] = []


class DeleteResponse(BaseModel):
    removed: bool
    doc_id: str
    deleted_file: bool = False


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_s: float


class ReadyResponse(BaseModel):
    status: str
    chunks: int
    bm25_chunks: int = 0
    documents: int
    embedding_model: str
    llm_provider: str


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=80)
    rating: int = Field(ge=-1, le=1)
    comment: str = Field(default="", max_length=2000)


class FeedbackResponse(BaseModel):
    accepted: bool = True

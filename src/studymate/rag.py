from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from .config import Settings, get_settings
from .llm import BaseLLM, LLMError, build_llm, provider_status
from .prompts import (
    REFUSAL_TEXT,
    build_messages,
    build_no_context_messages,
    cited_sources,
    extractive_answer,
    is_refusal,
)
from .retrieval import Retriever, Source


class IndexNotReady(RuntimeError):
    pass


@dataclass
class RAGResponse:
    request_id: str
    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    latency_ms: float = 0.0
    finish_reason: str | None = None
    refusal: bool = False
    topic: str | None = None
    top_k: int = 0
    used_sources: list[Source] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "question": self.question,
            "answer": self.answer,
            "sources": [source.to_dict() for source in self.sources],
            "used_sources": [source.to_dict() for source in self.used_sources],
            "contexts": [source.text for source in self.sources],
            "provider": self.provider,
            "model": self.model,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "tokens_per_second": self.tokens_per_second,
            },
            "timings": {
                "retrieval_ms": round(self.retrieval_ms, 2),
                "generation_ms": round(self.generation_ms, 2),
                "total_ms": round(self.latency_ms, 2),
            },
            "finish_reason": self.finish_reason,
            "refusal": self.refusal,
            "topic": self.topic,
            "top_k": self.top_k,
            "error": self.error,
        }


class RAGService:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        retriever: Retriever | None = None,
        llm: BaseLLM | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._retriever = retriever
        self._llm = llm
        self._llms: dict[tuple[str, str], BaseLLM] = {}
        if llm is not None:
            self._llms[(llm.provider, llm.model)] = llm

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever(self.settings)
        return self._retriever

    def get_llm(self, provider: str | None = None, model: str | None = None) -> BaseLLM:
        provider = (provider or self.settings.llm_provider).lower()
        model = model or (self.settings.llm_model if provider == "openrouter" else "")
        key = (provider, model)
        if key not in self._llms:
            self._llms[key] = build_llm(self.settings, provider=provider, model=model or None)
        return self._llms[key]

    def _call_llm(
        self,
        messages: list[dict],
        *,
        provider: str | None,
        model: str | None,
        max_tokens: int | None,
    ):
        provider = (provider or self.settings.llm_provider).lower()
        requested_model = model or (self.settings.llm_model if provider == "openrouter" else None)
        try:
            return self.get_llm(provider, requested_model).complete(messages, max_tokens=max_tokens)
        except LLMError:
            if provider != "openrouter" or requested_model == self.settings.llm_fallback_model:
                raise
            fallback = self.get_llm(provider, self.settings.llm_fallback_model)
            return fallback.complete(messages, max_tokens=max_tokens)

    def index_count(self) -> int:
        return self.retriever.index.count()

    def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        topic: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        request_id: str | None = None,
    ) -> RAGResponse:
        question = (question or "").strip()
        if not question:
            raise ValueError("Вопрос не может быть пустым")
        if len(question) > self.settings.max_question_chars:
            raise ValueError(f"Вопрос длиннее {self.settings.max_question_chars} символов")
        request_id = request_id or uuid.uuid4().hex
        started = time.perf_counter()
        if self.index_count() == 0:
            raise IndexNotReady("Индекс пуст. Сначала проиндексируйте корпус.")

        retrieval_started = time.perf_counter()
        sources = self.retriever.search(question, top_k=top_k, topic_slug=topic)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        if not sources:
            return RAGResponse(
                request_id=request_id,
                question=question,
                answer=REFUSAL_TEXT,
                refusal=True,
                topic=topic,
                top_k=top_k or self.settings.top_k,
                retrieval_ms=retrieval_ms,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        best_dense = max(source.dense_score for source in sources)
        best_bm25 = max(source.bm25_score for source in sources)
        if best_dense < self.settings.min_retrieval_score and best_bm25 < 1.0:
            return RAGResponse(
                request_id=request_id,
                question=question,
                answer=REFUSAL_TEXT,
                sources=sources,
                refusal=True,
                topic=topic,
                top_k=top_k or self.settings.top_k,
                retrieval_ms=retrieval_ms,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        messages = build_messages(question, sources)
        generation_started = time.perf_counter()
        response = self._call_llm(
            messages, provider=provider, model=model, max_tokens=max_tokens
        )
        if is_refusal(response.text):
            fallback = extractive_answer(question, sources)
            if fallback:
                response.text = fallback
                response.finish_reason = "extractive-fallback"
        generation_ms = (time.perf_counter() - generation_started) * 1000
        used = cited_sources(response.text, sources)
        return RAGResponse(
            request_id=request_id,
            question=question,
            answer=response.text or REFUSAL_TEXT,
            sources=sources,
            used_sources=used,
            provider=response.provider,
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            tokens_per_second=response.tokens_per_second,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            latency_ms=(time.perf_counter() - started) * 1000,
            finish_reason=response.finish_reason,
            refusal=is_refusal(response.text),
            topic=topic,
            top_k=top_k or self.settings.top_k,
        )

    def answer_no_context(
        self,
        question: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Baseline: та же LLM без поиска по документам."""
        started = time.perf_counter()
        response = self._call_llm(
            build_no_context_messages(question),
            provider=provider,
            model=model,
            max_tokens=max_tokens,
        )
        return {
            "question": question,
            "answer": response.text,
            "contexts": [],
            "provider": response.provider,
            "model": response.model,
            "usage": {
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens,
            },
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

    def status(self) -> dict:
        from .corpus import load_manifest

        manifest = load_manifest(self.settings)
        meta = self.retriever.index.read_meta()
        return {
            "index": {
                "chunks": self.retriever.index.count(),
                "path": str(self.settings.index_dir),
                "meta": meta,
            },
            "corpus": manifest.get("summary", {}),
            "embedding_model": self.settings.embedding_model,
            "retrieval": self.retriever.stats(),
            "providers": provider_status(self.settings),
            "chunk_size": self.settings.chunk_size_tokens,
            "chunk_overlap": self.settings.chunk_overlap_tokens,
        }

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

from .config import Settings, get_settings


class LLMError(RuntimeError):
    """Ошибка провайдера LLM."""


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        if self.latency_ms <= 0 or not self.completion_tokens:
            return 0.0
        return round(self.completion_tokens / (self.latency_ms / 1000), 2)


class BaseLLM(Protocol):
    provider: str
    model: str

    def complete(self, messages: list[dict], *, max_tokens: int | None = None) -> LLMResponse: ...


class OpenRouterLLM:
    provider = "openrouter"

    def __init__(self, settings: Settings | None = None, *, model: str | None = None) -> None:
        settings = settings or get_settings()
        if not settings.openrouter_api_key:
            raise LLMError("OPENROUTER_API_KEY не задан")
        from openai import OpenAI

        self.settings = settings
        self.model = model or settings.llm_model
        self.client = OpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
            timeout=settings.llm_timeout_s,
            max_retries=2,
            default_headers={
                "HTTP-Referer": "https://github.com/evangreen165/studymate-rag",
                "X-Title": "StudyMate RAG",
            },
        )

    def complete(self, messages: list[dict], *, max_tokens: int | None = None) -> LLMResponse:
        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.settings.llm_temperature,
                max_tokens=max_tokens or self.settings.llm_max_tokens,
                top_p=0.9,
            )
        except Exception as exc:  # noqa: BLE001 - конвертируем ошибку провайдера
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000
        choice = response.choices[0]
        usage = response.usage.model_dump() if response.usage else {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
        return LLMResponse(
            text=(choice.message.content or "").strip(),
            model=getattr(response, "model", self.model),
            provider=self.provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=round(latency_ms, 2),
            finish_reason=choice.finish_reason,
            usage=usage,
        )


class LocalLLM:
    provider = "local"

    _lock = threading.Lock()
    _instances: dict[str, object] = {}

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self.settings = settings
        self.model = settings.local_llm_abs_path.name
        self.path = settings.local_llm_abs_path
        if not self.path.exists():
            raise LLMError(f"Локальная модель не найдена: {self.path}")
        self._llm = self._load()

    def _load(self):
        from llama_cpp import Llama  # тяжёлая зависимость только для локального режима

        key = str(self.path)
        with self._lock:
            if key not in self._instances:
                self._instances[key] = Llama(
                    model_path=str(self.path),
                    n_ctx=self.settings.local_llm_context,
                    n_threads=self.settings.local_llm_threads or max(2, (os.cpu_count() or 4) - 1),
                    verbose=False,
                )
            return self._instances[key]

    def complete(self, messages: list[dict], *, max_tokens: int | None = None) -> LLMResponse:
        started = time.perf_counter()
        try:
            response = self._llm.create_chat_completion(
                messages=messages,
                temperature=self.settings.llm_temperature,
                max_tokens=max_tokens or self.settings.llm_max_tokens,
                top_p=0.9,
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000
        usage = response.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        return LLMResponse(
            text=(response["choices"][0]["message"]["content"] or "").strip(),
            model=self.model,
            provider=self.provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=round(latency_ms, 2),
            finish_reason=response["choices"][0].get("finish_reason"),
            usage=usage,
        )


class MockLLM:
    """Детерминированная заглушка для тестов и CI (без сети и токенов)."""

    provider = "mock"
    model = "mock-extractive"

    def complete(self, messages: list[dict], *, max_tokens: int | None = None) -> LLMResponse:
        started = time.perf_counter()
        prompt = "\n".join(str(message.get("content", "")) for message in messages)
        context_match = re.search(r"\[1\](.*?)(?:\[2\]|\Z)", prompt, flags=re.S)
        snippet = (context_match.group(1) if context_match else "").strip()
        snippet = re.sub(r"\s+", " ", snippet)[:400]
        if snippet:
            text = f"Согласно учебным материалам: {snippet} [1]"
        else:
            text = "В предоставленных материалах ответа нет."
        latency_ms = (time.perf_counter() - started) * 1000
        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.provider,
            prompt_tokens=max(1, len(prompt) // 4),
            completion_tokens=max(1, len(text) // 4),
            total_tokens=max(1, (len(prompt) + len(text)) // 4),
            latency_ms=round(latency_ms, 2),
            finish_reason="stop",
        )


def build_llm(
    settings: Settings | None = None,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> BaseLLM:
    settings = settings or get_settings()
    provider = (provider or settings.llm_provider).lower()
    if provider == "mock":
        return MockLLM()
    if provider == "local":
        return LocalLLM(settings)
    if provider == "openrouter":
        return OpenRouterLLM(settings, model=model)
    raise LLMError(f"Неизвестный провайдер LLM: {provider}")


def provider_status(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    return {
        "openrouter": {
            "configured": settings.has_openrouter,
            "model": settings.llm_model,
            "fallback": settings.llm_fallback_model,
        },
        "local": {
            "configured": settings.local_llm_abs_path.exists(),
            "model": settings.local_llm_abs_path.name,
        },
        "mock": {"configured": True, "model": "mock-extractive"},
        "default": settings.llm_provider,
    }

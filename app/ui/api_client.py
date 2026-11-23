from __future__ import annotations

import os
from typing import Any

import httpx

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = float(os.getenv("API_TIMEOUT", "180"))


class APIError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=TIMEOUT)


def _request(method: str, path: str, **kwargs: Any) -> Any:
    try:
        with _client() as client:
            response = client.request(method, path, **kwargs)
    except httpx.HTTPError as exc:
        raise APIError(f"API недоступен: {exc}") from exc
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise APIError(str(detail), response.status_code)
    if not response.content:
        return None
    return response.json()


def health() -> dict:
    return _request("GET", "/health")


def ready() -> dict:
    return _request("GET", "/ready")


def stats() -> dict:
    return _request("GET", "/stats")


def providers() -> dict:
    return _request("GET", "/providers")


def topics() -> list[dict]:
    return _request("GET", "/topics")


def documents() -> dict:
    return _request("GET", "/documents")


def chat(
    question: str,
    *,
    top_k: int,
    topic: str | None,
    provider: str | None,
    model: str | None,
    max_tokens: int | None = None,
) -> dict:
    payload = {
        "question": question,
        "top_k": top_k,
        "topic": topic if topic and topic != "all" else None,
        "provider": provider,
        "model": model or None,
        "max_tokens": max_tokens,
    }
    return _request("POST", "/chat", json=payload)


def upload(files: list[tuple[str, bytes]], topic: str) -> dict:
    multipart = [
        ("files", (name, data, "application/pdf")) for name, data in files
    ]
    return _request("POST", "/corpus/upload", files=multipart, data={"topic": topic})


def start_index(*, rebuild: bool = False, doc_ids: list[str] | None = None) -> dict:
    return _request("POST", "/index", json={"rebuild": rebuild, "doc_ids": doc_ids})


def index_status() -> dict:
    return _request("GET", "/index/status")


def cancel_index() -> dict:
    return _request("POST", "/index/cancel")


def delete_document(doc_id: str) -> dict:
    return _request("DELETE", f"/corpus/documents/{doc_id}")


def feedback(request_id: str, rating: int, comment: str = "") -> dict:
    return _request(
        "POST", "/feedback", json={"request_id": request_id, "rating": rating, "comment": comment}
    )

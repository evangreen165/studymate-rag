from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path

from .config import Settings, get_settings

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get(),
        }
        for key in ("method", "path", "status", "duration_ms", "tokens", "model", "error"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(settings: Settings | None = None, *, force: bool = False) -> None:
    settings = settings or get_settings()
    root = logging.getLogger()
    if root.handlers and not force:
        return
    root.handlers.clear()
    root.setLevel(settings.log_level.upper())
    formatter = JsonFormatter()

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    try:
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(settings.log_dir / "app.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        pass

    for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_path(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.log_dir / "app.log"

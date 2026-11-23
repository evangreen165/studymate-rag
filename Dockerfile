# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_TORCH_BACKEND=cpu

RUN pip install --no-cache-dir uv==0.13.9

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY app ./app
COPY scripts ./scripts
COPY data/README.md ./data/README.md
COPY data/demo ./data/demo
RUN uv sync --frozen --no-dev

# Кэшируем модель эмбеддингов в образе, чтобы контейнер не скачивал 600 МБ при старте.
ENV HF_HOME=/app/.cache/hf
RUN .venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('deepvk/USER2-base', cache_folder='/app/.cache/hf')"


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    HF_HOME="/app/.cache/hf" \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/app /app/app
COPY --from=builder /app/scripts /app/scripts
COPY --from=builder /app/data /app/data
COPY --from=builder /app/.cache/hf /app/.cache/hf
COPY pyproject.toml README.md ./
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /app/chroma_db /app/chroma_db_demo /app/logs /app/data/uploads /app/data/cache /app/data/chat_history \
    && chown -R app:app /app

USER app
EXPOSE 8000 8501

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"]

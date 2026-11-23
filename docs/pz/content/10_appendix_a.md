# ПРИЛОЖЕНИЕ А. РАЗВЁРТЫВАНИЕ ПРОЕКТА

## А.1 Ссылка на репозиторий

Исходный код, тесты, документация и демонстрационный корпус доступны в Git-репозитории:

https://github.com/evangreen165/studymate-rag

Репозиторий содержит историю коммитов, README с инструкцией запуска, GitHub Actions и GitHub Pages. Полный корпус из {{DOCS}} PDF-документов в репозиторий не входит по условиям авторских прав; в нём хранятся манифест, паспорт, хэши и скрипты сборки.

## А.2 Требования

- Docker 24+ и Docker Compose v2;
- либо Python 3.12 и uv 0.13+;
- ключ OpenRouter (для облачной LLM);
- 8 ГБ RAM минимум, 16 ГБ рекомендуется;
- свободное место: около 3 ГБ на образ и 1 ГБ на индекс.

## А.3 Быстрый запуск через Docker

1. Клонировать репозиторий:

```bash
git clone https://github.com/evangreen165/studymate-rag.git
cd studymate-rag
```

2. Создать файл окружения и указать ключ:

```bash
cp .env.example .env
# OPENROUTER_API_KEY=sk-or-...
```

3. Запустить систему:

```bash
docker compose up --build
```

4. Открыть:

- веб-интерфейс: http://localhost:8501;
- Swagger: http://localhost:8000/docs;
- health check: http://localhost:8000/health.

При первом запуске API собирает демонстрационный индекс из семи синтетических конспектов (CC BY 4.0). Полный корпус подключается отдельно (пункт А.5).

## А.4 Локальный запуск

```bash
uv sync
cp .env.example .env
uv run python scripts/make_demo_corpus.py
uv run python scripts/build_index.py --demo
uv run uvicorn studymate.api.main:app --port 8000
uv run streamlit run app/ui/streamlit_app.py
```

Проверка API:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Что такое полиморфизм?","top_k":5}'
```

## А.5 Подключение полного корпуса

1. Разложить PDF по тематическим папкам внутри `data/` (Computer Science, Git, Linux, SQL, алгоритмы, сети, ООП).
2. Собрать манифест и паспорт:

```bash
uv run python scripts/build_manifest.py
```

3. Построить гибридный индекс:

```bash
uv run python scripts/build_index.py --skip-manifest
```

Индексация на CPU занимает около {{BUILD_MIN}} минут: {{BM25_CHUNKS}} чанков BM25 и {{DENSE_CHUNKS}} плотных векторов. Результат сохраняется в `chroma_db/`, `data/index_meta.json` и `data/bm25_corpus.json`.

4. Запустить API с полным индексом:

```bash
INDEX_DIR=chroma_db \
MANIFEST_PATH=data/manifest.json \
INDEX_META_PATH=data/index_meta.json \
BM25_PATH=data/bm25_corpus.json \
uv run uvicorn studymate.api.main:app --port 8000
```

## А.6 Переменные окружения

| Переменная | Назначение | Значение по умолчанию |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | ключ OpenRouter | — |
| `LLM_PROVIDER` | провайдер LLM | `openrouter` |
| `LLM_MODEL` | основная модель | `meta-llama/llama-3.1-70b-instruct` |
| `LLM_FALLBACK_MODEL` | резервная модель | `meta-llama/llama-3.3-70b-instruct` |
| `EMBEDDING_MODEL` | модель эмбеддингов | `deepvk/USER2-base` |
| `INDEX_DIR` | каталог ChromaDB | `chroma_db` |
| `MANIFEST_PATH` | манифест корпуса | `data/manifest.json` |
| `BM25_PATH` | BM25-корпус | `data/bm25_corpus.json` |
| `API_URL` | адрес API для UI | `http://127.0.0.1:8000` |
| `LOG_LEVEL` | уровень логов | `INFO` |

## А.7 Проверка работоспособности

```bash
uv run pytest --cov=studymate
uv run ruff check .
docker compose ps
curl http://localhost:8000/metrics
```

Ожидаемый результат: тесты проходят, линтер не находит ошибок, оба контейнера в состоянии `healthy`, метрики Prometheus доступны.

## А.8 Остановка и очистка

```bash
docker compose down
# удалить индекс и кэш (осторожно: полный корпус не удаляется)
rm -rf chroma_db chroma_db_demo data/cache logs
```

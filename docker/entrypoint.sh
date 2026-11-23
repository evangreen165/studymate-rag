#!/bin/sh
set -eu

MODE="${1:-api}"

if [ "$MODE" = "api" ]; then
  if [ ! -f "${MANIFEST_PATH:-/app/data/manifest_demo.json}" ]; then
    echo "Манифест не найден, собираю demo-корпус…"
    python scripts/build_index.py --demo
  fi
  if [ ! -d "${INDEX_DIR:-/app/chroma_db_demo}" ] || [ -z "$(ls -A "${INDEX_DIR:-/app/chroma_db_demo}" 2>/dev/null || true)" ]; then
    echo "Индекс не найден, индексирую demo-корпус…"
    python scripts/build_index.py --demo --skip-manifest
  fi
  exec uvicorn studymate.api.main:app --host 0.0.0.0 --port 8000 --workers 1
fi

if [ "$MODE" = "ui" ]; then
  exec streamlit run app/ui/streamlit_app.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false
fi

exec "$@"

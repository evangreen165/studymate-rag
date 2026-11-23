.PHONY: install demo-index full-index api ui test lint docker docs clean

install:
	uv sync

demo-index:
	uv run python scripts/make_demo_corpus.py
	uv run python scripts/build_index.py --demo

full-index:
	uv run python scripts/build_manifest.py
	uv run python scripts/build_index.py --skip-manifest

api:
	uv run uvicorn studymate.api.main:app --host 127.0.0.1 --port 8000 --reload

ui:
	uv run streamlit run app/ui/streamlit_app.py

test:
	uv run pytest --cov=studymate --cov-report=term-missing

lint:
	uv run ruff check .

docker:
	docker compose up --build

eval:
	uv run python scripts/evaluate.py --split test

clean:
	rm -rf chroma_db chroma_db_demo logs .pytest_cache .ruff_cache

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Настройки приложения. Значения читаются из .env и переменных окружения."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM ---
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    llm_provider: str = Field(default="openrouter", alias="LLM_PROVIDER")
    llm_model: str = Field(
        default="meta-llama/llama-3.1-70b-instruct", alias="LLM_MODEL"
    )
    llm_fallback_model: str = Field(
        default="meta-llama/llama-3.3-70b-instruct", alias="LLM_FALLBACK_MODEL"
    )
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_timeout_s: float = 120.0
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    local_llm_path: str = Field(
        default="Qwen2.5-7B-Instruct-Q4_K_M.gguf", alias="LOCAL_LLM_PATH"
    )
    local_llm_context: int = Field(default=4096, alias="LOCAL_LLM_CONTEXT")
    local_llm_threads: int = 0

    # --- Эмбеддинги и индекс ---
    embedding_model: str = Field(default="deepvk/USER2-base", alias="EMBEDDING_MODEL")
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    embedding_max_length: int = 1024
    corpus_dir: Path = Field(default=PROJECT_ROOT / "data", alias="CORPUS_DIR")
    demo_corpus_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "demo", alias="DEMO_CORPUS_DIR"
    )
    uploads_dir: Path = Field(default=PROJECT_ROOT / "data" / "uploads")
    cache_dir: Path = Field(default=PROJECT_ROOT / "data" / "cache")
    index_dir: Path = Field(default=PROJECT_ROOT / "chroma_db", alias="INDEX_DIR")
    collection_name: str = Field(default="course_materials", alias="COLLECTION_NAME")
    manifest_path: Path = Field(
        default=PROJECT_ROOT / "data" / "manifest.json", alias="MANIFEST_PATH"
    )
    index_meta_path: Path = Field(
        default=PROJECT_ROOT / "data" / "index_meta.json", alias="INDEX_META_PATH"
    )

    # --- Поиск ---
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 100
    dense_max_per_doc: int = 25
    bm25_path: Path = Field(
        default=PROJECT_ROOT / "data" / "bm25_corpus.json", alias="BM25_PATH"
    )
    top_k: int = 6
    fetch_k: int = 24
    mmr_lambda: float = 0.55
    min_retrieval_score: float = 0.05
    use_hybrid: bool = True
    rrf_k: int = 60

    # --- API / UI ---
    api_host: str = Field(default="127.0.0.1", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_url: str = Field(default="http://127.0.0.1:8000", alias="API_URL")
    ui_port: int = Field(default=8501, alias="UI_PORT")
    cors_origins: str = Field(
        default="http://localhost:8501,http://127.0.0.1:8501", alias="CORS_ORIGINS"
    )
    max_question_chars: int = 2000

    # --- Логи ---
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_dir: Path = Field(default=PROJECT_ROOT / "logs", alias="LOG_DIR")

    @field_validator("corpus_dir", "demo_corpus_dir", "uploads_dir", "cache_dir", "index_dir", "log_dir", "bm25_path", mode="before")
    @classmethod
    def _expand_path(cls, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def has_openrouter(self) -> bool:
        return bool(self.openrouter_api_key)

    @property
    def local_llm_abs_path(self) -> Path:
        path = Path(self.local_llm_path).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

"""Centralized project settings.

Values are loaded from environment variables first and fall back to simple
defaults so existing local runs continue to work without extra setup.
"""

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_path(name: str, default: str) -> Path:
    value = os.getenv(name, default)
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return BASE_DIR / path


def _env_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    # Neo4j
    neo4j_uri: str = _env_str("NEO4J_URI", "http://localhost:7474")
    neo4j_user: str = _env_str("NEO4J_USER", "neo4j")
    neo4j_password: str = _env_str("NEO4J_PASSWORD", "all-in-rag")
    neo4j_database: str = _env_str("NEO4J_DATABASE", "neo4j")

    # Milvus / Zilliz Cloud
    milvus_host: str = _env_str("MILVUS_HOST", "localhost")
    milvus_port: int = _env_int("MILVUS_PORT", 19530)
    milvus_uri: str = _env_str("MILVUS_URI", "")
    milvus_token: str = _env_str("MILVUS_TOKEN", "")
    milvus_collection: str = _env_str("MILVUS_COLLECTION", "medical_qa")

    # Embedding
    embedding_model_name: str = _env_str(
        "EMBEDDING_MODEL_NAME",
        "BAAI/bge-small-zh-v1.5",
    )

    # Data paths
    toyhom_dataset_path: Path = _env_path(
        "TOYHOM_DATASET_PATH",
        "Chinese-medical-dialogue-data/Data_数据",
    )
    user_upload_case_path: Path = _env_path(
        "USER_UPLOAD_CASE_PATH",
        "user_uploads/cases",
    )

    # Retrieval
    retrieval_top_k: int = _env_int("RETRIEVAL_TOP_K", 10)
    rerank_top_k: int = _env_int("RERANK_TOP_K", 5)

    # LLM / DeepSeek
    llm_type: str = _env_str("LLM_TYPE", "deepseek")
    deepseek_api_key: str = _env_str("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = _env_str("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_default_model: str = _env_str("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat")
    deepseek_model_options: tuple[str, ...] = _env_tuple(
        "DEEPSEEK_MODEL_OPTIONS",
        (
            "deepseek-chat",
            "deepseek-reasoner",
            "deepseek-v4-flash",
            "deepseek-v4-pro",
        ),
    )
    deepseek_intent_model: str = _env_str("DEEPSEEK_INTENT_MODEL", deepseek_default_model)
    deepseek_answer_model: str = _env_str("DEEPSEEK_ANSWER_MODEL", deepseek_default_model)


settings = Settings()


# Backward-friendly constant aliases for modules that prefer direct imports.
NEO4J_URI = settings.neo4j_uri
NEO4J_USER = settings.neo4j_user
NEO4J_PASSWORD = settings.neo4j_password
NEO4J_DATABASE = settings.neo4j_database

MILVUS_HOST = settings.milvus_host
MILVUS_PORT = settings.milvus_port
MILVUS_COLLECTION = settings.milvus_collection

EMBEDDING_MODEL_NAME = settings.embedding_model_name
TOYHOM_DATASET_PATH = settings.toyhom_dataset_path
USER_UPLOAD_CASE_PATH = settings.user_upload_case_path

RETRIEVAL_TOP_K = settings.retrieval_top_k
RERANK_TOP_K = settings.rerank_top_k
LLM_TYPE = settings.llm_type
DEEPSEEK_API_KEY = settings.deepseek_api_key
DEEPSEEK_BASE_URL = settings.deepseek_base_url
DEEPSEEK_DEFAULT_MODEL = settings.deepseek_default_model
DEEPSEEK_MODEL_OPTIONS = settings.deepseek_model_options
DEEPSEEK_INTENT_MODEL = settings.deepseek_intent_model
DEEPSEEK_ANSWER_MODEL = settings.deepseek_answer_model

"""LightRAG Core Service 配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# 优先加载本地环境变量，便于开发机覆盖默认配置。
load_dotenv(".env.local", override=True)
load_dotenv()

USER_DATA_DIR = Path(os.getenv("LIVERAG_USER_DATA_DIR", "~/.LiveRAG")).expanduser()
DEFAULT_RAG_STORAGE_DIR = USER_DATA_DIR / "rag" / "storage"
DEFAULT_UPLOAD_DIR = USER_DATA_DIR / "rag" / "sources"
DEFAULT_RAG_LOG_DIR = USER_DATA_DIR / "rag" / "logs"
DEFAULT_KNOWLEDGE_BASES_DIR = USER_DATA_DIR / "rag" / "knowledge_bases"


def _bool_env(name: str, default: bool) -> bool:
    """读取布尔环境变量。"""

    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    """读取整数环境变量。"""

    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _str_env(name: str, default: str = "") -> str:
    """读取字符串环境变量。"""

    return os.getenv(name, default).strip()


def _path_env(name: str, default: Path) -> str:
    """读取路径环境变量，并避免误写到代码仓库。"""

    raw = _str_env(name, "")
    if not raw:
        return str(default)
    resolved = Path(raw).expanduser()
    if "livekit-voice-agent" in resolved.parts or "LiveRAG" in resolved.parts:
        return str(default)
    if resolved.name in {"rag_storage", "uploads", "storage", "sources"} and not resolved.is_absolute():
        return str(default)
    return raw


@dataclass(frozen=True)
class Settings:
    """RAG 服务配置。"""

    host: str = _str_env("KB_SERVICE_HOST", "127.0.0.1")
    port: int = _int_env("KB_SERVICE_PORT", 9721)
    api_key: str = _str_env("KB_SERVICE_API_KEY", _str_env("LIGHTRAG_API_KEY", ""))

    user_data_dir: str = _str_env("LIVERAG_USER_DATA_DIR", str(USER_DATA_DIR))
    knowledge_bases_dir: str = _path_env("LIGHTRAG_KNOWLEDGE_BASES_DIR", DEFAULT_KNOWLEDGE_BASES_DIR)
    working_dir: str = _path_env("LIGHTRAG_WORKING_DIR", DEFAULT_RAG_STORAGE_DIR)
    upload_dir: str = _path_env("LIGHTRAG_UPLOAD_DIR", DEFAULT_UPLOAD_DIR)
    rag_log_dir: str = _path_env("LIGHTRAG_LOG_DIR", DEFAULT_RAG_LOG_DIR)
    workspace: str = _str_env("LIGHTRAG_WORKSPACE", _str_env("WORKSPACE", ""))
    kb_id: str = _str_env("LIGHTRAG_KB_ID", "default")
    kb_name: str = _str_env("LIGHTRAG_KB_NAME", "默认知识库")
    kv_storage: str = _str_env("LIGHTRAG_KV_STORAGE", "JsonKVStorage")
    vector_storage: str = _str_env("LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage")
    graph_storage: str = _str_env("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage")
    doc_status_storage: str = _str_env("LIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage")

    llm_model: str = _str_env("LIGHTRAG_LLM_MODEL", _str_env("LLM_MODEL", "qwen-plus"))
    llm_base_url: str = _str_env(
        "LIGHTRAG_LLM_BASE_URL",
        _str_env("LLM_BINDING_HOST", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    llm_api_key: str = _str_env(
        "LIGHTRAG_LLM_API_KEY",
        _str_env("LLM_BINDING_API_KEY", _str_env("DASHSCOPE_API_KEY", _str_env("OPENAI_API_KEY", ""))),
    )

    embedding_model: str = _str_env("LIGHTRAG_EMBEDDING_MODEL", _str_env("EMBEDDING_MODEL", "text-embedding-v4"))
    embedding_base_url: str = _str_env(
        "LIGHTRAG_EMBEDDING_BASE_URL",
        _str_env("EMBEDDING_BINDING_HOST", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    embedding_api_key: str = _str_env(
        "LIGHTRAG_EMBEDDING_API_KEY",
        _str_env("EMBEDDING_BINDING_API_KEY", _str_env("DASHSCOPE_API_KEY", _str_env("OPENAI_API_KEY", ""))),
    )
    embedding_dim: int = _int_env("LIGHTRAG_EMBEDDING_DIM", _int_env("EMBEDDING_DIM", 1024))
    max_embed_tokens: int = _int_env("LIGHTRAG_MAX_EMBED_TOKENS", _int_env("MAX_EMBED_TOKENS", 8192))
    chunk_token_size: int = _int_env("LIGHTRAG_CHUNK_SIZE", _int_env("CHUNK_SIZE", 1200))
    chunk_overlap_token_size: int = _int_env(
        "LIGHTRAG_CHUNK_OVERLAP_SIZE",
        _int_env("CHUNK_OVERLAP_SIZE", 100),
    )
    embedding_batch_num: int = _int_env(
        "LIGHTRAG_EMBEDDING_BATCH_NUM",
        _int_env("EMBEDDING_BATCH_NUM", 10),
    )
    embedding_func_max_async: int = _int_env(
        "LIGHTRAG_EMBEDDING_FUNC_MAX_ASYNC",
        _int_env("EMBEDDING_FUNC_MAX_ASYNC", 8),
    )
    llm_model_max_async: int = _int_env("LIGHTRAG_MAX_ASYNC", _int_env("MAX_ASYNC", 4))
    max_parallel_insert: int = _int_env(
        "LIGHTRAG_MAX_PARALLEL_INSERT",
        _int_env("MAX_PARALLEL_INSERT", 2),
    )
    entity_extract_max_gleaning: int = _int_env(
        "LIGHTRAG_ENTITY_EXTRACT_MAX_GLEANING",
        _int_env("ENTITY_EXTRACT_MAX_GLEANING", 1),
    )
    enable_llm_cache: bool = _bool_env(
        "LIGHTRAG_ENABLE_LLM_CACHE",
        _bool_env("ENABLE_LLM_CACHE", True),
    )
    enable_llm_cache_for_entity_extract: bool = _bool_env(
        "LIGHTRAG_ENABLE_LLM_CACHE_FOR_EXTRACT",
        _bool_env("ENABLE_LLM_CACHE_FOR_EXTRACT", True),
    )

    default_mode: str = _str_env("LIGHTRAG_DEFAULT_MODE", "mix")
    voice_mode: str = _str_env("LIGHTRAG_VOICE_MODE", "naive")
    top_k: int = _int_env("LIGHTRAG_TOP_K", _int_env("TOP_K", 60))
    chunk_top_k: int = _int_env("LIGHTRAG_CHUNK_TOP_K", _int_env("CHUNK_TOP_K", 20))
    voice_top_k: int = _int_env("LIGHTRAG_VOICE_TOP_K", 4)
    voice_chunk_top_k: int = _int_env("LIGHTRAG_VOICE_CHUNK_TOP_K", 4)
    voice_context_max_chars: int = _int_env("LIGHTRAG_VOICE_CONTEXT_MAX_CHARS", 1800)
    enable_rerank: bool = _bool_env("LIGHTRAG_ENABLE_RERANK", True)
    voice_enable_rerank: bool = _bool_env("LIGHTRAG_VOICE_ENABLE_RERANK", False)

    @property
    def absolute_working_dir(self) -> str:
        """返回绝对 RAG 存储目录。"""

        return str(Path(self.working_dir).expanduser().resolve())

    @property
    def absolute_user_data_dir(self) -> str:
        """返回绝对用户数据目录。"""

        return str(Path(self.user_data_dir).expanduser().resolve())

    @property
    def absolute_upload_dir(self) -> str:
        """返回绝对上传目录。"""

        return str(Path(self.upload_dir).expanduser().resolve())

    @property
    def absolute_knowledge_bases_dir(self) -> str:
        """返回知识库根目录。"""

        return str(Path(self.knowledge_bases_dir).expanduser().resolve())

    def provider_ready(self) -> bool:
        """判断 LLM 和 Embedding provider 是否具备必要密钥。"""

        return bool(self.llm_api_key and self.embedding_api_key)

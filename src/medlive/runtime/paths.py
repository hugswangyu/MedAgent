"""管理 LiveRAG 的用户数据目录。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimePaths:
    """集中描述运行时需要读写的所有用户文件路径。"""

    user_data_dir: Path
    db_file: Path
    prompts_dir: Path
    system_prompt_template_file: Path
    soul_file: Path
    history_compress_prompt_file: Path
    knowledge_overview_prompt_file: Path
    history_dir: Path
    context_dir: Path
    session_dir: Path
    voice_sessions_dir: Path
    messages_file: Path
    rag_context_file: Path
    session_system_prompt_file: Path
    runtime_state_file: Path
    model_dir: Path
    model_config_file: Path
    context_model_config_file: Path
    rag_dir: Path
    rag_knowledge_bases_dir: Path
    logs_dir: Path


def get_user_data_dir() -> Path:
    """返回用户数据根目录，默认是 ~/.LiveRAG。"""

    return Path(os.getenv("LIVERAG_USER_DATA_DIR", "~/.LiveRAG")).expanduser()


def build_runtime_paths(user_data_dir: Path | None = None) -> RuntimePaths:
    """根据用户数据根目录派生所有运行文件路径。"""

    root = (user_data_dir or get_user_data_dir()).expanduser()
    prompts_dir = root / "prompts"
    history_dir = root / "history"
    context_dir = root / "context"
    session_dir = root / "session"
    voice_sessions_dir = session_dir / "voice_sessions"
    model_dir = root / "model"
    rag_dir = root / "rag"
    rag_knowledge_bases_dir = rag_dir / "knowledge_bases"
    logs_dir = root / "logs"
    return RuntimePaths(
        user_data_dir=root,
        db_file=root / "medlive.db",
        prompts_dir=prompts_dir,
        system_prompt_template_file=prompts_dir / "system_prompt_template.md",
        soul_file=prompts_dir / "SOUL.md",
        history_compress_prompt_file=prompts_dir / "history_compress_prompt.md",
        knowledge_overview_prompt_file=prompts_dir / "knowledge_overview_prompt.md",
        history_dir=history_dir,
        context_dir=context_dir,
        session_dir=session_dir,
        voice_sessions_dir=voice_sessions_dir,
        messages_file=session_dir / "messages.jsonl",
        rag_context_file=session_dir / "rag_context.jsonl",
        session_system_prompt_file=session_dir / "session_system_prompt.md",
        runtime_state_file=session_dir / "runtime_state.json",
        model_dir=model_dir,
        model_config_file=model_dir / "config.json",
        context_model_config_file=model_dir / "context_config.json",
        rag_dir=rag_dir,
        rag_knowledge_bases_dir=rag_knowledge_bases_dir,
        logs_dir=logs_dir,
    )


def ensure_runtime_dirs(paths: RuntimePaths) -> None:
    """创建运行所需目录。"""

    for directory in (
        paths.prompts_dir,
        paths.history_dir,
        paths.context_dir,
        paths.session_dir,
        paths.voice_sessions_dir,
        paths.model_dir,
        paths.rag_knowledge_bases_dir,
        paths.logs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

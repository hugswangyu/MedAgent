"""知识库概览生成。"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from medlive.config.settings import ContextModelSettings, RagClientSettings
from medlive.context.defaults import DEFAULT_KNOWLEDGE_OVERVIEW_FALLBACK
from medlive.context.store import ContextStore

logger = logging.getLogger("agent.context.overview")


class KnowledgeOverviewGenerator:
    """使用独立 Context Model 生成每个知识库的固定概览。"""

    def __init__(self, *, store: ContextStore, settings: ContextModelSettings) -> None:
        """绑定存储和模型配置。"""

        self.store = store
        self.settings = settings

    async def generate(
        self,
        *,
        kb_id: str,
        kb_name: str,
        raw_overview: dict[str, Any] | None,
        rag_settings: RagClientSettings,
        reason: str = "index_completed",
        source_job_id: str | None = None,
    ) -> dict[str, Any]:
        """调用 Context Model 生成知识库概览；失败时写入降级概览。"""

        if not self.settings.api_key:
            fallback = self._fallback(kb_name, "缺少 Context Model API Key")
            self.store.write_knowledge_overview(
                kb_id,
                fallback,
                stale=True,
                reason="missing_context_model_api_key",
                source="fallback",
                source_job_id=source_job_id,
                raw_overview=raw_overview,
            )
            return {"generated": False, "fallback": True, "reason": "missing_context_model_api_key", "content": fallback}

        prompt = self._build_user_prompt(kb_id, kb_name, raw_overview, rag_settings)
        try:
            client = AsyncOpenAI(
                api_key=self.settings.api_key,
                base_url=self.settings.base_url,
                timeout=max(self.settings.timeout_ms, 1000) / 1000.0,
            )
            response = await client.chat.completions.create(
                model=self.settings.model,
                messages=[
                    {"role": "system", "content": self.store.read_knowledge_overview_prompt()},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
            )
            content = self._clean_model_text(response.choices[0].message.content or "")
            if not content:
                raise ValueError("empty overview")
            self.store.write_knowledge_overview(
                kb_id,
                content,
                stale=False,
                reason=reason,
                source="context_model",
                source_job_id=source_job_id,
                raw_overview=raw_overview,
            )
            return {
                "generated": True,
                "fallback": False,
                "content": content,
                "meta": self.store.read_knowledge_overview_meta(kb_id),
            }
        except Exception as exc:
            logger.warning("knowledge_overview.generate_failed", extra={"kb_id": kb_id, "error": str(exc)})
            fallback = self._fallback(kb_name, f"生成失败：{type(exc).__name__}")
            self.store.write_knowledge_overview(
                kb_id,
                fallback,
                stale=True,
                reason=f"generate_failed: {type(exc).__name__}",
                source="fallback",
                source_job_id=source_job_id,
                raw_overview=raw_overview,
            )
            return {"generated": False, "fallback": True, "reason": str(exc), "content": fallback}

    def _build_user_prompt(
        self,
        kb_id: str,
        kb_name: str,
        raw_overview: dict[str, Any] | None,
        rag_settings: RagClientSettings,
    ) -> str:
        """构造概览生成输入。"""

        rag_params = {
            "query_mode": rag_settings.query_mode,
            "top_k": rag_settings.top_k,
            "chunk_top_k": rag_settings.chunk_top_k,
            "context_max_chars": rag_settings.context_max_chars,
            "enable_rerank": rag_settings.enable_rerank,
        }
        return "\n\n".join(
            [
                f"# 知识库\nkb_id: {kb_id}\nkb_name: {kb_name}",
                f"# 当前 RAG 参数\n```json\n{json.dumps(rag_params, ensure_ascii=False, indent=2)}\n```",
                f"# LightRAG 原始概览\n```json\n{json.dumps(raw_overview or {}, ensure_ascii=False, indent=2)[:12000]}\n```",
            ]
        )

    @staticmethod
    def _clean_model_text(text: str) -> str:
        """清理模型输出中的代码块包裹。"""

        content = text.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()
        return content

    @staticmethod
    def _fallback(kb_name: str, reason: str) -> str:
        """返回不会阻断通话的降级概览。"""

        return f"{DEFAULT_KNOWLEDGE_OVERVIEW_FALLBACK.strip()}\n\n## 当前知识库\n\n- 名称：{kb_name}\n- 概览状态：{reason}\n"

"""ReAct 多步推理循环 — Thought/Action/Observation。

用法::

    engine = ReActEngine(llm_client)
    engine.register_tool("retrieve_kg", "...", executor=kg.search)
    result = engine.run("高血压和糖尿病能同时吃什么药？")
    # → {"answer": "...", "steps": [...], "tool_results": {...}}
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from .tools import ReActTool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ReAct 系统提示词
# ---------------------------------------------------------------------------

_REACT_SYSTEM_PROMPT = """你是一个医疗智能助手，需要通过多步推理来解决复杂医疗问题。

## 工作方式
你将一步步思考，选择工具获取信息，最后给出答案。

## 输出格式

选择工具时，使用以下格式（每行一个字段）：
思考：<当前推理过程>
行动：<工具名称>
输入：<严格 JSON 格式的参数>

给出最终答案时，使用以下格式：
思考：<总结推理过程>
最终答案：<完整的医疗回答>

## 工具列表

{tool_descriptions}

## 规则
1. 每次只能调用一个工具。
2. 基于 Observation 中的结果继续推理。
3. 获得足够信息后给出最终答案。
4. 最终答案应基于中文医疗知识，标注信息来源。
5. 如果不确定，明确告知用户信息有限。
6. 不得给出超出工具结果范围的诊断结论。
7. 【强制】回答任何医疗相关问题前，必须先调用 retrieve_knowledge 工具检索知识库，不得直接依赖模型自身训练知识作答。"""


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 从 LLM 输出中抽取 Action 的正则
_ACTION_RE = re.compile(
    r"思考[：:]\s*(.*?)\s*行动[：:]\s*(\w+)\s*输入[：:]\s*(\{.*?\})",
    re.DOTALL,
)
_FINAL_ANSWER_RE = re.compile(
    r"思考[：:]\s*(.*?)\s*最终答案[：:]\s*(.*?)$",
    re.DOTALL,
)

_MAX_OBSERVATION_CHARS = 2000  # 观察结果截断，防 token 爆炸


# ---------------------------------------------------------------------------
# ReActEngine
# ---------------------------------------------------------------------------


class ReActEngine:
    """ReAct 多步推理循环。

    Attributes:
        llm: OpenAI-compatible client (``chat.completions.create``).
        model: 模型名称 (default from ``_DEFAULT_MODEL``).
        max_steps: 最大推理步数。
        tools: 已注册的工具字典。
    """

    def __init__(
        self,
        llm_client: Any,
        model: str = "",
        max_steps: int = 6,
        temperature: float = 0.1,
        request_timeout: float = 30.0,
    ):
        self.llm = llm_client
        self.model = model
        self.max_steps = max_steps
        self.temperature = temperature
        self.request_timeout = request_timeout
        self._tools: Dict[str, ReActTool] = {}

    # ------------------------------------------------------------------
    # 工具注册
    # ------------------------------------------------------------------

    def register_tool(
        self,
        name: str,
        description: str,
        executor=None,
        parameters: Optional[List[Dict]] = None,
    ) -> None:
        """注册一个 ReAct 工具。"""
        params = []
        if parameters:
            for p in parameters:
                params.append(ToolParamAdapter(p))
        self._tools[name] = ReActTool(
            name=name,
            description=description,
            parameters=params,
            executor=executor,
        )

    def register_tool_from_def(self, tool: ReActTool) -> None:
        """直接注册 ReActTool 实例。"""
        self._tools[tool.name] = tool

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        system_context: str = "",
    ) -> Dict:
        """执行 ReAct 推理循环。

        Args:
            query: 用户问题。
            system_context: 额外系统上下文（记忆、历史等）。

        Returns:
            ``{"answer": str, "steps": list[dict], "tool_results": dict}``
        """
        messages = self._build_initial_messages(query, system_context)
        steps: List[Dict] = []
        tool_results: Dict[str, Any] = {}

        for step_idx in range(1, self.max_steps + 1):
            logger.debug("ReAct step %d/%d", step_idx, self.max_steps)

            # ── 调用 LLM ──
            response_text = self._call_llm(messages)
            logger.debug("LLM response:\n%s", response_text[:300])

            # ── 检查是否为最终答案 ──
            final = self._parse_final_answer(response_text)
            if final:
                answer = self._build_final_answer(
                    query, final, steps, tool_results,
                )
                return answer

            # ── 解析工具调用 ──
            action = self._parse_action(response_text)
            if action is None:
                # LLM 输出格式不对，分析原因并引导重试
                error_reason = self._analyze_parse_error(response_text)
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "system",
                    "content": self._format_retry_guidance(step_idx, error_reason),
                })
                continue

            # ── 执行工具 ──
            tool_name = action["name"]
            tool_input = action["input"]
            tool = self._tools.get(tool_name)

            if tool is None:
                observation = f"工具「{tool_name}」不存在，可用工具：{', '.join(self._tools)}"
            else:
                observation = tool.call(**tool_input)
                if len(observation) > _MAX_OBSERVATION_CHARS:
                    keep_front = 500
                    keep_back = 200
                    truncated = len(observation) - keep_front - keep_back
                    observation = (
                        observation[:keep_front]
                        + f"\n…[截断 {truncated} 字]…\n"
                        + observation[-keep_back:]
                    )

            # 记录步骤
            step_record = {
                "step": step_idx,
                "thought": action["thought"],
                "action": tool_name,
                "input": tool_input,
                "observation": observation[:500],
            }
            steps.append(step_record)
            tool_results.setdefault(tool_name, []).append(observation)

            # ── 注入 Observation ──
            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "system",
                "content": f"Observation: {observation}",
            })

        # ── 达到最大步数，强制结束 ──
        logger.info("ReAct reached max steps (%d), forcing final answer", self.max_steps)
        steps_summary = self._build_steps_summary(steps)
        messages.append({
            "role": "system",
            "content": (
                f"你已经达到最大推理步数（{self.max_steps}），"
                "请基于以下已获取的信息直接给出最终答案。\n\n"
                f"已完成的推理步骤摘要：\n{steps_summary}"
            ),
        })
        forced = self._call_llm(messages)
        answer = self._parse_final_answer(forced) or forced
        return self._build_final_answer(query, answer, steps, tool_results)

    # ------------------------------------------------------------------
    # 消息构建
    # ------------------------------------------------------------------

    def _build_initial_messages(self, query: str, system_context: str) -> List[Dict]:
        """构建初始消息列表。"""
        tool_descriptions = "\n\n".join(
            t.to_prompt_block() for t in self._tools.values()
        )
        system = _REACT_SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)
        if system_context:
            system = f"{system_context}\n\n{system}"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ]

    # ------------------------------------------------------------------
    # LLM 调用
    # ------------------------------------------------------------------

    def _call_llm(self, messages: List[Dict]) -> str:
        """调用 LLM 并返回文本响应。"""
        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=2048,
                timeout=self.request_timeout,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("ReAct LLM call failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # 解析
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_action(text: str) -> Optional[Dict]:
        """从 LLM 输出中解析工具调用。"""
        m = _ACTION_RE.search(text)
        if not m:
            return None
        thought = m.group(1).strip()
        name = m.group(2).strip()
        raw_input = m.group(3).strip()
        try:
            parsed_input = json.loads(raw_input)
        except json.JSONDecodeError:
            # 尝试修复不合规 JSON
            try:
                parsed_input = json.loads(raw_input.replace("'", '"'))
            except json.JSONDecodeError:
                return None
        if not isinstance(parsed_input, dict):
            return None
        return {"thought": thought, "name": name, "input": parsed_input}

    @staticmethod
    def _parse_final_answer(text: str) -> Optional[str]:
        """从 LLM 输出中解析最终答案。"""
        m = _FINAL_ANSWER_RE.search(text)
        if m:
            return m.group(2).strip()
        # 也支持英文格式
        m_en = re.search(
            r"Thought[：:]\s*(.*?)\s*Final Answer[：:]\s*(.*?)$",
            text, re.DOTALL,
        )
        if m_en:
            return m_en.group(2).strip()
        return None

    @staticmethod
    def _analyze_parse_error(text: str) -> str:
        """分析 LLM 输出格式错误的具体原因。"""
        if not text.strip():
            return "输出为空"
        if _FINAL_ANSWER_RE.search(text):
            return "检测到最终答案但格式不清晰，请重新组织"
        if "行动" in text or "Action" in text:
            if "输入" in text or "Input" in text:
                return "工具调用格式正确但 JSON 参数解析失败，请检查 JSON 格式"
            return "工具调用缺少「输入」字段"
        return "未检测到标准格式（思考/行动/输入 或 思考/最终答案）"

    @staticmethod
    def _format_retry_guidance(step_idx: int, error_reason: str = "") -> str:
        """格式错误时的重试引导。"""
        reason = f"错误原因：{error_reason}\n\n" if error_reason else ""
        return (
            f"第 {step_idx} 步输出格式有误。{reason}"
            "请严格按以下格式之一输出：\n\n"
            "选择工具时：\n"
            "思考：<当前推理>\n"
            "行动：<工具名>\n"
            "输入：{\"参数名\": \"值\"}\n\n"
            "给出答案时：\n"
            "思考：<总结>\n"
            "最终答案：<回答>"
        )

    @staticmethod
    def _build_final_answer(
        query: str,
        answer: str,
        steps: List[Dict],
        tool_results: Dict,
    ) -> Dict:
        """组装最终返回结构。"""
        return {
            "answer": answer,
            "steps": steps,
            "tool_results": tool_results,
        }

    @staticmethod
    def _build_steps_summary(steps: List[Dict]) -> str:
        """构建步骤摘要（用于最大步数强制结束时的提示）。"""
        if not steps:
            return "（尚未执行任何步骤）"
        lines = []
        for s in steps:
            obs = s.get("observation", "")
            obs_preview = obs[:100] + "…" if len(obs) > 100 else obs
            lines.append(
                f"  步骤 {s['step']}: 调用 {s['action']}({s['input']}) → {obs_preview}"
            )
        return "\n".join(lines)


# 适配 ReActTool 参数类型
class ToolParamAdapter:
    """将 Dict 参数定义适配为 ToolParam。"""
    def __init__(self, p: Dict):
        self.name = p.get("name", "")
        self.description = p.get("description", "")
        self.type = p.get("type", "string")

    def __repr__(self):
        return f"{self.name}: {self.type} — {self.description}"

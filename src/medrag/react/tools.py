"""ReAct 工具定义 — 包装检索器与原生工具供 LLM 调用。

每个 ``ReActTool`` 包含名称、描述、参数列表和可调用方法。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolParam:
    """工具参数定义。"""
    name: str
    description: str
    type: str = "string"


@dataclass
class ReActTool:
    """ReAct 可调用工具。

    Attributes:
        name: 工具名（供 LLM 引用）。
        description: 用途说明（生成 LLM 提示词）。
        parameters: 参数列表。
        executor: 实际执行函数，接收 **(param_name → value)**。
    """
    name: str
    description: str
    parameters: List[ToolParam] = field(default_factory=list)
    executor: Optional[Callable] = None

    def to_prompt_block(self) -> str:
        """格式化为 LLM 提示词中的工具描述块。"""
        params_str = ", ".join(
            f"{p.name}: {p.type} — {p.description}" for p in self.parameters
        ) if self.parameters else "无参数"
        return f"{self.name}({params_str})\n   {self.description}"

    def call(self, **kwargs) -> str:
        """执行工具，返回字符串结果。"""
        if self.executor is None:
            return f"工具「{self.name}」不可用"
        try:
            result = self.executor(**kwargs)
            if result is None:
                return "未找到相关信息"
            return str(result)
        except Exception as exc:
            return f"工具执行出错：{exc}"


def base_tool_to_react_tool(base_tool) -> ReActTool:
    """将 ``BaseTool`` 子类适配为 ``ReActTool``。

    BaseTool（剂量计算/科室导诊/正常值查询）的 ``match/execute``
    接口与 ReAct 不同（ReAct 不需要 match，LLM 自主决定调用）。
    此适配器提取参数 schema 和执行器供 ReAct 引擎使用。
    """
    from medrag.tools.dosage_calculator import DosageCalculator
    from medrag.tools.department_guide import DepartmentGuide
    from medrag.tools.normal_range import NormalRangeTool

    params: list[ToolParam] = []

    if isinstance(base_tool, DosageCalculator):
        params = [
            ToolParam("drug", "药品名称（中文，如阿莫西林）", "string"),
            ToolParam("age", "患者年龄（岁，可选，传则按儿童剂量计算）", "number"),
            ToolParam("weight", "患者体重（kg，可选）", "number"),
        ]
    elif isinstance(base_tool, DepartmentGuide):
        params = [
            ToolParam("symptom", "症状或疾病名称", "string"),
        ]
    elif isinstance(base_tool, NormalRangeTool):
        params = [
            ToolParam("test", "检查项目名称（中文），如血红蛋白", "string"),
            ToolParam("value", "检查值（可选，传入则判断是否正常）", "string"),
        ]

    return ReActTool(
        name=base_tool.name,
        description=base_tool.description,
        parameters=params,
        executor=base_tool.execute,
    )

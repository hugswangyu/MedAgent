"""LLM-based medical case summarisation.

Generates a structured summary from de-identified case text using any
OpenAI-compatible LLM client (DeepSeek, OpenAI, etc.).
"""

from __future__ import annotations

from medrag.config.settings import settings

_SUMMARY_SYSTEM = """你是一位经验丰富的医疗记录整理专家。你的任务是根据用户提供的病例文本，提取关键信息并整理成结构化摘要。

**重要规则：**
1. 只提取病例中**明确存在**的信息，绝不凭空推断或补充。
2. 如果某个字段在病例中找不到对应信息，直接填写"未提供"。
3. 不要在摘要中做任何新的医学诊断或推断。
4. 保留原文中关键的数值、日期、药名、剂量等细节。
5. 输出的每个字段控制在 3-5 句话以内，简洁明了。"""


def build_case_summary_prompt(case_text: str) -> str:
    """Build the summarisation prompt wrapping *case_text*."""
    return f"""请阅读以下病例文本，并生成一个结构化的病例摘要。

<病例文本>
{case_text}
</病例文本>

请严格按以下格式输出，每个字段必须单独一行：

主诉：
现病史：
既往史：
检查/检验结果：
初步诊断：
当前用药：
医生建议：
异常指标：
需要关注的问题：

如果某个字段信息不足，请填写"未提供"（不要省略任何字段）。"""


def summarize_case(case_text: str, llm_client) -> str:
    """Run the case summarisation prompt against *llm_client*.

    *llm_client* must be an OpenAI-compatible client (``chat.completions.create``).
    """
    prompt = build_case_summary_prompt(case_text)
    response = llm_client.chat.completions.create(
        model=settings.deepseek_default_model,
        messages=[
            {"role": "system", "content": _SUMMARY_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content

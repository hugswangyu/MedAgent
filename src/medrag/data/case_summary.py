"""基于 LLM 的医疗病例摘要生成与病例文件处理流水线。

对脱敏后的病例文本，通过任意兼容 OpenAI 的 LLM 客户端（DeepSeek、OpenAI 等）生成结构化摘要。
"""

from __future__ import annotations

from pathlib import Path

from medrag.config.settings import settings
from medrag.llm import get_llm_client

_SUMMARY_SYSTEM = """你是一位经验丰富的医疗记录整理专家。你的任务是根据用户提供的病例文本，提取关键信息并整理成结构化摘要。

**重要规则：**
1. 只提取病例中**明确存在**的信息，绝不凭空推断或补充。
2. 如果某个字段在病例中找不到对应信息，直接填写"未提供"。
3. 不要在摘要中做任何新的医学诊断或推断。
4. 保留原文中关键的数值、日期、药名、剂量等细节。
5. 输出的每个字段控制在 3-5 句话以内，简洁明了。"""


def build_case_summary_prompt(case_text: str) -> str:
    """构建包装了 *case_text* 的摘要提示词。"""
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
    """通过 *llm_client* 对病例文本执行摘要提示词。

    *llm_client* 必须是兼容 OpenAI 的客户端（``chat.completions.create``）。
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


def process_case_file(uploaded_file, usname: str, llm_client=None) -> str:
    """端到端病例文件流水线：保存 → 解析 → 清洗 → 脱敏 → 生成摘要。

    *uploaded_file* 是 Streamlit ``UploadedFile`` 或任何具有
    ``.name`` 和 ``.getbuffer()`` 的对象。

    返回结构化病例摘要字符串。
    """
    from medrag.data.case_parser import parse_case_file as _parse
    from medrag.data.text_cleaner import clean_medical_text, desensitize_medical_text

    if llm_client is None:
        llm_client = get_llm_client("deepseek")

    user_dir = Path("user_uploads") / usname
    user_dir.mkdir(parents=True, exist_ok=True)

    dest = user_dir / uploaded_file.name
    with open(dest, "wb") as fh:
        fh.write(uploaded_file.getbuffer())

    raw = _parse(str(dest))
    cleaned = clean_medical_text(raw)
    safe = desensitize_medical_text(cleaned)
    return summarize_case(safe, llm_client)

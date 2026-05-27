"""医疗文本清洗与脱敏工具。"""

from __future__ import annotations

import re

def desensitize_medical_text(text: str) -> str:
    """替换 PII（姓名、身份证、手机号、地址等）为占位符。

    注意：基于正则表达式，并非完整的脱敏方案。涵盖中文病历中常见的模式：
      - 姓名（2–4 字中文姓名，尽力匹配）
      - 身份证号（18 位）
      - 手机号（11 位）
      - 固定电话
      - 就诊卡号 / 住院号（带标识的数字串）
      - 住址（结构化地址字符串）
    """
    # 去除首尾空白
    text = text.strip()

    # 姓名 — 替换 "姓名：XXX" 或 "患者：XXX" 模式
    text = re.sub(
        r"(姓名|患者|病人|联系人)[：:\s]*[一-龥]{2,4}(?![一-龥])",
        r"\1：【姓名***】",
        text,
    )
    # 文档开头的独立姓名（单独一行 2-4 个中文字符）
    text = re.sub(
        r"^([一-龥]{2,4})$",
        "【姓名***】",
        text,
        flags=re.MULTILINE,
    )

    # 带标签的 ID 和地址
    text = re.sub(
        r"(就诊卡号|住院号|病历号|病案号|门诊号)[：:\s]*[A-Za-z0-9]{4,30}",
        r"\1：【\1***】",
        text,
    )
    text = re.sub(
        r"(?:地址|住址|现住址|户籍地|居住地)[：:\s]*"
        r"[一-龥]{2,3}(?:省|自治区|特别行政区)"
        r"[一-龥]{2,}(?:市|地区|自治州|盟)"
        r".*(?:\d+号|\d+栋|\d+室|\d+楼|\d+单元|小区|街道|路|村|乡|镇)",
        "【住址***】",
        text,
    )

    # 裸 PII 模式
    text = re.sub(r"\b\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b",
                  "【身份证号***】", text)
    text = re.sub(r"\b1[3-9]\d{9}\b", "【手机号***】", text)
    text = re.sub(r"\b0\d{2,3}-\d{7,8}\b", "【电话***】", text)

    return text


def clean_medical_text(text: str) -> str:
    """格式化医疗文本：统一空白字符，去除明显噪声。

    不会删除医疗内容 —— 仅修复格式问题：
      - 合并重复的空行
      - 替换全角空格为半角
      - 去除首尾空白
    """
    text = text.strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("　", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

"""Medical text cleaning and desensitization utilities."""

from __future__ import annotations

import re

def desensitize_medical_text(text: str) -> str:
    """Replace PII (name, ID, phone, address, etc.) with placeholder tokens.

    NOTE: Regex-based, not a complete solution.  Covers common patterns
    found in Chinese medical records:
      - 姓名 (2–4 character Chinese names; best-effort)
      - 身份证号 (18-digit)
      - 手机号 (11-digit)
      - 固定电话
      - 就诊卡号 / 住院号 (labelled digit strings)
      - 住址 (structured address strings)
    """
    # Strip leading/trailing whitespace first
    text = text.strip()

    # 姓名 — replace "姓名：XXX" or "患者：XXX" patterns
    text = re.sub(
        r"(姓名|患者|病人|联系人)[：:\s]*[一-龥]{2,4}(?![一-龥])",
        r"\1：【姓名***】",
        text,
    )
    # Isolated name at document head (2-4 Chinese chars on their own line)
    text = re.sub(
        r"^([一-龥]{2,4})$",
        "【姓名***】",
        text,
        flags=re.MULTILINE,
    )

    # Labelled IDs and addresses
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

    # Bare PII patterns
    text = re.sub(r"\b\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b",
                  "【身份证号***】", text)
    text = re.sub(r"\b1[3-9]\d{9}\b", "【手机号***】", text)
    text = re.sub(r"\b0\d{2,3}-\d{7,8}\b", "【电话***】", text)

    return text


def clean_medical_text(text: str) -> str:
    """Normalise whitespace and remove obvious noise from medical text.

    Does NOT remove medical content — only fixes formatting:
      - Collapse repeated blank lines
      - Replace full-width spaces with half-width
      - Strip leading/trailing whitespace
    """
    text = text.strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("　", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

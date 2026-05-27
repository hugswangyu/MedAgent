"""病例文件解析器 —— 支持 txt、pdf、docx。"""

from __future__ import annotations

from pathlib import Path


def _parse_txt(path: Path) -> str:
    """读取 txt 文件，utf-8 优先，回退到 gbk。"""
    for encoding in ("utf-8", "gbk"):
        try:
            return path.read_text(encoding=encoding).strip()
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Cannot decode {path} with utf-8 or gbk")


def _parse_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def _parse_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs).strip()


_PARSERS = {
    ".txt": _parse_txt,
    ".pdf": _parse_pdf,
    ".docx": _parse_docx,
}


def parse_case_file(file_path: str) -> str:
    """解析病例文件（txt / pdf / docx），返回文本内容。

    不支持的扩展名将抛出 ValueError。
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix not in _PARSERS:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: {list(_PARSERS)}")
    return _PARSERS[suffix](path)

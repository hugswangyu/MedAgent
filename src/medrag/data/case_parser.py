"""Case file parser — supports txt, pdf, docx."""

from __future__ import annotations

from pathlib import Path


def _parse_txt(path: Path) -> str:
    """Read txt with utf-8, fall back to gbk."""
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
    """Parse a case file (txt / pdf / docx) and return its text content.

    Raises ValueError for unsupported extensions.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix not in _PARSERS:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: {list(_PARSERS)}")
    return _PARSERS[suffix](path)

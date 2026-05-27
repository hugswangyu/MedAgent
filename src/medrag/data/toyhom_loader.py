"""Loader for the Toyhom Chinese medical dialogue CSV dataset."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")

FIELD_ALIASES = {
    "department": (
        "department",
        "dept",
        "section",
        "category",
        "科室",
        "科别",
        "科目",
    ),
    "title": (
        "title",
        "subject",
        "topic",
        "标题",
        "题目",
    ),
    "question": (
        "question",
        "ask",
        "query",
        "content",
        "问题",
        "提问",
        "咨询",
        "患者问题",
    ),
    "answer": (
        "answer",
        "reply",
        "response",
        "doctor_answer",
        "答案",
        "回答",
        "回复",
        "医生回复",
    ),
}

WHITESPACE_RE = re.compile(r"\s+")


def _normalize_column_name(name: object) -> str:
    return re.sub(r"[\s_\-:：]+", "", str(name or "").strip().lower())


def _clean_text(value: object) -> str:
    text = "" if value is None else str(value)
    return WHITESPACE_RE.sub(" ", text).strip()


def _department_from_path(data_root: Path, csv_path: Path) -> str:
    try:
        relative_parts = csv_path.relative_to(data_root).parts
    except (IndexError, ValueError):
        return ""

    folder_name = relative_parts[0] if len(relative_parts) > 1 else data_root.name
    if "_" in folder_name:
        folder_name = folder_name.rsplit("_", 1)[-1]
    return _clean_text(folder_name)


def _build_text(record: Dict[str, str]) -> str:
    return "\n".join(
        [
            f"科室：{record['department']}",
            f"标题：{record['title']}",
            f"问题：{record['question']}",
            f"回答：{record['answer']}",
        ]
    )


def _resolve_columns(fieldnames: Optional[Iterable[str]]) -> Dict[str, Optional[str]]:
    names = list(fieldnames or [])
    normalized_to_original = {_normalize_column_name(name): name for name in names}
    resolved: Dict[str, Optional[str]] = {}

    for target, aliases in FIELD_ALIASES.items():
        resolved[target] = None
        for alias in aliases:
            original = normalized_to_original.get(_normalize_column_name(alias))
            if original is not None:
                resolved[target] = original
                break

    if len(names) >= 4:
        resolved["department"] = resolved["department"] or names[0]
        resolved["title"] = resolved["title"] or names[1]
        resolved["question"] = resolved["question"] or names[2]
        resolved["answer"] = resolved["answer"] or names[3]

    return resolved


def _read_sample(csv_path: Path, encoding: str, errors: str) -> str:
    with csv_path.open("r", encoding=encoding, errors=errors, newline="") as file:
        return file.read(4096)


def _encoding_candidates(csv_path: Path) -> Iterable[tuple[str, str]]:
    for encoding in ENCODINGS:
        try:
            _read_sample(csv_path, encoding, "strict")
            yield encoding, "strict"
        except UnicodeDecodeError:
            continue
    yield "gb18030", "replace"


def _iter_csv_rows(csv_path: Path) -> Iterable[Dict[str, str]]:
    last_error: Optional[Exception] = None

    for encoding, errors in _encoding_candidates(csv_path):
        try:
            with csv_path.open("r", encoding=encoding, errors=errors, newline="") as file:
                sample = file.read(4096)
                file.seek(0)
                dialect = csv.Sniffer().sniff(sample) if sample.strip() else csv.excel
                reader = csv.DictReader(file, dialect=dialect)
                columns = _resolve_columns(reader.fieldnames)
                if not columns["question"] or not columns["answer"]:
                    raise ValueError(f"missing question/answer columns: {reader.fieldnames}")

                for row in reader:
                    yield {
                        "department": _clean_text(row.get(columns["department"] or "")),
                        "title": _clean_text(row.get(columns["title"] or "")),
                        "question": _clean_text(row.get(columns["question"] or "")),
                        "answer": _clean_text(row.get(columns["answer"] or "")),
                    }
                return
        except Exception as exc:
            last_error = exc

    print(f"warning: skip bad csv file {csv_path}: {last_error}")


def load_toyhom_dataset(data_root, limit=None) -> List[Dict]:
    """Load Toyhom CSV files recursively into a unified list of dictionaries."""
    root = Path(data_root).expanduser()
    if not root.exists():
        print(f"warning: Toyhom data root does not exist: {root}")
        return []

    records: List[Dict] = []
    seen = set()
    max_count = None if limit is None else int(limit)

    for csv_path in sorted(root.rglob("*.csv")):
        path_department = _department_from_path(root, csv_path)
        for row in _iter_csv_rows(csv_path):
            department = path_department or row["department"]
            question = row["question"]
            answer = row["answer"]
            if len(question) < 2 or len(answer) < 5:
                continue

            dedupe_key = (department, row["title"], question, answer)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            record = {
                "id": f"toyhom-{len(records) + 1}",
                "department": department,
                "title": row["title"],
                "question": question,
                "answer": answer,
            }
            record["text"] = _build_text(record)
            records.append(record)

            if max_count is not None and len(records) >= max_count:
                return records

    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and preview Toyhom medical CSV data.")
    parser.add_argument(
        "data_root",
        nargs="?",
        default="Chinese-medical-dialogue-data/Data_数据",
        help="Toyhom dataset root directory.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum records to load.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    records = load_toyhom_dataset(args.data_root, limit=args.limit)
    print(f"total: {len(records)}")
    for record in records[:3]:
        print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

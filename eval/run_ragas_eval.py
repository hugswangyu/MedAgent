"""端到端 RAG/RAGAS 评估脚本。

默认会调用真实 MedicalChatService。若未安装 ragas，脚本仍会输出路由、
上下文关键词、安全提醒等自定义指标，便于先建立 baseline。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


def _load_jsonl(path: Path) -> list[dict]:
    cases: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def _contexts_from_result(result: dict) -> tuple[list[str], list[str]]:
    """返回 (ragas_contexts, kg_triplets)。

    ragas_contexts: KG 属性文本 + QA + Case，送 RAGAS 语义评测。
    kg_triplets:    KG 关系三元组文本，用关键词精确匹配评测。
    """
    ragas_contexts: list[str] = []
    kg_triplets: list[str] = []

    for item in result.get("kg_results") or []:
        evidence = item.get("evidence")
        answer = item.get("answer", "")
        if isinstance(evidence, list):
            # 关系三元组：evidence 是实体名列表
            kg_triplets.append(answer or "、".join(str(x) for x in evidence))
        else:
            # 属性文本：evidence 是字符串段落
            text = answer or str(evidence or "")
            if text:
                ragas_contexts.append(text)

    for key in ("qa_results", "case_results", "toyhom_results"):
        for item in result.get(key) or []:
            text = item.get("answer") or item.get("text") or item.get("evidence") or ""
            if isinstance(text, list):
                text = "、".join(str(x) for x in text)
            if text:
                ragas_contexts.append(str(text))

    return ragas_contexts, kg_triplets


def _run_pipeline(cases: list[dict], username: str) -> list[dict]:
    from medrag.service.chat_service import MedicalChatService

    service = MedicalChatService()
    rows: list[dict] = []
    for case in cases:
        result = service.chat_with_harness(
            case["question"],
            user_case_summary=case.get("case_context"),
            username=username,
        )
        ragas_contexts, kg_triplets = _contexts_from_result(result)
        rows.append(
            {
                "id": case.get("id", ""),
                "question": case["question"],
                "answer": result.get("answer", ""),
                "ground_truth": case.get("ground_truth", ""),
                "contexts": ragas_contexts,
                "kg_triplets": kg_triplets,
                "route": result.get("route", {}),
                "expected_route": case.get("expected_route", ""),
                "expected_context_keywords": case.get("expected_context_keywords", []),
                "risk_info": result.get("risk_info", {}),
                # 全链路 trace（供 bad case 分析用）
                "react_trace": result.get("react_trace") or {},
                "harness_trace": result.get("harness_trace") or {},
                "eval_ragas": case.get("eval_ragas", True),
            }
        )
    return rows


def _custom_metrics(rows: list[dict]) -> dict:
    route_hits = []
    safety_hits = []
    kg_triplet_hits = []

    for row in rows:
        expected_route = row.get("expected_route")
        if expected_route:
            route_hits.append(row.get("route", {}).get("query_type") == expected_route)

        if str(row.get("id", "")).startswith("safety_"):
            answer = row.get("answer") or ""
            risk = row.get("risk_info") or {}
            safety_hits.append(
                bool(risk.get("is_high_risk") or risk.get("is_moderate_risk"))
                or ("急诊" in answer or "120" in answer or "尽快就医" in answer)
            )

        keywords = row.get("expected_context_keywords") or []
        triplets = row.get("kg_triplets") or []
        if keywords and triplets:
            triplet_text = " ".join(triplets)
            hit = sum(1 for kw in keywords if kw in triplet_text) / len(keywords)
            kg_triplet_hits.append(hit)

    return {
        "route_accuracy": mean(route_hits) if route_hits else None,
        "safety_hit_rate": mean(safety_hits) if safety_hits else None,
        "kg_triplet_hit_rate": round(mean(kg_triplet_hits), 4) if kg_triplet_hits else None,
    }


def _try_ragas(rows: list[dict]) -> dict | None:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except Exception as exc:
        return {"skipped": True, "reason": f"ragas unavailable: {exc}"}

    dataset = Dataset.from_list(
        [
            {
                "question": row["question"],
                "answer": row["answer"],
                "contexts": row["contexts"] or [""],
                "ground_truth": row["ground_truth"],
            }
            for row in rows
        ]
    )
    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        if hasattr(result, "to_pandas"):
            return {"summary": result.to_pandas().mean(numeric_only=True).to_dict()}
        return {"summary": dict(result)}
    except Exception as exc:
        return {"skipped": True, "reason": f"ragas evaluation failed: {exc}"}


def _keyword_hit_rate(keywords: list[str], contexts: list[str], answer: str) -> float:
    """关键词命中率：检查 keywords 是否出现在 contexts 或 answer 中。"""
    if not keywords:
        return 1.0
    text = " ".join(contexts) + " " + answer
    hits = sum(1 for kw in keywords if kw in text)
    return hits / len(keywords)


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MedRAG end-to-end RAGAS evaluation.")
    parser.add_argument("--golden", default="eval/golden_cases.jsonl")
    parser.add_argument("--out", default="eval/reports/ragas_report.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--username", default="eval_user")
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--ragas-only", action="store_true",
                        help="只跑 eval_ragas=true 的案例")
    args = parser.parse_args()

    cases = _load_jsonl(Path(args.golden))
    if args.ragas_only:
        before = len(cases)
        cases = [c for c in cases if c.get("eval_ragas", True)]
        print(f"--ragas-only: {before} → {len(cases)} 条")
    if args.limit:
        cases = cases[: args.limit]

    rows = _run_pipeline(cases, username=args.username)
    custom = _custom_metrics(rows)
    ragas_result = {"skipped": True, "reason": "use run_ragas_claude.py for RAGAS scoring"}

    report = {
        "case_count": len(rows),
        "custom_metrics": custom,
        "ragas": ragas_result,
        "rows": rows,
    }
    _write_report(Path(args.out), report)

    print("MedRAG evaluation complete")
    print(json.dumps({"case_count": len(rows), "custom_metrics": custom, "ragas": ragas_result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

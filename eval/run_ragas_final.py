"""直接调用 RAGAS 评测，使用 Qwen 作为 judge LLM + 本地 embedding。

用法: uv run python eval/run_ragas_final.py [--report path/to/report.json]
      QWEN_JUDGE_MODELS=qwen-max,qwen-turbo uv run python eval/run_ragas_final.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)


class _FallbackChatOpenAI(ChatOpenAI):
    """ChatOpenAI 子类，在主模型失败时依序尝试 fallback_models。

    继承自 ChatOpenAI 以保留 temperature 等 Pydantic 字段，避免 RAGAS
    内部访问 .temperature 时报 ValueError。
    """
    fallback_models: list = []

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        try:
            return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except Exception as primary_exc:
            for model in self.fallback_models:
                try:
                    fb = ChatOpenAI(
                        model=model,
                        api_key=self.openai_api_key,
                        base_url=str(self.openai_api_base) if self.openai_api_base else None,
                    )
                    return await fb._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
                except Exception:
                    continue
            raise primary_exc

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        try:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except Exception as primary_exc:
            for model in self.fallback_models:
                try:
                    fb = ChatOpenAI(
                        model=model,
                        api_key=self.openai_api_key,
                        base_url=str(self.openai_api_base) if self.openai_api_base else None,
                    )
                    return fb._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                except Exception:
                    continue
            raise primary_exc


# 配置 Qwen 作为 judge LLM，按顺序 fallback
_api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY", "")
_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_model_priority = [
    m.strip() for m in os.environ.get(
        "QWEN_JUDGE_MODELS", "qwen-turbo,qwen-max"
    ).split(",")
]

llm = LangchainLLMWrapper(
    _FallbackChatOpenAI(
        model=_model_priority[0],
        api_key=_api_key,
        base_url=_base_url,
        fallback_models=_model_priority[1:],
    )
)

# 本地 embedding 模型（answer_relevancy 需要）
emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5"))

for m in [faithfulness, answer_relevancy, context_precision, context_recall]:
    m.llm = llm
    m.reproducibility = 1  # Qwen 不支持 n > 1
answer_relevancy.embeddings = emb
answer_relevancy.strictness = 1

_parser = argparse.ArgumentParser()
_parser.add_argument("--report", default=str(_root / "eval" / "reports" / "ragas_report.json"))
_args = _parser.parse_args()

# 从已生成的报告中读取 answers 和 contexts
report_path = Path(_args.report)
if report_path.exists():
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report["rows"]
else:
    print("Error: run_ragas_eval.py must be run first to generate answers")
    sys.exit(1)

print(f"Loaded {len(rows)} rows from {report_path}")

# 构造 RAGAS 数据集
dataset = Dataset.from_list([
    {
        "question": row["question"],
        "answer": row["answer"],
        "contexts": row["contexts"] or [""],
        "ground_truth": row["ground_truth"],
    }
    for row in rows
])

# 运行 RAGAS 评估
print(f"Running RAGAS evaluation with models: {_model_priority}")
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    run_config=RunConfig(max_workers=4, timeout=60),
)

import math

if hasattr(result, "to_pandas"):
    df = result.to_pandas()
    scores = df.mean(numeric_only=True).to_dict()
    per_row = df.to_dict(orient="records")
else:
    scores = dict(result)
    per_row = []

# 找出有 NaN 的行
metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
skipped_rows = []
for i, row_score in enumerate(per_row):
    nan_metrics = [c for c in metric_cols if c in row_score and (row_score[c] != row_score[c])]
    if nan_metrics:
        skipped_rows.append({
            "index": i,
            "id": rows[i].get("id", ""),
            "question": rows[i].get("question", ""),
            "nan_metrics": nan_metrics,
            **{c: row_score.get(c) for c in metric_cols},
        })

# 输出汇总
print("\n" + "=" * 50)
print(f"RAGAS Evaluation Results ({len(rows)} cases)")
print("=" * 50)
label_map = {
    "faithfulness": "Faithfulness（忠实度）",
    "answer_relevancy": "Answer Relevancy（答案相关性）",
    "context_precision": "Context Precision（上下文精确度）",
    "context_recall": "Context Recall（上下文召回率）",
}
for k, v in scores.items():
    label = label_map.get(k, k)
    val_str = f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else str(v)
    print(f"  {label}: {val_str}")

if skipped_rows:
    print(f"\n⚠  {len(skipped_rows)} 条有 NaN（API 超时或跳过）：")
    for r in skipped_rows:
        print(f"  [{r['index']}] {r['id']} — NaN: {r['nan_metrics']} — {r['question'][:40]}")

# 保存到报告
report["ragas"] = {
    "summary": {k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in scores.items()},
    "per_row": per_row,
    "skipped": skipped_rows,
}
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nReport updated: {report_path}")

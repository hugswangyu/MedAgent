"""RAGAS 评测 — 使用 Qwen 作为 judge LLM + 本地 embedding。

用法（key 已在 .env）:
  set -a && source .env && set +a
  uv run python eval/run_ragas_claude.py

可选环境变量:
  QWEN_API_KEY   Qwen API Key（读自 .env）
  QWEN_BASE_URL  API 地址，默认 https://dashscope.aliyuncs.com/compatible-mode/v1
  QWEN_MODEL     judge 模型，默认 qwen-plus
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

# 读取 .env
from dotenv import load_dotenv
load_dotenv(_root / ".env")

from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._answer_relevance import AnswerRelevancy
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

_API_KEY  = os.environ.get("QWEN_API_KEY", "")
_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
_MODEL    = os.environ.get("QWEN_MODEL", "qwen-plus")

if not _API_KEY:
    print("Error: QWEN_API_KEY 未设置，请检查 .env 文件")
    sys.exit(1)

print(f"Judge LLM : {_MODEL}")
print(f"Base URL  : {_BASE_URL}")

# Qwen 使用 DashScope OpenAI 兼容接口
qwen = ChatOpenAI(
    model=_MODEL,
    api_key=_API_KEY,
    base_url=_BASE_URL,
    temperature=0,
)
llm = LangchainLLMWrapper(qwen)
emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5"))

# 实例化指标并注入 LLM
faithfulness      = Faithfulness()
answer_relevancy  = AnswerRelevancy()
context_precision = ContextPrecision()
context_recall    = ContextRecall()

for m in [faithfulness, answer_relevancy, context_precision, context_recall]:
    m.llm = llm
    m.reproducibility = 1
answer_relevancy.embeddings = emb
answer_relevancy.strictness = 1

# 加载报告
report_path = _root / "eval" / "reports" / "ragas_report.json"
if not report_path.exists():
    print("Error: 请先运行 run_ragas_eval.py 生成 ragas_report.json")
    sys.exit(1)

report = json.loads(report_path.read_text(encoding="utf-8"))
rows = report["rows"]
print(f"Loaded {len(rows)} cases from report")

dataset = Dataset.from_list([
    {
        "question": row["question"],
        "answer": row["answer"],
        "contexts": row["contexts"] or [""],
        "ground_truth": row["ground_truth"],
    }
    for row in rows
])

print("Running RAGAS evaluation with Qwen judge...")
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)

if hasattr(result, "to_pandas"):
    scores = result.to_pandas().mean(numeric_only=True).to_dict()
else:
    scores = dict(result)

label_map = {
    "faithfulness":       "Faithfulness    （忠实度）",
    "answer_relevancy":   "Answer Relevancy（答案相关性）",
    "context_precision":  "Context Precision（上下文精确度）",
    "context_recall":     "Context Recall  （上下文召回率）",
}

print("\n" + "=" * 55)
print(f"RAGAS Evaluation Results — {_MODEL}")
print("=" * 55)
for k, v in scores.items():
    label = label_map.get(k, k)
    val_str = f"{v:.4f}" if isinstance(v, float) and v == v else str(v)
    print(f"  {label}: {val_str}")

report["qwen_ragas"] = {
    "model": _MODEL,
    "summary": {k: v for k, v in scores.items() if isinstance(v, float) and v == v},
}
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nReport updated: {report_path}")

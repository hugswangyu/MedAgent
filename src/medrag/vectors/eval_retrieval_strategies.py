"""Evaluate and compare Toyhom retrieval strategies.

Compares four indexing strategies on Precision@k and Recall@k:
  - title_only        clean medical question (proposed)
  - question_only     patient's verbose description
  - full_text         department + title + question + answer (baseline)
  - title_question    title + question combined

Methodology:
  1. Sample a diverse corpus from the Toyhom dataset (stratified by department).
  2. Define a set of test queries with manually labelled disease topics.
  3. For each query, ground-truth = records in the corpus whose title
     contains the same disease term (e.g., all "早泄" records for a
     query about "早泄").
  4. Compute Precision@k, Recall@k, and MRR for each strategy.

Usage:
    python vector_store/eval_retrieval_strategies.py \
        [--samples_per_dept 800] [--k 1 3 5 10]
"""

from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from medrag.data.toyhom_loader import load_toyhom_dataset

BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

# ── strategies ──────────────────────────────────────────────────────────
STRATEGIES: Dict[str, Callable[[dict], str]] = {
    "title_only":      lambda r: r["title"],
    "question_only":   lambda r: r["question"],
    "full_text":       lambda r: r["text"],
    "title_question":  lambda r: f"{r['title']} {r['question']}",
}

# ── manual test queries ─────────────────────────────────────────────────
# (query_text, disease_keyword_to_match_in_corpus_title)
TEST_QUERIES: List[Tuple[str, str]] = [
    # 男科 (4)
    ("早泄是什么原因",                    "早泄"),
    ("阳痿怎么治疗",                      "阳痿"),
    ("前列腺炎的症状",                    "前列腺炎"),
    ("精索静脉曲张怎么办",               "精索静脉曲张"),
    # 内科 (4)
    ("高血压吃什么药",                    "高血压"),
    ("糖尿病饮食注意什么",               "糖尿病"),
    ("感冒发烧怎么办",                    "感冒"),
    ("胃炎怎么调理",                      "胃炎"),
    # 妇产科 (4)
    ("阴道炎怎么治",                      "阴道炎"),
    ("月经不调是什么原因",               "月经不调"),
    ("盆腔炎的症状",                      "盆腔炎"),
    ("痛经怎么缓解",                      "痛经"),
    # 外科 (3)
    ("痔疮怎么治疗",                      "痔疮"),
    ("骨折后吃什么恢复快",               "骨折"),
    ("阑尾炎必须手术吗",                  "阑尾炎"),
    # 儿科 (3)
    ("小儿发烧怎么退烧",                  "发烧"),
    ("小儿腹泻怎么办",                    "腹泻"),
    ("手足口病有什么症状",               "手足口病"),
    # 肿瘤科 (2)
    ("肺癌早期症状",                      "肺癌"),
    ("胃癌怎么预防",                      "胃癌"),
]


def build_ground_truth(
    corpus_records: List[dict],
    test_queries: List[Tuple[str, str]],
) -> Dict[int, Set[int]]:
    """For each query, find all corpus records whose title contains
    the disease keyword (simple substring match)."""
    gt: Dict[int, Set[int]] = {}
    for q_idx, (_, disease) in enumerate(test_queries):
        relevant: Set[int] = set()
        for c_idx, rec in enumerate(corpus_records):
            if disease in rec["title"]:
                relevant.add(c_idx)
        gt[q_idx] = relevant
    return gt


# ── metrics ─────────────────────────────────────────────────────────────

def compute_metrics(
    query_emb: np.ndarray,
    corpus_embs: np.ndarray,
    relevant: Set[int],
    k_values: List[int],
) -> Dict[str, float]:
    """Precision@k, Recall@k, MRR for a single query."""
    sims = cosine_similarity(query_emb.reshape(1, -1), corpus_embs)[0]
    ranked = np.argsort(sims)[::-1]
    n_rel = len(relevant)

    results: Dict[str, float] = {}
    for k in k_values:
        top_k = set(ranked[:k])
        hits = len(top_k & relevant)
        results[f"P@{k}"] = hits / k
        results[f"R@{k}"] = hits / n_rel if n_rel > 0 else 0.0

    for rank_pos, idx in enumerate(ranked[:max(k_values)], 1):
        if idx in relevant:
            results["MRR"] = 1.0 / rank_pos
            break
    else:
        results["MRR"] = 0.0

    return results


def evaluate_strategy(
    name: str,
    strategy_fn: Callable[[dict], str],
    corpus_records: List[dict],
    corpus_embs: np.ndarray,
    test_queries: List[Tuple[str, str]],
    gt: Dict[int, Set[int]],
    k_values: List[int],
) -> Dict[str, float]:
    """Evaluate one strategy against pre-computed corpus embeddings."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    agg: Dict[str, list] = {f"P@{k}": [] for k in k_values}
    agg.update({f"R@{k}": [] for k in k_values})
    agg["MRR"] = []

    valid = 0
    for q_idx, (query_text, disease) in enumerate(test_queries):
        relevant = gt.get(q_idx, set())
        if not relevant:
            continue
        valid += 1
        query_emb = model.encode(
            BGE_QUERY_PREFIX + query_text,
            normalize_embeddings=True,
        )
        m = compute_metrics(query_emb, corpus_embs, relevant, k_values)
        for k in k_values:
            agg[f"P@{k}"].append(m[f"P@{k}"])
            agg[f"R@{k}"].append(m[f"R@{k}"])
        agg["MRR"].append(m["MRR"])

    return {
        **{key: float(np.mean(vals)) if vals else 0.0 for key, vals in agg.items()},
        "valid_queries": valid,
    }


def _stratified_sample(records: List[dict], per_dept: int, seed: int = 42) -> List[dict]:
    dept_buckets: Dict[str, list] = defaultdict(list)
    for r in records:
        dept_buckets[r["department"]].append(r)
    rng = np.random.default_rng(seed)
    sampled: List[dict] = []
    for dept, bucket in sorted(dept_buckets.items()):
        take = min(per_dept, len(bucket))
        chosen = [bucket[i] for i in rng.choice(len(bucket), take, replace=False)]
        sampled.extend(chosen)
        print(f"  {dept}: {take}/{len(bucket)}")
    return sampled


# ── main ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Toyhom retrieval strategies."
    )
    parser.add_argument("--data_root", default="Chinese-medical-dialogue-data/Data_数据")
    parser.add_argument("--samples_per_dept", type=int, default=1000)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── load & sample ──────────────────────────────────────────────────
    load_limit = 700_000  # enough to cover all 6 departments (~665k total)
    print(f"Loading up to {load_limit} records …")
    records = load_toyhom_dataset(args.data_root, limit=load_limit)
    depts = sorted({r["department"] for r in records})
    print(f"Loaded {len(records)} records across {len(depts)} departments: {depts}")

    print(f"\nStratified sampling (≤ {args.samples_per_dept} per dept) …")
    corpus = _stratified_sample(records, args.samples_per_dept, args.seed)
    print(f"Corpus: {len(corpus)} records\n")

    # ── ground truth ───────────────────────────────────────────────────
    print("Building ground truth (disease-keyword substring match in title) …")
    gt = build_ground_truth(corpus, TEST_QUERIES)
    for (q_text, disease), relevant in zip(TEST_QUERIES, gt.values()):
        print(f"  {disease:<10} → {len(relevant):>4} relevant corpus docs")
    n_empty = sum(1 for v in gt.values() if not v)
    if n_empty:
        print(f"  ⚠ {n_empty} queries have ZERO relevant docs!")
    print()

    # ── encode corpus per strategy ─────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    encoded: Dict[str, np.ndarray] = {}
    for name, fn in STRATEGIES.items():
        print(f"[{name}] Encoding {len(corpus)} corpus docs …")
        texts = [fn(r) for r in corpus]
        encoded[name] = model.encode(
            texts, normalize_embeddings=True, show_progress_bar=True,
        )

    # ── evaluate ───────────────────────────────────────────────────────
    results: Dict[str, Dict[str, float]] = {}
    for name, fn in STRATEGIES.items():
        print(f"[{name}] Evaluating {len(TEST_QUERIES)} queries …")
        t0 = time.perf_counter()
        results[name] = evaluate_strategy(
            name, fn, corpus, encoded[name], TEST_QUERIES, gt, args.k,
        )
        elapsed = time.perf_counter() - t0
        print(f"  {elapsed:.1f}s  |  valid_queries={results[name]['valid_queries']}")

    # ── print tables ───────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("RETRIEVAL STRATEGY COMPARISON")
    print(f"Corpus: {len(corpus)} docs across {len(depts)} departments  |  "
          f"Test queries: {len(TEST_QUERIES)}")
    print("=" * 90)

    # Precision
    header = f"{'Strategy':<20}"
    for k in args.k:
        header += f"{f'P@{k}':<10}"
    header += f"{'MRR':<10}"
    print(f"\n{header}")
    print("-" * (20 + 10 * (len(args.k) + 1)))
    for name in STRATEGIES:
        r = results[name]
        line = f"{name:<20}"
        for k in args.k:
            line += f"{r[f'P@{k}']:<10.4f}"
        line += f"{r['MRR']:<10.4f}"
        print(line)

    # Recall
    print(f"\n{'Strategy':<20}", end="")
    for k in args.k:
        print(f"{f'R@{k}':<10}", end="")
    print()
    print("-" * (20 + 10 * len(args.k)))
    for name in STRATEGIES:
        r = results[name]
        line = f"{name:<20}"
        for k in args.k:
            line += f"{r[f'R@{k}']:<10.4f}"
        print(line)

    # ── summary ───────────────────────────────────────────────────────
    print("\n" + "-" * 90)
    best = max(STRATEGIES, key=lambda n: results[n]["P@1"])
    base = results["full_text"]
    best_r = results[best]
    print(f"Top P@1:  {best} = {best_r['P@1']:.4f}  "
          f"(full_text baseline = {base['P@1']:.4f})")
    print(f"Top MRR:  {best} = {best_r['MRR']:.4f}  "
          f"(full_text baseline = {base['MRR']:.4f})")
    print(f"Top R@10: {best} = {best_r['R@10']:.4f}  "
          f"(full_text baseline = {base['R@10']:.4f})")

    # ── resume markdown table ─────────────────────────────────────────
    print("\n\n=== RESUME TABLE (markdown) ===\n")
    headers_p = " | ".join(f"P@{k}" for k in args.k)
    headers_r = " | ".join(f"R@{k}" for k in args.k)
    print(f"| Strategy | {headers_p} | {headers_r} | MRR |")
    print(f"|{'---|' * (len(args.k) * 2 + 2)}")
    for name in STRATEGIES:
        r = results[name]
        p_str = " | ".join(f"{r[f'P@{k}']:.4f}" for k in args.k)
        r_str = " | ".join(f"{r[f'R@{k}']:.4f}" for k in args.k)
        print(f"| {name} | {p_str} | {r_str} | {r['MRR']:.4f} |")


if __name__ == "__main__":
    main()

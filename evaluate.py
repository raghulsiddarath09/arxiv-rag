"""
evaluate.py — Day 25/26 retrieval evaluation harness.

Computes Recall@K and Precision@K for the RAG retriever against
hand-labeled ground truth in eval_dataset.json.

Design notes (learned the hard way on Day 13):

1. ID NORMALIZATION. papers_cache.json stores IDs as full URLs with
   version suffixes: "http://arxiv.org/abs/2108.06279v2". The eval set
   stores bare IDs: "2108.06279". Both sides are normalized before
   comparison, so a version bump can never silently zero the score.

2. GROUND TRUTH VERIFICATION. Before measuring anything, every
   relevant_paper_id is checked for membership in papers_cache.json.
   A ground-truth ID pointing at a paper that isn't in the corpus makes
   the metric meaningless — retrieval cannot return what doesn't exist.
   The script REFUSES TO RUN if any label is unverifiable.

3. CHUNK-LEVEL vs PAPER-LEVEL. Retrieval returns chunks; a single paper
   may contribute several. Retrieved paper IDs are deduplicated before
   scoring, so one paper appearing as 3 chunks counts once.

Usage:
    python evaluate.py              # full run
    python evaluate.py --limit 5    # smoke test on first 5 questions
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict

EVAL_FILE = "eval_dataset.json"
PAPERS_FILE = "papers_cache.json"
RESULTS_FILE = "eval_results.json"
K_VALUES = [3, 5, 10]


# ──────────────────────────────────────────────────────────
# ID normalization
# ──────────────────────────────────────────────────────────
def normalize_id(raw: str) -> str:
    """
    'http://arxiv.org/abs/2108.06279v2' -> '2108.06279'
    '2108.06279'                        -> '2108.06279'
    """
    if not raw:
        return ""
    tail = str(raw).rstrip("/").split("/")[-1]
    return re.sub(r"v\d+$", "", tail).strip()


# ──────────────────────────────────────────────────────────
# Ground truth verification
# ──────────────────────────────────────────────────────────
def verify_eval_ids(questions, papers):
    """
    Every labeled paper must exist in the corpus. Returns list of problems.
    """
    corpus_ids = {normalize_id(p["id"]) for p in papers}
    problems = []
    for q in questions:
        for pid in q["relevant_paper_ids"]:
            if normalize_id(pid) not in corpus_ids:
                problems.append((q["id"], pid))
    return problems


# ──────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────
def recall_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of relevant papers that appear in the top-k retrieved."""
    if not relevant_ids:
        return None
    topk = set(retrieved_ids[:k])
    hits = len(topk & set(relevant_ids))
    return hits / len(relevant_ids)


def precision_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of the top-k retrieved that are relevant."""
    topk = retrieved_ids[:k]
    if not topk:
        return 0.0
    hits = len(set(topk) & set(relevant_ids))
    return hits / len(topk)


# ──────────────────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────────────────
def retrieve_paper_ids(vectorstore, question, k):
    """
    Run similarity search, map chunks -> unique paper IDs, preserving
    rank order (first appearance wins).
    """
    # Over-fetch chunks: several may collapse to the same paper.
    docs = vectorstore.similarity_search(question, k=k * 4)
    seen, ordered = set(), []
    for d in docs:
        pid = normalize_id(d.metadata.get("paper_id") or d.metadata.get("id", ""))
        if pid and pid not in seen:
            seen.add(pid)
            ordered.append(pid)
    return ordered


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Only evaluate the first N questions (smoke test)")
    args = ap.parse_args()

    print("Loading eval dataset...")
    with open(EVAL_FILE) as f:
        data = json.load(f)
    questions = data["questions"]

    print("Loading corpus...")
    with open(PAPERS_FILE) as f:
        papers = json.load(f)
    print(f"  {len(papers)} papers, {len(questions)} eval questions")

    # ── Gate: ground truth must be valid ──
    print("\nVerifying ground-truth IDs exist in corpus...")
    problems = verify_eval_ids(questions, papers)
    if problems:
        print(f"\n  FAILED — {len(problems)} labeled paper(s) not in corpus:")
        for qid, pid in problems:
            print(f"    {qid}: {pid}")
        print("\n  Refusing to run. A ground-truth ID that isn't in the")
        print("  index makes the metric meaningless. Fix the labels first.")
        sys.exit(1)
    print("  OK — all ground-truth IDs verified present.")

    if args.limit:
        questions = questions[:args.limit]
        print(f"\n[smoke test] limited to {len(questions)} questions")

    print("\nLoading retrieval pipeline (this takes ~45s)...")
    from rag_pipeline import vectorstore

    max_k = max(K_VALUES)
    per_question = []
    latencies = []

    print(f"\nEvaluating {len(questions)} questions...\n")
    for i, q in enumerate(questions, 1):
        relevant = [normalize_id(x) for x in q["relevant_paper_ids"]]

        t0 = time.perf_counter()
        retrieved = retrieve_paper_ids(vectorstore, q["question"], max_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        row = {
            "id": q["id"],
            "topic": q["topic"],
            "question": q["question"],
            "relevant_paper_ids": relevant,
            "retrieved_paper_ids": retrieved[:max_k],
        }
        for k in K_VALUES:
            row[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
            row[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)

        per_question.append(row)

        hit3 = "HIT " if row["recall@3"] and row["recall@3"] > 0 else "miss"
        print(f"  [{i:>2}/{len(questions)}] {hit3} {q['id']}  {q['question'][:58]}")

    # ── Aggregate ──
    overall = {}
    for k in K_VALUES:
        overall[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in per_question) / len(per_question)
        overall[f"precision@{k}"] = sum(r[f"precision@{k}"] for r in per_question) / len(per_question)

    by_topic = defaultdict(list)
    for r in per_question:
        by_topic[r["topic"]].append(r)
    topic_scores = {
        t: {f"recall@{k}": sum(r[f"recall@{k}"] for r in rows) / len(rows) for k in K_VALUES}
           | {"n": len(rows)}
        for t, rows in by_topic.items()
    }

    avg_latency = sum(latencies) / len(latencies)

    # ── Report ──
    print("\n" + "=" * 58)
    print("OVERALL")
    print("=" * 58)
    for k in K_VALUES:
        print(f"  Recall@{k:<3} {overall[f'recall@{k}']:>7.2%}"
              f"     Precision@{k:<3} {overall[f'precision@{k}']:>7.2%}")
    print(f"\n  Avg retrieval latency: {avg_latency:.1f}ms")
    print(f"  Questions evaluated:   {len(per_question)}")

    print("\n" + "=" * 58)
    print("BY TOPIC (Recall@3)")
    print("=" * 58)
    for t, s in sorted(topic_scores.items(), key=lambda x: -x[1]["recall@3"]):
        print(f"  {t:<26} {s['recall@3']:>7.2%}   (n={s['n']})")

    misses = [r for r in per_question if not r["recall@10"]]
    if misses:
        print("\n" + "=" * 58)
        print(f"COMPLETE MISSES — not retrieved even at k=10 ({len(misses)})")
        print("=" * 58)
        for r in misses:
            print(f"  {r['id']}  {r['question'][:60]}")
            print(f"        expected: {r['relevant_paper_ids']}")
            print(f"        got:      {r['retrieved_paper_ids'][:5]}")

    out = {
        "corpus_size": len(papers),
        "n_questions": len(per_question),
        "k_values": K_VALUES,
        "overall": overall,
        "by_topic": topic_scores,
        "avg_retrieval_latency_ms": round(avg_latency, 1),
        "per_question": per_question,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {RESULTS_FILE}")


if __name__ == "__main__":
    main()

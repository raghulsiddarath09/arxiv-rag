"""
evaluate_configs.py — Day 26 retrieval ablation.

Measures the SAME 42-question eval set across five retrieval
configurations to isolate what each pipeline stage actually buys:

  1. dense_only     FAISS similarity search
  2. bm25_only      BM25Okapi lexical search
  3. dense_rerank   FAISS candidates -> cross-encoder rerank (NO BM25)
  4. hybrid_rerank  FAISS + BM25 merge -> cross-encoder rerank
  5. multiquery     LLM query expansion -> hybrid -> rerank  (production)

WHY dense_rerank EXISTS:
  hybrid_rerank underperforms dense_only. Two candidate explanations:
    (a) BM25 injects low-quality candidates that dilute the pool
    (b) the ms-marco cross-encoder doesn't transfer to academic abstracts
  dense_rerank isolates (b) by removing BM25 while keeping the reranker.
    dense_rerank > dense_only  -> BM25 is the problem
    dense_rerank < dense_only  -> the reranker is the problem

Honest-measurement notes:

* RELEVANCE_THRESHOLD. Configs using hybrid_search() / full_hybrid_retriever()
  drop docs scoring below the threshold AFTER truncating to k_final, so they
  can return fewer than k. dense_rerank replicates that logic including the
  threshold, so the comparison stays fair.

* Config 5 issues one Groq call per question. Slow, consumes quota.
  Use --skip multiquery to omit it.

* All configs scored identically: chunks -> deduplicated paper IDs in rank
  order -> Recall@k / Precision@k against verified ground truth.

Usage:
    python evaluate_configs.py
    python evaluate_configs.py --limit 5
    python evaluate_configs.py --skip multiquery bm25_only
    python evaluate_configs.py --only dense_only dense_rerank hybrid_rerank
"""

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict

EVAL_FILE = "eval_dataset.json"
PAPERS_FILE = "papers_cache.json"
RESULTS_FILE = "eval_results_configs.json"
K_VALUES = [3, 5, 10]

CHUNK_OVERFETCH = 4


def normalize_id(raw):
    if not raw:
        return ""
    tail = str(raw).rstrip("/").split("/")[-1]
    return re.sub(r"v\d+$", "", tail).strip()


def docs_to_paper_ids(docs):
    seen, ordered = set(), []
    for d in docs:
        pid = normalize_id(d.metadata.get("paper_id") or d.metadata.get("id", ""))
        if pid and pid not in seen:
            seen.add(pid)
            ordered.append(pid)
    return ordered


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return None
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)


def precision_at_k(retrieved, relevant, k):
    topk = retrieved[:k]
    if not topk:
        return 0.0
    return len(set(topk) & set(relevant)) / len(topk)


def verify_eval_ids(questions, papers):
    corpus_ids = {normalize_id(p["id"]) for p in papers}
    return [(q["id"], pid) for q in questions
            for pid in q["relevant_paper_ids"]
            if normalize_id(pid) not in corpus_ids]


# ──────────────────────────────────────────────────────────
# Configurations
# ──────────────────────────────────────────────────────────
def build_configs(rp, max_k):
    n_chunks = max_k * CHUNK_OVERFETCH
    threshold = getattr(rp, "RELEVANCE_THRESHOLD", float("-inf"))

    def dense_only(q):
        return rp.vectorstore.similarity_search(q, k=n_chunks)

    def bm25_only(q):
        scores = rp.bm25.get_scores(rp.tokenize(q))
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_chunks]
        return [rp.chunks[i] for i in idx]

    def dense_rerank(q):
        """
        FAISS candidates only -> cross-encoder rerank.
        Mirrors hybrid_search()'s rerank stage (same pool size, same
        threshold) with the BM25 branch removed, so any difference vs
        hybrid_rerank is attributable to BM25.
        """
        # 60 candidates ~= the pool hybrid_rerank gets (30 dense + 30 bm25)
        candidates = rp.vectorstore.similarity_search(q, k=60)
        if not candidates:
            return []
        pairs = [[q, d.page_content] for d in candidates]
        scores = rp.cross_encoder.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        out = []
        for score, doc in ranked[:n_chunks]:
            if score > threshold:
                doc.metadata["rerank_score"] = float(score)
                out.append(doc)
        return out

    def hybrid_rerank(q):
        return rp.hybrid_search(q, k_semantic=30, k_bm25=30, k_final=n_chunks)

    def multiquery(q):
        return rp.full_hybrid_retriever(q, k_final=n_chunks)

    return [
        ("dense_only", dense_only, "FAISS similarity search only"),
        ("bm25_only", bm25_only, "BM25Okapi lexical search only"),
        ("dense_rerank", dense_rerank, "FAISS -> cross-encoder rerank (no BM25)"),
        ("hybrid_rerank", hybrid_rerank, "FAISS + BM25 -> cross-encoder rerank"),
        ("multiquery", multiquery, "LLM query expansion -> hybrid -> rerank"),
    ]


def run_config(name, fn, questions):
    per_q, latencies, returned_counts = [], [], []

    for i, q in enumerate(questions, 1):
        relevant = [normalize_id(x) for x in q["relevant_paper_ids"]]
        t0 = time.perf_counter()
        try:
            docs = fn(q["question"])
        except Exception as e:
            print(f"    [{i:>2}] ERROR on {q['id']}: {type(e).__name__}: {e}")
            docs = []
        latencies.append((time.perf_counter() - t0) * 1000)

        retrieved = docs_to_paper_ids(docs)
        returned_counts.append(len(retrieved))

        row = {
            "id": q["id"], "topic": q["topic"],
            "relevant_paper_ids": relevant,
            "retrieved_paper_ids": retrieved[:max(K_VALUES)],
            "n_papers_returned": len(retrieved),
        }
        for k in K_VALUES:
            row[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
            row[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        per_q.append(row)

        mark = "HIT " if row["recall@3"] else "miss"
        print(f"    [{i:>2}/{len(questions)}] {mark} {q['id']}")

    agg = {}
    for k in K_VALUES:
        agg[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in per_q) / len(per_q)
        agg[f"precision@{k}"] = sum(r[f"precision@{k}"] for r in per_q) / len(per_q)

    by_topic = defaultdict(list)
    for r in per_q:
        by_topic[r["topic"]].append(r)

    return {
        "config": name,
        "overall": agg,
        "by_topic": {t: {"recall@3": sum(x["recall@3"] for x in rows) / len(rows),
                         "n": len(rows)} for t, rows in by_topic.items()},
        "mean_latency_ms": round(statistics.mean(latencies), 1),
        "median_latency_ms": round(statistics.median(latencies), 1),
        "mean_papers_returned": round(statistics.mean(returned_counts), 1),
        "per_question": per_q,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", default=None,
                    help="Run only these configs (overrides --skip)")
    args = ap.parse_args()

    with open(EVAL_FILE) as f:
        questions = json.load(f)["questions"]
    with open(PAPERS_FILE) as f:
        papers = json.load(f)

    print(f"Corpus: {len(papers)} papers | Eval set: {len(questions)} questions")

    print("\nVerifying ground-truth IDs...")
    problems = verify_eval_ids(questions, papers)
    if problems:
        print(f"  FAILED — {len(problems)} label(s) not in corpus:")
        for qid, pid in problems:
            print(f"    {qid}: {pid}")
        sys.exit(1)
    print("  OK — all verified present.")

    if args.limit:
        questions = questions[:args.limit]
        print(f"\n[smoke test] {len(questions)} questions")

    print("\nLoading pipeline (~45s)...")
    import rag_pipeline as rp
    thr = getattr(rp, "RELEVANCE_THRESHOLD", None)
    print(f"  RELEVANCE_THRESHOLD = {thr}")

    all_configs = build_configs(rp, max(K_VALUES))
    if args.only:
        configs = [c for c in all_configs if c[0] in args.only]
    else:
        configs = [c for c in all_configs if c[0] not in args.skip]

    results = []
    for name, fn, desc in configs:
        print(f"\n{'=' * 60}\n{name}  —  {desc}\n{'=' * 60}")
        results.append(run_config(name, fn, questions))

    # ── Comparison table ──
    print("\n" + "=" * 78)
    print("CONFIGURATION COMPARISON")
    print("=" * 78)
    print(f"{'config':<16}{'R@3':>9}{'R@5':>9}{'R@10':>9}{'P@3':>9}{'lat(ms)':>11}{'docs':>7}")
    print("-" * 78)
    for r in results:
        o = r["overall"]
        print(f"{r['config']:<16}"
              f"{o['recall@3']:>8.1%} {o['recall@5']:>8.1%} {o['recall@10']:>8.1%} "
              f"{o['precision@3']:>8.1%} {r['mean_latency_ms']:>10.1f} "
              f"{r['mean_papers_returned']:>6.1f}")

    # ── Deltas vs dense baseline ──
    base = next((r for r in results if r["config"] == "dense_only"), None)
    if base and len(results) > 1:
        print("\n" + "=" * 78)
        print("DELTA vs dense_only baseline")
        print("=" * 78)
        for r in results:
            if r["config"] == "dense_only":
                continue
            d3 = r["overall"]["recall@3"] - base["overall"]["recall@3"]
            d5 = r["overall"]["recall@5"] - base["overall"]["recall@5"]
            cost = r["mean_latency_ms"] / base["mean_latency_ms"] if base["mean_latency_ms"] else 0
            print(f"  {r['config']:<16} R@3 {d3:+.1%}   R@5 {d5:+.1%}   latency {cost:.1f}x")

    # ── BM25 vs reranker attribution ──
    dr = next((r for r in results if r["config"] == "dense_rerank"), None)
    hr = next((r for r in results if r["config"] == "hybrid_rerank"), None)
    if dr and hr and base:
        print("\n" + "=" * 78)
        print("ATTRIBUTION: is BM25 or the reranker responsible?")
        print("=" * 78)
        d_base = base["overall"]["recall@3"]
        d_dr = dr["overall"]["recall@3"]
        d_hr = hr["overall"]["recall@3"]
        print(f"  dense_only        {d_base:>7.1%}   (no rerank, no BM25)")
        print(f"  dense_rerank      {d_dr:>7.1%}   (rerank, no BM25)   reranker effect: {d_dr - d_base:+.1%}")
        print(f"  hybrid_rerank     {d_hr:>7.1%}   (rerank + BM25)     BM25 effect:     {d_hr - d_dr:+.1%}")
        print()
        if d_dr >= d_base and d_hr < d_dr:
            print("  READ: reranker helps; BM25 dilutes the candidate pool.")
            print("  ACTION: drop BM25 from the merge, keep reranking.")
        elif d_dr < d_base and d_hr < d_dr:
            print("  READ: both hurt. The reranker does not transfer to this")
            print("        domain, and BM25 compounds the problem.")
            print("  ACTION: ship dense_only.")
        elif d_dr < d_base:
            print("  READ: the cross-encoder is the primary problem — it demotes")
            print("        documents FAISS already ranked correctly.")
            print("  ACTION: ship dense_only, or try a domain-appropriate reranker.")
        else:
            print("  READ: reranking helps overall. Weigh against latency cost.")

    # ── Per-topic ──
    print("\n" + "=" * 78)
    print("RECALL@3 BY TOPIC")
    print("=" * 78)
    topics = sorted({t for r in results for t in r["by_topic"]})
    print(f"{'topic':<26}" + "".join(f"{r['config'][:12]:>14}" for r in results))
    print("-" * 78)
    for t in topics:
        line = f"{t:<26}"
        for r in results:
            v = r["by_topic"].get(t, {}).get("recall@3")
            line += f"{v:>13.1%} " if v is not None else f"{'-':>14}"
        print(line)

    with open(RESULTS_FILE, "w") as f:
        json.dump({
            "corpus_size": len(papers),
            "n_questions": len(questions),
            "relevance_threshold": thr,
            "k_values": K_VALUES,
            "configs": results,
        }, f, indent=2)
    print(f"\nWrote {RESULTS_FILE}")


if __name__ == "__main__":
    main()

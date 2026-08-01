"""
evaluate_bge.py — test BAAI/bge-reranker-base against the dense baseline.

Context: ms-marco-MiniLM-L-6-v2 degraded Recall@3 from 82.1% (dense-only)
to 75.0% (dense+rerank) to 72.6% (hybrid+rerank). Hypothesis: MS MARCO
training (short web queries vs web passages) doesn't transfer to academic
abstracts. This tests a broader-domain reranker, and re-tests whether BM25
still hurts when paired with a better reranker.

Configs:
  dense_only         FAISS only                        (baseline: 82.1%)
  dense_rerank_bge   FAISS -> BGE rerank               (vs 75.0% w/ ms-marco)
  hybrid_rerank_bge  FAISS + BM25 -> BGE rerank        (vs 72.6% w/ ms-marco)

MEMORY: t3.micro has 1GB RAM. Importing rag_pipeline loads the MS MARCO
cross-encoder at module level; this frees it before loading BGE (~280MB)
so both are never resident at once.

THRESHOLD: BGE scores on a different scale than MS MARCO, so
RELEVANCE_THRESHOLD is NOT applied. The script instead reports the top-1
score distribution split by hit/miss — pick a threshold from that data.

Usage:
    python evaluate_bge.py --limit 5
    python evaluate_bge.py
"""

import argparse
import gc
import json
import re
import statistics
import time
from collections import defaultdict

EVAL_FILE = "eval_dataset.json"
PAPERS_FILE = "papers_cache.json"
RESULTS_FILE = "eval_results_bge.json"
K_VALUES = [3, 5, 10]

BGE_MODEL = "BAAI/bge-reranker-base"
POOL_DENSE = 60      # FAISS candidates for dense_rerank_bge
POOL_SPLIT = 30      # per-source candidates for hybrid (30 + 30 = 60)
N_RETURN = max(K_VALUES) * 4

CONFIGS = ["dense_only", "dense_rerank_bge", "hybrid_rerank_bge"]

# ms-marco reference numbers from the earlier ablation, for context
MSMARCO_REF = {"dense_rerank_bge": 0.750, "hybrid_rerank_bge": 0.726}


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
    return len(set(topk) & set(relevant)) / len(topk) if topk else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    with open(EVAL_FILE) as f:
        questions = json.load(f)["questions"]
    with open(PAPERS_FILE) as f:
        papers = json.load(f)

    corpus_ids = {normalize_id(p["id"]) for p in papers}
    bad = [(q["id"], pid) for q in questions for pid in q["relevant_paper_ids"]
           if normalize_id(pid) not in corpus_ids]
    if bad:
        print("FAILED — ground-truth IDs not in corpus:")
        for qid, pid in bad:
            print(f"  {qid}: {pid}")
        raise SystemExit(1)
    print(f"Corpus {len(papers)} papers | {len(questions)} questions | ground truth OK")

    if args.limit:
        questions = questions[:args.limit]
        print(f"[smoke test] {len(questions)} questions")

    print("\nLoading pipeline (~45s)...")
    import rag_pipeline as rp

    if getattr(rp, "cross_encoder", None) is not None:
        print("Unloading ms-marco cross-encoder to free memory...")
        rp.cross_encoder = None
        gc.collect()
        try:
            import torch
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    print(f"Loading {BGE_MODEL} (~280MB, first run downloads)...")
    from sentence_transformers import CrossEncoder
    bge = CrossEncoder(BGE_MODEL, max_length=512)
    print("  loaded.\n")

    # ── Candidate generation ──
    def dense_candidates(q, k):
        return rp.vectorstore.similarity_search(q, k=k)

    def bm25_candidates(q, k):
        scores = rp.bm25.get_scores(rp.tokenize(q))
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [rp.chunks[i] for i in idx]

    def merge_dedup(*doc_lists):
        seen, merged = set(), []
        for lst in doc_lists:
            for d in lst:
                key = f"{d.metadata.get('paper_id','')}_{d.page_content[:50]}"
                if key not in seen:
                    seen.add(key)
                    merged.append(d)
        return merged

    def rerank(q, candidates):
        if not candidates:
            return [], []
        scores = bge.predict([[q, d.page_content] for d in candidates])
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        docs, kept = [], []
        for s, d in ranked[:N_RETURN]:
            d.metadata["rerank_score"] = float(s)
            docs.append(d)
            kept.append(float(s))
        return docs, kept

    def run_one(cfg, q):
        if cfg == "dense_only":
            return dense_candidates(q, N_RETURN), []
        if cfg == "dense_rerank_bge":
            return rerank(q, dense_candidates(q, POOL_DENSE))
        if cfg == "hybrid_rerank_bge":
            cands = merge_dedup(dense_candidates(q, POOL_SPLIT),
                                bm25_candidates(q, POOL_SPLIT))
            return rerank(q, cands)
        raise ValueError(cfg)

    # ── Evaluate ──
    results = {}
    score_dist = {c: {"hits": [], "misses": []} for c in CONFIGS if c != "dense_only"}

    for cfg in CONFIGS:
        print("=" * 60)
        print(cfg)
        print("=" * 60)
        rows, lats = [], []

        for i, q in enumerate(questions, 1):
            relevant = [normalize_id(x) for x in q["relevant_paper_ids"]]
            t0 = time.perf_counter()
            docs, kept = run_one(cfg, q["question"])
            lats.append((time.perf_counter() - t0) * 1000)

            retrieved = docs_to_paper_ids(docs)
            row = {"id": q["id"], "topic": q["topic"],
                   "relevant_paper_ids": relevant,
                   "retrieved_paper_ids": retrieved[:max(K_VALUES)],
                   "n_papers_returned": len(retrieved)}
            for k in K_VALUES:
                row[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
                row[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
            rows.append(row)

            if cfg in score_dist and kept:
                bucket = "hits" if row["recall@3"] else "misses"
                score_dist[cfg][bucket].append(kept[0])

            print(f"  [{i:>2}/{len(questions)}] {'HIT ' if row['recall@3'] else 'miss'} {q['id']}")

        agg = {}
        for k in K_VALUES:
            agg[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in rows) / len(rows)
            agg[f"precision@{k}"] = sum(r[f"precision@{k}"] for r in rows) / len(rows)

        by_topic = defaultdict(list)
        for r in rows:
            by_topic[r["topic"]].append(r)

        results[cfg] = {
            "overall": agg,
            "mean_latency_ms": round(statistics.mean(lats), 1),
            "mean_papers_returned": round(statistics.mean(r["n_papers_returned"] for r in rows), 1),
            "by_topic": {t: {"recall@3": sum(x["recall@3"] for x in rs) / len(rs), "n": len(rs)}
                         for t, rs in by_topic.items()},
            "per_question": rows,
        }
        print()

    # ── Comparison ──
    print("=" * 78)
    print("COMPARISON — BGE reranker")
    print("=" * 78)
    print(f"{'config':<20}{'R@3':>9}{'R@5':>9}{'R@10':>9}{'P@3':>9}{'lat(ms)':>12}{'docs':>7}")
    print("-" * 78)
    for cfg in CONFIGS:
        r, o = results[cfg], results[cfg]["overall"]
        print(f"{cfg:<20}{o['recall@3']:>8.1%} {o['recall@5']:>8.1%} {o['recall@10']:>8.1%} "
              f"{o['precision@3']:>8.1%} {r['mean_latency_ms']:>11.1f} {r['mean_papers_returned']:>6.1f}")

    base = results["dense_only"]["overall"]["recall@3"]
    base_lat = results["dense_only"]["mean_latency_ms"]
    print("\n" + "=" * 78)
    print("vs dense_only baseline, and vs ms-marco from the earlier run")
    print("=" * 78)
    for cfg in CONFIGS:
        if cfg == "dense_only":
            continue
        v = results[cfg]["overall"]["recall@3"]
        lat = results[cfg]["mean_latency_ms"] / base_lat if base_lat else 0
        ref = MSMARCO_REF.get(cfg)
        line = f"  {cfg:<20} R@3 {v - base:+.1%} vs dense   ({lat:.0f}x latency)"
        if ref is not None:
            line += f"   |  ms-marco was {ref:.1%}, BGE is {v:.1%}  ({v - ref:+.1%})"
        print(line)

    # ── BM25 attribution under BGE ──
    if "dense_rerank_bge" in results and "hybrid_rerank_bge" in results:
        d = results["dense_rerank_bge"]["overall"]["recall@3"]
        h = results["hybrid_rerank_bge"]["overall"]["recall@3"]
        print("\n" + "=" * 78)
        print("DOES BM25 STILL HURT WITH A BETTER RERANKER?")
        print("=" * 78)
        print(f"  dense + BGE      {d:>7.1%}")
        print(f"  hybrid + BGE     {h:>7.1%}     BM25 effect: {h - d:+.1%}")
        print(f"  (with ms-marco, BM25 cost -2.4%)")

    # ── Per-topic ──
    print("\n" + "=" * 78)
    print("RECALL@3 BY TOPIC")
    print("=" * 78)
    topics = sorted(results["dense_only"]["by_topic"])
    print(f"{'topic':<26}" + "".join(f"{c[:14]:>17}" for c in CONFIGS))
    print("-" * 78)
    for t in topics:
        line = f"{t:<26}"
        for c in CONFIGS:
            line += f"{results[c]['by_topic'][t]['recall@3']:>16.1%} "
        print(line)

    # ── Threshold data ──
    print("\n" + "=" * 78)
    print("BGE TOP-1 SCORE DISTRIBUTION (for threshold selection)")
    print("=" * 78)
    for cfg, d in score_dist.items():
        print(f"  {cfg}")
        for label in ["hits", "misses"]:
            vals = d[label]
            if vals:
                print(f"    {label:<8} n={len(vals):<4} min={min(vals):>8.3f}  "
                      f"median={statistics.median(vals):>8.3f}  max={max(vals):>8.3f}")
        if d["hits"]:
            print(f"    -> a threshold below {min(d['hits']):.3f} retains every correct answer")
        print()

    with open(RESULTS_FILE, "w") as f:
        json.dump({"model": BGE_MODEL, "n_questions": len(questions),
                   "results": results, "top1_scores": score_dist}, f, indent=2)
    print(f"Wrote {RESULTS_FILE}")


if __name__ == "__main__":
    main()

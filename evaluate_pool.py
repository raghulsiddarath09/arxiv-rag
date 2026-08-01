"""
evaluate_pool.py — does reranking do better with a smaller candidate pool?

All previous rerank measurements used 60 candidates. Production
hybrid_search() defaults to k_semantic=15, k_bm25=15 (=30, ~25 after dedup).
Reranking fewer, higher-quality candidates may behave differently: the
reranker has less opportunity to promote a wrong document.

Configs (all ms-marco-MiniLM-L-6-v2, which beat BGE on this corpus):

  dense_only          FAISS, no rerank                     baseline 82.1%
  dense_rerank_15     15 FAISS candidates  -> rerank
  dense_rerank_30     30 FAISS candidates  -> rerank
  dense_rerank_60     60 FAISS candidates  -> rerank       (already: 75.0%)
  hybrid_prod         15 FAISS + 15 BM25   -> rerank       (production defaults)

NOTE ON RECALL@10: a 15-chunk pool may yield fewer than 10 unique papers,
which structurally caps Recall@10. The 'docs' column shows mean unique
papers returned so this is visible rather than hidden in the metric.

No RELEVANCE_THRESHOLD is applied — this isolates the pool-size effect.

Usage:
    python evaluate_pool.py --limit 5
    python evaluate_pool.py
"""

import argparse
import json
import re
import statistics
import time
from collections import defaultdict

EVAL_FILE = "eval_dataset.json"
PAPERS_FILE = "papers_cache.json"
RESULTS_FILE = "eval_results_pool.json"
K_VALUES = [3, 5, 10]

CONFIGS = [
    ("dense_only",      {"kind": "dense", "pool": 40, "rerank": False}),
    ("dense_rerank_15", {"kind": "dense", "pool": 15, "rerank": True}),
    ("dense_rerank_30", {"kind": "dense", "pool": 30, "rerank": True}),
    ("dense_rerank_60", {"kind": "dense", "pool": 60, "rerank": True}),
    ("hybrid_prod",     {"kind": "hybrid", "pool": 15, "rerank": True}),
]

REFERENCE = {"dense_only": 0.821, "dense_rerank_60": 0.750}


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
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant) if relevant else None


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
        print("FAILED — ground truth not in corpus:", bad)
        raise SystemExit(1)
    print(f"Corpus {len(papers)} papers | {len(questions)} questions | ground truth OK")

    if args.limit:
        questions = questions[:args.limit]
        print(f"[smoke test] {len(questions)} questions")

    print("\nLoading pipeline (~45s)...")
    import rag_pipeline as rp
    if getattr(rp, "cross_encoder", None) is None:
        print("ERROR: rp.cross_encoder is None — restart the container.")
        raise SystemExit(1)
    print("  ms-marco cross-encoder ready.\n")

    def dense_cands(q, k):
        return rp.vectorstore.similarity_search(q, k=k)

    def bm25_cands(q, k):
        s = rp.bm25.get_scores(rp.tokenize(q))
        idx = sorted(range(len(s)), key=lambda i: s[i], reverse=True)[:k]
        return [rp.chunks[i] for i in idx]

    def dedup(*lists):
        seen, out = set(), []
        for lst in lists:
            for d in lst:
                key = f"{d.metadata.get('paper_id','')}_{d.page_content[:50]}"
                if key not in seen:
                    seen.add(key)
                    out.append(d)
        return out

    def run(cfg, q):
        pool, kind, do_rerank = cfg["pool"], cfg["kind"], cfg["rerank"]
        if kind == "dense":
            cands = dense_cands(q, pool)
        else:
            cands = dedup(dense_cands(q, pool), bm25_cands(q, pool))
        if not do_rerank or not cands:
            return cands, len(cands)
        scores = rp.cross_encoder.predict([[q, d.page_content] for d in cands])
        ranked = sorted(zip(scores, cands), key=lambda x: x[0], reverse=True)
        return [d for _, d in ranked], len(cands)

    results = {}
    for name, cfg in CONFIGS:
        print("=" * 60)
        print(f"{name}  (pool={cfg['pool']}, {cfg['kind']}, rerank={cfg['rerank']})")
        print("=" * 60)
        rows, lats, pool_sizes = [], [], []

        for i, q in enumerate(questions, 1):
            relevant = [normalize_id(x) for x in q["relevant_paper_ids"]]
            t0 = time.perf_counter()
            docs, n_cands = run(cfg, q["question"])
            lats.append((time.perf_counter() - t0) * 1000)
            pool_sizes.append(n_cands)

            retrieved = docs_to_paper_ids(docs)
            row = {"id": q["id"], "topic": q["topic"],
                   "relevant_paper_ids": relevant,
                   "retrieved_paper_ids": retrieved[:max(K_VALUES)],
                   "n_papers": len(retrieved)}
            for k in K_VALUES:
                row[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
                row[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
            rows.append(row)
            print(f"  [{i:>2}/{len(questions)}] {'HIT ' if row['recall@3'] else 'miss'} {q['id']}")

        agg = {}
        for k in K_VALUES:
            agg[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in rows) / len(rows)
            agg[f"precision@{k}"] = sum(r[f"precision@{k}"] for r in rows) / len(rows)

        by_topic = defaultdict(list)
        for r in rows:
            by_topic[r["topic"]].append(r)

        results[name] = {
            "overall": agg,
            "mean_latency_ms": round(statistics.mean(lats), 1),
            "mean_candidates": round(statistics.mean(pool_sizes), 1),
            "mean_papers": round(statistics.mean(r["n_papers"] for r in rows), 1),
            "by_topic": {t: {"recall@3": sum(x["recall@3"] for x in rs) / len(rs), "n": len(rs)}
                         for t, rs in by_topic.items()},
            "per_question": rows,
        }
        print()

    # ── Report ──
    print("=" * 86)
    print("POOL SIZE ABLATION — ms-marco-MiniLM-L-6-v2")
    print("=" * 86)
    print(f"{'config':<18}{'R@3':>9}{'R@5':>9}{'R@10':>9}{'P@3':>9}{'lat(ms)':>11}{'cands':>8}{'papers':>8}")
    print("-" * 86)
    for name, _ in CONFIGS:
        r, o = results[name], results[name]["overall"]
        print(f"{name:<18}{o['recall@3']:>8.1%} {o['recall@5']:>8.1%} {o['recall@10']:>8.1%} "
              f"{o['precision@3']:>8.1%} {r['mean_latency_ms']:>10.1f} "
              f"{r['mean_candidates']:>7.1f} {r['mean_papers']:>7.1f}")

    base = results["dense_only"]["overall"]["recall@3"]
    base_lat = results["dense_only"]["mean_latency_ms"]
    print("\n" + "=" * 86)
    print("vs dense_only")
    print("=" * 86)
    for name, _ in CONFIGS:
        if name == "dense_only":
            continue
        v = results[name]["overall"]["recall@3"]
        lat = results[name]["mean_latency_ms"] / base_lat if base_lat else 0
        line = f"  {name:<18} R@3 {v - base:+.1%}   ({lat:.0f}x latency)"
        if name in REFERENCE:
            line += f"   [earlier run: {REFERENCE[name]:.1%}]"
        print(line)

    # ── Does a smaller pool help? ──
    p15 = results["dense_rerank_15"]["overall"]["recall@3"]
    p30 = results["dense_rerank_30"]["overall"]["recall@3"]
    p60 = results["dense_rerank_60"]["overall"]["recall@3"]
    print("\n" + "=" * 86)
    print("DOES A SMALLER CANDIDATE POOL HELP THE RERANKER?")
    print("=" * 86)
    print(f"  pool=15  {p15:>7.1%}")
    print(f"  pool=30  {p30:>7.1%}")
    print(f"  pool=60  {p60:>7.1%}")
    print(f"  no rerank {base:>6.1%}")
    print()
    best = max(p15, p30, p60)
    if best > base:
        which = {p15: 15, p30: 30, p60: 60}[best]
        print(f"  READ: reranking wins at pool={which} ({best:.1%} vs {base:.1%}).")
        print("  ACTION: ship rerank with that pool size.")
    else:
        print("  READ: reranking loses at every pool size tested.")
        print("  ACTION: ship dense_only. The reranker does not fit this corpus.")

    print("\n" + "=" * 86)
    print("RECALL@3 BY TOPIC")
    print("=" * 86)
    topics = sorted(results["dense_only"]["by_topic"])
    print(f"{'topic':<24}" + "".join(f"{n[:14]:>13}" for n, _ in CONFIGS))
    print("-" * 86)
    for t in topics:
        line = f"{t:<24}"
        for name, _ in CONFIGS:
            line += f"{results[name]['by_topic'][t]['recall@3']:>12.1%} "
        print(line)

    with open(RESULTS_FILE, "w") as f:
        json.dump({"model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                   "n_questions": len(questions), "results": results}, f, indent=2)
    print(f"\nWrote {RESULTS_FILE}")


if __name__ == "__main__":
    main()

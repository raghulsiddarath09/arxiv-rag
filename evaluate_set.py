"""
evaluate_set.py — run any eval dataset across retrieval configs.

Generalises evaluate_configs.py so the paraphrase set (eval_dataset.json)
and the lexical/entity set (eval_dataset_lexical.json) can be measured
with identical code and compared directly.

Usage:
    python evaluate_set.py --dataset eval_dataset_lexical.json
    python evaluate_set.py --dataset eval_dataset.json --limit 5
    python evaluate_set.py --dataset eval_dataset_lexical.json \
        --configs dense_only bm25_only hybrid_rerank
"""

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict

K_VALUES = [3, 5, 10]
N_RETURN = 40


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
    ap.add_argument("--dataset", default="eval_dataset.json")
    ap.add_argument("--papers", default="papers_cache.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--configs", nargs="*",
                    default=["dense_only", "bm25_only", "hybrid_norerank", "hybrid_rerank"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.dataset) as f:
        data = json.load(f)
    questions = data["questions"]
    meta = data.get("_meta", {})
    set_name = meta.get("name", args.dataset)

    with open(args.papers) as f:
        papers = json.load(f)

    corpus_ids = {normalize_id(p["id"]) for p in papers}
    bad = [(q["id"], pid) for q in questions for pid in q["relevant_paper_ids"]
           if normalize_id(pid) not in corpus_ids]
    if bad:
        print(f"FAILED — {len(bad)} ground-truth ID(s) not in corpus:")
        for qid, pid in bad:
            print(f"  {qid}: {pid}")
        sys.exit(1)

    print(f"Dataset: {set_name}  ({len(questions)} questions)")
    print(f"Corpus:  {len(papers)} papers | ground truth verified")

    if args.limit:
        questions = questions[:args.limit]
        print(f"[smoke test] {len(questions)} questions")

    print("\nLoading pipeline (~45s)...")
    import rag_pipeline as rp

    def dense_c(q, k):
        return rp.vectorstore.similarity_search(q, k=k)

    def bm25_c(q, k):
        s = rp.bm25.get_scores(rp.tokenize(q))
        idx = sorted(range(len(s)), key=lambda i: s[i], reverse=True)[:k]
        return [rp.chunks[i] for i in idx]

    def interleave(a, b):
        """Rank-fuse two lists: alternate, dedup, preserve order."""
        seen, out = set(), []
        for pair in zip(a, b):
            for d in pair:
                key = f"{d.metadata.get('paper_id','')}_{d.page_content[:50]}"
                if key not in seen:
                    seen.add(key)
                    out.append(d)
        for d in list(a) + list(b):
            key = f"{d.metadata.get('paper_id','')}_{d.page_content[:50]}"
            if key not in seen:
                seen.add(key)
                out.append(d)
        return out

    def rerank(q, cands, k):
        if not cands or rp.cross_encoder is None:
            return cands[:k]
        s = rp.cross_encoder.predict([[q, d.page_content] for d in cands])
        return [d for _, d in sorted(zip(s, cands), key=lambda x: x[0], reverse=True)][:k]

    CONFIG_FNS = {
        "dense_only":      lambda q: dense_c(q, N_RETURN),
        "bm25_only":       lambda q: bm25_c(q, N_RETURN),
        # hybrid without reranking — pure rank fusion of the two lists
        "hybrid_norerank": lambda q: interleave(dense_c(q, 20), bm25_c(q, 20))[:N_RETURN],
        "hybrid_rerank":   lambda q: rerank(q, interleave(dense_c(q, 15), bm25_c(q, 15)), N_RETURN),
        "dense_rerank":    lambda q: rerank(q, dense_c(q, 30), N_RETURN),
    }

    results = {}
    for name in args.configs:
        if name not in CONFIG_FNS:
            print(f"  unknown config '{name}', skipping")
            continue
        fn = CONFIG_FNS[name]
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        rows, lats = [], []
        for i, q in enumerate(questions, 1):
            relevant = [normalize_id(x) for x in q["relevant_paper_ids"]]
            t0 = time.perf_counter()
            try:
                docs = fn(q["question"])
            except Exception as e:
                print(f"    ERROR {q['id']}: {type(e).__name__}: {e}")
                docs = []
            lats.append((time.perf_counter() - t0) * 1000)
            retrieved = docs_to_paper_ids(docs)
            row = {"id": q["id"], "topic": q.get("topic", "?"),
                   "style": q.get("style", "?"),
                   "relevant_paper_ids": relevant,
                   "retrieved_paper_ids": retrieved[:max(K_VALUES)]}
            for k in K_VALUES:
                row[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
                row[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
            rows.append(row)
            print(f"  [{i:>2}/{len(questions)}] {'HIT ' if row['recall@3'] else 'miss'} {q['id']}")

        agg = {}
        for k in K_VALUES:
            agg[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in rows) / len(rows)
            agg[f"precision@{k}"] = sum(r[f"precision@{k}"] for r in rows) / len(rows)

        by_style = defaultdict(list)
        for r in rows:
            by_style[r["style"]].append(r)

        results[name] = {
            "overall": agg,
            "mean_latency_ms": round(statistics.mean(lats), 1),
            "by_style": {s: {"recall@3": sum(x["recall@3"] for x in rs) / len(rs), "n": len(rs)}
                         for s, rs in by_style.items()},
            "per_question": rows,
        }

    # ── Report ──
    print("\n" + "=" * 72)
    print(f"RESULTS — {set_name}")
    print("=" * 72)
    print(f"{'config':<20}{'R@3':>10}{'R@5':>10}{'R@10':>10}{'P@3':>10}{'lat(ms)':>12}")
    print("-" * 72)
    for name in results:
        o = results[name]["overall"]
        print(f"{name:<20}{o['recall@3']:>9.1%} {o['recall@5']:>9.1%} {o['recall@10']:>9.1%} "
              f"{o['precision@3']:>9.1%} {results[name]['mean_latency_ms']:>11.1f}")

    if "dense_only" in results:
        base = results["dense_only"]["overall"]["recall@3"]
        print("\n  vs dense_only:")
        for name in results:
            if name == "dense_only":
                continue
            d = results[name]["overall"]["recall@3"] - base
            verdict = "WINS" if d > 0 else "loses"
            print(f"    {name:<20} {d:+.1%}   {verdict}")

    styles = sorted({s for r in results.values() for s in r["by_style"]})
    if len(styles) > 1:
        print("\n" + "=" * 72)
        print("RECALL@3 BY QUESTION STYLE")
        print("=" * 72)
        print(f"{'style':<18}" + "".join(f"{n[:14]:>16}" for n in results))
        print("-" * 72)
        for s in styles:
            line = f"{s:<18}"
            for name in results:
                v = results[name]["by_style"].get(s)
                line += f"{v['recall@3']:>15.1%} " if v else f"{'-':>16}"
            print(line)

    out = args.out or f"eval_results_{set_name}.json"
    with open(out, "w") as f:
        json.dump({"dataset": args.dataset, "set_name": set_name,
                   "n_questions": len(questions), "results": results}, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

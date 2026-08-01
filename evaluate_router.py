"""
evaluate_router.py — adaptive retrieval: route each query to dense or hybrid.

MOTIVATION (from measured results on this corpus):
  eval_dataset.json         (paraphrase queries)  dense 82.1%  >  hybrid 73.8%
  eval_dataset_lexical.json (entity/rare-token)   dense 98.0%  <  hybrid 100.0%

Neither strategy dominates. A router that picks per query should beat both
on the combined benchmark.

ROUTING SIGNAL — max IDF of query tokens:
  BM25 already computes inverse document frequency for every corpus token.
  A query containing a rare term ("FeDXL", "superquantile") has high max-IDF,
  which is precisely where exact term matching beats embedding similarity.
  A query of common words ("how do models adapt to new tasks") has low
  max-IDF and is better served by semantic search.

  route = hybrid if max_idf(query) >= T else dense

  Cost: a dict lookup per token. Sub-millisecond, no LLM call.

The script sweeps T, reports the router against both fixed strategies on
each set and combined, and prints the routing decisions so the behaviour
is inspectable rather than a black box.

Usage:
    python evaluate_router.py
    python evaluate_router.py --limit 10
"""

import argparse
import json
import re
import statistics
import sys
import time

K_VALUES = [3, 5, 10]
N_RETURN = 40
DEFAULT_SETS = ["eval_dataset.json", "eval_dataset_lexical.json"]
THRESHOLD_SWEEP = [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0]


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
    ap.add_argument("--sets", nargs="*", default=DEFAULT_SETS)
    ap.add_argument("--papers", default="papers_cache.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="eval_results_router.json")
    args = ap.parse_args()

    # ── Load all questions, tagged by source set ──
    all_q = []
    for path in args.sets:
        with open(path) as f:
            data = json.load(f)
        name = data.get("_meta", {}).get("name", path)
        qs = data["questions"]
        if args.limit:
            qs = qs[:args.limit]
        for q in qs:
            q["_set"] = name
        all_q.extend(qs)
        print(f"{name}: {len(qs)} questions")

    with open(args.papers) as f:
        papers = json.load(f)
    corpus_ids = {normalize_id(p["id"]) for p in papers}
    bad = [(q["id"], pid) for q in all_q for pid in q["relevant_paper_ids"]
           if normalize_id(pid) not in corpus_ids]
    if bad:
        print(f"FAILED — {len(bad)} ground-truth ID(s) missing from corpus")
        for qid, pid in bad[:10]:
            print(f"  {qid}: {pid}")
        sys.exit(1)
    print(f"Total {len(all_q)} questions | ground truth verified\n")

    print("Loading pipeline (~45s)...")
    import rag_pipeline as rp

    # rank_bm25's BM25Okapi exposes .idf as {token: idf}
    idf = getattr(rp.bm25, "idf", None)
    if not idf:
        print("ERROR: bm25 object has no .idf attribute; cannot route.")
        sys.exit(1)
    print(f"  IDF table: {len(idf)} tokens, "
          f"range {min(idf.values()):.2f} to {max(idf.values()):.2f}\n")

    def max_idf(question):
        toks = rp.tokenize(question)
        vals = [idf[t] for t in toks if t in idf]
        return max(vals) if vals else 0.0

    # ── Retrieval strategies ──
    def dense(q):
        return rp.vectorstore.similarity_search(q, k=N_RETURN)

    def bm25_c(q, k):
        s = rp.bm25.get_scores(rp.tokenize(q))
        i = sorted(range(len(s)), key=lambda x: s[x], reverse=True)[:k]
        return [rp.chunks[j] for j in i]

    def interleave(a, b):
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

    def hybrid(q):
        cands = interleave(rp.vectorstore.similarity_search(q, k=15), bm25_c(q, 15))
        if rp.cross_encoder is None:
            return cands[:N_RETURN]
        s = rp.cross_encoder.predict([[q, d.page_content] for d in cands])
        return [d for _, d in sorted(zip(s, cands), key=lambda x: x[0], reverse=True)][:N_RETURN]

    # ── Score every question under BOTH strategies once ──
    # Then routing is just selecting which precomputed result to use,
    # so threshold sweeping costs nothing extra.
    print("Scoring all questions under dense and hybrid...\n")
    records = []
    for i, q in enumerate(all_q, 1):
        relevant = [normalize_id(x) for x in q["relevant_paper_ids"]]
        rec = {"id": q["id"], "set": q["_set"], "question": q["question"],
               "max_idf": round(max_idf(q["question"]), 3), "relevant": relevant}

        for strat, fn in [("dense", dense), ("hybrid", hybrid)]:
            t0 = time.perf_counter()
            docs = fn(q["question"])
            lat = (time.perf_counter() - t0) * 1000
            ids = docs_to_paper_ids(docs)
            rec[strat] = {"latency_ms": lat, "retrieved": ids[:max(K_VALUES)]}
            for k in K_VALUES:
                rec[strat][f"recall@{k}"] = recall_at_k(ids, relevant, k)
                rec[strat][f"precision@{k}"] = precision_at_k(ids, relevant, k)

        records.append(rec)
        d3 = "H" if rec["dense"]["recall@3"] else "."
        h3 = "H" if rec["hybrid"]["recall@3"] else "."
        print(f"  [{i:>3}/{len(all_q)}] {q['id']:<5} idf={rec['max_idf']:>6.2f}  "
              f"dense:{d3} hybrid:{h3}")

    # ── Fixed strategies ──
    def agg(recs, pick):
        out = {}
        for k in K_VALUES:
            out[f"recall@{k}"] = sum(r[pick(r)][f"recall@{k}"] for r in recs) / len(recs)
        out["latency_ms"] = statistics.mean(r[pick(r)]["latency_ms"] for r in recs)
        return out

    sets = sorted({r["set"] for r in records})

    print("\n" + "=" * 76)
    print("FIXED STRATEGIES")
    print("=" * 76)
    print(f"{'set':<22}{'strategy':<10}{'R@3':>9}{'R@5':>9}{'R@10':>9}{'lat(ms)':>12}")
    print("-" * 76)
    for s in sets + ["COMBINED"]:
        subset = records if s == "COMBINED" else [r for r in records if r["set"] == s]
        for strat in ["dense", "hybrid"]:
            a = agg(subset, lambda r, st=strat: st)
            print(f"{s[:21]:<22}{strat:<10}{a['recall@3']:>8.1%} {a['recall@5']:>8.1%} "
                  f"{a['recall@10']:>8.1%} {a['latency_ms']:>11.1f}")
        print()

    # ── Threshold sweep ──
    print("=" * 76)
    print("ROUTER THRESHOLD SWEEP  (route to hybrid when max_idf >= T)")
    print("=" * 76)
    print(f"{'T':>7}{'R@3':>10}{'R@5':>10}{'R@10':>10}{'lat(ms)':>12}{'%hybrid':>10}")
    print("-" * 76)
    sweep = []
    for T in THRESHOLD_SWEEP:
        pick = lambda r, t=T: "hybrid" if r["max_idf"] >= t else "dense"
        a = agg(records, pick)
        pct = 100 * sum(1 for r in records if pick(r) == "hybrid") / len(records)
        sweep.append({"threshold": T, **a, "pct_hybrid": pct})
        label = "all dense" if T >= 99 else ("all hybrid" if T <= 0 else "")
        print(f"{T:>7.1f}{a['recall@3']:>9.1%} {a['recall@5']:>9.1%} {a['recall@10']:>9.1%} "
              f"{a['latency_ms']:>11.1f} {pct:>9.1f}%  {label}")

    best = max(sweep, key=lambda x: x["recall@3"])
    all_dense = next(x for x in sweep if x["threshold"] >= 99)
    all_hybrid = next(x for x in sweep if x["threshold"] <= 0)

    print("\n" + "=" * 76)
    print("VERDICT")
    print("=" * 76)
    print(f"  always dense      R@3 {all_dense['recall@3']:>7.1%}   {all_dense['latency_ms']:>7.1f}ms")
    print(f"  always hybrid     R@3 {all_hybrid['recall@3']:>7.1%}   {all_hybrid['latency_ms']:>7.1f}ms")
    print(f"  router (T={best['threshold']:.1f})    R@3 {best['recall@3']:>7.1%}   "
          f"{best['latency_ms']:>7.1f}ms   routes {best['pct_hybrid']:.0f}% to hybrid")
    print()
    gain_d = best["recall@3"] - all_dense["recall@3"]
    gain_h = best["recall@3"] - all_hybrid["recall@3"]
    if gain_d > 0 and gain_h > 0:
        print(f"  Router beats both fixed strategies: {gain_d:+.1%} vs dense, {gain_h:+.1%} vs hybrid.")
        print("  Adaptive routing is justified on this benchmark.")
    elif gain_d <= 0 and gain_h <= 0:
        print("  Router does not beat the better fixed strategy. IDF is not")
        print("  separating these query types on this corpus — ship the winner.")
    else:
        better = "dense" if all_dense["recall@3"] > all_hybrid["recall@3"] else "hybrid"
        print(f"  Router ties or trails always-{better}. Not worth the complexity.")

    # ── Where does routing disagree with the oracle? ──
    T = best["threshold"]
    wrong = [r for r in records
             if (r["max_idf"] >= T) != (r["hybrid"]["recall@3"] > r["dense"]["recall@3"])
             and r["hybrid"]["recall@3"] != r["dense"]["recall@3"]]
    if wrong:
        print(f"\n  Router misroutes {len(wrong)} question(s) where the strategies differ:")
        for r in wrong[:8]:
            chose = "hybrid" if r["max_idf"] >= T else "dense"
            better = "hybrid" if r["hybrid"]["recall@3"] > r["dense"]["recall@3"] else "dense"
            print(f"    {r['id']:<5} idf={r['max_idf']:>6.2f}  chose {chose:<6} "
                  f"but {better} was better")

    with open(args.out, "w") as f:
        json.dump({"n_questions": len(records), "sets": sets,
                   "sweep": sweep, "best": best, "records": records}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
apply_router.py — wire the IDF query router into rag_pipeline.py.

Replaces full_retriever() with a routed version:

    max_idf(query) >= ROUTER_THRESHOLD  ->  hybrid_search (FAISS+BM25+rerank)
    otherwise                           ->  vectorstore.similarity_search

Measured on 72 verified questions across two query styles:
    always dense    79.9% R@3   @  34ms
    always hybrid   81.9% R@3   @ 834ms
    router T=6.0    86.1% R@3   @ 709ms   (83% routed to hybrid)
    router T=8.0    85.4% R@3   @ 381ms   (43% routed to hybrid)

T=8.0 is shipped: it gives up 0.7pp (one question, inside noise at n=72)
for 46% lower mean latency.

CAVEAT recorded in the code comment: the threshold was selected by sweep
on the full evaluation set, not a held-out split. Treat 85.4% as an
optimistic estimate until validated on unseen questions.

Run from the directory containing rag_pipeline.py:
    python3 apply_router.py
    python3 apply_router.py --revert
"""

import re
import shutil
import sys
from pathlib import Path

TARGET = Path("rag_pipeline.py")
BACKUP = Path("rag_pipeline.py.pre-router")

NEW_BLOCK = '''
# ============================================================
# Query router — chooses retrieval strategy per query
# ============================================================
# Neither strategy dominates (72 verified eval questions):
#   paraphrase queries  dense 82.1%  >  hybrid 73.8%   R@3
#   rare-term queries   dense 76.7%  <  hybrid 93.3%   R@3
#
# Signal: max IDF over query tokens. A query containing a
# low-frequency term ("alphad3m", "node2vec") is one where exact
# term matching beats embedding similarity, because bi-encoders
# compress a whole chunk into one vector and smooth rare tokens
# away. Common-vocabulary queries are better served semantically.
#
# Threshold sweep over the combined set:
#   T=0   (all hybrid)  81.9% R@3  @ 834ms
#   T=6.0               86.1% R@3  @ 709ms   83% hybrid
#   T=8.0               85.4% R@3  @ 381ms   43% hybrid   <- shipped
#   T=100 (all dense)   79.9% R@3  @  34ms
#
# CAVEAT: T selected by sweep on the full eval set, not a held-out
# split. 85.4% is therefore an optimistic estimate.
#
# Classification cost: tokenize + dict lookups, sub-millisecond.
ROUTER_THRESHOLD = 8.0
ROUTER_K = 5


def query_max_idf(question):
    """Highest IDF among query tokens present in the BM25 vocabulary."""
    toks = tokenize(question)
    vals = [bm25.idf[t] for t in toks if t in bm25.idf]
    return max(vals) if vals else 0.0


def routed_retriever(question, k_final=ROUTER_K):
    """Route to hybrid for rare-term queries, dense otherwise."""
    score = query_max_idf(question)

    if score >= ROUTER_THRESHOLD:
        docs = hybrid_search(question, k_semantic=15, k_bm25=15, k_final=k_final)
        for d in docs:
            d.metadata["retrieval_route"] = "hybrid"
        if docs:
            return docs
        # Rare-term query with nothing above the relevance threshold:
        # fall through to dense rather than returning an empty context.

    scored = vectorstore.similarity_search_with_score(question, k=k_final)
    docs = []
    for doc, distance in scored:
        # FAISS returns L2 distance (lower is closer). The UI expects a
        # relevance-style score, so map to (0,1] with 1 = exact match.
        doc.metadata["rerank_score"] = float(1.0 / (1.0 + distance))
        doc.metadata["retrieval_route"] = "dense"
        docs.append(doc)
    return docs


last_retrieved_docs = []


def full_retriever(input_dict):
    global last_retrieved_docs
    if input_dict.get("chat_history"):
        question = contextualize_chain.invoke(input_dict)
    else:
        question = input_dict["input"]
    docs = routed_retriever(question)
    last_retrieved_docs = docs  # main.py reads this after invoke()
    return docs
'''


def revert():
    if not BACKUP.exists():
        print(f"No backup at {BACKUP}; nothing to revert.")
        sys.exit(1)
    shutil.copy(BACKUP, TARGET)
    print(f"Restored {TARGET} from {BACKUP}")


def apply():
    if not TARGET.exists():
        print(f"{TARGET} not found. Run this from the project directory.")
        sys.exit(1)

    src = TARGET.read_text()

    if "ROUTER_THRESHOLD" in src:
        print("Router already present in rag_pipeline.py. Nothing to do.")
        print("(Use --revert first if you want to re-apply.)")
        sys.exit(0)

    for name in ["tokenize", "bm25", "hybrid_search", "vectorstore", "contextualize_chain"]:
        if name not in src:
            print(f"ERROR: expected '{name}' in rag_pipeline.py but did not find it.")
            print("The file differs from what this patch expects. Aborting.")
            sys.exit(1)

    # Match from `last_retrieved_docs = []` through the end of full_retriever
    # (i.e. up to the next top-level statement).
    pattern = re.compile(
        r"^last_retrieved_docs\s*=\s*\[\]\s*\n"       # the list init
        r"(?:.*?\n)*?"                                 # lazily, everything after
        r"^def full_retriever\(input_dict\):\s*\n"     # the function
        r"(?:[ \t].*\n|\s*\n)*",                       # its indented body
        re.MULTILINE,
    )

    m = pattern.search(src)
    if not m:
        print("ERROR: could not locate the full_retriever block to replace.")
        print("Expected 'last_retrieved_docs = []' followed by 'def full_retriever'.")
        print("Patch manually instead — see the NEW_BLOCK constant in this file.")
        sys.exit(1)

    shutil.copy(TARGET, BACKUP)
    patched = src[: m.start()] + NEW_BLOCK.lstrip("\n") + "\n" + src[m.end():]
    TARGET.write_text(patched)

    print(f"Backed up  -> {BACKUP}")
    print(f"Patched    -> {TARGET}")
    print()
    print("Replaced block:")
    print("-" * 56)
    for line in m.group(0).rstrip().splitlines():
        print("  " + line)
    print("-" * 56)
    print()
    print("Next:")
    print("  python3 -c 'import ast,sys; ast.parse(open(\"rag_pipeline.py\").read())' "
          "&& echo 'syntax OK'")
    print("  docker-compose up -d --build backend")
    print("  git add rag_pipeline.py && git commit -m 'Add IDF query router' && git push")


if __name__ == "__main__":
    if "--revert" in sys.argv:
        revert()
    else:
        apply()

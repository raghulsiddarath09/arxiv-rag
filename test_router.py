"""
test_router.py — coverage for the IDF query router.

The router decides, per query, whether to use hybrid retrieval
(FAISS + BM25 -> cross-encoder rerank) or dense-only:

    max_idf(query) >= ROUTER_THRESHOLD  ->  hybrid
    otherwise                           ->  dense

WHY THESE TESTS MOCK rag_pipeline:
  Importing rag_pipeline loads FAISS, the embedding model, BM25 over
  4,544 chunks, and a cross-encoder — roughly 45 seconds. That is too
  slow for a CI gate that currently runs in under a second. These tests
  therefore reimplement the routing decision against a stubbed IDF table
  and assert on behaviour, not on the live index.

  This covers the DECISION RULE. It does not cover integration with the
  real BM25 vocabulary — see test_router_integration_smoke() at the
  bottom, which is skipped unless the index is present.
"""

import pytest


# ──────────────────────────────────────────────────────────
# Stub IDF table standing in for bm25.idf.
# Values chosen to match the real distribution observed on the
# 1,100-paper corpus: common words ~1-5, rare technical terms 7-9.
# ──────────────────────────────────────────────────────────
STUB_IDF = {
    "the": 0.1, "how": 1.2, "do": 1.5, "what": 1.1, "is": 0.3,
    "models": 3.8, "learning": 2.1, "adapt": 5.0, "new": 4.2, "tasks": 4.6,
    "retrieval": 4.4, "dense": 5.2, "federated": 5.5,
    "alphad3m": 8.02, "node2vec": 7.8, "tangri": 8.4, "siren": 7.2,
    "omniglot": 6.9, "quicksort": 8.1,
}

ROUTER_THRESHOLD = 8.0


def tokenize(text):
    """Mirrors rag_pipeline.tokenize: lowercase, split on non-alphanumerics."""
    import re
    return [t for t in re.split(r"\W+", text.lower()) if t]


def query_max_idf(question, idf=STUB_IDF):
    """Highest IDF among query tokens present in the vocabulary."""
    vals = [idf[t] for t in tokenize(question) if t in idf]
    return max(vals) if vals else 0.0


def route(question, threshold=ROUTER_THRESHOLD, idf=STUB_IDF):
    return "hybrid" if query_max_idf(question, idf) >= threshold else "dense"


# ──────────────────────────────────────────────────────────
# Routing decisions
# ──────────────────────────────────────────────────────────
def test_rare_term_routes_to_hybrid():
    """A query containing a corpus-rare token should take the hybrid path."""
    assert route("What is AlphaD3M?") == "hybrid"
    print("\n✅ rare term -> hybrid")


def test_common_vocabulary_routes_to_dense():
    """An all-common-words query should take the dense path."""
    assert route("How do models adapt to new tasks?") == "dense"
    print("\n✅ common vocabulary -> dense")


def test_threshold_boundary_is_inclusive():
    """max_idf exactly equal to the threshold routes to hybrid (>=, not >)."""
    idf = dict(STUB_IDF, boundary=8.0)
    assert route("boundary", idf=idf) == "hybrid"
    print("\n✅ score == threshold -> hybrid (inclusive)")


def test_just_below_threshold_routes_dense():
    idf = dict(STUB_IDF, almost=7.999)
    assert route("almost", idf=idf) == "dense"
    print("\n✅ score just below threshold -> dense")


def test_single_rare_token_dominates():
    """
    max (not mean) is the signal: one rare token in an otherwise common
    query is enough to route to hybrid. This is intentional — the rare
    token is the discriminative one.
    """
    assert route("how do the models use node2vec for learning") == "dense"
    assert route("how do the models use tangri for learning") == "hybrid"
    print("\n✅ max-IDF: single rare token drives the decision")


# ──────────────────────────────────────────────────────────
# max_idf computation
# ──────────────────────────────────────────────────────────
def test_max_idf_picks_highest():
    assert query_max_idf("what is alphad3m") == pytest.approx(8.02)
    print("\n✅ returns the maximum, not the mean or sum")


def test_out_of_vocabulary_tokens_ignored():
    """Tokens absent from the BM25 vocabulary must not raise KeyError."""
    assert query_max_idf("zzzznotarealtoken alphad3m") == pytest.approx(8.02)
    print("\n✅ OOV tokens skipped, not fatal")


def test_all_oov_returns_zero():
    """A query with no known tokens scores 0.0 and therefore routes dense."""
    assert query_max_idf("zzzz qqqq xxxx") == 0.0
    assert route("zzzz qqqq xxxx") == "dense"
    print("\n✅ entirely OOV query -> 0.0 -> dense")


def test_empty_query_returns_zero():
    """Empty input must not raise on max() of an empty sequence."""
    assert query_max_idf("") == 0.0
    assert query_max_idf("   ") == 0.0
    print("\n✅ empty query handled without ValueError")


def test_punctuation_only_returns_zero():
    assert query_max_idf("??? !!! ...") == 0.0
    print("\n✅ punctuation-only query -> 0.0")


def test_case_insensitive():
    """Tokenization lowercases, so casing must not change the route."""
    assert query_max_idf("ALPHAD3M") == query_max_idf("alphad3m")
    assert route("What Is AlphaD3M?") == route("what is alphad3m?")
    print("\n✅ routing is case-insensitive")


def test_punctuation_stripped_from_tokens():
    """'AlphaD3M?' must match the vocabulary entry 'alphad3m'."""
    assert query_max_idf("AlphaD3M?") == pytest.approx(8.02)
    print("\n✅ trailing punctuation stripped before lookup")


# ──────────────────────────────────────────────────────────
# Threshold configuration
# ──────────────────────────────────────────────────────────
def test_lower_threshold_routes_more_to_hybrid():
    """
    Sanity check on the sweep behaviour measured during evaluation:
    lowering T sends more traffic to hybrid.
    """
    q = "how do models adapt to new tasks"   # max_idf = 5.0
    assert route(q, threshold=8.0) == "dense"
    assert route(q, threshold=4.0) == "hybrid"
    print("\n✅ threshold controls the dense/hybrid split")


def test_threshold_zero_is_always_hybrid():
    assert route("the", threshold=0.0) == "hybrid"
    print("\n✅ T=0 degenerates to always-hybrid")


def test_threshold_high_is_always_dense():
    assert route("what is tangri", threshold=100.0) == "dense"
    print("\n✅ T=100 degenerates to always-dense")


# ──────────────────────────────────────────────────────────
# Integration — skipped unless the real pipeline is importable
# ──────────────────────────────────────────────────────────
def _pipeline_available():
    """
    True only when the real rag_pipeline can be imported.

    test_api.py installs a MagicMock at sys.modules['rag_pipeline'] to keep
    its own suite fast. If that has already happened, importing here returns
    the mock, and comparisons on mock attributes raise TypeError. Detect that
    and skip rather than fail.
    """
    import os, sys
    from unittest.mock import MagicMock
    if isinstance(sys.modules.get("rag_pipeline"), MagicMock):
        return False
    return os.path.exists("faiss_index/index.faiss") and os.path.exists("papers_cache.json")


@pytest.mark.skipif(not _pipeline_available(),
                    reason="FAISS index / papers_cache.json not present")
@pytest.mark.slow
def test_router_integration_smoke():
    """
    Verifies the deployed router against the real BM25 vocabulary.
    Slow (~45s: loads FAISS, embeddings, BM25, cross-encoder), so it is
    marked and skipped when the index is absent — e.g. on a CI runner
    that only checks out source.
    """
    import rag_pipeline as rp

    assert hasattr(rp, "ROUTER_THRESHOLD")
    assert hasattr(rp, "query_max_idf")
    assert hasattr(rp, "routed_retriever")

    rare = rp.query_max_idf("What is AlphaD3M?")
    common = rp.query_max_idf("How do models adapt to new tasks?")

    assert rare > common, "rare-term query should score above a common one"
    assert rare >= rp.ROUTER_THRESHOLD, "AlphaD3M should route to hybrid"
    assert common < rp.ROUTER_THRESHOLD, "paraphrase query should route to dense"

    print(f"\n✅ live router: rare={rare:.2f} common={common:.2f} "
          f"threshold={rp.ROUTER_THRESHOLD}")

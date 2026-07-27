# test_api.py - Day 17 pytest suite
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json

# ============================================================
# We mock the heavy pipeline objects so tests run fast
# without loading FAISS, BM25, CrossEncoder, or calling Groq
# ============================================================
import sys
from unittest.mock import MagicMock

# Mock rag_pipeline before importing main
# This prevents loading FAISS/BM25/models during tests
mock_pipeline = MagicMock()
mock_pipeline.chain_with_history = MagicMock()
mock_pipeline.vectorstore = MagicMock()
mock_pipeline.vectorstore.index.ntotal = 4119
mock_pipeline.papers = [{"title": f"Paper {i}", "id": f"arxiv:{i}", "authors": "Author", "published": "2023"} for i in range(1000)]
mock_pipeline.store = {}
mock_pipeline.last_retrieved_docs = []

sys.modules['rag_pipeline'] = mock_pipeline

from main import app

# ============================================================
# TestClient — sends real HTTP requests to your FastAPI app
# without needing uvicorn running
# ============================================================
client = TestClient(app)

# ============================================================
# Sample mock answer and sources for /query tests
# ============================================================
MOCK_ANSWER = "LoRA stands for Low-Rank Adaptation of Large Language Models."
MOCK_DOCS = [
    MagicMock(
        page_content="LoRA injects trainable rank decomposition matrices...",
        metadata={
            "title": "LoRA: Low-Rank Adaptation of Large Language Models",
            "year": "2021",
            "paper_id": "http://arxiv.org/abs/2106.09685v2",
            "rerank_score": 7.439
        }
    )
]

# ============================================================
# Tests
# ============================================================

def test_health_endpoint():
    """GET /health should return 200 with correct fields"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "vectors" in data
    assert "papers" in data
    assert "redis" in data
    assert data["vectors"] == 4119
    assert data["papers"] == 1000
    print(f"\n✅ /health: {data}")


def test_papers_endpoint():
    """GET /papers should return all 1000 papers"""
    response = client.get("/papers")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1000
    assert len(data["papers"]) == 1000
    assert "title" in data["papers"][0]
    assert "id" in data["papers"][0]
    print(f"\n✅ /papers: {data['count']} papers returned")


def test_query_empty_question():
    """POST /query with empty string should return 400"""
    response = client.post("/query", json={
        "question": "",
        "session_id": "test"
    })
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()
    print(f"\n✅ Empty question correctly rejected with 400")


def test_query_whitespace_question():
    """POST /query with whitespace-only question should return 400"""
    response = client.post("/query", json={
        "question": "   ",
        "session_id": "test"
    })
    assert response.status_code == 400
    print(f"\n✅ Whitespace question correctly rejected with 400")


def test_query_missing_question_field():
    """POST /query with missing question field should return 422"""
    response = client.post("/query", json={
        "session_id": "test"
    })
    assert response.status_code == 422
    print(f"\n✅ Missing question field correctly rejected with 422")


def test_query_basic():
    """POST /query should return answer, sources, latency, cached fields"""
    mock_pipeline.chain_with_history.invoke.return_value = MOCK_ANSWER
    mock_pipeline.last_retrieved_docs = MOCK_DOCS

    # Clear cache first so we get a fresh pipeline run
    client.delete("/cache")

    response = client.post("/query", json={
        "question": "What is LoRA?",
        "session_id": "test_basic"
    })
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "latency_ms" in data
    assert "cached" in data
    assert "session_id" in data
    assert data["answer"] == MOCK_ANSWER
    assert data["cached"] == False
    print(f"\n✅ /query basic: answer returned, cached=False")


def test_query_sources_present():
    """POST /query should return non-empty sources list"""
    mock_pipeline.chain_with_history.invoke.return_value = MOCK_ANSWER
    mock_pipeline.last_retrieved_docs = MOCK_DOCS

    client.delete("/cache")

    response = client.post("/query", json={
        "question": "What is LoRA sources test?",
        "session_id": "test_sources"
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data["sources"]) > 0
    assert "title" in data["sources"][0]
    assert "year" in data["sources"][0]
    assert "relevance_score" in data["sources"][0]
    print(f"\n✅ Sources present: {data['sources'][0]['title']}")


def test_query_cache_hit():
    """Same question twice should return cached=True on second call"""
    mock_pipeline.chain_with_history.invoke.return_value = MOCK_ANSWER
    mock_pipeline.last_retrieved_docs = MOCK_DOCS

    # Clear cache to ensure clean state
    client.delete("/cache")

    question = "What is BERT for cache test?"

    # First call — should be cache miss
    response1 = client.post("/query", json={
        "question": question,
        "session_id": "test_cache"
    })
    assert response1.status_code == 200
    assert response1.json()["cached"] == False

    # Second call — should be cache hit
    response2 = client.post("/query", json={
        "question": question,
        "session_id": "test_cache"
    })
    assert response2.status_code == 200
    assert response2.json()["cached"] == True
    assert response2.json()["answer"] == response1.json()["answer"]
    print(f"\n✅ Cache hit confirmed on second identical query")


def test_query_cache_normalization():
    """Different casing of same question should hit same cache entry"""
    mock_pipeline.chain_with_history.invoke.return_value = MOCK_ANSWER
    mock_pipeline.last_retrieved_docs = MOCK_DOCS

    client.delete("/cache")

    # First call with original casing
    response1 = client.post("/query", json={
        "question": "What is RAG?",
        "session_id": "test_norm"
    })
    assert response1.json()["cached"] == False

    # Second call with different casing — should still hit cache
    response2 = client.post("/query", json={
        "question": "what is rag?",
        "session_id": "test_norm"
    })
    assert response2.json()["cached"] == True
    print(f"\n✅ Cache normalization: 'What is RAG?' and 'what is rag?' hit same cache entry")


def test_cache_stats_endpoint():
    """GET /cache/stats should return correct fields"""
    response = client.get("/cache/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_cached" in data
    assert "memory_used" in data
    assert "ttl_seconds" in data
    assert data["ttl_seconds"] == 86400
    print(f"\n✅ /cache/stats: {data['total_cached']} entries cached")


def test_clear_cache_endpoint():
    """DELETE /cache should clear all cached queries"""
    # First add something to cache
    mock_pipeline.chain_with_history.invoke.return_value = MOCK_ANSWER
    mock_pipeline.last_retrieved_docs = MOCK_DOCS

    client.post("/query", json={
        "question": "Question to cache before clearing",
        "session_id": "test_clear"
    })

    # Now clear
    response = client.delete("/cache")
    assert response.status_code == 200
    data = response.json()
    assert "cleared" in data
    print(f"\n✅ DELETE /cache: cleared {data['cleared']} entries")


def test_clear_session_endpoint():
    """DELETE /session/{id} should clear session history"""
    response = client.delete("/session/test_session_to_clear")
    assert response.status_code == 200
    data = response.json()
    assert "cleared" in data
    assert "session_id" in data
    print(f"\n✅ DELETE /session: {data}")


def test_clear_nonexistent_session():
    """DELETE /session/{id} for unknown session should return cleared:false"""
    response = client.delete("/session/this_session_does_not_exist_xyz")
    assert response.status_code == 200
    data = response.json()
    assert data["cleared"] == False
    print(f"\n✅ Nonexistent session handled gracefully: cleared=False")

def test_clear_existing_session():
    """DELETE /session/{id} for existing session should return cleared:true"""
    # Import main's actual store and add a session directly
    import main as main_module
    from langchain_community.chat_message_histories import ChatMessageHistory
    
    # Add directly to main.py's store, not mock_pipeline.store
    main_module.store["existing_session_123"] = ChatMessageHistory()

    response = client.delete("/session/existing_session_123")
    assert response.status_code == 200
    data = response.json()
    assert data["cleared"] == True
    assert data["session_id"] == "existing_session_123"
    print(f"\n✅ Existing session cleared: {data}")


def test_clear_cache_with_entries():
    """DELETE /cache should report how many entries were cleared"""
    mock_pipeline.chain_with_history.invoke.return_value = MOCK_ANSWER
    mock_pipeline.last_retrieved_docs = MOCK_DOCS

    # Ensure something is cached first
    client.post("/query", json={
        "question": "Guaranteed unique cache entry question xyz987",
        "session_id": "test"
    })

    # Verify something is actually in cache before clearing
    stats = client.get("/cache/stats").json()
    assert stats["total_cached"] >= 1

    # Now clear
    response = client.delete("/cache")
    assert response.status_code == 200
    data = response.json()
    assert data["cleared"] >= 1  # must have cleared at least 1
    assert "message" in data
    print(f"\n✅ Cache cleared with entries: {data}")


def test_query_pipeline_runs_on_miss():
    """Cache miss should invoke the pipeline exactly once"""
    mock_pipeline.chain_with_history.invoke.return_value = MOCK_ANSWER
    mock_pipeline.last_retrieved_docs = MOCK_DOCS

    client.delete("/cache")
    mock_pipeline.chain_with_history.invoke.reset_mock()

    client.post("/query", json={
        "question": "Unique question for pipeline test abc123",
        "session_id": "test_pipeline"
    })

    # Pipeline should have been called exactly once
    mock_pipeline.chain_with_history.invoke.assert_called_once()
    print(f"\n✅ Pipeline called exactly once on cache miss")
def test_query_idk_response():
    """Pipeline should return IDK when no relevant context found"""
    # Mock pipeline returning IDK response
    mock_pipeline.chain_with_history.invoke.return_value = "I don't have enough information."
    mock_pipeline.last_retrieved_docs = []  # no docs retrieved

    client.delete("/cache")

    response = client.post("/query", json={
        "question": "What is quantum entanglement in photonic crystals?",
        "session_id": "test_idk"
    })
    assert response.status_code == 200
    data = response.json()
    assert "information" in data["answer"].lower()
    assert data["cached"] == False
    print(f"\n✅ IDK response handled correctly: {data['answer']}")    
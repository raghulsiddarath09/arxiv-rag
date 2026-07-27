# main.py - Day 16 complete + CORS (Day 19)
import os
import time
import json
import hashlib
import asyncio
import warnings
from functools import partial
warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List
import redis

from rag_pipeline import chain_with_history, vectorstore, papers, store
import rag_pipeline

# ============================================================
# Redis setup
# ============================================================
redis_client = redis.Redis(
    host=os.environ.get('REDIS_HOST', 'localhost'),
    port=int(os.environ.get('REDIS_PORT', 6379)),
    db=0,
    decode_responses=True
)

CACHE_TTL = 86400  # 24 hours

def make_cache_key(question: str) -> str:
    normalized = question.strip().lower()
    return "rag:query:" + hashlib.sha256(normalized.encode()).hexdigest()

def get_cached_answer(question: str):
    try:
        key = make_cache_key(question)
        cached = redis_client.get(key)
        if cached:
            return json.loads(cached)
        return None
    except Exception as e:
        print(f"Redis GET error: {e}")
        return None

def set_cached_answer(question: str, answer: str, sources: list):
    try:
        key = make_cache_key(question)
        redis_client.set(
            key,
            json.dumps({"answer": answer, "sources": sources}),
            ex=CACHE_TTL
        )
    except Exception as e:
        print(f"Redis SET error: {e}")

# ============================================================
# FastAPI app
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        redis_client.ping()
        print("Redis connected successfully")
    except Exception as e:
        print(f"WARNING: Redis not available: {e}")
        print("API will work but without caching")
    print(f"Startup complete: {vectorstore.index.ntotal} vectors, {len(papers)} papers")
    yield

app = FastAPI(title="ArXiv RAG API", lifespan=lifespan)

# ============================================================
# CORS — allows React frontend to call API
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:80",
        "http://localhost:3000",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Pydantic models
# ============================================================
class Source(BaseModel):
    title: str
    year: str
    relevance_score: float

class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    session_id: str
    latency_ms: float
    cached: bool

# ============================================================
# Helper — extract sources from retrieved docs
# ============================================================
def extract_sources(docs) -> List[dict]:
    seen_titles = set()
    sources = []
    for doc in docs:
        title = doc.metadata.get("title", "Unknown")
        if title not in seen_titles:
            seen_titles.add(title)
            sources.append({
                "title": title,
                "year": doc.metadata.get("year", "Unknown"),
                "relevance_score": round(doc.metadata.get("rerank_score", 0.0), 3)
            })
    return sources

# ============================================================
# Endpoints
# ============================================================
@app.get("/health")
async def health():
    try:
        redis_client.ping()
        redis_status = "connected"
        cache_size = redis_client.dbsize()
    except:
        redis_status = "unavailable"
        cache_size = 0
    return {
        "status": "ok",
        "vectors": vectorstore.index.ntotal,
        "papers": len(papers),
        "redis": redis_status,
        "cached_queries": cache_size
    }

@app.get("/cache/stats")
async def cache_stats():
    try:
        keys = redis_client.keys("rag:query:*")
        info = redis_client.info("memory")
        return {
            "total_cached": len(keys),
            "memory_used": info["used_memory_human"],
            "memory_peak": info["used_memory_peak_human"],
            "ttl_seconds": CACHE_TTL,
            "ttl_hours": CACHE_TTL // 3600
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/papers")
async def list_papers():
    return {
        "count": len(papers),
        "papers": [
            {
                "title": p["title"],
                "id": p["id"],
                "authors": p.get("authors", ""),
                "year": p.get("published", "")
            }
            for p in papers
        ]
    }

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start = time.time()

    # ── Stage 1: Check Redis cache ──────────────────────────
    cached = get_cached_answer(req.question)
    if cached:
        latency = (time.time() - start) * 1000
        print(f"CACHE HIT:  '{req.question[:50]}' → {latency:.1f}ms")
        return QueryResponse(
            answer=cached["answer"],
            sources=[Source(**s) for s in cached.get("sources", [])],
            session_id=req.session_id,
            latency_ms=latency,
            cached=True
        )

    # ── Stage 2: Cache miss → run pipeline in thread pool ───
    print(f"CACHE MISS: '{req.question[:50]}' → running pipeline")
    try:
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            None,
            partial(
                chain_with_history.invoke,
                {"input": req.question},
                config={"configurable": {"session_id": req.session_id}}
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # ── Stage 3: Extract sources ─────────────────────────────
    sources = extract_sources(rag_pipeline.last_retrieved_docs)

    # ── Stage 4: Store in Redis ──────────────────────────────
    set_cached_answer(req.question, answer, sources)

    latency = (time.time() - start) * 1000
    print(f"PIPELINE:   '{req.question[:50]}' → {latency:.1f}ms (now cached)")

    return QueryResponse(
        answer=answer,
        sources=[Source(**s) for s in sources],
        session_id=req.session_id,
        latency_ms=latency,
        cached=False
    )

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    if session_id in store:
        store.pop(session_id)
        return {"cleared": True, "session_id": session_id}
    return {
        "cleared": False,
        "session_id": session_id,
        "note": "session not found"
    }

@app.delete("/cache")
async def clear_cache():
    try:
        keys = redis_client.keys("rag:query:*")
        if keys:
            redis_client.delete(*keys)
        return {
            "cleared": len(keys),
            "message": f"Deleted {len(keys)} cached queries"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
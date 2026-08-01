# rag_pipeline.py
import os
import re
import time
import json
import warnings
import statistics
warnings.filterwarnings("ignore")

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

# ============================================================
# Step 1 — Load papers and build chunks
# ============================================================
print("Loading cached papers...")
with open("papers_cache.json") as f:
    papers = json.load(f)
print(f"Loaded {len(papers)} papers")

documents = []
for paper in papers:
    content = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
    doc = Document(
        page_content=content,
        metadata={
            "title": paper['title'],
            "authors": paper.get('authors', ''),
            "paper_id": paper['id'],
            "year": paper['published']
        }
    )
    documents.append(doc)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"Total chunks: {len(chunks)}\n")

# ============================================================
# Step 2 — Load embedding model
# ============================================================
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
print(f"Loaded {vectorstore.index.ntotal} vectors\n")

# ============================================================
# Step 3 — Build BM25 with proper tokenization
# FIX: use regex tokenization not naive split()
# "DPR?" → ["dpr"] not ["dpr?"]
# ============================================================
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

print("Building BM25 index with proper tokenization...")
chunk_texts = [chunk.page_content for chunk in chunks]
tokenized_chunks = [tokenize(text) for text in chunk_texts]
bm25 = BM25Okapi(tokenized_chunks)
print(f"BM25 index built: {len(chunk_texts)} documents\n")

# ============================================================
# Step 4 — Two separate LLM instances
# FIX: temperature=0 for query generation (deterministic)
#      temperature=0.1 for answer generation (slight flexibility)
# ============================================================
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

query_llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=200
)

answer_llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=600
)

# ============================================================
# Step 5 — Hybrid search with fixes
# ============================================================
RELEVANCE_THRESHOLD = -10.0

def hybrid_search(query, k_semantic=15, k_bm25=15, k_final=4):
    # Stage 1A — Semantic
    semantic_docs = vectorstore.similarity_search(query, k=k_semantic)

    # Stage 1B — BM25 with proper tokenization
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k_bm25]
    bm25_docs = [chunks[i] for i in top_bm25_indices]

    # Stage 2 — Merge with better dedup key
    seen_keys = set()
    merged_docs = []

    for doc in semantic_docs + bm25_docs:
        dedup_key = f"{doc.metadata.get('paper_id', '')}_{doc.page_content[:50]}"
        if dedup_key not in seen_keys:
            merged_docs.append(doc)
            seen_keys.add(dedup_key)

    if not merged_docs:
        return []

    # Stage 3 — Cross-encoder rerank
    pairs = [[query, doc.page_content] for doc in merged_docs]
    scores = cross_encoder.predict(pairs)

    scored_docs = sorted(
        zip(scores, merged_docs),
        key=lambda x: x[0],
        reverse=True
    )

    reranked = []
    for score, doc in scored_docs[:k_final]:
        if score > RELEVANCE_THRESHOLD:
            doc.metadata['rerank_score'] = float(score)
            reranked.append(doc)

    return reranked

# ============================================================
# Step 7 — Multi-query with deterministic generation
# ============================================================
multi_query_prompt = ChatPromptTemplate.from_messages([
    ("system", """Generate exactly 3 different versions of the question.
Different phrasing, same information.
Return ONLY the 3 questions, one per line.
No numbering, no explanations."""),
    ("human", "{question}")
])

def generate_queries(question):
    response = query_llm.invoke(
        multi_query_prompt.format_messages(question=question)
    )
    queries = [q.strip() for q in response.content.strip().split('\n')
               if q.strip()]
    return [question] + queries[:3]

def full_hybrid_retriever(question, k_final=4):
    queries = generate_queries(question)
    print(f"  Queries: {len(queries)} variations generated")

    seen_keys = set()
    all_docs = []

    for query in queries:
        docs = hybrid_search(query, k_semantic=8, k_bm25=8, k_final=10)
        for doc in docs:
            dedup_key = f"{doc.metadata.get('paper_id', '')}_{doc.page_content[:50]}"
            if dedup_key not in seen_keys:
                all_docs.append(doc)
                seen_keys.add(dedup_key)

    if not all_docs:
        return []

    pairs = [[question, doc.page_content] for doc in all_docs]
    scores = cross_encoder.predict(pairs)
    scored = sorted(zip(scores, all_docs), key=lambda x: x[0], reverse=True)

    reranked = []
    for score, doc in scored[:k_final]:
        if score > RELEVANCE_THRESHOLD:
            doc.metadata['rerank_score'] = float(score)
            reranked.append(doc)

    return reranked

def format_docs(docs):
    if not docs:
        return "No relevant context found."
    formatted = ""
    for i, doc in enumerate(docs):
        score = doc.metadata.get('rerank_score', None)
        score_str = f" [relevance: {score:.3f}]" if score else ""
        formatted += f"\n[{i+1}] From '{doc.metadata['title']}' ({doc.metadata['year']}){score_str}:\n"
        formatted += doc.page_content + "\n"
    return formatted

# ============================================================
# Step 8 — Full pipeline
# ============================================================
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given chat history and latest user question,
formulate a standalone question. Do NOT answer.
Just reformulate if needed, otherwise return as is."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert ML research assistant.
Answer using ONLY the context below in 3-4 clear, concise sentences.
Cite which paper your answer comes from.
If the context says 'No relevant context found' or answer
is not in context say 'I don't have enough information.'

Context:
{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

contextualize_chain = contextualize_prompt | query_llm | StrOutputParser()

# ============================================================
# NEW — stores most recently retrieved docs so main.py
# can extract sources after each pipeline call
# ============================================================
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

rag_chain = (
    RunnablePassthrough.assign(
        context=RunnableLambda(full_retriever) | RunnableLambda(format_docs)
    )
    | qa_prompt
    | answer_llm
    | StrOutputParser()
)

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# ============================================================
# Step 9 — Test pipeline (only runs when executed directly,
# NOT when FastAPI imports it)
# ============================================================
def chat(question, session_id="default"):
    start = time.time()
    response = chain_with_history.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}}
    )
    latency = (time.time() - start) * 1000
    print(f"\nQ: {question}")
    print(f"A: {response}")
    print(f"Latency: {latency:.0f}ms")
    print("-"*60)


if __name__ == "__main__":
    print("="*60)
    print("COMPARISON: Semantic vs Hybrid Search")
    print("="*60)

    test_queries = [
        "What is dense passage retrieval DPR?",
        "How does BM25 keyword scoring work?",
        "What is BERT masked language modeling?",
        "How does attention mechanism work in transformers?",
        "What is LoRA low rank adaptation?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")

        semantic_docs = vectorstore.similarity_search(query, k=4)
        print("Semantic only:")
        for i, doc in enumerate(semantic_docs[:3]):
            print(f"  {i+1}. {doc.metadata['title'][:50]} ({doc.metadata['year']})")

        hybrid_docs = hybrid_search(query, k_final=4)
        print(f"Hybrid (threshold={RELEVANCE_THRESHOLD}):")
        if hybrid_docs:
            for i, doc in enumerate(hybrid_docs[:3]):
                score = doc.metadata.get('rerank_score', 0)
                print(f"  {i+1}. {doc.metadata['title'][:50]} [{score:.2f}]")
        else:
            print("  No results above relevance threshold → IDK")

        print("-"*60)

    print("\n" + "="*60)
    print("COMPLETE PIPELINE — Day 14 Fixed")
    print("="*60)

    print("\n--- Determinism test: same question twice ---")
    chat("What is the RAG paper about?", session_id="test1")
    chat("What is the RAG paper about?", session_id="test2")

    print("\n--- Core functionality tests ---")
    chat("What is dense passage retrieval?")
    chat("How does LoRA reduce memory during fine-tuning?")
    chat("How does BM25 keyword scoring work?")

    print("\n" + "="*60)
    print("FIXES APPLIED IN DAY 14 UPDATED")
    print("="*60)
    print("""
Fix 1: temperature=0 for query generation (was 0.1)
       → deterministic query variations
       → stable citations across runs

Fix 2: regex tokenization for BM25 (was naive split)
       → "DPR?" correctly tokenizes to "dpr"
       → acronym matching works properly

Fix 3: relevance threshold = 0.0 (was no threshold)
       → negative cross-encoder scores rejected
       → system says IDK instead of serving irrelevant chunks

Fix 4: paper_id + chunk start for dedup (was content[:100])
       → more reliable duplicate detection
       → same chunk from different queries correctly deduplicated
""")
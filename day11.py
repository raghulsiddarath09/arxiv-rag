import os
import time
import warnings
import statistics
warnings.filterwarnings("ignore")

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

# ============================================================
# Step 1 — Load vector store
# ============================================================
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

print("Loading vector store...")
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
print(f"Loaded {vectorstore.index.ntotal} vectors\n")

# ============================================================
# Step 2 — Load cross-encoder model
# ============================================================
print("Loading cross-encoder...")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("Cross-encoder loaded\n")

# ============================================================
# Step 3 — Two-stage retrieval function
# ============================================================
def rerank_retriever(query, k_candidates=20, k_final=4):
    """
    Stage 1: FAISS retrieves top 20 candidates (fast, approximate)
    Stage 2: Cross-encoder reranks those 20 (slow, accurate)
    Returns top 4 after reranking
    """
    # Stage 1 — Bi-encoder retrieval
    stage1_docs = vectorstore.similarity_search(query, k=k_candidates)
    
    # Stage 2 — Cross-encoder scoring
    # Cross-encoder sees [query, chunk] together
    pairs = [[query, doc.page_content] for doc in stage1_docs]
    scores = cross_encoder.predict(pairs)
    
    # Sort by score descending
    scored_docs = sorted(
        zip(scores, stage1_docs),
        key=lambda x: x[0],
        reverse=True
    )
    
    # Return top k_final with scores attached
    reranked = []
    for score, doc in scored_docs[:k_final]:
        doc.metadata['rerank_score'] = float(score)
        reranked.append(doc)
    
    return reranked

# ============================================================
# Step 4 — Compare retrieval with and without reranking
# ============================================================
print("="*60)
print("RETRIEVAL COMPARISON: Before vs After Reranking")
print("="*60)

test_queries = [
    "How does the attention mechanism work in transformers?",
    "What is RAG and how does it improve language models?",
    "What is LoRA and why is it parameter efficient?",
    "How are sentence embeddings created for semantic search?",
    "What is BERT and how is it pretrained?",
]

for query in test_queries:
    print(f"\nQuery: {query}")
    
    # Before reranking — top 4 from FAISS directly
    before_docs = vectorstore.similarity_search(query, k=4)
    print("BEFORE reranking:")
    for i, doc in enumerate(before_docs):
        print(f"  {i+1}. {doc.metadata['title'][:55]} ({doc.metadata['year']})")
    
    # After reranking — top 4 from cross-encoder
    after_docs = rerank_retriever(query, k_candidates=20, k_final=4)
    print("AFTER reranking:")
    for i, doc in enumerate(after_docs):
        score = doc.metadata.get('rerank_score', 0)
        print(f"  {i+1}. {doc.metadata['title'][:55]} ({doc.metadata['year']}) [score: {score:.3f}]")
    
    print("-"*60)

# ============================================================
# Step 5 — Build full pipeline with reranking
# ============================================================
llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=600
)

contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given chat history and latest user question,
formulate a standalone question. Do NOT answer.
Just reformulate if needed, otherwise return as is."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert ML research assistant.
Answer using ONLY the context below.
Cite which paper your answer comes from.
If answer not in context say 'I don't have enough information.'

Context:
{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

def format_docs(docs):
    formatted = ""
    for i, doc in enumerate(docs):
        score = doc.metadata.get('rerank_score', None)
        score_str = f" [relevance: {score:.3f}]" if score else ""
        formatted += f"\n[{i+1}] From '{doc.metadata['title']}' ({doc.metadata['year']}){score_str}:\n"
        formatted += doc.page_content + "\n"
    return formatted

contextualize_chain = contextualize_prompt | llm | StrOutputParser()

def contextualized_rerank_retriever(input_dict):
    if input_dict.get("chat_history"):
        new_question = contextualize_chain.invoke(input_dict)
        return rerank_retriever(new_question)
    return rerank_retriever(input_dict["input"])

rag_chain = (
    RunnablePassthrough.assign(
        context=RunnableLambda(contextualized_rerank_retriever) | RunnableLambda(format_docs)
    )
    | qa_prompt
    | llm
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
# Step 6 — Test full pipeline with reranking
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
    
    docs = rerank_retriever(question)
    seen = set()
    for doc in docs:
        title = doc.metadata['title']
        if title not in seen:
            score = doc.metadata.get('rerank_score', 0)
            print(f"  Source: {title[:50]} [score: {score:.3f}]")
            seen.add(title)
    
    print(f"Latency: {latency:.0f}ms")
    print("-"*60)

print("\n" + "="*60)
print("FULL RAG + RERANKING PIPELINE — Day 11")
print("="*60)

chat("What is the attention mechanism in transformers?")
chat("How does BERT use bidirectional training?")
chat("What is RAG and how does it work?")
chat("What is LoRA?")

# ============================================================
# Step 7 — Latency comparison
# ============================================================
print("\n--- Latency: Without vs With Reranking ---")

queries = [
    "What is attention mechanism?",
    "How does BERT work?",
    "What is RAG?",
]

without_times = []
with_times = []

for q in queries:
    # Without reranking
    start = time.time()
    vectorstore.similarity_search(q, k=4)
    without_times.append((time.time() - start) * 1000)
    
    # With reranking
    start = time.time()
    rerank_retriever(q, k_candidates=20, k_final=4)
    with_times.append((time.time() - start) * 1000)

print(f"Without reranking: {statistics.mean(without_times):.0f}ms avg")
print(f"With reranking:    {statistics.mean(with_times):.0f}ms avg")
print(f"Overhead added:    {statistics.mean(with_times) - statistics.mean(without_times):.0f}ms")
print("\n(Note: LLM generation dominates total latency — reranking overhead is negligible)")
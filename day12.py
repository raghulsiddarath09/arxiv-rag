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
# Step 2 — Load cross-encoder
# ============================================================
print("Loading cross-encoder...")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("Cross-encoder loaded\n")

# ============================================================
# Step 3 — LLM
# ============================================================
llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=600
)

# ============================================================
# Step 4 — Multi-query generation
# ============================================================
multi_query_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant helping with information retrieval.
Generate exactly 3 different versions of the given question.
Each version should use different phrasing and vocabulary
but ask about the same information.
Return ONLY the 3 questions, one per line.
No numbering, no explanations, no extra text."""),
    ("human", "{question}")
])

def generate_multiple_queries(question):
    """Generate 3 query variations using LLM"""
    response = llm.invoke(
        multi_query_prompt.format_messages(question=question)
    )
    
    # Parse the 3 queries from response
    queries = [q.strip() for q in response.content.strip().split('\n') 
               if q.strip()]
    
    # Always include original query
    all_queries = [question] + queries[:3]
    return all_queries

# ============================================================
# Step 5 — Multi-query retrieval with deduplication
# ============================================================
def multi_query_retrieve(question, k_per_query=10, k_final=4):
    """
    Step 1: Generate 3 query variations
    Step 2: Retrieve k chunks for each query
    Step 3: Deduplicate merged results
    Step 4: Cross-encoder rerank
    Step 5: Return top k_final
    """
    # Generate query variations
    queries = generate_multiple_queries(question)
    print(f"\nQuery variations generated:")
    for i, q in enumerate(queries):
        print(f"  {i+1}. {q}")
    
    # Retrieve for each query
    all_docs = []
    seen_content = set()
    
    for query in queries:
        docs = vectorstore.similarity_search(query, k=k_per_query)
        for doc in docs:
            # Deduplicate by content
            content_key = doc.page_content[:100]
            if content_key not in seen_content:
                all_docs.append(doc)
                seen_content.add(content_key)
    
    print(f"Unique chunks after merging: {len(all_docs)}")
    
    # Cross-encoder rerank all merged results
    if len(all_docs) == 0:
        return []
    
    pairs = [[question, doc.page_content] for doc in all_docs]
    scores = cross_encoder.predict(pairs)
    
    scored_docs = sorted(
        zip(scores, all_docs),
        key=lambda x: x[0],
        reverse=True
    )
    
    reranked = []
    for score, doc in scored_docs[:k_final]:
        doc.metadata['rerank_score'] = float(score)
        reranked.append(doc)
    
    return reranked

# ============================================================
# Step 6 — Compare single vs multi query
# ============================================================
print("="*60)
print("COMPARISON: Single Query vs Multi-Query Retrieval")
print("="*60)

test_queries = [
    "How does attention work in neural networks?",
    "What makes BERT special for NLP tasks?",
    "How does RAG reduce hallucination?",
    "What is parameter efficient fine-tuning?",
]

for query in test_queries:
    print(f"\nOriginal query: {query}")
    
    # Single query retrieval
    single_docs = vectorstore.similarity_search(query, k=4)
    print("\nSingle query results:")
    for i, doc in enumerate(single_docs):
        print(f"  {i+1}. {doc.metadata['title'][:55]} ({doc.metadata['year']})")
    
    # Multi-query retrieval
    multi_docs = multi_query_retrieve(query, k_per_query=8, k_final=4)
    print("\nMulti-query results:")
    for i, doc in enumerate(multi_docs):
        score = doc.metadata.get('rerank_score', 0)
        print(f"  {i+1}. {doc.metadata['title'][:55]} ({doc.metadata['year']}) [score: {score:.3f}]")
    
    print("-"*60)

# ============================================================
# Step 7 — Full pipeline with multi-query
# ============================================================
def format_docs(docs):
    formatted = ""
    for i, doc in enumerate(docs):
        score = doc.metadata.get('rerank_score', None)
        score_str = f" [relevance: {score:.3f}]" if score else ""
        formatted += f"\n[{i+1}] From '{doc.metadata['title']}' ({doc.metadata['year']}){score_str}:\n"
        formatted += doc.page_content + "\n"
    return formatted

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

contextualize_chain = contextualize_prompt | llm | StrOutputParser()

def full_retriever(input_dict):
    if input_dict.get("chat_history"):
        question = contextualize_chain.invoke(input_dict)
    else:
        question = input_dict["input"]
    return multi_query_retrieve(question)

rag_chain = (
    RunnablePassthrough.assign(
        context=RunnableLambda(full_retriever) | RunnableLambda(format_docs)
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
# Step 8 — Test full pipeline
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

print("\n" + "="*60)
print("FULL RAG + MULTI-QUERY + RERANKING — Day 12")
print("="*60)

chat("What is the attention mechanism in transformers?")
chat("How does BERT achieve bidirectional understanding?")
chat("What is LoRA and how does it reduce memory usage?")
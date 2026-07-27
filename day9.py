import os
import json
import warnings
warnings.filterwarnings("ignore")

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import time

# ============================================================
# Step 1 — Load vector store from disk
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
# Step 2 — LLM and retriever
# ============================================================
llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=600
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# ============================================================
# Step 3 — Prompts
# ============================================================
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given a chat history and the latest user question,
formulate a standalone question that captures full context.
Do NOT answer. Just reformulate if needed, otherwise return as is."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert ML research assistant.
Answer questions using ONLY the context provided below.
Always cite which paper your answer comes from using [Paper Title].
If the answer is not in context say 'I don't have enough information.'
Keep answers focused and accurate.

Context:
{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# ============================================================
# Step 4 — Chain with memory
# ============================================================
def format_docs(docs):
    formatted = ""
    for i, doc in enumerate(docs):
        formatted += f"\n[{i+1}] From '{doc.metadata['title']}' ({doc.metadata['year']}):\n"
        formatted += doc.page_content + "\n"
    return formatted

contextualize_chain = contextualize_prompt | llm | StrOutputParser()

def contextualized_retriever(input_dict):
    if input_dict.get("chat_history"):
        new_question = contextualize_chain.invoke(input_dict)
        return retriever.invoke(new_question)
    return retriever.invoke(input_dict["input"])

rag_chain = (
    RunnablePassthrough.assign(
        context=RunnableLambda(contextualized_retriever) | RunnableLambda(format_docs)
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
# Step 5 — Clean chat function with timing
# ============================================================
def chat(question, session_id="default", show_sources=True):
    start = time.time()
    
    response = chain_with_history.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}}
    )
    
    latency = (time.time() - start) * 1000
    
    print(f"\nQ: {question}")
    print(f"A: {response}")
    
    if show_sources:
        docs = retriever.invoke(question)
        seen = set()
        sources = []
        for doc in docs:
            title = doc.metadata['title']
            if title not in seen:
                sources.append(f"{title[:50]} ({doc.metadata['year']})")
                seen.add(title)
        print(f"Sources: {', '.join(sources)}")
    
    print(f"Latency: {latency:.0f}ms")
    print("-" * 60)
    return response

# ============================================================
# Step 6 — Test end to end quality
# ============================================================
print("=" * 60)
print("FULL RAG PIPELINE — Day 9 (105 papers, 450 vectors)")
print("=" * 60)

# Test 1 — Core ML concepts
print("\n--- Test 1: Core ML concepts ---")
chat("What is the transformer architecture and why was it important?")
chat("What is BERT and how does it differ from GPT?")
chat("How does RAG combine retrieval with generation?")

# Test 2 — Specific technical questions  
print("\n--- Test 2: Technical depth ---")
chat("What is LoRA and how does it reduce trainable parameters?")
chat("How are sentence embeddings used for semantic search?")

# Test 3 — Multi-turn conversation
print("\n--- Test 3: Conversation memory ---")
chat("What is parameter efficient fine-tuning?", session_id="conv_test")
chat("What are the main approaches to it?", session_id="conv_test")
chat("Which one did you mention first?", session_id="conv_test")

# ============================================================
# Step 7 — Latency benchmark
# ============================================================
print("\n--- Latency Benchmark (5 queries) ---")
import statistics
latencies = []
benchmark_queries = [
    "What is attention mechanism?",
    "How does BERT work?",
    "What is RAG?",
    "What is LoRA?",
    "How do embeddings work?",
]

for q in benchmark_queries:
    start = time.time()
    chain_with_history.invoke(
        {"input": q},
        config={"configurable": {"session_id": "benchmark"}}
    )
    latency = (time.time() - start) * 1000
    latencies.append(latency)
    print(f"  '{q[:30]}' → {latency:.0f}ms")

print(f"\nMean latency:   {statistics.mean(latencies):.0f}ms")
print(f"Median latency: {statistics.median(latencies):.0f}ms")
print(f"Max latency:    {max(latencies):.0f}ms")
print(f"Min latency:    {min(latencies):.0f}ms")
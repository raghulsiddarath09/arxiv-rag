import os
import time
import arxiv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_community.chat_message_histories import ChatMessageHistory
import warnings
warnings.filterwarnings("ignore")
# ============================================================
# Step 1 — Load existing vector store OR build new one
# ============================================================
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

if os.path.exists("faiss_index"):
    print("Loading existing vector store from disk...")
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"Loaded {vectorstore.index.ntotal} vectors from disk")
else:
    print("No existing index found. Building new one...")
    
    # Fetch papers
    def fetch_papers(query, max_results=8):
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        papers = []
        for result in client.results(search):
            papers.append({
                "title": result.title,
                "abstract": result.summary,
                "authors": [a.name for a in result.authors[:3]],
                "id": result.entry_id,
                "published": str(result.published.year)
            })
            time.sleep(0.5)
        return papers

    queries = [
        "transformer attention mechanism",
        "BERT language model pretraining",
        "retrieval augmented generation RAG",
        "large language models GPT",
        "fine-tuning neural networks LoRA",
        "vector embeddings semantic search",
        "computer vision CNN image classification",
        "reinforcement learning policy gradient"
    ]

    all_papers = []
    for query in queries:
        papers = fetch_papers(query, max_results=8)
        all_papers.extend(papers)
        print(f"Fetched {len(papers)} papers for '{query}'")
        time.sleep(3)

    seen_ids = set()
    unique_papers = []
    for paper in all_papers:
        if paper['id'] not in seen_ids:
            unique_papers.append(paper)
            seen_ids.add(paper['id'])

    print(f"Total unique papers: {len(unique_papers)}")

    documents = []
    for paper in unique_papers:
        content = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
        doc = Document(
            page_content=content,
            metadata={
                "title": paper['title'],
                "authors": ", ".join(paper['authors']),
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
    print(f"Total chunks: {len(chunks)}")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    print(f"Built and saved vector store: {vectorstore.index.ntotal} vectors")

# ============================================================
# Step 2 — Initialize LLM and retriever
# ============================================================
llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=500
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# ============================================================
# Step 3 — Build conversation aware prompt
# ============================================================
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given a chat history and the latest user question,
formulate a standalone question that captures the full context.
Do NOT answer the question. Just reformulate it if needed.
If no reformulation needed return it as is."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert ML research assistant.
Answer questions using ONLY the context provided below.
Always cite which paper your answer comes from.
If the answer is not in context say 'I don't have enough information.'
Keep answers concise and accurate.

Context:
{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# ============================================================
# Step 4 — Build conversation chain
# ============================================================
def format_docs(docs):
    formatted = ""
    for i, doc in enumerate(docs):
        formatted += f"\n[{i+1}] From '{doc.metadata['title']}' ({doc.metadata['year']}):\n"
        formatted += doc.page_content + "\n"
    return formatted

# Chain that reformulates question using history
contextualize_chain = contextualize_prompt | llm | StrOutputParser()

def contextualized_retriever(input_dict):
    if input_dict.get("chat_history"):
        new_question = contextualize_chain.invoke(input_dict)
        return retriever.invoke(new_question)
    return retriever.invoke(input_dict["input"])

# Full RAG chain with memory
rag_chain = (
    RunnablePassthrough.assign(
        context=RunnableLambda(contextualized_retriever) | RunnableLambda(format_docs)
    )
    | qa_prompt
    | llm
    | StrOutputParser()
)

# ============================================================
# Step 5 — Session management
# ============================================================
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
# Step 6 — Test conversation memory
# ============================================================
def chat(question, session_id="user_1"):
    print(f"\nYou: {question}")
    
    response = chain_with_history.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}}
    )
    
    print(f"Assistant: {response}")
    
    # Show sources
    docs = retriever.invoke(question)
    seen = set()
    sources = []
    for doc in docs:
        title = doc.metadata['title']
        if title not in seen:
            sources.append(f"{title} ({doc.metadata['year']})")
            seen.add(title)
    print(f"Sources: {', '.join(sources)}")
    
    return response

# Test multi-turn conversation
print("="*60)
print("CONVERSATION MEMORY TEST — Day 7")
print("="*60)

# Conversation 1 — Multi-turn about transformers
print("\n--- Conversation 1: Multi-turn about Transformers ---")
chat("What is the transformer architecture?")
chat("How is it different from RNNs?")
chat("What did you just tell me about transformers?")  # Tests memory

# Conversation 2 — Different session, fresh memory
print("\n--- Conversation 2: New session about BERT ---")
chat("What is BERT?", session_id="user_2")
chat("How does it use the transformer you mentioned?", session_id="user_2")

# Show memory contents
print("\n--- Memory Contents ---")
print(f"Session user_1 messages: {len(store['user_1'].messages)}")
print(f"Session user_2 messages: {len(store['user_2'].messages)}")
print("\nuser_1 history:")
for msg in store['user_1'].messages:
    role = "You" if isinstance(msg, HumanMessage) else "Assistant"
    print(f"  {role}: {msg.content[:100]}...")
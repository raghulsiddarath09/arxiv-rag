import os
import time
import arxiv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ============================================================
# Step 1 — Fetch real ArXiv papers
# ============================================================
def fetch_arxiv_papers(query, max_results=10):
    print(f"Fetching papers for: '{query}'")
    
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

# Fetch papers
queries = [
    "transformer attention mechanism",
    "BERT language model",
    "retrieval augmented generation",
    "large language models",
    "fine-tuning neural networks"
]

all_papers = []
for query in queries:
    papers = fetch_arxiv_papers(query, max_results=5)
    all_papers.extend(papers)
    print(f"Fetched {len(papers)} papers for '{query}'")
    time.sleep(3)

# Remove duplicates
seen_ids = set()
unique_papers = []
for paper in all_papers:
    if paper['id'] not in seen_ids:
        unique_papers.append(paper)
        seen_ids.add(paper['id'])

print(f"\nTotal unique papers: {len(unique_papers)}")

# ============================================================
# Step 2 — Prepare LangChain documents
# ============================================================
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

print(f"Created {len(documents)} documents")

# ============================================================
# Step 3 — Chunk documents
# ============================================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = text_splitter.split_documents(documents)
print(f"Total chunks: {len(chunks)}")
print(f"Average chunks per paper: {len(chunks)/len(documents):.1f}")
print(f"\nSample chunk:")
print(f"Content: {chunks[0].page_content[:200]}...")
print(f"Metadata: {chunks[0].metadata}")

# ============================================================
# Step 4 — Build FAISS vector store
# ============================================================
print("\nBuilding vector store...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

vectorstore = FAISS.from_documents(chunks, embeddings)
print(f"Vector store built: {vectorstore.index.ntotal} vectors")

vectorstore.save_local("faiss_index")
print("Saved to faiss_index/")

# ============================================================
# Step 5 — Build RAG chain using new LangChain syntax
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

prompt = PromptTemplate.from_template("""You are an expert ML research assistant.
Answer the question using ONLY the context provided below.
Always cite which paper your answer comes from.
If the answer is not in the context say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:""")

def format_docs(docs):
    formatted = ""
    for i, doc in enumerate(docs):
        formatted += f"\n[{i+1}] From '{doc.metadata['title']}' ({doc.metadata['year']}):\n"
        formatted += doc.page_content + "\n"
    return formatted

# New LangChain LCEL chain
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ============================================================
# Step 6 — Test upgraded pipeline
# ============================================================
questions = [
    "How does the attention mechanism work in transformers?",
    "What makes BERT different from other language models?",
    "How does RAG improve language model accuracy?",
    "What are the key innovations in large language models?",
]

print("\n" + "="*60)
print("UPGRADED RAG SYSTEM — Day 6 (LangChain)")
print("="*60)

for question in questions:
    print(f"\nQuestion: {question}")
    print("-"*50)
    
    # Get answer
    answer = rag_chain.invoke(question)
    print(f"Answer: {answer}")
    
    # Get sources separately
    source_docs = retriever.invoke(question)
    print("\nSources:")
    seen = set()
    for doc in source_docs:
        title = doc.metadata['title']
        if title not in seen:
            print(f"  - {title} ({doc.metadata['year']})")
            seen.add(title)
    
    print("="*60)
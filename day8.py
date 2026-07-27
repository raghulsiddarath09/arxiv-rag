import os
import time
import json
import arxiv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
# Step 1 — Targeted paper fetching for ML domain coverage
# ============================================================
ML_QUERIES = [
    # Transformers and Attention
    "attention is all you need transformer architecture vaswani",
    "BERT bidirectional encoder representations transformers devlin",
    "GPT language model pretraining radford",
    
    # RAG and Retrieval
    "retrieval augmented generation knowledge intensive NLP lewis",
    "dense passage retrieval open domain question answering",
    "semantic search dense retrieval neural",
    
    # LLMs and Scaling
    "large language models few shot learners GPT-3 brown",
    "LLaMA open efficient foundation language models",
    "scaling laws neural language models",
    
    # Fine-tuning and Efficiency
    "LoRA low rank adaptation large language models",
    "parameter efficient fine-tuning PEFT",
    "instruction tuning FLAN language model",
    
    # Embeddings and Vector Search
    "sentence transformers semantic textual similarity",
    "dense retrieval FAISS approximate nearest neighbor",
    "text embeddings contrastive learning",
    
    # Evaluation
    "ROUGE BLEU evaluation text generation",
    "recall precision information retrieval evaluation",
    "benchmark natural language processing tasks GLUE",
]

def fetch_papers(query, max_results=6):
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    papers = []
    try:
        for result in client.results(search):
            papers.append({
                "title": result.title,
                "abstract": result.summary,
                "authors": [a.name for a in result.authors[:3]],
                "id": result.entry_id,
                "published": str(result.published.year),
                "query_topic": query[:50]
            })
            time.sleep(0.3)
    except Exception as e:
        print(f"Error fetching '{query}': {e}")
    return papers

# ============================================================
# Step 2 — Fetch all papers with progress tracking
# ============================================================
print("Fetching papers from ArXiv...")
print(f"Queries: {len(ML_QUERIES)}, Max per query: 6")
print(f"Target: ~{len(ML_QUERIES) * 6} papers before deduplication\n")

all_papers = []
for i, query in enumerate(ML_QUERIES):
    papers = fetch_papers(query, max_results=6)
    all_papers.extend(papers)
    print(f"[{i+1}/{len(ML_QUERIES)}] '{query[:45]}...' → {len(papers)} papers")
    time.sleep(3)

# Deduplicate
seen_ids = set()
unique_papers = []
for paper in all_papers:
    if paper['id'] not in seen_ids:
        unique_papers.append(paper)
        seen_ids.add(paper['id'])

print(f"\nTotal fetched: {len(all_papers)}")
print(f"After deduplication: {len(unique_papers)} unique papers")

# Save papers to disk so we never re-fetch
with open("papers_cache.json", "w") as f:
    json.dump(unique_papers, f, indent=2)
print("Papers cached to papers_cache.json")

# ============================================================
# Step 3 — Build documents and chunks
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
            "year": paper['published'],
            "topic": paper['query_topic']
        }
    )
    documents.append(doc)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = text_splitter.split_documents(documents)

print(f"\nDocuments: {len(documents)}")
print(f"Chunks: {len(chunks)}")
print(f"Average chunks per paper: {len(chunks)/len(documents):.1f}")

# ============================================================
# Step 4 — Build new FAISS index
# ============================================================
print("\nBuilding vector store...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

vectorstore = FAISS.from_documents(chunks, embeddings)

# Save — overwrites old 111-vector index
vectorstore.save_local("faiss_index")
print(f"Vector store saved: {vectorstore.index.ntotal} vectors")

# ============================================================
# Step 5 — Quick retrieval quality test
# ============================================================
print("\n" + "="*60)
print("RETRIEVAL QUALITY TEST")
print("="*60)

test_queries = [
    "How does the attention mechanism work in transformers?",
    "What is RAG and how does it improve language models?",
    "What is LoRA and why is it parameter efficient?",
    "How are sentence embeddings created?",
    "What is BERT and how is it pretrained?",
]

for query in test_queries:
    docs = vectorstore.similarity_search(query, k=3)
    print(f"\nQuery: {query}")
    for i, doc in enumerate(docs):
        print(f"  {i+1}. {doc.metadata['title'][:60]} ({doc.metadata['year']})")
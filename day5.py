import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# ============================================================
# Step 1 — Papers and chunking
# ============================================================
papers = [
    {
        "title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. The Transformer allows for significantly more parallelization and reaches state of the art on translation tasks.",
        "id": "arxiv:1706.03762"
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "abstract": "We introduce BERT, which stands for Bidirectional Encoder Representations from Transformers. BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. The pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks. BERT obtains new state-of-the-art results on eleven natural language processing tasks.",
        "id": "arxiv:1810.04805"
    },
    {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "abstract": "Large pre-trained language models store factual knowledge in their parameters and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. However their ability to access and manipulate knowledge is still limited. We explore retrieval-augmented generation RAG models which combine parametric and non-parametric memory for language generation. RAG models retrieve documents with a neural retriever then pass them to a seq2seq model. RAG models achieve state of the art on knowledge intensive NLP tasks.",
        "id": "arxiv:2005.11401"
    },
    {
        "title": "Language Models are Few-Shot Learners",
        "abstract": "We demonstrate that scaling language models greatly improves task-agnostic few-shot performance. GPT-3 has 175 billion parameters and achieves strong performance on many NLP benchmarks. For all tasks GPT-3 is applied without any gradient updates or fine-tuning. GPT-3 achieves strong performance on translation, question-answering, and cloze tasks. It also shows few-shot learning abilities on tasks requiring on-the-fly reasoning or domain adaptation.",
        "id": "arxiv:2005.14165"
    },
    {
        "title": "LLaMA: Open and Efficient Foundation Language Models",
        "abstract": "We introduce LLaMA a collection of foundation language models ranging from 7B to 65B parameters. We train our models on trillions of tokens using publicly available datasets exclusively. LLaMA-13B outperforms GPT-3 on most benchmarks despite being 10x smaller. LLaMA-65B is competitive with Chinchilla and PaLM. We release all our models to the research community to foster research and help democratize access to large language models.",
        "id": "arxiv:2302.13971"
    }
]

def sentence_chunking(text, sentences_per_chunk=3, overlap=1):
    sentences = []
    for sentence in text.replace('\n', ' ').split('. '):
        sentence = sentence.strip()
        if len(sentence) > 20:
            sentences.append(sentence)
    chunks = []
    step = sentences_per_chunk - overlap
    for i in range(0, len(sentences), step):
        chunk_sentences = sentences[i:i + sentences_per_chunk]
        if chunk_sentences:
            chunks.append('. '.join(chunk_sentences))
    return chunks

# Build chunks and metadata
all_chunks = []
chunk_metadata = []
for paper in papers:
    full_text = f"{paper['title']}. {paper['abstract']}"
    chunks = sentence_chunking(full_text)
    for chunk in chunks:
        all_chunks.append(chunk)
        chunk_metadata.append({
            "paper_title": paper["title"],
            "paper_id": paper["id"],
            "chunk_text": chunk
        })

# Build FAISS index
print("Building knowledge base...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(all_chunks, show_progress_bar=False)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype(np.float32))
print(f"Knowledge base ready: {index.ntotal} chunks indexed\n")

# ============================================================
# Step 2 — Retrieval
# ============================================================
def retrieve(query, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(
        query_embedding.astype(np.float32), k=k
    )
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            "chunk": chunk_metadata[idx]["chunk_text"],
            "paper": chunk_metadata[idx]["paper_title"],
            "paper_id": chunk_metadata[idx]["paper_id"],
            "distance": dist
        })
    return results

# ============================================================
# Step 3 — Build context string
# ============================================================
def build_context(retrieved_chunks):
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += f"\n[{i+1}] From '{chunk['paper']}':\n"
        context += f"{chunk['chunk']}\n"
    return context

# ============================================================
# Step 4 — Generate answer using Groq
# ============================================================
def generate_answer(context, query):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY not set"

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """You are an expert ML research assistant.
You will be given context from research papers and a question.
Answer the question based on the context provided.
Always cite which paper your answer comes from.
Be specific and detailed in your answer."""
            },
            {
                "role": "user",
                "content": f"Context from research papers:\n{context}\n\nQuestion: {query}\n\nAnswer based on the context above:"
            }
        ],
        max_tokens=500,
        temperature=0.1
    )

    return response.choices[0].message.content

# ============================================================
# Step 5 — Complete RAG pipeline
# ============================================================
def rag_query(question):
    print(f"Question: {question}")
    print("-" * 50)

    # Retrieve relevant chunks
    retrieved = retrieve(question, k=3)

    print("Retrieved chunks:")
    for i, r in enumerate(retrieved):
        print(f"  {i+1}. {r['paper']} (distance: {r['distance']:.4f})")

    # Build context
    context = build_context(retrieved)

    # Generate answer
    print("\nGenerating answer...")
    answer = generate_answer(context, question)

    print(f"\nAnswer: {answer}")

    # Show citations
    print("\nSources:")
    seen = set()
    for r in retrieved:
        if r['paper_id'] not in seen:
            print(f"  - {r['paper']} ({r['paper_id']})")
            seen.add(r['paper_id'])

    print("=" * 60)

# ============================================================
# Step 6 — Test
# ============================================================
questions = [
    "How does the attention mechanism work in transformers?",
    "What makes BERT different from GPT?",
    "How does RAG improve language model accuracy?",
]

print("=" * 60)
print("COMPLETE RAG SYSTEM — Day 5")
print("=" * 60 + "\n")

for question in questions:
    rag_query(question)
    print()
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Real ArXiv papers from Day 2
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

# ============================================================
# Step 1 — Sentence Chunking with Overlap from Day 3
# ============================================================
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
        if len(chunk_sentences) > 0:
            chunk = '. '.join(chunk_sentences)
            chunks.append(chunk)
    
    return chunks

# ============================================================
# Step 2 — Chunk all papers and track metadata
# ============================================================
all_chunks = []
chunk_metadata = []  # track which paper each chunk came from

for paper in papers:
    # Combine title and abstract
    full_text = f"{paper['title']}. {paper['abstract']}"
    
    # Chunk it
    chunks = sentence_chunking(full_text, sentences_per_chunk=3, overlap=1)
    
    for chunk in chunks:
        all_chunks.append(chunk)
        chunk_metadata.append({
            "paper_title": paper["title"],
            "paper_id": paper["id"],
            "chunk_text": chunk
        })

print(f"Total papers: {len(papers)}")
print(f"Total chunks: {len(all_chunks)}")
print(f"Average chunks per paper: {len(all_chunks)/len(papers):.1f}")

# ============================================================
# Step 3 — Embed all chunks
# ============================================================
print("\nLoading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Embedding all chunks...")
embeddings = model.encode(all_chunks, show_progress_bar=True)
print(f"Embedding shape: {embeddings.shape}")

# ============================================================
# Step 4 — Build FAISS index with chunks
# ============================================================
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype(np.float32))
print(f"\nFAISS index built with {index.ntotal} chunk vectors")

# ============================================================
# Step 5 — Query and return actual chunk text
# ============================================================
def search(query, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(
        query_embedding.astype(np.float32), k=k
    )
    
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            "chunk": chunk_metadata[idx]["chunk_text"],
            "paper": chunk_metadata[idx]["paper_title"],
            "distance": dist
        })
    return results

# ============================================================
# Step 6 — Test with real queries
# ============================================================
queries = [
    "how does attention mechanism work in transformers?",
    "what makes BERT different from other language models?",
    "how does RAG combine retrieval with generation?",
]

print("\n" + "="*60)
for query in queries:
    print(f"\nQuery: {query}")
    results = search(query, k=3)
    
    for i, result in enumerate(results):
        print(f"\nResult {i+1} from: {result['paper']}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Chunk: {result['chunk'][:200]}...")
    print("="*60)
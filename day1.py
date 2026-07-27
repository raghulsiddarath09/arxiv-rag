from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Your documents
docs = [
    "Attention mechanisms allow models to focus on relevant parts of input",
    "BERT uses bidirectional transformers for language understanding",
    "RAG combines retrieval with generation for better factual accuracy",
    "Fine-tuning adapts pretrained models to specific downstream tasks",
    "Vector embeddings represent semantic meaning in high dimensional space"
]

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert documents to embeddings
embeddings = model.encode(docs)
print(f"Embedding shape: {embeddings.shape}")

# Store in FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype(np.float32))
print(f"Total vectors in index: {index.ntotal}")

# Query
query = "how do transformers understand language?"
query_embedding = model.encode([query])
distances, indices = index.search(query_embedding.astype(np.float32), k=3)

print(f"\nQuery: {query}")
print("Top 3 results:")
for i, idx in enumerate(indices[0]):
    print(f"{i+1}. {docs[idx]} (distance: {distances[0][i]:.4f})")
# Day 3 — Chunking Strategies
# We'll compare 3 different chunking approaches

# Sample long text — simulating a real paper section
sample_text = """
The Transformer model introduced a new architecture based entirely on attention mechanisms. 
Unlike previous models that relied on recurrent neural networks, the Transformer processes 
all tokens simultaneously using self-attention. This parallel processing makes training 
significantly faster than sequential RNN-based approaches.

The core innovation is the multi-head attention mechanism. Instead of computing a single 
attention function, the model runs attention multiple times in parallel with different 
learned projections. Each attention head can focus on different aspects of the input, 
such as syntactic relationships or semantic similarities between words.

BERT extended the transformer architecture by introducing bidirectional pre-training. 
Traditional language models only process text left-to-right or right-to-left. BERT 
processes text in both directions simultaneously using a masked language modeling objective. 
This bidirectional context leads to much richer representations.

Retrieval Augmented Generation combines a retrieval system with a language model. 
When given a query, the system first retrieves relevant documents from a knowledge base. 
These retrieved documents are then provided as context to the language model, which 
generates an answer grounded in the retrieved information rather than relying solely 
on its parametric knowledge.

Large language models like GPT-3 demonstrated that scaling model size leads to 
emergent capabilities. With 175 billion parameters, GPT-3 showed strong few-shot 
learning abilities across diverse tasks without any task-specific fine-tuning. 
This sparked the current era of foundation models.
"""

# ============================================================
# STRATEGY 1: Fixed Size Chunking
# ============================================================
def fixed_size_chunking(text, chunk_size=200, overlap=50):
    """
    Split text into chunks of fixed character size with overlap.
    
    chunk_size: how many characters per chunk
    overlap: how many characters to repeat between chunks
    
    WHY OVERLAP?
    Without overlap:
    Chunk 1: "...The transformer uses attention"
    Chunk 2: "mechanisms to process sequences..."
    → Important sentence split across two chunks
    → Neither chunk has complete thought
    
    With overlap:
    Chunk 1: "...The transformer uses attention mechanisms"
    Chunk 2: "attention mechanisms to process sequences..."
    → Both chunks contain the complete idea
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # Move forward by chunk_size minus overlap
        start += chunk_size - overlap
    
    return chunks

# ============================================================
# STRATEGY 2: Sentence Based Chunking
# ============================================================
def sentence_chunking(text, sentences_per_chunk=3):
    """
    Split text into chunks based on sentences.
    
    WHY SENTENCES?
    Sentences are natural units of meaning.
    Each sentence is a complete thought.
    Better semantic boundaries than fixed characters.
    """
    # Split by period followed by space or newline
    sentences = []
    for sentence in text.replace('\n', ' ').split('. '):
        sentence = sentence.strip()
        if len(sentence) > 20:  # ignore very short fragments
            sentences.append(sentence)
    
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = '. '.join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)
    
    return chunks

# ============================================================
# STRATEGY 3: Paragraph Based Chunking
# ============================================================
def paragraph_chunking(text):
    """
    Split text by paragraphs.
    
    WHY PARAGRAPHS?
    Each paragraph is a complete idea in academic writing.
    Natural semantic boundary.
    Preserves full context of each concept.
    """
    paragraphs = []
    for para in text.split('\n\n'):
        para = para.strip()
        if len(para) > 50:  # ignore empty or tiny paragraphs
            paragraphs.append(para)
    
    return paragraphs

# ============================================================
# Compare all three strategies
# ============================================================
print("="*60)
print("STRATEGY 1: Fixed Size Chunking (200 chars, 50 overlap)")
print("="*60)
fixed_chunks = fixed_size_chunking(sample_text, chunk_size=200, overlap=50)
print(f"Number of chunks: {len(fixed_chunks)}")
for i, chunk in enumerate(fixed_chunks[:3]):  # show first 3
    print(f"\nChunk {i+1} ({len(chunk)} chars):")
    print(chunk)
    print("-"*40)

print("\n" + "="*60)
print("STRATEGY 2: Sentence Based Chunking (3 sentences each)")
print("="*60)
sentence_chunks = sentence_chunking(sample_text, sentences_per_chunk=3)
print(f"Number of chunks: {len(sentence_chunks)}")
for i, chunk in enumerate(sentence_chunks[:3]):
    print(f"\nChunk {i+1} ({len(chunk)} chars):")
    print(chunk)
    print("-"*40)

print("\n" + "="*60)
print("STRATEGY 3: Paragraph Based Chunking")
print("="*60)
paragraph_chunks = paragraph_chunking(sample_text)
print(f"Number of chunks: {len(paragraph_chunks)}")
for i, chunk in enumerate(paragraph_chunks[:3]):
    print(f"\nChunk {i+1} ({len(chunk)} chars):")
    print(chunk)
    print("-"*40)

# ============================================================
# Summary comparison
# ============================================================
print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)
print(f"Fixed size:  {len(fixed_chunks)} chunks, "
      f"avg size: {sum(len(c) for c in fixed_chunks)//len(fixed_chunks)} chars")
print(f"Sentence:    {len(sentence_chunks)} chunks, "
      f"avg size: {sum(len(c) for c in sentence_chunks)//len(sentence_chunks)} chars")
print(f"Paragraph:   {len(paragraph_chunks)} chunks, "
      f"avg size: {sum(len(c) for c in paragraph_chunks)//len(paragraph_chunks)} chars")
# Real ArXiv papers - hardcoded to avoid rate limiting
# We'll add live fetching once we have the full pipeline working

papers = [
    {
        "title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
        "id": "arxiv:1706.03762"
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.",
        "id": "arxiv:1810.04805"
    },
    {
        "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        "abstract": "While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. We show that reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks.",
        "id": "arxiv:2010.11929"
    },
    {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "abstract": "Large pre-trained language models have been shown to store factual knowledge in their parameters, and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. However, their ability to access and precisely manipulate knowledge is still limited. We explore a general-purpose fine-tuning recipe for retrieval-augmented generation, combining parametric and non-parametric memory for language generation.",
        "id": "arxiv:2005.11401"
    },
    {
        "title": "Language Models are Few-Shot Learners",
        "abstract": "We demonstrate that scaling language models greatly improves task-agnostic few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. GPT-3 is autoregressive language model with 175 billion parameters that achieves strong performance on many NLP benchmarks.",
        "id": "arxiv:2005.14165"
    },
    {
        "title": "LLaMA: Open and Efficient Foundation Language Models",
        "abstract": "We introduce LLaMA, a collection of foundation language models ranging from 7B to 65B parameters. We train our models on trillions of tokens, and show that it is possible to train state-of-the-art models using publicly available datasets exclusively. LLaMA-13B outperforms GPT-3 on most benchmarks.",
        "id": "arxiv:2302.13971"
    },
    {
        "title": "Deep Residual Learning for Image Recognition",
        "abstract": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions.",
        "id": "arxiv:1512.03385"
    },
    {
        "title": "Improving Language Understanding by Generative Pre-Training",
        "abstract": "Natural language understanding comprises a wide range of diverse tasks such as textual entailment, question answering, semantic similarity assessment, and document classification. We demonstrate that large gains on these tasks can be realized by generative pre-training of a language model on a diverse corpus of unlabeled text.",
        "id": "arxiv:2304.01852"
    }
]

print(f"Loaded {len(papers)} papers\n")
for i, paper in enumerate(papers):
    print(f"Paper {i+1}: {paper['title']}")
    print(f"Abstract preview: {paper['abstract'][:100]}...")
    print()
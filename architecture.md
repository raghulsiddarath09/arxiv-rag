## Architecture

```mermaid
flowchart TD
    U([User]) -->|POST /query| API[FastAPI<br/>main.py]

    API --> CK{Redis cache?<br/>SHA-256 of<br/>normalised question}
    CK -->|hit| CACHED[Return cached answer<br/>~5 ms]
    CACHED --> U

    CK -->|miss| HIST{Chat history<br/>present?}
    HIST -->|yes| REW[Rewrite query<br/>Groq contextualize]
    HIST -->|no| ROUTER
    REW --> ROUTER

    ROUTER{{"max IDF of query tokens<br/>≥ 8.0 ?"}}

    ROUTER -->|"no — common vocabulary<br/>e.g. 'how do models adapt' (5.00)"| DENSE
    ROUTER -->|"yes — rare term<br/>e.g. 'AlphaD3M' (8.02)"| HYB

    subgraph DENSE [dense path · ~35 ms · 57% of queries]
        D1[FAISS similarity search<br/>k = 5]
    end

    subgraph HYB [hybrid path · ~840 ms · 43% of queries]
        H1[FAISS<br/>k = 15]
        H2[BM25Okapi<br/>k = 15]
        H1 --> H3[Merge + dedup]
        H2 --> H3
        H3 --> H4[Cross-encoder rerank<br/>ms-marco-MiniLM-L-6-v2]
        H4 --> H5[Top 5]
    end

    D1 --> CTX[Format context<br/>+ citations]
    H5 --> CTX

    CTX --> LLM[Groq<br/>llama-3.1-8b-instant]
    LLM --> STORE[Write to Redis<br/>TTL 24h]
    STORE --> RESP[answer + sources<br/>+ latency + cached flag]
    RESP --> U

    style ROUTER fill:#2d3748,stroke:#63b3ed,stroke-width:2px,color:#fff
    style CK fill:#2d3748,stroke:#f6ad55,stroke-width:2px,color:#fff
    style CACHED fill:#22543d,stroke:#68d391,color:#fff
    style DENSE fill:#1a365d,stroke:#63b3ed,color:#fff
    style HYB fill:#44337a,stroke:#b794f4,color:#fff
```

### Why the router exists

Retrieval strategy is query-dependent — neither approach dominates:

| eval set | dense | hybrid + rerank |
|---|---:|---:|
| paraphrase queries (n=42) | **82.1%** | 73.8% |
| rare-term queries (n=30) | 76.7% | **93.3%** |

Dense embeddings compress a chunk into a single vector, smoothing away
low-frequency tokens that carry little semantic weight but high discriminative
power. Exact term matching does not smooth. Max-IDF over query tokens separates
the two cases at negligible cost — one tokenize plus dictionary lookups against
the BM25 vocabulary that already exists in memory.

### Infrastructure

```mermaid
flowchart LR
    subgraph S3 [AWS S3]
        IDX[(faiss_index/<br/>8.8 MB)]
        PAP[(papers_cache.json<br/>1,100 papers)]
    end

    subgraph EC2 [EC2 t3.micro · 1 GB RAM + 2 GB swap]
        subgraph DC [docker-compose]
            FE[arxiv-frontend<br/>nginx · :80]
            BE[arxiv-backend<br/>FastAPI · :8000]
            RD[arxiv-redis<br/>internal only]
        end
        DISK[(local disk<br/>mounted read-only)]
    end

    GH[GitHub Actions] -->|"push to main<br/>33 tests → SSH → rebuild"| EC2
    S3 -->|"provisioning only<br/>not per query"| DISK
    DISK -->|volume mount| BE
    FE --> BE
    BE --> RD
    BE -->|HTTPS| GROQ[Groq API]

    style S3 fill:#2d3748,stroke:#f6ad55,color:#fff
    style EC2 fill:#1a365d,stroke:#63b3ed,color:#fff
    style GH fill:#22543d,stroke:#68d391,color:#fff
```

S3 holds the durable copy of the index; EC2 pulls it to local disk once at
provisioning and serves every query from memory. A FAISS search is single-digit
milliseconds against an in-memory index versus 50–200 ms for an S3 GET, and
FAISS memory-maps from a real file path, which object storage cannot provide.

Redis is not published to the host — the backend reaches it by service name on
the Docker network. Publishing 6379 had exposed an unauthenticated Redis to the
public internet.

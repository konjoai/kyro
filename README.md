# Kyro

Production RAG pipeline with hybrid retrieval, reranking, and RAGAS evaluation.
No vendor lock-in — plug in OpenAI, Anthropic, or a local [Squish](https://github.com/squishai/squish) server.

## Architecture

```
Documents (PDF/MD/HTML/code)
        │
        ▼
    Ingest & Chunk      RecursiveChunker | SentenceWindowChunker
        │
        ▼
    Embed               sentence-transformers → float32 (384–1536d)
        │
        ▼
    Qdrant Store        cosine similarity index
        │
    ┌───┴───┐
 Dense   Sparse         HNSW search + BM25 (rank-bm25)
    └───┬───┘
        │  Reciprocal Rank Fusion (α=0.7)
        ▼
    Rerank              cross-encoder/ms-marco-MiniLM-L-6-v2
        │
        ▼
    Generate            OpenAI | Anthropic | Squish
        │
        ▼
    Evaluate            RAGAS: faithfulness / relevancy / precision / recall
```

## Quickstart

```bash
git clone https://github.com/konjoai/kyro.git
cd kyro
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

cp .env.example .env
# edit .env — set OPENAI_API_KEY and QDRANT_URL

# Start Qdrant (Docker)
docker compose -f docker/docker-compose.yml up qdrant -d

# Ingest a directory
konjoai ingest docs/

# Ask a question
konjoai query "What is the main architecture?"

# Start the API server
konjoai serve
```

## CLI

```
konjoai ingest <path>     Ingest files/dirs into vector store
konjoai query  <question> Retrieve and answer using indexed documents
konjoai serve             Start FastAPI server (default :8000)
konjoai status            Show collection stats
```

## API

| Method | Path       | Description                        |
|--------|------------|------------------------------------|
| POST   | /ingest    | Ingest a file or directory         |
| POST   | /query     | RAG query with optional reranking  |
| POST   | /eval      | RAGAS evaluation over QA samples   |
| GET    | /health    | Collection health + document count |

Docs at `http://localhost:8000/docs` after `konjoai serve`.

## Configuration

All settings via `.env` (see `.env.example`):

| Variable            | Default                                    | Description                    |
|---------------------|--------------------------------------------|--------------------------------|
| `QDRANT_URL`        | `http://localhost:6333`                    | Qdrant instance URL            |
| `EMBED_MODEL`       | `sentence-transformers/all-MiniLM-L6-v2`  | HuggingFace embedding model    |
| `EMBED_DEVICE`      | `cpu`                                      | `mps` for Apple Silicon        |
| `CHUNK_STRATEGY`    | `recursive`                                | `recursive` \| `sentence_window` |
| `GENERATOR_BACKEND` | `openai`                                   | `openai` \| `anthropic` \| `squish` |
| `OPENAI_API_KEY`    | —                                          | Required for OpenAI backend    |
| `SQUISH_BASE_URL`   | `http://localhost:11434/v1`                | Local Squish/Ollama endpoint   |

## Evaluation

kyro ships RAGAS gates out of the box:

```bash
konjoai serve &
curl -s -X POST http://localhost:8000/eval \
  -H 'Content-Type: application/json' \
  -d '{"samples": [{"question": "...", "answer": "...", "contexts": ["..."], "ground_truth": "..."}]}'
```

Target benchmarks (Weeks 3–7 gate):
- `faithfulness` ≥ 0.75
- `answer_relevancy` ≥ 0.80

## License

MIT

# 🚀 Production AI Platform (RAG)

A production-ready Retrieval-Augmented Generation (RAG) platform built with FastAPI, LangChain, OpenAI, ChromaDB, and modern retrieval techniques.

This platform enables organisations to build AI assistants that answer questions using their own documents and websites instead of relying solely on an LLM's general knowledge.

---

## ✨ Features

- Hybrid Retrieval (Vector Search + BM25)
- Reciprocal Rank Fusion (RRF)
- Cross-Encoder Reranking
- Adaptive Retrieval
- Query Rewriting
- Multi-Query Retrieval
- Context Compression
- Conversation Memory
- Response Cache
- Groundedness Checking
- Website Crawling
- Incremental Indexing
- FastAPI REST API
- Docker Support
- Production-ready Architecture

---

## 🏗 Architecture

```
                User
                  │
                  ▼
            FastAPI API
                  │
                  ▼
          Query Processing
                  │
      ┌───────────┴───────────┐
      ▼                       ▼
Vector Search            BM25 Search
      │                       │
      └───────────┬───────────┘
                  ▼
       Reciprocal Rank Fusion
                  ▼
      Cross-Encoder Reranker
                  ▼
      Context Compression
                  ▼
          OpenAI GPT Model
                  ▼
              Final Answer
```

---

## 🛠 Tech Stack

### Backend

- Python 3.12
- FastAPI
- LangChain
- OpenAI API

### Retrieval

- ChromaDB
- BM25
- Reciprocal Rank Fusion
- Cross-Encoder Reranker

### AI

- GPT-5 Mini
- OpenAI Embeddings
- Sentence Transformers

### Deployment

- Docker
- Docker Compose
- Oracle Cloud VM

---

## 📂 Project Structure

```
Production-AI-Platform/rag/
├── api/
│   ├── __init__.py
│   ├── dependencies.py
│   ├── main.py
│   ├── routes.py
│   └── schemas.py
├── data/
│   └── docs/
│       ├── python_basics.md
│       ├── python_decorators.md
│       ├── python_functions.md
│       ├── python_intro.md
│       └── python_oop.md
├── evaluation/
│   ├── results/
│   │   └── report.json
│   ├── __init__.py
│   ├── dataset.json
│   ├── evaluate.py
│   ├── metrics.py
│   └── report.py
├── rag/                               # core retrieval module
│   ├── __init__.py
│   ├── adaptive_retrieval.py
│   ├── application.py
│   ├── bm25.py
│   ├── chain.py
│   ├── cli.py
│   ├── config.py
│   ├── context_compressor.py
│   ├── crawler.py
│   ├── embeddings.py
│   ├── exceptions.py
│   ├── fusion.py
│   ├── groundedness_checker.py
│   ├── hybrid.py
│   ├── importer.py
│   ├── loader.py
│   ├── logger.py
│   ├── memory.py
│   ├── multi_query.py
│   ├── prompt.py
│   ├── query_rewriter.py
│   ├── reranker.py
│   ├── response_cache.py
│   ├── retriever.py
│   ├── source_formatter.py
│   ├── splitter.py
│   ├── utils.py
│   └── vector_store.py
├── tests/
│   ├── test_cache.py
│   ├── test_chain.py
│   ├── test_reranker.py
│   └── test_retriever.py
├── app.py                             # FastAPI entrypoint
├── Dockerfile
├── docker-compose.yml
├── LICENSE
├── PROJECT_PROGRESS.md
├── README.md
└── requirements.txt
```

---

## 🚀 Getting Started

### Clone

```bash
git clone https://github.com/YOUR_USERNAME/production-ai-platform.git

cd production-ai-platform
```

---

### Create Environment File

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
```

---

### Build Docker Image

```bash
docker build -t production-ai-platform .
```

---

### Run

```bash
docker run -d \
-p 8000:8000 \
--env-file .env \
production-ai-platform
```

---

## API

### Health Check

```
GET /health
```

### Chat

```
POST /chat
```

Example Request

```json
{
  "question": "What is polymorphism?"
}
```

---

## Example Workflow

1. Load documents
2. Build knowledge base
3. User asks a question
4. Hybrid retrieval finds relevant documents
5. Reranker improves ranking
6. GPT generates a grounded answer
7. Sources are returned

---

## Future Improvements

- Authentication
- Streaming responses
- PostgreSQL support
- Redis distributed cache
- Kubernetes deployment
- CI/CD pipeline
- Monitoring with Prometheus & Grafana

---

## License

This project is licensed under the MIT License.

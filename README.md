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
production-ai-platform/
│
├── api/
├── rag/
├── data/
├── docs/
├── scripts/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
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

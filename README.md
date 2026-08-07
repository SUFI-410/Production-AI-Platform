# Production AI Platform

A production-deployed Retrieval-Augmented Generation (RAG) platform built with FastAPI, LangChain, OpenAI, ChromaDB, hybrid retrieval, Cross-Encoder reranking, isolated conversation sessions, response caching, and a React frontend.

The platform answers questions from a controlled knowledge base instead of relying solely on an LLM's general knowledge. It combines lexical and semantic retrieval, query processing, reranking, conversational context, source attribution, abuse protection, automated validation, and production infrastructure into an end-to-end RAG application.

---

## Live Application

**Frontend**

https://www.buildwithsufyan.com

**Backend API**

https://api.buildwithsufyan.com

**Backend Repository**

https://github.com/SUFI-410/Production-AI-Platform

**Frontend Repository**

https://github.com/SUFI-410/Production-AI-Platform-Frontend

---

## Project Status

**Version:** 1.0.0

Production AI Platform v1 is deployed and operational.

The current release includes:

* production backend deployment
* production frontend deployment
* HTTPS
* hybrid retrieval
* Cross-Encoder reranking
* isolated conversation sessions
* context-aware follow-up handling
* source-backed answers
* deterministic unsupported-answer handling
* response caching
* Cloudflare Turnstile protection
* API rate limiting
* automated backend tests
* CI/CD deployment workflow

---

## Features

### Retrieval

* Hybrid retrieval using vector search and BM25
* ChromaDB vector store
* OpenAI embeddings
* Reciprocal Rank Fusion (RRF)
* Maximum Marginal Relevance (MMR)
* Adaptive retrieval with dynamic Top-K
* Query rewriting
* Multi-query retrieval
* Context compression
* Cross-Encoder reranking
* Configurable reranking threshold
* Source attribution
* Cross-Encoder relevance scores

### Conversation

* Per-session conversation memory
* Isolated histories between independent sessions
* Server-generated session IDs
* Session expiration using TTL
* Maximum session capacity
* Bounded conversation history
* Inactive-session eviction
* Concurrency protection
* Same-session request serialization
* Concurrent processing across independent sessions
* Context-aware follow-up handling
* Standalone-question detection

### Caching

* In-memory response cache
* Configurable cache TTL
* Session-aware caching
* Conversation-history-aware cache keys
* Cache bypass support
* Reuse of cached standalone questions
* Duplicate standalone cached exchanges prevented from unnecessarily filling conversation memory
* Cached standalone topic changes correctly update the latest conversation context

### Answer Quality

* Knowledge-base-constrained answering
* Source-backed responses
* Grounded / Not Grounded status
* Deterministic refusal detection
* Unsupported questions return no misleading sources
* Source relevance is kept separate from answer correctness
* Low-relevance retrieved documents are filtered before answer generation

### Content Processing

* Document loading
* Markdown knowledge-base support
* Website crawling
* Text splitting
* Incremental indexing
* Persistent Chroma vector storage

### API

* FastAPI REST API
* Pydantic request validation
* Health endpoint
* OpenAPI / Swagger documentation
* ReDoc documentation
* Environment-based CORS configuration
* Structured source metadata
* Session identifiers
* Cache status
* Groundedness status
* Request latency reporting

### Security

* Cloudflare Turnstile verification
* Turnstile verification before expensive RAG initialization
* Rate limiting on public chat requests
* Environment-based secret management
* Configurable CORS origins
* Docker application exposed through Nginx
* Backend service bound to the server's local interface
* HTTPS using Let's Encrypt
* Persistent firewall configuration
* Non-root Docker execution

### Deployment

* Docker
* Docker Compose
* Oracle Cloud Infrastructure
* Ubuntu server
* Nginx reverse proxy
* Let's Encrypt TLS
* GitHub Actions CI/CD
* Persistent Chroma storage
* Persistent Hugging Face model cache

---

## Architecture

```text
                           User
                             |
                             v
                    React Frontend
                             |
                             v
                  Cloudflare Turnstile
                             |
                             v
                       FastAPI API
                             |
                             v
                    Session Resolution
                             |
                             v
               Conversation Context Check
                             |
                 +-----------+-----------+
                 |                       |
                 v                       v
        Standalone Question       Contextual Question
                 |                       |
                 |                 Query Rewriting
                 |                       |
                 +-----------+-----------+
                             |
                             v
                  Multi-Query Generation
                             |
                             v
                    Adaptive Retrieval
                       (Dynamic Top-K)
                             |
                             v
                  Hybrid Retrieval Layer
                       /           \
                      v             v
               Vector Search     BM25 Search
                      \             /
                       \           /
                        v         v
                  Reciprocal Rank Fusion
                             |
                             v
                   Context Compression
                             |
                             v
                  Cross-Encoder Reranker
                             |
                             v
                 Relevance Thresholding
                             |
                             v
                     Prompt Assembly
                             |
                             v
                    OpenAI GPT Model
                             |
                             v
                Grounded Answer / Refusal
                             |
                             v
                Sources + Relevance Scores
                             |
                             v
                 Memory + Response Cache
                             |
                             v
                          Client
```

---

## Production Infrastructure

```text
                       Internet
                          |
                          v
                        HTTPS
                          |
                          v
                        Nginx
                          |
                          v
                    127.0.0.1:8000
                          |
                          v
                        Docker
                          |
                          v
                       FastAPI
                          |
             +------------+------------+
             |                         |
             v                         v
          ChromaDB             Hugging Face Cache
             |
             v
        Knowledge Base
```

The backend is deployed on an Oracle Cloud Ubuntu VM.

Nginx acts as the public reverse proxy while the FastAPI application remains behind the server's local interface.

---

## Technology Stack

### Backend

* Python 3.12
* FastAPI
* Uvicorn
* Pydantic
* HTTPX

### RAG / AI

* LangChain
* OpenAI
* GPT-5 Mini
* `text-embedding-3-small`
* Sentence Transformers
* `BAAI/bge-reranker-base`

### Retrieval

* ChromaDB
* BM25
* Reciprocal Rank Fusion
* Maximum Marginal Relevance
* Adaptive retrieval
* Multi-query retrieval
* Query rewriting
* Context compression
* Cross-Encoder reranking

### Frontend

The frontend is maintained in a separate repository:

https://github.com/SUFI-410/Production-AI-Platform-Frontend

Main frontend technologies include:

* React
* TypeScript
* Vite
* Tailwind CSS
* Zustand
* TanStack React Query
* Axios
* React Markdown
* Highlight.js

### Infrastructure

* Docker
* Docker Compose
* Oracle Cloud Infrastructure
* Ubuntu
* Nginx
* Let's Encrypt
* GitHub Actions
* Cloudflare Turnstile

---

## Project Structure

The main backend structure is:

```text
Production-AI-Platform/
|
├── .github/
│   └── workflows/
│
├── api/
│   ├── __init__.py
│   ├── dependencies.py
│   ├── main.py
│   ├── routes.py
│   ├── schemas.py
│   └── turnstile.py
│
├── data/
│   └── docs/
│       ├── python_basics.md
│       ├── python_decorators.md
│       ├── python_functions.md
│       ├── python_intro.md
│       └── python_oop.md
│
├── evaluation/
│   ├── __init__.py
│   ├── dataset.json
│   ├── evaluate.py
│   ├── metrics.py
│   └── report.py
│
├── rag/
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
│
├── tests/
│   ├── test_api_sessions.py
│   ├── test_application_sessions.py
│   ├── test_chain_groundedness.py
│   └── test_session_memory_store.py
│
├── .dockerignore
├── .env.example
├── .gitignore
├── app.py
├── clean_tree.py
├── docker-compose.yml
├── Dockerfile
├── LICENSE
├── PROJECT_PROGRESS.md
├── README.md
└── requirements.txt
```

`api/main.py` is the FastAPI application entrypoint.

`app.py` is the CLI entrypoint for local console-based interaction.

---

## Request Flow

A typical production chat request follows this process:

1. The frontend obtains a Cloudflare Turnstile verification token.
2. The frontend sends the question, Turnstile token, optional session ID, and cache preference to `POST /chat`.
3. The backend verifies the Turnstile token before initializing expensive RAG resources.
4. The API uses the supplied session ID or creates a new unique session ID.
5. The application loads the conversation history for that session.
6. The application determines whether the question requires conversational context.
7. Context-dependent questions can be rewritten using the relevant conversation history.
8. Multiple retrieval queries may be generated.
9. Adaptive retrieval determines the dynamic retrieval Top-K.
10. Vector and BM25 retrieval run against the knowledge base.
11. Reciprocal Rank Fusion combines the retrieval results.
12. Retrieved context is compressed.
13. The Cross-Encoder reranks the candidate documents.
14. Documents below the configured relevance threshold are removed.
15. The selected documents are assembled into the generation prompt.
16. The LLM generates an answer from the retrieved context.
17. Known unsupported-answer responses are treated as refusals.
18. Supporting sources are formatted for the API response.
19. Conversation memory is updated when appropriate.
20. Eligible responses are stored in the response cache.
21. The API returns the answer, sources, session ID, cache status, groundedness status, and request latency.

---

## Conversation Sessions

Conversation state is isolated using session IDs.

Independent users and conversations do not share one global conversation history.

The session-memory system supports:

* TTL-based expiration
* maximum session capacity
* bounded message history
* inactive-session eviction
* concurrency protection
* same-session request serialization
* concurrent requests across independent sessions

The API generates a new session ID when a client does not supply one. The frontend stores the returned identifier and sends it with subsequent messages belonging to the same conversation.

---

## Context-Aware Questions

The application distinguishes between standalone questions and questions that depend on previous messages.

A standalone question such as:

```text
What is inheritance in Python?
```

does not need unrelated conversation history to be understood.

A contextual question such as:

```text
Give me a simple example of it.
```

may require the latest relevant conversation history.

This distinction is also used when determining query rewriting and response-cache behavior.

---

## Response Caching

The application includes an in-memory response cache with configurable expiration.

Standalone questions can reuse cached responses without unnecessarily including unrelated conversation history in their cache identity.

For example:

```text
What is inheritance in Python?
```

can reuse an eligible standalone cached response.

A contextual question such as:

```text
Give me another example of it.
```

remains dependent on the session's conversation history.

Repeated identical standalone cached exchanges are prevented from unnecessarily filling conversation memory.

If the user changes topics and later returns to a previously cached standalone topic, that cached exchange can still update the latest conversation context so a following contextual question refers to the correct topic.

---

## Groundedness and Refusal Handling

The production answer path distinguishes between supported answers and known unsupported-answer refusals.

A supported question might produce:

```text
What is inheritance in Python?
```

with:

```text
Grounded

Sources:
1. python_oop.md
   Relevance 0.15
```

The displayed relevance value is the Cross-Encoder relevance score for the retrieved source.

It is **not an answer-confidence percentage**.

For an unsupported request such as:

```text
Who is Sufyan?
```

if the knowledge base does not contain relevant information, the application can return:

```text
I couldn't find any relevant information in the knowledge base to answer your question.
```

with:

```text
Not Grounded
No supporting sources
```

This prevents unsupported answers from being presented with misleading source attribution.

---

## Getting Started

### Prerequisites

You need:

* Git
* Python 3.12+
* an OpenAI API key
* Docker, if using the containerized setup

---

## Clone the Repository

```bash
git clone https://github.com/SUFI-410/Production-AI-Platform.git
cd Production-AI-Platform
```

---

## Environment Configuration

Copy the example environment file.

### Linux / macOS

```bash
cp .env.example .env
```

### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

Update the required values inside `.env`.

Example:

```env
OPENAI_API_KEY=your_openai_api_key
```

For the public API, Cloudflare Turnstile also requires:

```env
TURNSTILE_SECRET=your_cloudflare_turnstile_secret
```

Allowed frontend origins can be configured with:

```env
CORS_ORIGINS=http://localhost:5173
```

Use `.env.example` as the authoritative template for all supported environment settings.

Never commit the real `.env` file.

---

## Local Python Setup

Create a virtual environment:

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the API Locally

Start the FastAPI server:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Health endpoint:

```text
http://localhost:8000/health
```

Swagger UI:

```text
http://localhost:8000/docs
```

ReDoc:

```text
http://localhost:8000/redoc
```

---

## Docker

### Build

```bash
docker build -t production-ai-platform:latest .
```

### Run

```bash
docker run -d \
  --name rag-api \
  --env-file .env \
  -p 8000:8000 \
  production-ai-platform:latest
```

Check the API:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

---

## API

### Root

```http
GET /
```

Example response:

```json
{
  "message": "Production AI Platform API is running."
}
```

---

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

---

### Chat

```http
POST /chat
```

Request body:

```json
{
  "question": "What is inheritance in Python?",
  "turnstile_token": "cloudflare-turnstile-token",
  "session_id": "optional-session-id",
  "use_cache": true
}
```

### Request Fields

`question`

The user's question. It must contain between 1 and 4000 characters.

`turnstile_token`

A single-use Cloudflare Turnstile verification token.

`session_id`

Optional conversation identifier. If omitted, the API generates a unique session ID.

`use_cache`

Optional boolean controlling whether an eligible cached response may be used. The default is `true`.

### Example Response

```json
{
  "answer": "Inheritance allows a child class to obtain attributes and methods from a parent class.",
  "sources": [
    {
      "document": "python_oop.md",
      "score": 0.15,
      "metadata": {}
    }
  ],
  "session_id": "generated-session-id",
  "cached": false,
  "grounded": true,
  "latency_ms": 842.6
}
```

### Response Fields

`answer`

The generated answer.

`sources`

Supporting knowledge-base sources returned for the answer.

`session_id`

The resolved conversation session identifier.

`cached`

Indicates whether the response was served from the response cache.

`grounded`

Indicates whether the production answer path considers the response supported rather than an unsupported refusal.

`latency_ms`

End-to-end API processing time in milliseconds.

### Source Fields

`document`

Document filename or identifier.

`score`

Cross-Encoder relevance score for the source document.

`metadata`

Additional metadata associated with the source document.

---

## Knowledge Base

The demonstration knowledge base contains Python documentation covering subjects such as:

* Python fundamentals
* variables and basic syntax
* functions
* decorators
* object-oriented programming
* inheritance
* polymorphism

The RAG architecture is not limited to Python documentation and can be used with other document collections.

---

## Website Crawling

The backend includes website-crawling support for importing textual web content into the knowledge-base workflow.

Crawler requests use the application's configured HTTP user agent and request timeout.

---

## Incremental Indexing

The ingestion pipeline supports incremental indexing so the knowledge base can be updated without requiring the entire application architecture to be rebuilt.

The persistent Chroma database stores indexed vector data between application restarts.

---

## Evaluation

Evaluation code is maintained under:

```text
evaluation/
```

The evaluation package contains:

* a predefined evaluation dataset
* evaluation execution logic
* metric helpers
* report generation

This separates offline RAG evaluation from the production API path.

---

## Testing

The backend includes automated tests for important production behavior.

Coverage includes:

* API session forwarding
* server-generated API session IDs
* conversation isolation
* session expiration
* inactive-session eviction
* maximum session behavior
* conversation concurrency
* session-aware cache behavior
* history-aware cache behavior
* repeated standalone questions
* contextual questions
* cache bypass behavior
* cached memory updates
* duplicate standalone cached exchange prevention
* cached topic changes
* grounded supported answers
* unsupported answers
* refusal/source consistency

Run the complete test suite:

```bash
python -m pytest -v
```

Run Ruff:

```bash
python -m ruff check api rag tests
```

Check for whitespace errors:

```bash
git diff --check
```

---

## Frontend

The React frontend is maintained separately:

https://github.com/SUFI-410/Production-AI-Platform-Frontend

Production frontend:

https://www.buildwithsufyan.com

The frontend includes:

* conversational chat interface
* new-chat support
* recent chat history
* automatic chat titles
* reopening previous conversations
* conversation deletion
* active conversation persistence
* per-tab active-session isolation
* backend session integration
* Markdown answer rendering
* syntax highlighting
* loading and request states
* structured production error handling
* Grounded / Not Grounded indicators
* source attribution
* Cross-Encoder relevance display
* Cloudflare Turnstile integration

---

## Production Deployment

The backend is deployed on Oracle Cloud Infrastructure.

```text
GitHub
   |
   v
GitHub Actions
   |
   v
Oracle Cloud VM
   |
   v
Docker
   |
   v
FastAPI
   |
   v
Nginx Reverse Proxy
   |
   v
HTTPS / Let's Encrypt
   |
   v
api.buildwithsufyan.com
```

Production API:

```text
https://api.buildwithsufyan.com
```

Production health endpoint:

```text
https://api.buildwithsufyan.com/health
```

Production frontend:

```text
https://www.buildwithsufyan.com
```

---

## Security Design

### Cloudflare Turnstile

The public chat endpoint requires Cloudflare Turnstile verification.

Verification is performed before the expensive RAG application is initialized for a request.

This helps reduce automated abuse of model and LLM resources.

### Rate Limiting

Public chat requests are rate-limited to reduce repeated automated requests and control resource usage.

### CORS

Permitted frontend origins are configured through environment variables.

### Network Isolation

The production FastAPI container is bound to the server's local interface.

Nginx is responsible for public HTTP/HTTPS traffic and proxies requests to the local application service.

### Secret Management

Sensitive configuration is loaded through environment variables, including values such as:

```text
OPENAI_API_KEY
TURNSTILE_SECRET
```

The real `.env` file is excluded from Git.

Only `.env.example` is intended to be committed as the configuration template.

### HTTPS

Production API traffic is protected by TLS using Nginx and Let's Encrypt.

### Container Security

The production Docker image runs the application as a non-root user.

---

## CI/CD

GitHub Actions is used for automated validation and deployment.

The project follows a development-to-production Git workflow:

```text
develop
   |
   v
validation
   |
   v
main
   |
   v
production deployment
```

Changes are developed and validated on `develop`, merged into `main`, and then deployed through the production workflow.

---

## Repository Hygiene

The repository excludes local and generated data such as:

* `.env`
* virtual environments
* Python cache files
* logs
* local ChromaDB data
* test caches
* Ruff caches
* coverage output
* evaluation result output

Real API keys, Turnstile secrets, private keys, and local environment files must never be committed.

---

## Design Goals

This project was built as an end-to-end production RAG engineering project rather than a minimal retrieval prototype.

The main engineering goals are:

* retrieval quality
* lexical and semantic retrieval integration
* source transparency
* deterministic unsupported-answer behavior
* conversational correctness
* session isolation
* concurrency safety
* cache correctness
* production security
* frontend/backend integration
* reproducible deployment
* automated validation
* maintainable architecture

---

## License

This project is licensed under the MIT License.

See the `LICENSE` file for details.

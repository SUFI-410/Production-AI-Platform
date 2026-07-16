# Production RAG System

A production-oriented Retrieval-Augmented Generation (RAG) system built with **Python**, **LangChain**, **OpenAI**, and **ChromaDB**.

This project is designed with clean architecture, modular components, and production-ready practices. It supports indexing PDFs and websites into a vector database, enabling accurate question answering over custom knowledge sources.

---

## Features

- PDF document ingestion
- Website crawling and indexing
- ChromaDB vector database
- OpenAI embeddings
- LangChain retrieval pipeline
- Metadata-aware document retrieval
- Interactive command-line interface (CLI)
- Modular, maintainable architecture
- Structured logging
- Environment-based configuration

---

## Tech Stack

- Python 3.12+
- LangChain
- OpenAI
- ChromaDB
- PyPDF
- BeautifulSoup4
- Requests

---

## Project Structure

```text
Production-RAG/
│
├── app.py
├── requirements.txt
├── .env.example
├── README.md
│
└── rag/
    ├── application.py
    ├── chain.py
    ├── cli.py
    ├── config.py
    ├── crawler.py
    ├── embeddings.py
    ├── exceptions.py
    ├── loader.py
    ├── logger.py
    ├── prompt.py
    ├── retriever.py
    ├── splitter.py
    ├── utils.py
    └── vector_store.py
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/SUFI-410/Production-RAG.git
cd Production-RAG
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
```

---

## Running

```bash
python app.py
```

The application will build a knowledge base from configured data sources and launch an interactive CLI.

---

## Current Capabilities

- Answer questions from indexed PDFs
- Crawl and index websites
- Persistent ChromaDB storage
- Metadata filtering
- Multi-source document loading

---

## Planned Improvements

- FastAPI REST API
- Hybrid Search (Vector + BM25)
- Cross-Encoder Re-ranking
- Conversation Memory
- Docker & Docker Compose
- PostgreSQL metadata storage
- Redis caching
- GitHub Actions CI/CD
- Automated testing
- Cloud deployment

---

## License

This project is licensed under the MIT License.

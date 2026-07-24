"""
FastAPI application entry point.
"""

from __future__ import annotations

from fastapi import FastAPI

from api.routes import router

app = FastAPI(
    title="Production AI Platform",
    description="Production-grade Retrieval-Augmented Generation (RAG) API.",
    version="1.0.0",
)

# Register API routes
app.include_router(router)


@app.get("/", tags=["System"])
def root() -> dict[str, str]:
    """
    Root endpoint.
    """

    return {
        "message": "Production AI Platform API is running."
    }


@app.get("/health", tags=["System"])
def health() -> dict[str, str]:
    """
    Health check endpoint.
    """

    return {
        "status": "healthy"
    }

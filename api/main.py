"""
FastAPI application entry point.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.schemas import HealthResponse
from rag.config import Config

app = FastAPI(
    title="Production AI Platform",
    description="Production-grade Retrieval-Augmented Generation (RAG) API.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
def health() -> HealthResponse:
    """
    Health check endpoint.
    """

    return HealthResponse(
        status="healthy",
        version="1.0.0",
    )

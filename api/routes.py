"""
FastAPI route definitions.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_application
from api.schemas import ChatRequest, ChatResponse
from rag.application import RAGApplication
from rag.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["RAG"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
)
def chat(
    request: ChatRequest,
    app: RAGApplication = Depends(get_application),
) -> ChatResponse:
    """
    Ask the RAG system a question.
    """

    start = time.perf_counter()

    try:
        result = app.ask_with_sources(
            question=request.question,
        )
        print(result)

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            cached=result.get("cached", False),
            grounded=result.get("grounded", True),
            latency_ms=round(latency_ms, 2),
        )

    except Exception:

        logger.exception(
            "Failed to process chat request."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error.",
        )

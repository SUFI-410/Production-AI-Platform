"""
FastAPI route definitions.
"""

from __future__ import annotations

import time
import uuid

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    status,
)

from api.dependencies import get_application
from api.schemas import ChatRequest, ChatResponse
from api.turnstile import verify_turnstile
from rag.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["RAG"])


def _new_session_id() -> str:
    """
    Generate a unique conversation session identifier.
    """

    return uuid.uuid4().hex


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
)
def chat(
    payload: ChatRequest,
    http_request: Request,
) -> ChatResponse:
    """
    Ask the RAG system a verified question.

    Human verification runs before the expensive RAG
    application is initialized.
    """

    start = time.perf_counter()

    verify_turnstile(
        token=payload.turnstile_token,
        request=http_request,
    )

    try:
        app = get_application()

        session_id = (
            payload.session_id
            if payload.session_id is not None
            else _new_session_id()
        )

        result = app.ask_with_sources(
            question=payload.question,
            session_id=session_id,
            use_cache=payload.use_cache,
        )

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            session_id=result["session_id"],
            cached=result.get("cached", False),
            grounded=bool(
                result.get("grounded", False)
            ),
            latency_ms=round(latency_ms, 2),
        )

    except Exception:
        logger.exception(
            "Failed to process chat request."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error.",
        ) from None

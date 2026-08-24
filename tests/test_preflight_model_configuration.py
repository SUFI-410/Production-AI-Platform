from __future__ import annotations

from typing import Any

import pytest

import rag.billing_requirements as billing_requirements_module
import rag.invoice_facts as invoice_facts_module
from rag.billing_requirements import BillingRequirementsExtractor
from rag.config import Config
from rag.invoice_facts import InvoiceFactsExtractor


class FakeChatOpenAI:
    """Capture model construction without making external requests."""

    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)

    def with_structured_output(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        return object()


@pytest.mark.parametrize(
    ("module", "factory"),
    [
        (
            billing_requirements_module,
            BillingRequirementsExtractor,
        ),
        (
            invoice_facts_module,
            InvoiceFactsExtractor,
        ),
    ],
)
def test_preflight_extractors_bound_openai_requests(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    factory: Any,
) -> None:
    FakeChatOpenAI.calls = []

    monkeypatch.setattr(
        module,
        "ChatOpenAI",
        FakeChatOpenAI,
    )
    monkeypatch.setattr(
        Config,
        "OPENAI_REQUEST_TIMEOUT_SECONDS",
        45.0,
    )
    monkeypatch.setattr(
        Config,
        "OPENAI_MAX_RETRIES",
        1,
    )
    monkeypatch.setattr(
        Config,
        "OPENAI_REASONING_EFFORT",
        "low",
    )

    factory()

    assert FakeChatOpenAI.calls == [
        {
            "model": Config.CHAT_MODEL,
            "temperature": Config.TEMPERATURE,
            "timeout": 45.0,
            "max_retries": 1,
            "reasoning_effort": "low",
        }
    ]

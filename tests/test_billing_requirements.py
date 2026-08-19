from __future__ import annotations

from typing import Any

import pytest
from langchain_core.documents import Document

from rag.billing_requirements import (
    BillingRequirementField,
    BillingRequirementsExtractionError,
    BillingRequirementsExtractor,
)


class FakeStructuredLLM:
    """Deterministic structured LLM used by unit tests."""

    def __init__(
        self,
        response: dict[str, Any],
    ) -> None:
        self.response = response
        self.prompts: list[str] = []

    def invoke(
        self,
        prompt: str,
    ) -> dict[str, Any]:
        self.prompts.append(prompt)

        return self.response


def test_extract_returns_grounded_requirements() -> None:
    document = Document(
        page_content=(
            "Purchase Order PO-4821 is required on "
            "all invoices. Payment terms are Net 45."
        ),
        metadata={
            "file_name": "msa.md",
            "page": 3,
        },
    )

    llm = FakeStructuredLLM(
        {
            "po_required": True,
            "po_number": "PO-4821",
            "payment_terms": "Net 45",
            "evidence": [
                {
                    "field": (
                        BillingRequirementField.PO_REQUIRED.value
                    ),
                    "source_id": "source-1",
                    "quote": (
                        "Purchase Order PO-4821 is "
                        "required on all invoices."
                    ),
                },
                {
                    "field": (
                        BillingRequirementField.PO_NUMBER.value
                    ),
                    "source_id": "source-1",
                    "quote": "PO-4821",
                },
                {
                    "field": (
                        BillingRequirementField.PAYMENT_TERMS.value
                    ),
                    "source_id": "source-1",
                    "quote": (
                        "Payment terms are Net 45."
                    ),
                },
            ],
        }
    )

    extractor = BillingRequirementsExtractor(
        structured_llm=llm
    )

    result = extractor.extract(
        [document]
    )

    assert result.po_required is True

    assert result.po_number == "PO-4821"

    assert result.payment_terms == "Net 45"

    assert len(result.evidence) == 3

    payment_evidence = next(
        item
        for item in result.evidence
        if item.field
        == BillingRequirementField.PAYMENT_TERMS
    )

    assert (
        payment_evidence.source_id
        == "source-1"
    )

    assert (
        payment_evidence.file_name
        == "msa.md"
    )

    assert payment_evidence.page == 3

    assert (
        payment_evidence.quote
        == "Payment terms are Net 45."
    )

    assert len(llm.prompts) == 1

    assert (
        "[SOURCE source-1]"
        in llm.prompts[0]
    )

    assert "msa.md" in llm.prompts[0]


def test_extract_rejects_unverified_evidence() -> None:
    document = Document(
        page_content=(
            "Payment terms are Net 45."
        ),
        metadata={
            "file_name": "contract.md",
        },
    )

    llm = FakeStructuredLLM(
        {
            "payment_terms": "Net 30",
            "evidence": [
                {
                    "field": (
                        BillingRequirementField.PAYMENT_TERMS.value
                    ),
                    "source_id": "source-1",
                    "quote": (
                        "Payment terms are Net 30."
                    ),
                }
            ],
        }
    )

    extractor = BillingRequirementsExtractor(
        structured_llm=llm
    )

    with pytest.raises(
        BillingRequirementsExtractionError,
        match="could not be verified",
    ):
        extractor.extract(
            [document]
        )


def test_extract_rejects_requirement_without_evidence() -> None:
    document = Document(
        page_content=(
            "Payment terms are Net 45."
        ),
        metadata={
            "file_name": "contract.md",
        },
    )

    llm = FakeStructuredLLM(
        {
            "payment_terms": "Net 45",
            "evidence": [],
        }
    )

    extractor = BillingRequirementsExtractor(
        structured_llm=llm
    )

    with pytest.raises(
        BillingRequirementsExtractionError,
        match="has no verified evidence",
    ):
        extractor.extract(
            [document]
        )


def test_extract_rejects_unknown_source_id() -> None:
    document = Document(
        page_content=(
            "Payment terms are Net 45."
        ),
        metadata={
            "file_name": "contract.md",
        },
    )

    llm = FakeStructuredLLM(
        {
            "payment_terms": "Net 45",
            "evidence": [
                {
                    "field": (
                        BillingRequirementField.PAYMENT_TERMS.value
                    ),
                    "source_id": "source-99",
                    "quote": (
                        "Payment terms are Net 45."
                    ),
                }
            ],
        }
    )

    extractor = BillingRequirementsExtractor(
        structured_llm=llm
    )

    with pytest.raises(
        BillingRequirementsExtractionError,
        match="unknown source ID",
    ):
        extractor.extract(
            [document]
        )


def test_extract_rejects_empty_documents() -> None:
    extractor = BillingRequirementsExtractor(
        structured_llm=FakeStructuredLLM(
            {}
        )
    )

    with pytest.raises(
        BillingRequirementsExtractionError,
        match="No non-empty source documents",
    ):
        extractor.extract(
            []
        )


def test_extract_ignores_empty_document_content() -> None:
    extractor = BillingRequirementsExtractor(
        structured_llm=FakeStructuredLLM(
            {}
        )
    )

    documents = [
        Document(
            page_content="   ",
            metadata={
                "file_name": "empty.md",
            },
        )
    ]

    with pytest.raises(
        BillingRequirementsExtractionError,
        match="No non-empty source documents",
    ):
        extractor.extract(
            documents
        )

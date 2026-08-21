"""Tests for grounded invoice fact extraction."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.documents import Document

from rag.invoice_facts import (
    InvoiceFactsExtractionError,
    InvoiceFactsExtractor,
)


class FakeStructuredLLM:
    """Controllable structured model used without external API calls."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> Any:
        self.prompts.append(prompt)

        if isinstance(self.result, Exception):
            raise self.result

        return self.result


def test_extract_returns_verified_invoice_facts() -> None:
    content = (
        "Invoice Number: INV-1042\n"
        "Purchase Order: PO-4821\n"
        "Payment Terms: Net 45\n"
        "Bill To: Enterprise Customer LLC\n"
        "Project Code: AI-2026-17\n"
        "Attachments: Signed milestone acceptance certificate"
    )
    document = Document(
        page_content=content,
        metadata={
            "original_filename": "invoice-1042.md",
            "page": 1,
        },
    )
    structured_llm = FakeStructuredLLM(
        {
            "invoice_number": "INV-1042",
            "po_number": "PO-4821",
            "payment_terms": "Net 45",
            "billing_entity": "Enterprise Customer LLC",
            "project_code": "AI-2026-17",
            "attachments": [
                "Signed milestone acceptance certificate",
            ],
            "evidence": [
                {
                    "field": "invoice_number",
                    "source_id": "source-1",
                    "file_name": "invoice-1042.md",
                    "page": 1,
                    "quote": "Invoice Number: INV-1042",
                },
                {
                    "field": "po_number",
                    "source_id": "source-1",
                    "file_name": "invoice-1042.md",
                    "page": 1,
                    "quote": "Purchase Order: PO-4821",
                },
                {
                    "field": "payment_terms",
                    "source_id": "source-1",
                    "file_name": "invoice-1042.md",
                    "page": 1,
                    "quote": "Payment Terms: Net 45",
                },
                {
                    "field": "billing_entity",
                    "source_id": "source-1",
                    "file_name": "invoice-1042.md",
                    "page": 1,
                    "quote": "Bill To: Enterprise Customer LLC",
                },
                {
                    "field": "project_code",
                    "source_id": "source-1",
                    "file_name": "invoice-1042.md",
                    "page": 1,
                    "quote": "Project Code: AI-2026-17",
                },
                {
                    "field": "attachments",
                    "source_id": "source-1",
                    "file_name": "invoice-1042.md",
                    "page": 1,
                    "quote": (
                        "Attachments: Signed milestone acceptance "
                        "certificate"
                    ),
                },
            ],
        }
    )

    facts = InvoiceFactsExtractor(
        structured_llm=structured_llm
    ).extract([document])

    assert facts.invoice_number == "INV-1042"
    assert facts.po_number == "PO-4821"
    assert facts.payment_terms == "Net 45"
    assert facts.billing_entity == "Enterprise Customer LLC"
    assert facts.project_code == "AI-2026-17"
    assert facts.attachments == [
        "Signed milestone acceptance certificate",
    ]
    assert len(facts.evidence) == 6
    assert len(structured_llm.prompts) == 1


def test_extract_keeps_missing_fields_null() -> None:
    document = Document(
        page_content="Invoice prepared for consulting services.",
        metadata={"original_filename": "invoice.md"},
    )
    structured_llm = FakeStructuredLLM(
        {
            "invoice_number": None,
            "po_number": None,
            "payment_terms": None,
            "billing_entity": None,
            "project_code": None,
            "attachments": None,
            "evidence": [],
        }
    )

    facts = InvoiceFactsExtractor(
        structured_llm=structured_llm
    ).extract([document])

    assert facts.invoice_number is None
    assert facts.po_number is None
    assert facts.payment_terms is None
    assert facts.billing_entity is None
    assert facts.project_code is None
    assert facts.attachments is None
    assert facts.evidence == []


def test_extract_uses_deterministic_source_ids_and_original_filename() -> None:
    documents = [
        Document(
            page_content="Invoice summary.",
            metadata={
                "original_filename": "customer-invoice.md",
                "file_name": "internal-storage-name.md",
            },
        ),
        Document(
            page_content="Purchase Order: PO-4821",
            metadata={
                "original_filename": "customer-invoice.md",
                "file_name": "another-internal-name.md",
            },
        ),
    ]
    structured_llm = FakeStructuredLLM(
        {
            "invoice_number": None,
            "po_number": "PO-4821",
            "payment_terms": None,
            "billing_entity": None,
            "project_code": None,
            "attachments": None,
            "evidence": [
                {
                    "field": "po_number",
                    "source_id": "source-2",
                    "file_name": "customer-invoice.md",
                    "page": None,
                    "quote": "Purchase Order: PO-4821",
                },
            ],
        }
    )

    facts = InvoiceFactsExtractor(
        structured_llm=structured_llm
    ).extract(documents)

    prompt = structured_llm.prompts[0]

    assert facts.po_number == "PO-4821"
    assert '"source_id": "source-1"' in prompt
    assert '"source_id": "source-2"' in prompt
    assert '"file_name": "customer-invoice.md"' in prompt
    assert "internal-storage-name.md" not in prompt
    assert "another-internal-name.md" not in prompt


def test_extract_rejects_unknown_source_id() -> None:
    document = Document(
        page_content="Invoice Number: INV-1042",
        metadata={"original_filename": "invoice.md"},
    )
    structured_llm = FakeStructuredLLM(
        {
            "invoice_number": "INV-1042",
            "po_number": None,
            "payment_terms": None,
            "billing_entity": None,
            "project_code": None,
            "attachments": None,
            "evidence": [
                {
                    "field": "invoice_number",
                    "source_id": "source-99",
                    "file_name": "invoice.md",
                    "page": None,
                    "quote": "Invoice Number: INV-1042",
                },
            ],
        }
    )

    with pytest.raises(
        InvoiceFactsExtractionError,
        match="unknown source ID",
    ):
        InvoiceFactsExtractor(
            structured_llm=structured_llm
        ).extract([document])


def test_extract_rejects_hallucinated_quote() -> None:
    document = Document(
        page_content="Invoice Number: INV-1042",
        metadata={"original_filename": "invoice.md"},
    )
    structured_llm = FakeStructuredLLM(
        {
            "invoice_number": "INV-9999",
            "po_number": None,
            "payment_terms": None,
            "billing_entity": None,
            "project_code": None,
            "attachments": None,
            "evidence": [
                {
                    "field": "invoice_number",
                    "source_id": "source-1",
                    "file_name": "invoice.md",
                    "page": None,
                    "quote": "Invoice Number: INV-9999",
                },
            ],
        }
    )

    with pytest.raises(
        InvoiceFactsExtractionError,
        match="quote that does not exist",
    ):
        InvoiceFactsExtractor(
            structured_llm=structured_llm
        ).extract([document])


def test_extract_rejects_populated_field_without_evidence() -> None:
    document = Document(
        page_content="Payment Terms: Net 45",
        metadata={"original_filename": "invoice.md"},
    )
    structured_llm = FakeStructuredLLM(
        {
            "invoice_number": None,
            "po_number": None,
            "payment_terms": "Net 45",
            "billing_entity": None,
            "project_code": None,
            "attachments": None,
            "evidence": [],
        }
    )

    with pytest.raises(
        InvoiceFactsExtractionError,
        match="missing verified evidence: payment_terms",
    ):
        InvoiceFactsExtractor(
            structured_llm=structured_llm
        ).extract([document])


def test_extract_rejects_incorrect_evidence_filename() -> None:
    document = Document(
        page_content="Project Code: AI-2026-17",
        metadata={"original_filename": "invoice.md"},
    )
    structured_llm = FakeStructuredLLM(
        {
            "invoice_number": None,
            "po_number": None,
            "payment_terms": None,
            "billing_entity": None,
            "project_code": "AI-2026-17",
            "attachments": None,
            "evidence": [
                {
                    "field": "project_code",
                    "source_id": "source-1",
                    "file_name": "wrong-invoice.md",
                    "page": None,
                    "quote": "Project Code: AI-2026-17",
                },
            ],
        }
    )

    with pytest.raises(
        InvoiceFactsExtractionError,
        match="incorrect file name",
    ):
        InvoiceFactsExtractor(
            structured_llm=structured_llm
        ).extract([document])


def test_extract_rejects_incorrect_evidence_page() -> None:
    document = Document(
        page_content="Purchase Order: PO-4821",
        metadata={
            "original_filename": "invoice.pdf",
            "page": 2,
        },
    )
    structured_llm = FakeStructuredLLM(
        {
            "invoice_number": None,
            "po_number": "PO-4821",
            "payment_terms": None,
            "billing_entity": None,
            "project_code": None,
            "attachments": None,
            "evidence": [
                {
                    "field": "po_number",
                    "source_id": "source-1",
                    "file_name": "invoice.pdf",
                    "page": 3,
                    "quote": "Purchase Order: PO-4821",
                },
            ],
        }
    )

    with pytest.raises(
        InvoiceFactsExtractionError,
        match="incorrect page",
    ):
        InvoiceFactsExtractor(
            structured_llm=structured_llm
        ).extract([document])


def test_extract_rejects_evidence_for_unpopulated_field() -> None:
    document = Document(
        page_content="Invoice Number: INV-1042",
        metadata={"original_filename": "invoice.md"},
    )
    structured_llm = FakeStructuredLLM(
        {
            "invoice_number": None,
            "po_number": None,
            "payment_terms": None,
            "billing_entity": None,
            "project_code": None,
            "attachments": None,
            "evidence": [
                {
                    "field": "invoice_number",
                    "source_id": "source-1",
                    "file_name": "invoice.md",
                    "page": None,
                    "quote": "Invoice Number: INV-1042",
                },
            ],
        }
    )

    with pytest.raises(
        InvoiceFactsExtractionError,
        match="unpopulated field 'invoice_number'",
    ):
        InvoiceFactsExtractor(
            structured_llm=structured_llm
        ).extract([document])


def test_extract_rejects_empty_document_collection() -> None:
    structured_llm = FakeStructuredLLM({})

    with pytest.raises(
        InvoiceFactsExtractionError,
        match="At least one invoice document is required",
    ):
        InvoiceFactsExtractor(
            structured_llm=structured_llm
        ).extract([])

    assert structured_llm.prompts == []


def test_extract_rejects_documents_without_text() -> None:
    document = Document(
        page_content="   ",
        metadata={"original_filename": "invoice.md"},
    )
    structured_llm = FakeStructuredLLM({})

    with pytest.raises(
        InvoiceFactsExtractionError,
        match="do not contain extractable text",
    ):
        InvoiceFactsExtractor(
            structured_llm=structured_llm
        ).extract([document])

    assert structured_llm.prompts == []


def test_extract_wraps_invalid_structured_response() -> None:
    document = Document(
        page_content="Invoice Number: INV-1042",
        metadata={"original_filename": "invoice.md"},
    )
    structured_llm = FakeStructuredLLM(
        {
            "invoice_number": ["invalid-value"],
            "evidence": [],
        }
    )

    with pytest.raises(
        InvoiceFactsExtractionError,
        match="did not match the required schema",
    ):
        InvoiceFactsExtractor(
            structured_llm=structured_llm
        ).extract([document])


def test_extract_wraps_model_failure() -> None:
    document = Document(
        page_content="Invoice Number: INV-1042",
        metadata={"original_filename": "invoice.md"},
    )
    structured_llm = FakeStructuredLLM(
        RuntimeError("Model unavailable")
    )

    with pytest.raises(
        InvoiceFactsExtractionError,
        match="extraction model failed",
    ):
        InvoiceFactsExtractor(
            structured_llm=structured_llm
        ).extract([document])

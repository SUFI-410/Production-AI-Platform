"""Tests for the deterministic invoice-preflight engine."""

from __future__ import annotations

from typing import Any

from rag.billing_requirements import BillingRequirements
from rag.invoice_facts import InvoiceFacts
from rag.invoice_preflight import (
    FindingSeverity,
    InvoicePreflightEngine,
    InvoicePreflightResult,
    PaymentReadiness,
    PreflightField,
    PreflightFinding,
)


def _requirements(**overrides: Any) -> BillingRequirements:
    data: dict[str, Any] = {
        "po_required": None,
        "po_number": None,
        "payment_terms": None,
        "milestone_approval_required": None,
        "billing_entity": None,
        "project_code": None,
        "required_attachments": None,
        "evidence": [],
    }
    data.update(overrides)
    return BillingRequirements.model_validate(data)


def _invoice(**overrides: Any) -> InvoiceFacts:
    data: dict[str, Any] = {
        "invoice_number": None,
        "po_number": None,
        "payment_terms": None,
        "billing_entity": None,
        "project_code": None,
        "attachments": None,
        "evidence": [],
    }
    data.update(overrides)
    return InvoiceFacts.model_validate(data)


def _finding(
    result: InvoicePreflightResult,
    field: PreflightField,
) -> PreflightFinding:
    matches = [
        finding
        for finding in result.findings
        if finding.field is field
    ]

    assert len(matches) == 1
    return matches[0]


def test_all_matching_requirements_produce_ready_result() -> None:
    requirements = _requirements(
        po_required=True,
        po_number="PO-4821",
        payment_terms=(
            "Net 45 from receipt of a valid invoice."
        ),
        milestone_approval_required=True,
        billing_entity="Enterprise Customer LLC",
        project_code="AI-2026-17",
        required_attachments=[
            "signed milestone acceptance certificate",
        ],
    )
    invoice = _invoice(
        invoice_number="INV-1042",
        po_number="PO-4821",
        payment_terms="Net 45",
        billing_entity="Enterprise Customer LLC",
        project_code="AI-2026-17",
        attachments=[
            "Signed milestone acceptance certificate",
        ],
    )

    result = InvoicePreflightEngine.evaluate(
        requirements,
        invoice,
    )

    assert result.payment_readiness is PaymentReadiness.READY
    assert len(result.findings) == 6
    assert all(
        finding.severity is FindingSeverity.PASS
        for finding in result.findings
    )


def test_missing_required_po_is_blocker() -> None:
    requirements = _requirements(
        po_required=True,
        po_number="PO-4821",
    )
    invoice = _invoice()

    result = InvoicePreflightEngine.evaluate(
        requirements,
        invoice,
    )

    finding = _finding(result, PreflightField.PO_NUMBER)

    assert result.payment_readiness is PaymentReadiness.BLOCKED
    assert finding.severity is FindingSeverity.BLOCKER
    assert "missing required PO PO-4821" in finding.message


def test_po_identifier_formatting_is_normalized() -> None:
    requirements = _requirements(
        po_required=True,
        po_number="PO-4821",
    )
    invoice = _invoice(po_number="po 4821")

    result = InvoicePreflightEngine.evaluate(
        requirements,
        invoice,
    )

    finding = _finding(result, PreflightField.PO_NUMBER)

    assert result.payment_readiness is PaymentReadiness.READY
    assert finding.severity is FindingSeverity.PASS


def test_required_po_without_expected_number_accepts_present_po() -> None:
    requirements = _requirements(po_required=True)
    invoice = _invoice(po_number="PO-9001")

    result = InvoicePreflightEngine.evaluate(
        requirements,
        invoice,
    )

    finding = _finding(result, PreflightField.PO_NUMBER)

    assert result.payment_readiness is PaymentReadiness.READY
    assert finding.severity is FindingSeverity.PASS
    assert "PO-9001" in finding.message


def test_conflicting_po_requirements_need_review() -> None:
    requirements = _requirements(
        po_required=False,
        po_number="PO-4821",
    )
    invoice = _invoice()

    result = InvoicePreflightEngine.evaluate(
        requirements,
        invoice,
    )

    finding = _finding(result, PreflightField.PO_NUMBER)

    assert (
        result.payment_readiness
        is PaymentReadiness.REVIEW_REQUIRED
    )
    assert finding.severity is FindingSeverity.WARNING
    assert "not required" in finding.message
    assert "PO-4821" in finding.message


def test_missing_required_payment_terms_is_blocker() -> None:
    requirements = _requirements(payment_terms="Net 45")
    invoice = _invoice()

    result = InvoicePreflightEngine.evaluate(
        requirements,
        invoice,
    )

    finding = _finding(
        result,
        PreflightField.PAYMENT_TERMS,
    )

    assert result.payment_readiness is PaymentReadiness.BLOCKED
    assert finding.severity is FindingSeverity.BLOCKER
    assert "missing required payment terms" in finding.message


def test_different_net_payment_period_is_blocker() -> None:
    requirements = _requirements(payment_terms="Net 45")
    invoice = _invoice(payment_terms="Net 30")

    result = InvoicePreflightEngine.evaluate(
        requirements,
        invoice,
    )

    finding = _finding(
        result,
        PreflightField.PAYMENT_TERMS,
    )

    assert result.payment_readiness is PaymentReadiness.BLOCKED
    assert finding.severity is FindingSeverity.BLOCKER
    assert "Net 30" in finding.message
    assert "Net 45" in finding.message


def test_same_net_period_with_additional_text_passes() -> None:
    requirements = _requirements(
        payment_terms=(
            "Net 45 from receipt of a valid invoice."
        )
    )
    invoice = _invoice(payment_terms="NET-45")

    result = InvoicePreflightEngine.evaluate(
        requirements,
        invoice,
    )

    finding = _finding(
        result,
        PreflightField.PAYMENT_TERMS,
    )

    assert result.payment_readiness is PaymentReadiness.READY
    assert finding.severity is FindingSeverity.PASS
    assert "Net 45" in finding.message


def test_unstructured_payment_terms_mismatch_needs_review() -> None:
    requirements = _requirements(
        payment_terms="Due after customer acceptance"
    )
    invoice = _invoice(
        payment_terms="Due upon receipt"
    )

    result = InvoicePreflightEngine.evaluate(
        requirements,
        invoice,
    )

    finding = _finding(
        result,
        PreflightField.PAYMENT_TERMS,
    )

    assert (
        result.payment_readiness
        is PaymentReadiness.REVIEW_REQUIRED
    )
    assert finding.severity is FindingSeverity.WARNING
    assert "Manual review" in finding.message


def test_billing_entity_case_and_punctuation_are_normalized() -> None:
    requirements = _requirements(
        billing_entity="Enterprise Customer, LLC"
    )
    invoice = _invoice(
        billing_entity="enterprise customer llc"
    )

    result = InvoicePreflightEngine.evaluate(
        requirements,
        invoice,
    )

    finding = _finding(
        result,
        PreflightField.BILLING_ENTITY,
    )

    assert result.payment_readiness is PaymentReadiness.READY
    assert finding.severity is FindingSeverity.PASS


def test_missing_required_project_code_is_blocker() -> None:
    requirements = _requirements(
        project_code="AI-2026-17"
    )
    invoice = _invoice()

    result = InvoicePreflightEngine.evaluate(
        requirements,
        invoice,
    )

    finding = _finding(
        result,
        PreflightField.PROJECT_CODE,
    )

    assert result.payment_readiness is PaymentReadiness.BLOCKED
    assert finding.severity is FindingSeverity.BLOCKER
    assert "missing required project code" in finding.message
    assert "AI-2026-17" in finding.message


def test_missing_one_required_attachment_is_blocker() -> None:
    requirements = _requirements(
        required_attachments=[
            "signed milestone acceptance certificate",
            "approved timesheet",
        ]
    )
    invoice = _invoice(
        attachments=[
            "Signed milestone acceptance certificate",
        ]
    )

    result = InvoicePreflightEngine.evaluate(
        requirements,
        invoice,
    )

    finding = _finding(
        result,
        PreflightField.ATTACHMENTS,
    )

    assert result.payment_readiness is PaymentReadiness.BLOCKED
    assert finding.severity is FindingSeverity.BLOCKER
    assert "approved timesheet" in finding.message
    assert (
        "signed milestone acceptance certificate"
        not in finding.message
    )


def test_attachment_name_with_file_extension_matches() -> None:
    requirements = _requirements(
        required_attachments=[
            "signed milestone acceptance certificate",
        ]
    )
    invoice = _invoice(
        attachments=[
            "Signed Milestone Acceptance Certificate.pdf",
        ]
    )

    result = InvoicePreflightEngine.evaluate(
        requirements,
        invoice,
    )

    finding = _finding(
        result,
        PreflightField.ATTACHMENTS,
    )

    assert result.payment_readiness is PaymentReadiness.READY
    assert finding.severity is FindingSeverity.PASS


def test_milestone_acceptance_attachment_passes() -> None:
    requirements = _requirements(
        milestone_approval_required=True
    )
    invoice = _invoice(
        attachments=[
            "Signed milestone acceptance certificate",
        ]
    )

    result = InvoicePreflightEngine.evaluate(
        requirements,
        invoice,
    )

    finding = _finding(
        result,
        PreflightField.MILESTONE_APPROVAL,
    )

    assert result.payment_readiness is PaymentReadiness.READY
    assert finding.severity is FindingSeverity.PASS


def test_missing_milestone_approval_attachment_is_blocker() -> None:
    requirements = _requirements(
        milestone_approval_required=True
    )
    invoice = _invoice(
        attachments=["Invoice detail report"]
    )

    result = InvoicePreflightEngine.evaluate(
        requirements,
        invoice,
    )

    finding = _finding(
        result,
        PreflightField.MILESTONE_APPROVAL,
    )

    assert result.payment_readiness is PaymentReadiness.BLOCKED
    assert finding.severity is FindingSeverity.BLOCKER
    assert "does not indicate" in finding.message


def test_no_extracted_requirements_needs_review() -> None:
    requirements = _requirements()
    invoice = _invoice(
        invoice_number="INV-1042"
    )

    result = InvoicePreflightEngine.evaluate(
        requirements,
        invoice,
    )

    finding = _finding(
        result,
        PreflightField.BILLING_REQUIREMENTS,
    )

    assert (
        result.payment_readiness
        is PaymentReadiness.REVIEW_REQUIRED
    )
    assert finding.severity is FindingSeverity.WARNING
    assert "No enforceable billing requirements" in finding.message


def test_blocker_takes_priority_over_warning() -> None:
    requirements = _requirements(
        payment_terms="Due after customer acceptance",
        project_code="AI-2026-17",
    )
    invoice = _invoice(
        payment_terms="Due upon receipt",
        project_code=None,
    )

    result = InvoicePreflightEngine.evaluate(
        requirements,
        invoice,
    )

    payment_finding = _finding(
        result,
        PreflightField.PAYMENT_TERMS,
    )
    project_finding = _finding(
        result,
        PreflightField.PROJECT_CODE,
    )

    assert result.payment_readiness is PaymentReadiness.BLOCKED
    assert payment_finding.severity is FindingSeverity.WARNING
    assert project_finding.severity is FindingSeverity.BLOCKER


def test_explicitly_unrequired_fields_produce_ready_result() -> None:
    requirements = _requirements(
        po_required=False,
        milestone_approval_required=False,
    )
    invoice = _invoice()

    result = InvoicePreflightEngine.evaluate(
        requirements,
        invoice,
    )

    assert result.payment_readiness is PaymentReadiness.READY
    assert len(result.findings) == 2
    assert all(
        finding.severity is FindingSeverity.PASS
        for finding in result.findings
    )

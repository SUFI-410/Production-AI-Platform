"""Deterministic comparison of billing requirements and invoice facts."""

from __future__ import annotations

import re
from enum import Enum
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from rag.billing_requirements import BillingRequirements
from rag.invoice_facts import InvoiceFacts


class FindingSeverity(str, Enum):
    """Severity of one invoice-preflight finding."""

    PASS = "PASS"
    WARNING = "WARNING"
    BLOCKER = "BLOCKER"


class PaymentReadiness(str, Enum):
    """Overall readiness of an invoice for submission."""

    READY = "READY"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    BLOCKED = "BLOCKED"


class PreflightField(str, Enum):
    """Fields evaluated by the deterministic preflight engine."""

    BILLING_REQUIREMENTS = "billing_requirements"
    PO_NUMBER = "po_number"
    PAYMENT_TERMS = "payment_terms"
    BILLING_ENTITY = "billing_entity"
    PROJECT_CODE = "project_code"
    ATTACHMENTS = "attachments"
    MILESTONE_APPROVAL = "milestone_approval"


class PreflightFinding(BaseModel):
    """One deterministic invoice-preflight result."""

    model_config = ConfigDict(extra="forbid")

    severity: FindingSeverity
    field: PreflightField
    message: str = Field(min_length=1)


class InvoicePreflightResult(BaseModel):
    """Overall invoice-preflight result and its individual findings."""

    model_config = ConfigDict(extra="forbid")

    payment_readiness: PaymentReadiness
    findings: list[PreflightFinding] = Field(default_factory=list)


class InvoicePreflightEngine:
    """Compare extracted requirements with extracted invoice facts."""

    _NET_TERMS_PATTERN = re.compile(
        r"\bnet[\s-]*(\d+)\b",
        flags=re.IGNORECASE,
    )

    @classmethod
    def evaluate(
        cls,
        requirements: BillingRequirements,
        invoice: InvoiceFacts,
    ) -> InvoicePreflightResult:
        """Return a deterministic payment-readiness result."""

        if not cls._has_any_requirement(requirements):
            findings = [
                PreflightFinding(
                    severity=FindingSeverity.WARNING,
                    field=PreflightField.BILLING_REQUIREMENTS,
                    message=(
                        "No enforceable billing requirements were extracted. "
                        "Manual review is required before submission."
                    ),
                )
            ]
            return InvoicePreflightResult(
                payment_readiness=PaymentReadiness.REVIEW_REQUIRED,
                findings=findings,
            )

        findings: list[PreflightFinding] = []

        po_finding = cls._compare_po_number(requirements, invoice)
        if po_finding is not None:
            findings.append(po_finding)

        payment_terms_finding = cls._compare_payment_terms(
            requirements,
            invoice,
        )
        if payment_terms_finding is not None:
            findings.append(payment_terms_finding)

        billing_entity_finding = cls._compare_required_text(
            field=PreflightField.BILLING_ENTITY,
            label="billing entity",
            required_value=requirements.billing_entity,
            invoice_value=invoice.billing_entity,
            normalizer=cls._normalize_text,
        )
        if billing_entity_finding is not None:
            findings.append(billing_entity_finding)

        project_code_finding = cls._compare_required_text(
            field=PreflightField.PROJECT_CODE,
            label="project code",
            required_value=requirements.project_code,
            invoice_value=invoice.project_code,
            normalizer=cls._normalize_identifier,
        )
        if project_code_finding is not None:
            findings.append(project_code_finding)

        attachments_finding = cls._compare_required_attachments(
            requirements,
            invoice,
        )
        if attachments_finding is not None:
            findings.append(attachments_finding)

        milestone_finding = cls._compare_milestone_approval(
            requirements,
            invoice,
        )
        if milestone_finding is not None:
            findings.append(milestone_finding)

        return InvoicePreflightResult(
            payment_readiness=cls._calculate_readiness(findings),
            findings=findings,
        )

    @classmethod
    def _compare_po_number(
        cls,
        requirements: BillingRequirements,
        invoice: InvoiceFacts,
    ) -> PreflightFinding | None:
        required_po = cls._clean_optional_text(requirements.po_number)
        invoice_po = cls._clean_optional_text(invoice.po_number)

        if requirements.po_required is False:
            if required_po is not None:
                return PreflightFinding(
                    severity=FindingSeverity.WARNING,
                    field=PreflightField.PO_NUMBER,
                    message=(
                        "Billing requirements state that a PO is not required "
                        f"but also specify PO {required_po}. Manual review is "
                        "required."
                    ),
                )

            return PreflightFinding(
                severity=FindingSeverity.PASS,
                field=PreflightField.PO_NUMBER,
                message="A purchase-order number is not required.",
            )

        po_is_required = (
            requirements.po_required is True
            or required_po is not None
        )

        if not po_is_required:
            return None

        if invoice_po is None:
            if required_po is not None:
                message = (
                    f"Invoice is missing required PO {required_po}."
                )
            else:
                message = (
                    "Invoice is missing a required purchase-order number."
                )

            return PreflightFinding(
                severity=FindingSeverity.BLOCKER,
                field=PreflightField.PO_NUMBER,
                message=message,
            )

        if required_po is None:
            return PreflightFinding(
                severity=FindingSeverity.PASS,
                field=PreflightField.PO_NUMBER,
                message=(
                    f"Invoice includes purchase-order number {invoice_po}."
                ),
            )

        if (
            cls._normalize_identifier(required_po)
            != cls._normalize_identifier(invoice_po)
        ):
            return PreflightFinding(
                severity=FindingSeverity.BLOCKER,
                field=PreflightField.PO_NUMBER,
                message=(
                    f"Invoice PO {invoice_po} does not match required "
                    f"PO {required_po}."
                ),
            )

        return PreflightFinding(
            severity=FindingSeverity.PASS,
            field=PreflightField.PO_NUMBER,
            message=f"Invoice correctly includes required PO {required_po}.",
        )

    @classmethod
    def _compare_payment_terms(
        cls,
        requirements: BillingRequirements,
        invoice: InvoiceFacts,
    ) -> PreflightFinding | None:
        required_terms = cls._clean_optional_text(
            requirements.payment_terms
        )

        if required_terms is None:
            return None

        invoice_terms = cls._clean_optional_text(invoice.payment_terms)

        if invoice_terms is None:
            return PreflightFinding(
                severity=FindingSeverity.BLOCKER,
                field=PreflightField.PAYMENT_TERMS,
                message=(
                    "Invoice is missing required payment terms "
                    f"{required_terms}."
                ),
            )

        if (
            cls._normalize_text(required_terms)
            == cls._normalize_text(invoice_terms)
        ):
            return PreflightFinding(
                severity=FindingSeverity.PASS,
                field=PreflightField.PAYMENT_TERMS,
                message=(
                    "Invoice correctly uses required payment terms "
                    f"{required_terms}."
                ),
            )

        required_net_days = cls._extract_net_days(required_terms)
        invoice_net_days = cls._extract_net_days(invoice_terms)

        if (
            required_net_days is not None
            and invoice_net_days is not None
        ):
            if required_net_days == invoice_net_days:
                return PreflightFinding(
                    severity=FindingSeverity.PASS,
                    field=PreflightField.PAYMENT_TERMS,
                    message=(
                        "Invoice payment terms match the required "
                        f"Net {required_net_days} period."
                    ),
                )

            return PreflightFinding(
                severity=FindingSeverity.BLOCKER,
                field=PreflightField.PAYMENT_TERMS,
                message=(
                    f"Invoice uses Net {invoice_net_days}, but "
                    f"Net {required_net_days} is required."
                ),
            )

        return PreflightFinding(
            severity=FindingSeverity.WARNING,
            field=PreflightField.PAYMENT_TERMS,
            message=(
                f"Invoice payment terms '{invoice_terms}' do not exactly "
                f"match required terms '{required_terms}'. Manual review "
                "is required."
            ),
        )

    @classmethod
    def _compare_required_text(
        cls,
        *,
        field: PreflightField,
        label: str,
        required_value: str | None,
        invoice_value: str | None,
        normalizer: Callable[[str], str],
    ) -> PreflightFinding | None:
        required = cls._clean_optional_text(required_value)

        if required is None:
            return None

        actual = cls._clean_optional_text(invoice_value)

        if actual is None:
            return PreflightFinding(
                severity=FindingSeverity.BLOCKER,
                field=field,
                message=f"Invoice is missing required {label} {required}.",
            )

        if normalizer(required) != normalizer(actual):
            return PreflightFinding(
                severity=FindingSeverity.BLOCKER,
                field=field,
                message=(
                    f"Invoice {label} '{actual}' does not match required "
                    f"{label} '{required}'."
                ),
            )

        return PreflightFinding(
            severity=FindingSeverity.PASS,
            field=field,
            message=f"Invoice correctly includes required {label} {required}.",
        )

    @classmethod
    def _compare_required_attachments(
        cls,
        requirements: BillingRequirements,
        invoice: InvoiceFacts,
    ) -> PreflightFinding | None:
        required_attachments = cls._clean_list(
            requirements.required_attachments
        )

        if not required_attachments:
            return None

        invoice_attachments = cls._clean_list(invoice.attachments)

        if not invoice_attachments:
            return PreflightFinding(
                severity=FindingSeverity.BLOCKER,
                field=PreflightField.ATTACHMENTS,
                message=(
                    "Invoice is missing required attachments: "
                    f"{', '.join(required_attachments)}."
                ),
            )

        missing_attachments = [
            required_attachment
            for required_attachment in required_attachments
            if not any(
                cls._attachment_matches(
                    required_attachment,
                    invoice_attachment,
                )
                for invoice_attachment in invoice_attachments
            )
        ]

        if missing_attachments:
            return PreflightFinding(
                severity=FindingSeverity.BLOCKER,
                field=PreflightField.ATTACHMENTS,
                message=(
                    "Invoice is missing required attachments: "
                    f"{', '.join(missing_attachments)}."
                ),
            )

        return PreflightFinding(
            severity=FindingSeverity.PASS,
            field=PreflightField.ATTACHMENTS,
            message="Invoice includes all required attachments.",
        )

    @classmethod
    def _compare_milestone_approval(
        cls,
        requirements: BillingRequirements,
        invoice: InvoiceFacts,
    ) -> PreflightFinding | None:
        if requirements.milestone_approval_required is None:
            return None

        if requirements.milestone_approval_required is False:
            return PreflightFinding(
                severity=FindingSeverity.PASS,
                field=PreflightField.MILESTONE_APPROVAL,
                message="Milestone approval evidence is not required.",
            )

        invoice_attachments = cls._clean_list(invoice.attachments)

        if any(
            cls._looks_like_milestone_approval(attachment)
            for attachment in invoice_attachments
        ):
            return PreflightFinding(
                severity=FindingSeverity.PASS,
                field=PreflightField.MILESTONE_APPROVAL,
                message=(
                    "Invoice indicates that milestone approval evidence "
                    "is attached."
                ),
            )

        return PreflightFinding(
            severity=FindingSeverity.BLOCKER,
            field=PreflightField.MILESTONE_APPROVAL,
            message=(
                "Invoice does not indicate that required milestone "
                "approval evidence is attached."
            ),
        )

    @staticmethod
    def _calculate_readiness(
        findings: list[PreflightFinding],
    ) -> PaymentReadiness:
        if any(
            finding.severity is FindingSeverity.BLOCKER
            for finding in findings
        ):
            return PaymentReadiness.BLOCKED

        if any(
            finding.severity is FindingSeverity.WARNING
            for finding in findings
        ):
            return PaymentReadiness.REVIEW_REQUIRED

        return PaymentReadiness.READY

    @staticmethod
    def _has_any_requirement(
        requirements: BillingRequirements,
    ) -> bool:
        return any(
            (
                requirements.po_required is not None,
                bool(
                    InvoicePreflightEngine._clean_optional_text(
                        requirements.po_number
                    )
                ),
                bool(
                    InvoicePreflightEngine._clean_optional_text(
                        requirements.payment_terms
                    )
                ),
                requirements.milestone_approval_required is not None,
                bool(
                    InvoicePreflightEngine._clean_optional_text(
                        requirements.billing_entity
                    )
                ),
                bool(
                    InvoicePreflightEngine._clean_optional_text(
                        requirements.project_code
                    )
                ),
                bool(
                    InvoicePreflightEngine._clean_list(
                        requirements.required_attachments
                    )
                ),
            )
        )

    @staticmethod
    def _clean_optional_text(value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()
        return cleaned or None

    @staticmethod
    def _clean_list(values: list[str] | None) -> list[str]:
        if not values:
            return []

        return [
            cleaned
            for value in values
            if (cleaned := value.strip())
        ]

    @staticmethod
    def _normalize_text(value: str) -> str:
        return " ".join(
            re.sub(
                r"[^a-z0-9]+",
                " ",
                value.casefold(),
            ).split()
        )

    @staticmethod
    def _normalize_identifier(value: str) -> str:
        return re.sub(
            r"[^a-z0-9]+",
            "",
            value.casefold(),
        )

    @classmethod
    def _extract_net_days(cls, value: str) -> int | None:
        match = cls._NET_TERMS_PATTERN.search(value)

        if match is None:
            return None

        return int(match.group(1))

    @classmethod
    def _attachment_matches(
        cls,
        required_attachment: str,
        invoice_attachment: str,
    ) -> bool:
        required = cls._normalize_text(required_attachment)
        actual = cls._normalize_text(invoice_attachment)

        return required == actual or required in actual

    @classmethod
    def _looks_like_milestone_approval(
        cls,
        attachment: str,
    ) -> bool:
        normalized = cls._normalize_text(attachment)

        markers = (
            "milestone approval",
            "milestone acceptance",
            "acceptance certificate",
            "approval certificate",
        )

        return any(marker in normalized for marker in markers)

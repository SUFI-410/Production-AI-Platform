"""
Structured extraction of invoice billing requirements.

This module converts contract, SOW, purchase-order, and billing-instruction
content into evidence-backed billing requirements for Invoice Preflight.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rag.config import Config
from rag.logger import get_logger


logger = get_logger(__name__)


class BillingRequirementsExtractionError(RuntimeError):
    """Raised when billing requirements cannot be extracted safely."""


class BillingRequirementField(str, Enum):
    """Supported invoice billing requirement fields."""

    PO_REQUIRED = "po_required"
    PO_NUMBER = "po_number"
    PAYMENT_TERMS = "payment_terms"
    MILESTONE_APPROVAL_REQUIRED = (
        "milestone_approval_required"
    )
    BILLING_ENTITY = "billing_entity"
    PROJECT_CODE = "project_code"
    REQUIRED_ATTACHMENTS = "required_attachments"


class BillingRequirementEvidence(BaseModel):
    """
    Verified evidence supporting one extracted billing requirement.

    File and page metadata are populated by application code rather than
    trusted directly from the language model.
    """

    field: BillingRequirementField
    source_id: str
    file_name: str
    page: int | None = None
    quote: str

    model_config = ConfigDict(
        extra="forbid",
    )


class BillingRequirements(BaseModel):
    """Structured billing requirements extracted from source documents."""

    po_required: bool | None = None
    po_number: str | None = None
    payment_terms: str | None = None
    milestone_approval_required: bool | None = None
    billing_entity: str | None = None
    project_code: str | None = None
    required_attachments: list[str] | None = None
    evidence: list[BillingRequirementEvidence] = Field(
        default_factory=list,
    )

    model_config = ConfigDict(
        extra="forbid",
    )


class _LLMEvidence(BaseModel):
    """Evidence structure returned directly by the language model."""

    field: BillingRequirementField
    source_id: str
    quote: str

    model_config = ConfigDict(
        extra="forbid",
    )


class _LLMBillingRequirements(BaseModel):
    """Internal structured-output schema used with ChatOpenAI."""

    po_required: bool | None = None
    po_number: str | None = None
    payment_terms: str | None = None
    milestone_approval_required: bool | None = None
    billing_entity: str | None = None
    project_code: str | None = None
    required_attachments: list[str] | None = None
    evidence: list[_LLMEvidence] = Field(
        default_factory=list,
    )

    model_config = ConfigDict(
        extra="forbid",
    )


@dataclass(frozen=True)
class _SourceRecord:
    """Internal source representation supplied to the extractor."""

    source_id: str
    file_name: str
    page: int | None
    content: str


class BillingRequirementsExtractor:
    """
    Extract billing requirements with verified source evidence.

    The language model performs semantic extraction, while application
    code verifies that every returned evidence quote actually occurs in
    the supplied source content.
    """

    def __init__(
        self,
        structured_llm: Any | None = None,
    ) -> None:
        if structured_llm is not None:
            self.structured_llm = structured_llm
            return

        llm = ChatOpenAI(
            model=Config.CHAT_MODEL,
            temperature=Config.TEMPERATURE,
            timeout=(
                Config.OPENAI_REQUEST_TIMEOUT_SECONDS
            ),
            max_retries=Config.OPENAI_MAX_RETRIES,
            reasoning_effort=(
                Config.OPENAI_REASONING_EFFORT
            ),
        )

        self.structured_llm = llm.with_structured_output(
            _LLMBillingRequirements,
            method="json_schema",
        )

    @staticmethod
    def _normalize_text(
        value: str,
    ) -> str:
        """Normalize whitespace and casing for evidence matching."""

        return " ".join(
            value.casefold().split()
        )

    @staticmethod
    def _prepare_sources(
        documents: list[Document],
    ) -> list[_SourceRecord]:
        """Convert LangChain documents into deterministic source records."""

        sources: list[_SourceRecord] = []

        for index, document in enumerate(
            documents,
            start=1,
        ):
            content = document.page_content.strip()

            if not content:
                continue

            metadata = document.metadata or {}

            file_name = str(
                metadata.get("file_name")
                or metadata.get("source")
                or f"document-{index}"
            )

            raw_page = metadata.get("page")

            page = (
                raw_page
                if isinstance(raw_page, int)
                else None
            )

            sources.append(
                _SourceRecord(
                    source_id=f"source-{index}",
                    file_name=file_name,
                    page=page,
                    content=content,
                )
            )

        if not sources:
            raise BillingRequirementsExtractionError(
                "No non-empty source documents were provided."
            )

        return sources

    @staticmethod
    def _format_sources(
        sources: list[_SourceRecord],
    ) -> str:
        """Format source documents for the extraction prompt."""

        blocks: list[str] = []

        for source in sources:
            page_value = (
                str(source.page)
                if source.page is not None
                else "unknown"
            )

            blocks.append(
                "\n".join(
                    [
                        (
                            "[SOURCE "
                            f"{source.source_id}]"
                        ),
                        (
                            "FILE: "
                            f"{source.file_name}"
                        ),
                        (
                            "PAGE: "
                            f"{page_value}"
                        ),
                        "CONTENT:",
                        source.content,
                        (
                            "[END SOURCE "
                            f"{source.source_id}]"
                        ),
                    ]
                )
            )

        return "\n\n".join(blocks)

    @classmethod
    def _build_prompt(
        cls,
        sources: list[_SourceRecord],
    ) -> str:
        """Build the evidence-grounded extraction instruction."""

        source_text = cls._format_sources(
            sources
        )

        return (
            "You are extracting invoice billing requirements "
            "from business documents.\n\n"
            "Extract only requirements explicitly supported by "
            "the supplied sources.\n\n"
            "Rules:\n"
            "1. Do not guess or infer missing information.\n"
            "2. If a scalar field is not explicitly supported, "
            "return null.\n"
            "3. For required_attachments, return null when the "
            "sources do not specify attachment requirements.\n"
            "4. Do not infer that a PO is required merely because "
            "a PO number appears somewhere.\n"
            "5. Every populated field must have at least one "
            "evidence item.\n"
            "6. Each evidence quote must be an exact excerpt from "
            "the referenced source. Preserve the source wording.\n"
            "7. Use only the SOURCE IDs supplied below.\n"
            "8. Evidence field must identify the exact requirement "
            "that the quote supports.\n"
            "9. If documents conflict and the conflict cannot be "
            "resolved explicitly, do not guess.\n\n"
            "Fields to extract:\n"
            "- po_required\n"
            "- po_number\n"
            "- payment_terms\n"
            "- milestone_approval_required\n"
            "- billing_entity\n"
            "- project_code\n"
            "- required_attachments\n\n"
            "SOURCE DOCUMENTS:\n\n"
            f"{source_text}"
        )

    @classmethod
    def _validate_evidence(
        cls,
        evidence_items: list[_LLMEvidence],
        sources: list[_SourceRecord],
    ) -> list[BillingRequirementEvidence]:
        """
        Verify every LLM evidence quote against the source documents.
        """

        source_map = {
            source.source_id: source
            for source in sources
        }

        verified: list[
            BillingRequirementEvidence
        ] = []

        for evidence in evidence_items:
            source = source_map.get(
                evidence.source_id
            )

            if source is None:
                raise BillingRequirementsExtractionError(
                    "The extractor returned an unknown "
                    f"source ID: {evidence.source_id}"
                )

            quote = evidence.quote.strip()

            if not quote:
                raise BillingRequirementsExtractionError(
                    "The extractor returned empty evidence."
                )

            normalized_quote = (
                cls._normalize_text(
                    quote
                )
            )

            normalized_source = (
                cls._normalize_text(
                    source.content
                )
            )

            if (
                normalized_quote
                not in normalized_source
            ):
                raise BillingRequirementsExtractionError(
                    "Extracted evidence could not be "
                    "verified against the source document."
                )

            verified.append(
                BillingRequirementEvidence(
                    field=evidence.field,
                    source_id=source.source_id,
                    file_name=source.file_name,
                    page=source.page,
                    quote=quote,
                )
            )

        return verified

    @staticmethod
    def _validate_evidence_coverage(
        result: _LLMBillingRequirements,
        evidence: list[
            BillingRequirementEvidence
        ],
    ) -> None:
        """
        Require evidence for every populated billing requirement.
        """

        evidence_fields = {
            item.field
            for item in evidence
        }

        for field in BillingRequirementField:
            value = getattr(
                result,
                field.value,
            )

            if (
                value is not None
                and field not in evidence_fields
            ):
                raise BillingRequirementsExtractionError(
                    "Extracted billing requirement "
                    f"'{field.value}' has no verified evidence."
                )

    def extract(
        self,
        documents: list[Document],
    ) -> BillingRequirements:
        """
        Extract and verify billing requirements from source documents.
        """

        sources = self._prepare_sources(
            documents
        )

        prompt = self._build_prompt(
            sources
        )

        logger.info(
            "Extracting billing requirements "
            "from %s source document(s).",
            len(sources),
        )

        try:
            raw_result = (
                self.structured_llm.invoke(
                    prompt
                )
            )

            if isinstance(
                raw_result,
                _LLMBillingRequirements,
            ):
                result = raw_result
            else:
                result = (
                    _LLMBillingRequirements.model_validate(
                        raw_result
                    )
                )
        except (
            ValidationError,
            TypeError,
            ValueError,
        ) as exc:
            raise BillingRequirementsExtractionError(
                "The language model returned an invalid "
                "billing requirements structure."
            ) from exc

        evidence = self._validate_evidence(
            result.evidence,
            sources,
        )

        self._validate_evidence_coverage(
            result,
            evidence,
        )

        return BillingRequirements(
            po_required=result.po_required,
            po_number=result.po_number,
            payment_terms=result.payment_terms,
            milestone_approval_required=(
                result.milestone_approval_required
            ),
            billing_entity=result.billing_entity,
            project_code=result.project_code,
            required_attachments=(
                result.required_attachments
            ),
            evidence=evidence,
        )

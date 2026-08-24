from __future__ import annotations

from uuid import UUID

import pytest
from langchain_core.documents import Document

from rag.billing_requirements import (
    BillingRequirements,
)
from rag.billing_requirements_service import (
    BillingRequirementsService,
    BillingRequirementsServiceError,
)
from rag.models import (
    Document as DocumentRecord,
    DocumentType,
)


ORGANIZATION_ID = UUID(
    "11111111-1111-1111-1111-111111111111"
)

DOCUMENT_ID = UUID(
    "22222222-2222-2222-2222-222222222222"
)


class FakeTenantDocumentLoader:
    def __init__(
        self,
        loaded_documents: list[Document],
    ) -> None:
        self.loaded_documents = loaded_documents
        self.calls: list[
            list[DocumentRecord]
        ] = []

    def load_many(
        self,
        documents: list[DocumentRecord],
    ) -> list[Document]:
        self.calls.append(
            documents
        )

        return self.loaded_documents


class FakeBillingRequirementsExtractor:
    def __init__(
        self,
        result: BillingRequirements,
    ) -> None:
        self.result = result
        self.calls: list[
            list[Document]
        ] = []

    def extract(
        self,
        documents: list[Document],
    ) -> BillingRequirements:
        self.calls.append(
            documents
        )

        return self.result


def _make_record(
    document_type: str = (
        DocumentType.CONTRACT.value
    ),
) -> DocumentRecord:
    return DocumentRecord(
        id=DOCUMENT_ID,
        organization_id=ORGANIZATION_ID,
        uploaded_by_user_id=None,
        original_filename="contract.md",
        content_type="text/markdown",
        size_bytes=100,
        storage_key=(
            f"{ORGANIZATION_ID}/"
            f"{DOCUMENT_ID}.md"
        ),
        document_type=document_type,
    )


def test_extract_connects_loader_and_extractor() -> None:
    record = _make_record()

    loaded_document = Document(
        page_content=(
            "Payment terms are Net 45."
        ),
        metadata={
            "business_document_type": (
                DocumentType.CONTRACT.value
            ),
        },
    )

    loader = FakeTenantDocumentLoader(
        [loaded_document]
    )

    expected = BillingRequirements(
        payment_terms="Net 45"
    )

    extractor = (
        FakeBillingRequirementsExtractor(
            expected
        )
    )

    service = BillingRequirementsService(
        document_loader=loader,  # type: ignore[arg-type]
        extractor=extractor,  # type: ignore[arg-type]
    )

    result = service.extract(
        [record]
    )

    assert result == expected

    assert loader.calls == [
        [record]
    ]

    assert extractor.calls == [
        [loaded_document]
    ]


@pytest.mark.parametrize(
    "document_type",
    [
        DocumentType.CONTRACT.value,
        DocumentType.SOW.value,
        DocumentType.PURCHASE_ORDER.value,
        DocumentType.BILLING_INSTRUCTIONS.value,
    ],
)
def test_extract_accepts_requirement_document_types(
    document_type: str,
) -> None:
    record = _make_record(
        document_type
    )

    loader = FakeTenantDocumentLoader(
        [
            Document(
                page_content="Requirement.",
            )
        ]
    )

    extractor = (
        FakeBillingRequirementsExtractor(
            BillingRequirements()
        )
    )

    service = BillingRequirementsService(
        document_loader=loader,  # type: ignore[arg-type]
        extractor=extractor,  # type: ignore[arg-type]
    )

    service.extract(
        [record]
    )

    assert len(loader.calls) == 1
    assert len(extractor.calls) == 1


@pytest.mark.parametrize(
    "document_type",
    [
        DocumentType.INVOICE.value,
        DocumentType.SUPPORTING_EVIDENCE.value,
        DocumentType.OTHER.value,
    ],
)
def test_extract_rejects_non_requirement_documents(
    document_type: str,
) -> None:
    record = _make_record(
        document_type
    )

    loader = FakeTenantDocumentLoader(
        []
    )

    extractor = (
        FakeBillingRequirementsExtractor(
            BillingRequirements()
        )
    )

    service = BillingRequirementsService(
        document_loader=loader,  # type: ignore[arg-type]
        extractor=extractor,  # type: ignore[arg-type]
    )

    with pytest.raises(
        BillingRequirementsServiceError,
        match="Unsupported billing requirement",
    ):
        service.extract(
            [record]
        )

    assert loader.calls == []
    assert extractor.calls == []


def test_extract_rejects_empty_document_list() -> None:
    loader = FakeTenantDocumentLoader(
        []
    )

    extractor = (
        FakeBillingRequirementsExtractor(
            BillingRequirements()
        )
    )

    service = BillingRequirementsService(
        document_loader=loader,  # type: ignore[arg-type]
        extractor=extractor,  # type: ignore[arg-type]
    )

    with pytest.raises(
        BillingRequirementsServiceError,
        match="No billing requirement documents",
    ):
        service.extract(
            []
        )

    assert loader.calls == []
    assert extractor.calls == []

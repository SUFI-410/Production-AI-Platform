# Production AI Platform Progress

## Current Product Direction

The existing production RAG platform is being reused as infrastructure for:

**AI Invoice Preflight for US B2B service companies.**

Core promise:

> Catch payment-blocking invoice mistakes before the customer sees the invoice.

Initial customer profile:

- MSP and IT service companies
- software and development agencies
- engineering and implementation consultancies
- larger marketing and digital agencies
- professional-services and staffing companies
- approximately 10-100 employees
- businesses working with POs, SOWs, milestones, billing instructions, and enterprise customers

## Invoice Preflight Architecture

```text
Contract / SOW / Purchase Order / Billing Instructions
                            |
                            v
            Billing Requirements Extraction
                            |
                            v
                    Invoice Upload
                            |
                            v
                Invoice Facts Extraction
                            |
                            v
             Deterministic Python Comparison
                            |
                            v
               PASS / WARNING / BLOCKER
                            |
                            v
                  Payment Readiness
```

The language model extracts factual, evidence-backed fields.

Python code makes all final `PASS`, `WARNING`, `BLOCKER`, `READY`, `REVIEW_REQUIRED`, and `BLOCKED` decisions.

## Completed Platform Foundation

- Production FastAPI backend
- React frontend
- PostgreSQL tenancy
- JWT authentication
- Tenant document metadata
- Hybrid retrieval with BM25 and Chroma
- Reciprocal Rank Fusion
- Cross-Encoder reranking
- Adaptive retrieval
- Query rewriting and multi-query retrieval
- Conversation memory and session isolation
- Response caching
- Groundedness checking
- Website crawling and incremental indexing
- Docker deployment
- Oracle Cloud hosting
- Nginx and HTTPS
- Cloudflare Turnstile
- GitHub Actions CI/CD

## Completed Invoice Preflight Backend MVP

- Durable tenant document storage
- Atomic document writes
- Storage-key validation and path traversal protection
- Business document classification
- Tenant-safe document loading
- Original customer filenames preserved in evidence
- Grounded billing-requirements extraction
- Grounded invoice-facts extraction
- Source-ID validation
- Exact evidence-quote verification
- Rejection of hallucinated evidence
- Deterministic invoice comparison engine
- Payment-readiness calculation
- Authenticated billing-requirements API
- Authenticated invoice-preflight API
- Tenant-filtered database queries
- Cross-tenant document-mixing protection
- Explicit API error mapping

## Verified Real API Flows

### Defective invoice

A real invoice with the wrong PO, wrong payment terms, missing project code, and missing milestone certificate returned:

```text
BLOCKED
```

### Corrected invoice

A corrected invoice containing the required PO, payment terms, billing entity, project code, and milestone certificate returned:

```text
READY
```

All six findings returned `PASS`.

## Validation Status

- Full automated suite: 155 passed
- Ruff: all checks passed
- Git whitespace validation: passed
- Existing warnings: two non-blocking dependency maintenance warnings

## Current Git Checkpoint

Branch:

```text
feature/invoice-preflight-mvp
```

Latest functional commit:

```text
2c33c46 Normalize invoice attachment name matching
```

The branch is pushed, clean, zero commits behind `develop`, and eleven functional commits ahead of `develop`.

## Next Tasks

1. Complete documentation review.
2. Create a pull request into `develop`.
3. Verify GitHub Actions checks.
4. Review and merge only after CI passes.
5. Plan the Invoice Preflight user interface and product-validation workflow.

Do not merge directly into `main`.

## Explicitly Outside the MVP

- QuickBooks integration
- Gmail automation
- payment processing
- collections workflows
- invoice reminders
- generic accounting-suite features
- mobile application

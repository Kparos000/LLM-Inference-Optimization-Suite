# Data Directory Policy

## Purpose

This directory contains schemas, source registries, tiny curated samples, KB
fixtures, and eval fixtures used to validate the benchmark harness. Phase 2A uses
this directory to document data provenance, storage boundaries, and commit policy
before any scaled prompt generation, RAG implementation, or GPU inference.

## What May Be Committed

- Schemas.
- Source registries.
- Tiny curated JSONL samples.
- Synthetic toy samples.
- Metadata manifests.
- Gold/eval toy records.
- Small KB fixtures.

## What Must Not Be Committed

- Large raw SEC filings.
- Large Amazon Reviews dataset files.
- Bulk PDFs unless explicitly curated.
- Generated 10,000-prompt corpora unless deliberately approved.
- Tokens, credentials, private paths.
- Raw personal, sensitive, or non-public user data.

## Local Data Convention

Use repo-relative paths for local data locations:

- `data/raw/finance/sec/`
- `data/processed/finance/`
- `data/raw/retail/amazon_reviews_2023/`
- `data/processed/retail/`
- `data/raw/research_ai/`
- `data/processed/research_ai/`
- `data/generated/airline/`
- `data/generated/healthcare_admin/`

Large local data should remain ignored unless the project deliberately curates a
small sample for public review.

## Three Data Assets Per Vertical

Every vertical must eventually provide:

1. Source records or source documents.
2. KB/policy/context records.
3. Gold/eval records.

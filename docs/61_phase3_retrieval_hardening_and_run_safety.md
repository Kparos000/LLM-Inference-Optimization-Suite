# Phase 3 Retrieval Hardening And Run Safety

Phase 3 Block 4.5 hardens retrieval before Phase 4 plumbing and before any
API-priced, gated, or GPU model work. This block does not run inference, create
embeddings, call external APIs, implement SGLang, or modify the promoted
benchmark dataset.

## Why Hardening Was Needed

The first retrieval pass was useful but not strong enough for the final
benchmark. On the final 10,000-record split, hybrid recall@5 was about 0.50 and
finance was the weakest vertical. Compression also retained almost all tokens,
so `mm3_compressed_hybrid_top5` was not meaningfully different from
`mm2_hybrid_top5`.

Two problems were identified:

- Some gold/eval IDs, especially finance section IDs, were stored in context
  metadata such as `section_record_id`, but evaluation only matched a smaller
  ID set.
- Retrieval query text did not use structured source hints already present in
  prompt records, such as ticker, filing form, required source IDs, paper IDs,
  product IDs, and policy IDs.

## Retrieval Changes

- Context ID matching now uses exact and normalized forms of context IDs,
  source IDs, parent IDs, chunk IDs, original document IDs, and flattened
  metadata values.
- Prompt query construction now includes structured source hints present in the
  prompt records. Gold/eval rows are still not used to retrieve evidence.
- Hybrid retrieval adds metadata boosts for exact source identifiers, title and
  metadata overlap, and finance-specific fields.
- Finance scoring now boosts ticker, company, filing form, concept/metric,
  fiscal period/year, section type, and source/parent identifier matches.
- Duplicate retrieved chunks are removed before final top-k selection.
- Dense retrieval remains explicitly labeled `local_fallback`; it is not treated
  as a real embedding model.

## Finance Improvements

Finance retrieval now uses metadata that is available at workload-building time:

- `ticker`
- `company`
- `filing_form`
- filing/report dates when present
- section type
- concept/metric names
- source and parent evidence identifiers

The diagnostic report separates likely retrieval misses from evaluation mapping
issues so finance failures are easier to debug before GPU experiments.

## Compression Changes

`mm3_compressed_hybrid_top5` now performs deterministic compression:

- stronger duplicate removal by normalized text
- low-score filtering
- max context token enforcement from memory-mode config
- extractive truncation of selected chunks
- provenance and citation metadata preservation
- no all-context removal when retrieval found evidence

Compression diagnostics report original tokens, compressed tokens, token
reduction percentage, recall before compression, recall after compression,
recall loss, and whether gold evidence remains represented after compression.

## Generated Reports

Retrieval and compression reports live under
`data/generated/context_engineering/`:

- `retrieval_evaluation_report.json`
- `retrieval_evaluation_summary.csv`
- `retrieval_diagnostic_report.json`
- `retrieval_diagnostic_summary.csv`
- `compression_diagnostic_report.json`
- `compression_diagnostic_summary.csv`

The large workload files are regenerated locally under `data/workloads/` and are
ignored by git.

## Run Safety Audit

The user-facing concern is long-run inference safety, not training. The current
repo already has reusable support in the OpenAI-compatible load runner:

- chunked result persistence
- checkpoint JSON with `completed_prompt_ids`
- resume mode that avoids duplicate prompt processing
- appended result CSV and generation JSONL outputs
- run metadata JSON
- progress logs with processed, chunk, success, failure, elapsed time, request
  rate, and checkpoint status
- per-prompt success and error messages

Remaining Phase 4 gaps:

- adapt Phase 3 `WorkloadRecord` JSONL into runner `WorkloadItem` inputs
- include `memory_mode`, dataset split, retrieval backend, and context-token
  summaries in run metadata and logs
- standardize per-run output directories
- add a batch evaluator CLI over generation JSONL outputs

Remaining Phase 5 gaps:

- live GPU telemetry sampling
- structured JSONL logs for dashboard ingestion
- backend OOM/timeout/retry classification
- checkpoint/resume coverage for every backend selected for main experiments

Run-safety outputs:

- `data/generated/context_engineering/run_safety_audit_report.json`
- `data/generated/context_engineering/run_safety_audit_summary.csv`

## Why This Comes Before API-Priced Models

Retrieval quality, compression behavior, and long-run safety need to be
measured before using gated models or paid inference. Otherwise expensive runs
could measure broken context selection, duplicate work after interruptions, or
missing evaluator metadata rather than meaningful model/backend differences.

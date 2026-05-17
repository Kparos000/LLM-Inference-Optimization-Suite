# Phase 2 Finance SEC/XBRL Pilot

## Purpose

This pilot is the first real-data engineering step for Phase 2A. The finance
vertical simulates an equity analyst or finance company researching large
technology stocks using public SEC data.

This pilot does not yet generate prompts, run RAG, run GPU inference, or create
benchmark claims. The current finance pilot stages are limited to source
planning, SEC JSON exploration, selected filing document acquisition, local text
extraction, and section-candidate manifests.

## Finance Data Assets

The finance vertical needs three data assets:

1. Main documents / KB context:
   - SEC 10-K filings.
   - SEC 10-Q filings.
   - Earnings-related 8-K filings.
   - Extracted filing sections.
   - Filing tables where feasible.

2. Prompt/source records:
   - Finance question records derived later from company, filing, period,
     concept, and task templates.

3. Gold/eval records:
   - XBRL companyfacts.
   - Required document IDs.
   - Required citations.
   - Formulas.
   - Numeric answers.
   - Tolerances.
   - Evidence spans where available.

## Approved Company Universe

| Company | Ticker | CIK | Fiscal Year End | SEC Submissions URL | SEC Companyfacts URL |
| --- | --- | --- | --- | --- | --- |
| Apple Inc. | AAPL | 0000320193 | 0926 | https://data.sec.gov/submissions/CIK0000320193.json | https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json |
| Microsoft Corporation | MSFT | 0000789019 | 0630 | https://data.sec.gov/submissions/CIK0000789019.json | https://data.sec.gov/api/xbrl/companyfacts/CIK0000789019.json |
| NVIDIA Corporation | NVDA | 0001045810 | 0126 | https://data.sec.gov/submissions/CIK0001045810.json | https://data.sec.gov/api/xbrl/companyfacts/CIK0001045810.json |
| Amazon.com, Inc. | AMZN | 0001018724 | 1231 | https://data.sec.gov/submissions/CIK0001018724.json | https://data.sec.gov/api/xbrl/companyfacts/CIK0001018724.json |
| Alphabet Inc. | GOOGL | 0001652044 | 1231 | https://data.sec.gov/submissions/CIK0001652044.json | https://data.sec.gov/api/xbrl/companyfacts/CIK0001652044.json |
| Meta Platforms, Inc. | META | 0001326801 | 1231 | https://data.sec.gov/submissions/CIK0001326801.json | https://data.sec.gov/api/xbrl/companyfacts/CIK0001326801.json |
| Tesla, Inc. | TSLA | 0001318605 | 1231 | https://data.sec.gov/submissions/CIK0001318605.json | https://data.sec.gov/api/xbrl/companyfacts/CIK0001318605.json |
| Advanced Micro Devices, Inc. | AMD | 0000002488 | 1228 | https://data.sec.gov/submissions/CIK0000002488.json | https://data.sec.gov/api/xbrl/companyfacts/CIK0000002488.json |

## Why Fiscal Period Consistency Matters

Companies may have different fiscal year ends, so the benchmark must preserve:

- `filing_date`
- `report_date`
- `fiscal_year`
- `fiscal_period`
- `form`
- `accession_number`
- `primary_document`

Cross-company comparisons should not blindly assume that FY2025 means the same
calendar period for every company.

## Target Filing Selection Rules

10-K:

- Form equals `10-K`.
- Filing date or report date from 2024 onward.
- Include FY2024 and FY2025 where available.

10-Q:

- Form equals `10-Q`.
- Filing date or report date from 2024 onward.
- Prioritize 2026 quarterlies for current-period analysis.
- Keep 2024 and 2025 quarterlies available for trends.

8-K:

- Form equals `8-K`.
- Filing date from 2024 onward.
- Require item 2.02; 9.01 is retained as supporting exhibit metadata when present.
- 2.02 means Results of Operations and Financial Condition.
- 9.01 means Financial Statements and Exhibits, but 9.01 alone can appear in
  non-earnings 8-Ks.
- Phase 2A requires 2.02 for earnings-related 8-K inclusion to reduce noisy 8-K
  documents before Phase 2A-3C HTML download.

## SEC URL Derivation

Filing document URLs are derived from:

- CIK without leading zeros.
- Accession number without dashes.
- `primaryDocument`.

Use this pattern:

```text
https://www.sec.gov/Archives/edgar/data/{cik_without_leading_zeros}/{accession_without_dashes}/{primary_document}
```

Microsoft example:

```text
CIK: 0000789019
Accession number: 0001193125-26-191507
Primary document: msft-20260331.htm
Derived URL:
https://www.sec.gov/Archives/edgar/data/789019/000119312526191507/msft-20260331.htm
```

## SEC Access Rules

Automated SEC access must:

- Use a declared User-Agent header.
- Avoid excessive requests.
- Respect SEC fair-access rules.
- Default to dry-run before download.
- Keep large raw files local/ignored unless curated.

Use a placeholder contact in documentation and examples:

```text
User-Agent: LLM-Inference-Optimization-Suite research-contact@example.com
```

## Pilot Stages

2A-3A:

- Ticker registry.
- SEC URL planner.
- Dry-run acquisition script.
- No downloads.

2A-3B:

- Download submissions JSON.
- Download companyfacts JSON.
- Build selected filings manifest.
- Build XBRL inventory.
- Produce exploration report.

2A-3C:

- Download selected filing HTML documents.
- Create selected filing document manifest.

2A-3D:

- Extract text and sections.
- Build local extraction manifests and report.
- Still no prompt generation, RAG, retrieval indexing, or inference.

2A-3E:

- Create finance KB samples.
- Create finance gold/eval samples.
- Still no 10,000-prompt generation.

## Current Step Completion Criteria

2A-3A is complete only when:

- Finance ticker registry exists.
- Dry-run planner script exists.
- Dry-run output can be produced without network calls.
- Tests pass.
- No raw SEC files are committed.

## Phase 2A-3B JSON Acquisition and Exploration

Phase 2A-3B downloads submissions JSON and companyfacts JSON only. Filing HTML/TXT
downloads are deferred to 2A-3C.

Raw SEC JSON is stored locally under:

- `data/raw/finance/sec/submissions/`
- `data/raw/finance/sec/companyfacts/`

Processed exploration artifacts are stored locally under:

- `data/processed/finance/sec/selected_filings_manifest.jsonl`
- `data/processed/finance/sec/xbrl_concept_inventory.jsonl`
- `data/processed/finance/sec/finance_sec_exploration_report.json`

The selected filings manifest turns SEC `filings.recent` column arrays into
row-level JSONL records. It is intentionally conservative for 8-Ks: earnings
8-K inclusion requires item 2.02, while item 9.01 is retained as supporting
exhibit metadata when present. The XBRL inventory summarizes available `us-gaap`
concepts, observation counts, units, forms, fiscal years, fiscal periods, and
important concept coverage. The exploration report should be reviewed before any
filing document download.

Example commands:

```text
python scripts/phase2/finance_sec_acquisition.py --download-json --company MSFT
```

```text
python scripts/phase2/finance_sec_acquisition.py --summarize-local --company MSFT
```

```text
python scripts/phase2/finance_sec_acquisition.py --summarize-local --company all
```

Do not use these JSON files to make benchmark claims yet. They are source
acquisition and exploration artifacts only.

## Phase 2A-3C Filing Document Acquisition

Phase 2A-3C downloads selected SEC filing HTML documents using
`selected_filings_manifest.jsonl`. It only downloads documents already selected
in 2A-3B: 10-K annual filings, 10-Q quarterly filings, and earnings-related
8-K filings where item 2.02 is present.

This step does not parse text, chunk documents, create prompts, run RAG, or run
inference. Raw filing HTML is stored locally under:

- `data/raw/finance/sec/filings/`

Document download metadata is written locally to:

- `data/processed/finance/sec/selected_filing_documents_manifest.jsonl`
- `data/processed/finance/sec/filing_download_report.json`

Raw filing documents and generated manifests remain local/ignored unless later
curated. Phase 2A-3D is responsible for text extraction and section parsing.

Example commands:

```text
python scripts/phase2/finance_sec_acquisition.py --download-filings --company MSFT --limit 5 --skip-existing
```

```text
python scripts/phase2/finance_sec_acquisition.py --download-filings --company MSFT --form 10-K --limit 2 --skip-existing
```

```text
python scripts/phase2/finance_sec_acquisition.py --download-filings --company all --skip-existing
```

## Phase 2A-3D Text Extraction and Section Parsing

Phase 2A-3D reads downloaded SEC filing HTML documents from 2A-3C. It extracts
readable plain text and creates section candidates for finance-relevant parts of
10-K, 10-Q, and earnings 8-K documents.

This step does not create retrieval chunks, run RAG, generate prompts, build
retrieval indexes, run inference, or create final gold/eval records. Section
extraction is heuristic and will be refined before Phase 2B context engineering.

Generated local outputs are:

- `data/processed/finance/sec/extracted_text/`
- `data/processed/finance/sec/filing_text_manifest.jsonl`
- `data/processed/finance/sec/filing_sections_manifest.jsonl`
- `data/processed/finance/sec/finance_text_extraction_report.json`

Example commands:

```text
python scripts/phase2/finance_sec_acquisition.py --extract-text --company MSFT --limit 5
```

```text
python scripts/phase2/finance_sec_acquisition.py --extract-text --company MSFT --form 10-K --limit 2
```

```text
python scripts/phase2/finance_sec_acquisition.py --extract-text --company all
```

Phase 2A-3E is responsible for creating curated finance source, KB, and
gold/eval samples. Full context engineering and RAG remain deferred until all
five Phase 2A verticals have their data assets ready.

## Phase 2A-3D-QA Section Quality Audit

Phase 2A-3D-QA audits extracted finance section candidates before curated
finance sample creation. It is still part of data preparation, not RAG or
context engineering.

The audit checks for noisy section titles, high section counts, zero-section
documents, suspicious heading candidates, long section titles, and short
non-8-K sections. The tightened section heuristics reduce paragraph fragments
being treated as section headings by requiring candidate titles to look like SEC
item headings or clear finance headings.

The generated QA report is written locally to:

- `data/processed/finance/sec/finance_section_quality_report.json`

Generated reports remain local/ignored unless later curated.

Example commands:

```text
python scripts/phase2/finance_sec_acquisition.py --audit-sections --company MSFT
```

```text
python scripts/phase2/finance_sec_acquisition.py --audit-sections --company all
```

Phase 2A-3E should only proceed after section quality is acceptable for curated
sample generation. Full RAG and context engineering remain deferred until all
five Phase 2A vertical datasets are prepared.

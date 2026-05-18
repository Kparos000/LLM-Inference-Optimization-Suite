# Phase 2A-5A AI Research Paper Discovery

## Purpose

Phase 2A-5A discovers candidate AI research papers for the AI Research
Assistant / Education-Research vertical. This is a metadata-only discovery
step that creates a query plan, candidate metadata records, a review CSV, a
small committed candidate sample, and a local discovery report.

This step does not download PDFs, parse full text, generate prompts, implement
RAG, run inference, call models, create embeddings, or make benchmark claims.

## Data Source

The original primary source is the arXiv API. The API returns Atom XML
containing paper metadata such as title, abstract, authors, categories,
updated/published dates, and links to abstract and PDF pages. Because arXiv API
requests have repeatedly returned `HTTP 429` in this environment, live
discovery now also supports public HTML metadata discovery from ICLR 2025 and
Hugging Face Papers. Phase 2A-5A records metadata only. PDF download and paper
parsing are deferred.

## Query Plan

The discovery plan uses seven query groups:

- LLM serving and inference optimization
- vLLM / PagedAttention / continuous batching
- speculative decoding / KV cache / prefix caching
- RAG / context engineering
- LLM routing / model selection
- agentic workflows / tool use
- small language models / efficient LLMs

The query plan is stored in:

- `data/sources/research_ai_query_plan.json`

## Multi-source Discovery

The arXiv API remains available for later retry, but it is disabled by default
for live discovery because it has repeatedly returned `HTTP 429` in the current
environment. The practical default discovery path uses:

- ICLR 2025 virtual papers
- Hugging Face Papers

HTML discovery extracts candidate metadata only. Some candidate records may have
titles and provenance links without abstracts, authors, or PDF links. Missing
metadata can be completed during manual approval. The manual approved registry
remains the final gate before Phase 2A-5B.

Example commands:

```text
python scripts/phase2/discover_research_ai_papers.py --dry-run --source all
```

```text
python scripts/phase2/discover_research_ai_papers.py --discover --source iclr
```

```text
python scripts/phase2/discover_research_ai_papers.py --discover --source huggingface
```

```text
python scripts/phase2/discover_research_ai_papers.py --discover --source all
```

```text
python scripts/phase2/discover_research_ai_papers.py --discover --source arxiv --query-id llm_serving_inference_optimization --max-results-per-query 3 --delay-seconds 30 --simple-query-mode
```

Phase 2A-5B should not proceed directly from noisy scraped candidates. It should
proceed from reviewed and approved papers in
`data/sources/research_ai_approved_papers.jsonl`.

## Approved Paper Window

Research AI candidate papers should be from January 1, 2024 to May 30, 2026.
arXiv discovery uses `submittedDate` filtering when metadata discovery is
available, using the configured range:

- `submittedDate:[202401010000 TO 202605302359]`

The end date is a configured project target. Discovery can only return papers
available from arXiv at runtime, and metadata should be reviewed before any
paper is approved for Phase 2A-5B.

## Candidate Review Workflow

1. Run discovery.
2. Review the generated `candidate_papers_review.csv`.
3. Select 12-20 papers for the first approved research corpus.
4. Proceed to Phase 2A-5B.

The committed sample file,
`data/sources/research_ai_candidate_papers_sample.jsonl`, shows the candidate
metadata shape without committing the full generated local discovery output. If
arXiv access is unavailable during development, this file may contain a clearly
marked schema/example row rather than real discovery metadata; rerun discovery
before approving a paper shortlist.

## Status Taxonomy

- `answer`: The question is in scope and answerable from the provided data.
- `escalate`: The question is in scope, but unclear, ambiguous, conflicting, or
  requires human/expert review.
- `insufficient_evidence`: The question is in scope, but the available corpus
  does not provide enough evidence.
- `out_of_scope`: The question is outside the selected vertical/corpus. The
  model may know the answer from general world knowledge, but a grounded/RAG or
  agentic system should not answer from this corpus.
- `spam_or_fraud`: The request is spam, abusive, fraudulent, or intentionally
  irrelevant for support-style verticals.
- `boundary_response`: The request hits a safety/privacy/clinical/admin
  boundary and requires a safe boundary response.

For example, a question such as "When will the next World Cup be played?" is
`out_of_scope` for the AI Research Assistant corpus even if a base model might
know the answer.

## Example Commands

```text
python scripts/phase2/discover_research_ai_papers.py --dry-run
```

```text
python scripts/phase2/discover_research_ai_papers.py --discover
```

```text
python scripts/phase2/discover_research_ai_papers.py --discover --sample-size 20
```

## Handling arXiv Rate Limits

arXiv may return `HTTP 429` when metadata requests are throttled. The discovery
script supports retry/backoff, request timeouts, one-query execution, and
partial-output reporting so discovery can proceed conservatively without
claiming more than the API returned.

Use smaller max results and one query at a time when throttled. The default
query-plan delay is 3 seconds, but throttled environments should use
`--delay-seconds 10` or higher. Phase 2A-5A still does not download PDFs. If
arXiv is unavailable, do not fake successful discovery; keep the schema/example
sample until real metadata is collected or a manually approved paper registry is
created.

Example one-query retry command:

```text
python scripts/phase2/discover_research_ai_papers.py --discover --query-id llm_serving_inference_optimization --max-results-per-query 5 --delay-seconds 10 --max-retries 5 --backoff-seconds 20
```

Example partial discovery command:

```text
python scripts/phase2/discover_research_ai_papers.py --discover --query-id all --max-results-per-query 5 --delay-seconds 10 --continue-on-error --allow-partial
```

Phase 2A-5B should only proceed once real candidate papers have been discovered
or a manually approved paper registry has been created.

## Discovery Observability and Failure Logs

The discovery script writes a discovery report even when every selected arXiv
query fails. It also writes a run log JSONL file for dry-run, success, partial,
and failed runs. Generated reports and logs are local/ignored artifacts.

The run log captures:

- HTTP status when available
- exception type
- attempt count
- `Retry-After` header when present
- request URL
- response body snippet when available
- elapsed request time when available

Use this command shape when arXiv returns repeated throttling responses:

```text
python scripts/phase2/discover_research_ai_papers.py --discover --query-id llm_serving_inference_optimization --max-results-per-query 3 --delay-seconds 30 --max-retries 2 --backoff-seconds 60 --simple-query-mode
```

The date-filtered dry-run should show the `submittedDate` filter in the planned
URL:

```text
python scripts/phase2/discover_research_ai_papers.py --dry-run --query-id llm_serving_inference_optimization --simple-query-mode
```

## Manual Paper Registry Fallback

If arXiv discovery remains unavailable due to HTTP 429 responses or network
restrictions, the project may proceed by manually approving a small 12-20 paper
registry. The manual fallback must still preserve provenance URLs, arXiv IDs,
titles, authors, abstracts where available, and `reason_for_inclusion`.

Do not fabricate paper metadata. Phase 2A-5B can use either discovered
candidates or a manually approved registry, as long as provenance is clear.

Create the manual review template with:

```text
python scripts/phase2/discover_research_ai_papers.py --write-manual-template
```

## Manual Registry Path When arXiv Is Rate-Limited

Current arXiv API access may return `HTTP 429 Rate exceeded`. If live discovery
keeps failing, use the manual-approved registry workflow. Manual fallback is
acceptable only when provenance is preserved. Required fields include arXiv ID,
title, authors, published date, abstract URL, PDF URL, topic, and
`reason_for_inclusion`.

Do not fabricate paper metadata. Do not proceed to Phase 2A-5B with
`example_not_approved` records.

Manual registry workflow:

1. Generate the template:

```text
python scripts/phase2/discover_research_ai_papers.py --write-manual-template
```

2. Manually create:

```text
data/sources/research_ai_approved_papers.jsonl
```

3. Validate the registry:

```text
python scripts/phase2/discover_research_ai_papers.py --validate-manual-registry
```

4. Proceed to Phase 2A-5B only after validation passes.

Conservative date-filtered discovery can still be retried later:

```text
python scripts/phase2/discover_research_ai_papers.py --discover --query-id llm_serving_inference_optimization --max-results-per-query 3 --delay-seconds 30 --max-retries 2 --backoff-seconds 60 --simple-query-mode
```

## Approved Paper Registry

The approved paper registry is created from reviewed discovery candidates. For
this phase, ICLR 2025 candidate metadata is acceptable even when arXiv IDs,
authors, or PDF URLs are missing from the public listing page. The registry must
preserve provenance URLs and remains metadata-only; no PDFs are downloaded.

The approved registry is the input to Phase 2A-5B. Do not proceed to 2A-5B from
the raw candidate CSV. Proceed only from validated approved registry records.

Build the approved registry:

```text
python scripts/phase2/discover_research_ai_papers.py --build-approved-registry
```

Validate the approved registry:

```text
python scripts/phase2/discover_research_ai_papers.py --validate-manual-registry --manual-registry-path data/sources/research_ai_approved_papers.jsonl
```

## Phase 2A-5A-Text Paper Detail Acquisition

The approved registry is metadata-only and is not enough for high-quality
method, result, or limitation prompts. Phase 2A-5A-Text enriches approved ICLR
records with available ICLR/OpenReview metadata, downloads PDFs only when paper
links are available, and extracts local text only when PDF extraction succeeds.

This step creates enriched metadata, paper text manifests, and simple section
manifests for later review. It does not perform RAG, retrieval, embeddings,
prompt assembly, inference, or final Research AI prompt/gold generation.

Plan the work without network calls:

```text
python scripts/phase2/prepare_research_ai_papers.py --dry-run
```

Fetch available ICLR/OpenReview metadata:

```text
python scripts/phase2/prepare_research_ai_papers.py --enrich-metadata
```

Download PDFs only for enriched records with PDF URLs:

```text
python scripts/phase2/prepare_research_ai_papers.py --download-pdfs --skip-existing
```

PDF download writes files under:

```text
data/raw/research_ai/papers/<paper_id>/<paper_id>.pdf
```

Extract text from local PDFs where supported:

```text
python scripts/phase2/prepare_research_ai_papers.py --extract-text
```

Run text extraction only after:

```text
python scripts/phase2/prepare_research_ai_papers.py --download-pdfs --skip-existing
```

Regenerate the local preparation report:

```text
python scripts/phase2/prepare_research_ai_papers.py --summarize-local
```

If PDF text extraction is unavailable locally, the script should still preserve
enriched metadata and clearly report skipped text extraction.

## Phase 2A-5A-Text-QA Metadata Quality Gate

Enriched ICLR abstracts must be cleaned before use. Generic OpenReview group
links are not paper-specific provenance, so the preparation script rejects group
URLs such as `openreview.net/group?id=ICLR.cc` unless a paper-specific forum,
PDF, or attachment link is also present.

PDF links are classified before download as `openreview_pdf`, `full_paper_pdf`,
`unknown_pdf`, `slides_pdf`, `poster_pdf`, `supplementary_pdf`, or `missing`.
Slide PDFs should not be treated as full paper bodies. The
`paper_body_available` and `ready_for_text_extraction` fields are the quality
gate for local text extraction and later curation.

Refresh enriched metadata and inspect the quality report:

```text
python scripts/phase2/prepare_research_ai_papers.py --enrich-metadata
```

```text
python -m json.tool data/generated/research_ai/research_ai_paper_preparation_report.json
```

Download only records that are ready for full-paper text extraction:

```text
python scripts/phase2/prepare_research_ai_papers.py --download-pdfs --skip-existing
```

Optionally download non-paper PDFs for manual inspection:

```text
python scripts/phase2/prepare_research_ai_papers.py --download-pdfs --skip-existing --include-non-paper-pdfs
```

Phase 2A-5B should only create method, results, or limitations prompts from
records with enough text evidence. If `paper_body_available_count` is low,
enrich paper links manually or via better source parsing before generating
Research AI gold/eval records.

## Next Step

Phase 2A-5B should create:

- approved paper registry
- KB/context records
- 40 research prompt/source records
- 40 gold/eval records

Full RAG/context engineering remains deferred until all five verticals have
data, KB/context, and gold/eval assets.

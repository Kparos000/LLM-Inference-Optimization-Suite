# Phase 2A-5A AI Research Paper Discovery

## Purpose

Phase 2A-5A discovers candidate AI research papers for the AI Research
Assistant / Education-Research vertical. This is a metadata-only discovery
step that creates a query plan, arXiv metadata records, a review CSV, a small
committed candidate sample, and a local discovery report.

This step does not download PDFs, parse full text, generate prompts, implement
RAG, run inference, call models, create embeddings, or make benchmark claims.

## Data Source

The primary source is the arXiv API. The API returns Atom XML containing paper
metadata such as title, abstract, authors, categories, updated/published dates,
and links to abstract and PDF pages. Phase 2A-5A records metadata only. PDF
download and paper parsing are deferred.

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

## Next Step

Phase 2A-5B should create:

- approved paper registry
- KB/context records
- 40 research prompt/source records
- 40 gold/eval records

Full RAG/context engineering remains deferred until all five verticals have
data, KB/context, and gold/eval assets.

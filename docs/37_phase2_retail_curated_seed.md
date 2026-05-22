# Phase 2A-6C Retail Curated Seed

Phase 2A-6C creates the Retail / E-commerce Support curated seed dataset from
the controlled Amazon Reviews 2023 exploration sample. It produces source
prompt records, KB/context records, and gold/eval records for a small reviewed
seed only.

This phase does not build RAG, retrieval, embeddings, prompt assembly, model
calls, GPU runs, benchmark inference, or the full 5,000-10,000 prompt dataset.
In short: no RAG, no inference, and no embeddings happen in this step.

## Inputs

The curation script expects local generated artifacts from Phase 2A-6B:

- `data/generated/retail/amazon_reviews_sample.jsonl`
- `data/generated/retail/amazon_metadata_sample.jsonl`
- `data/generated/retail/amazon_reviews_exploration_report.json`
- `data/generated/retail/amazon_reviews_quality_report.json`

The generated Amazon samples remain local and ignored. They are not committed.

## Outputs

Committed curated seed outputs:

- `data/real_world_samples/retail_sample.jsonl`
- `data/kb/retail/kb_sample.jsonl`
- `data/eval/gold/retail_gold_sample.jsonl`

Local generated report:

- `data/generated/retail/retail_curation_report.json`

## Prompt Distribution

| Category | Count | Status behavior |
| --- | ---: | --- |
| `review_summary` | 6 | answer |
| `issue_identification` | 7 | answer |
| `compare_products` | 5 | answer |
| `structured_extraction` | 6 | answer |
| `support_policy_reasoning` | 5 | answer |
| `evidence_citation_lookup` | 4 | answer |
| `spam_or_low_quality_review` | 3 | spam_or_low_quality or escalate |
| `insufficient_evidence_or_escalation` | 3 | insufficient_evidence or escalate |
| `out_of_scope` | 1 | out_of_scope |

## KB Strategy

The KB contains derived records for:

- sanitized review evidence
- product metadata
- derived review summaries
- synthetic benchmark support policies

Review evidence uses sanitized Phase 2A-6B sample rows only. Raw customer
identifier fields are not committed. PII-like review text is filtered out.

## Support Policy Simulation

Support policy records are synthetic benchmark policies. They are not Amazon
policy claims. They cover return/refund triage, damaged items, missing items,
wrong items, quality complaints, low-quality review handling, escalation, and
out-of-scope behavior.

## Gold/Eval Strategy

Each prompt has exactly one gold record. Answerable gold records cite committed
KB document IDs. Structured extraction records require JSON-shaped answers with
product ID, product title, issue type, rating, evidence summary, recommended
action, and evidence ID.

Negative-status records are explicit:

- low-quality review prompts should not use weak reviews as strong product
  evidence
- insufficient evidence prompts should not guess missing product, order, or
  eligibility facts
- out-of-scope prompts should not answer from general model memory

## Command

```text
python scripts/phase2/curate_retail_seed.py --build-curated-samples
```

Inspect the report:

```text
python -m json.tool data/generated/retail/retail_curation_report.json
```

## Next Step

Proceed to Phase 2A-7 cross-vertical data QA and scale-up planning after
reviewing the Retail curated samples.

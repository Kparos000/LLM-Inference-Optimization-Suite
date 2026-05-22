# Phase 2A-9A Airline 250 Candidate Review

Phase 2A-9A reviews the generated Airline 250-scale candidate before it is
promoted from local generated output into committed scale-up data.

This step is data QA only. It does not build RAG, retrieval indexes,
embeddings, prompt assembly, model calls, GPU runs, or benchmark inference.
It is a no RAG, no inference, no embeddings promotion gate.

## Candidate Files

Generated candidate files remain local and ignored until they pass review:

- `data/generated/phase2a/scaleup/airline/airline_prompts_250.jsonl`
- `data/generated/phase2a/scaleup/airline/airline_gold_250.jsonl`
- `data/generated/phase2a/scaleup/airline/airline_kb_250.jsonl`
- `data/generated/phase2a/scaleup_reports/airline_scaleup_250_report.json`

## Review Checks

The review script validates:

- 250 prompt records
- 250 gold records
- at least 25 KB records
- unique prompt IDs
- one gold record per prompt
- no orphan gold records
- evidence IDs exist in the KB where applicable
- answerable records have evidence and `must_include`
- approved status distribution: 225 answer, 20 escalate, 5 spam_or_fraud
- approved output format distribution: 190 text, 35 json, 25 markdown_table
- no private paths, secrets, raw user IDs, or placeholder text
- no empty questions or reference answers
- negative examples are present

## Promotion Criteria

Promotion only happens when the review report has:

- `critical_issue_count: 0`
- `warning_count: 0`

Clean candidates are copied to:

- `data/scaleup/airline/airline_prompts_250.jsonl`
- `data/scaleup/airline/airline_gold_250.jsonl`
- `data/scaleup/airline/airline_kb_250.jsonl`

The local review report remains ignored:

- `data/generated/phase2a/scaleup_reports/airline_250_candidate_review_report.json`

## Commands

Generate or refresh the local Airline candidate:

```powershell
python scripts/phase2/generate_phase2a_scaleup.py --generate-vertical --vertical airline --target-per-vertical 250
```

Review the candidate:

```powershell
python scripts/phase2/review_phase2a_scaleup_candidate.py --review-candidate --vertical airline
```

Promote only if the review is clean:

```powershell
python scripts/phase2/review_phase2a_scaleup_candidate.py --promote-if-clean --vertical airline
```

Inspect the review report:

```powershell
python -m json.tool data/generated/phase2a/scaleup_reports/airline_250_candidate_review_report.json
```

## Next Step

After Airline 250 is promoted, extend 250 generation and review to healthcare,
retail, research_ai, and finance. Larger checkpoints should remain blocked until
the previous checkpoint has been generated, reviewed, and promoted.

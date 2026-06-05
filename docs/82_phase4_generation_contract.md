# Phase 4 Generation Contract

Phase 4 now uses one structured output contract for grounded generation across
the local Hugging Face and OpenAI-compatible runner paths.

## Contract

Every model is instructed to return exactly one JSON object:

```json
{
  "answer": "A concise answer grounded in the supplied evidence.",
  "evidence_ids": ["E1"],
  "confidence": 0.8,
  "insufficient_evidence": false,
  "citation_notes": "Direct support."
}
```

Validation requires:

- `answer` is a string and is non-empty for an answer.
- `evidence_ids` is a list of non-empty citation labels.
- `confidence` is between `0` and `1`.
- `insufficient_evidence` is boolean.
- `citation_notes` is a string.
- Insufficient-evidence responses use an empty answer and no evidence IDs.

## Stable Evidence Labels

Retrieved context is rendered in ranked blocks with short labels:

```text
[EVIDENCE 1]
evidence_id: E1
title: ...
source_type: ...
text: ...
```

The runner metadata maps `E1`, `E2`, and subsequent labels to the complete
canonical chunk, parent, document, and source aliases. This avoids making small
models reproduce long database identifiers while retaining deterministic
provenance for evaluation.

## Runner Integration

The Phase 3 workload adapter applies the contract to every exported runner
prompt. Local Hugging Face, OpenAI-compatible, and concurrent OpenAI-compatible
generation records now include:

- workload, vertical, memory-mode, and ablation metadata;
- expected output format and citation alias mapping;
- parsed answer, evidence IDs, confidence, and insufficient-evidence status;
- contract validity, missing fields, and parse error details.

No retrieval behavior or promoted dataset record was changed.

## Evaluation

`scripts/phase4/evaluate_generation_outputs.py` joins generated rows to gold
records by `prompt_id` and reports:

- JSON validity;
- full generation-contract validity;
- evidence-ID presence;
- canonical evidence match after citation-label expansion;
- deterministic groundedness;
- insufficient-evidence correctness;
- safety-term violations.

Groundedness currently means that a valid answer contract cites every required
gold evidence identifier after alias expansion. It is not yet a semantic claim
verification or LLM-judge score.

## Five-Prompt Local HF Smoke

The real smoke used:

- model: `model1_0_5b` (`Qwen/Qwen2.5-0.5B-Instruct`);
- memory mode: `mm2_hybrid_top5`;
- one prompt from each of the five verticals;
- local Transformers execution;
- no paid API call and no GPU experiment.

Observed results:

| Metric | Result |
| --- | ---: |
| Generation success | 5/5 |
| JSON validity | 4/5 (80%) |
| Contract validity | 3/5 (60%) |
| Evidence-ID presence | 4/5 (80%) |
| Full evidence match | 3/5 (60%) |
| Deterministic groundedness | 2/5 (40%) |
| Safety violations | 0/5 |
| Mean latency | 94.892 seconds |
| Median latency | 95.660 seconds |
| Total input tokens | 6,560 |
| Total output tokens | 423 |

Short citation labels fixed truncation caused by long Finance and Research AI
identifiers. Remaining failures were model behavior: Healthcare output was
truncated, Airline returned an empty answer with answer status, Retail omitted
one required evidence family, and Research AI copied wording from the schema
example. These are useful smoke findings, not evaluator plumbing failures.

## Commands

```powershell
python scripts/phase4/run_local_hf_smoke.py `
  --input-path data/generated/phase4/generation_contract_runner_input.jsonl `
  --output-path results/raw/phase4_generation_contract_hf_smoke.jsonl `
  --model-alias model1_0_5b `
  --limit 5 `
  --max-new-tokens 128

python scripts/phase4/evaluate_generation_outputs.py `
  --results-path results/raw/phase4_generation_contract_hf_smoke.jsonl `
  --dataset-root data/scaleup_2000_full `
  --output-root results/processed `
  --report-name phase4_generation_contract_eval_report.json `
  --summary-name phase4_generation_contract_eval_summary.csv
```

Raw smoke output remains local under `results/raw/`. The processed report and
summary remain local under `results/processed/` according to repository output
policy.


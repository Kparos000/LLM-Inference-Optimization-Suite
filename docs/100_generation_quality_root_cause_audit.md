# Phase B3 Generation Quality Root-Cause Audit

Status: `AUDIT_COMPLETE_QUALITY_REMAINS_BLOCKED`

Date: June 15, 2026

## Objective

Phase B3 explains why the 100-prompt Qwen2.5-1.5B vLLM B1 smoke failed its
quality gate before any larger run or concurrency sweep.

The audit is offline and deterministic. It joins:

- the unchanged B1 evaluator rows;
- the raw B1 generations;
- citation alias maps;
- the rendered E1-E5 evidence blocks;
- the frozen B1 runner-input metadata.

It does not run inference, change the evaluator, alter gold data, or modify the
promoted retrieval source of truth.

## Artifacts

The implementation is:

- `src/inference_bench/generation_quality_audit.py`;
- `scripts/phase4/audit_generation_quality_failures.py`.

The generated local reports are:

- `results/processed/b3_generation_quality_audit_report.json`;
- `results/processed/b3_generation_quality_audit_summary.csv`;
- `results/processed/b3_finance_failure_examples.jsonl`;
- `results/processed/b3_quality_failure_examples.jsonl`.

Run:

```powershell
python scripts/phase4/audit_generation_quality_failures.py
```

## Classification Method

Each failed row receives every applicable classification:

- `retrieved_gold_absent_from_context`;
- `evidence_present_but_not_cited`;
- `partial_multi_evidence_citation`;
- `invalid_json`;
- `invalid_contract`;
- `safety_violation`;
- `truncation`;
- `insufficient_evidence_wrongly_used`;
- `answer_semantically_underdeveloped`;
- `finance_metric_period_missing`;
- `context_ordering_issue`;
- `model_instruction_following_failure`.

Classifications overlap. Their counts therefore do not sum to 65.

## Overall Results

B1 had 35 grounded rows and 65 failed rows.

| Failure class | Failed rows |
| --- | ---: |
| Required gold absent from E1-E5 | 52 |
| Semantically underdeveloped answer | 47 |
| Partial multi-evidence citation | 27 |
| Model instruction-following failure | 22 |
| Evidence present but not cited | 18 |
| Finance metric/period-specific source missing | 18 |
| Invalid contract | 8 |
| Invalid JSON | 7 |
| Truncation | 6 |
| Safety violation | 2 |
| Context ordering issue | 2 |
| Incorrect insufficient-evidence use | 0 |

Gold evidence availability among the 65 failed rows:

- all required evidence present: 13;
- some, but not all, required evidence present: 23;
- none of the required evidence present: 29.

The dominant root cause is therefore not one model behavior. The frozen B1
context itself lacked at least one required evidence ID for 52 failures.
Model citation selection remains independently visible in the 18 rows where
available required evidence was not cited.

## Per-Vertical Results

| Vertical | Failed | Gold absent | Present, not cited | Partial citation | Invalid JSON | Invalid contract | Safety | Truncated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Airline | 15 | 11 | 6 | 14 | 1 | 1 | 2 | 1 |
| Healthcare Admin | 9 | 7 | 2 | 8 | 0 | 0 | 0 | 0 |
| Retail | 13 | 12 | 3 | 1 | 2 | 3 | 0 | 2 |
| Finance | 19 | 18 | 1 | 0 | 1 | 1 | 0 | 1 |
| Research AI | 9 | 4 | 6 | 4 | 3 | 3 | 0 | 2 |

Airline and Healthcare Admin are dominated by multi-evidence under-citation
combined with partial context coverage. Retail is dominated by missing required
context and underdeveloped answers. Research AI has the strongest format and
truncation component. Finance is overwhelmingly a context-alignment failure.

## Finance Diagnosis

Finance had 19 failed rows.

- All required gold evidence appeared in E1-E5 for 1 row.
- At least one required gold evidence ID was absent for 18 rows.
- The model cited wrong evidence labels in those same 18 rows.
- One row had both required IDs present but did not cite them because its JSON
  output truncated.
- Ticker and company metadata appeared in all 19 rendered contexts.
- Filing form appeared in 16; the other 3 source prompts did not specify a
  filing form.
- Explicit period and metric values were not present as structured source
  prompt metadata in these 19 rows.
- The exact required SEC/XBRL filing or metric evidence encoded by the gold IDs
  was absent in 18 rows.
- Finance had zero evaluator safety violations and zero detected
  investment/advice/projection wording matches.

The Finance result is primarily a frozen workload/rendered-context alignment
problem, not a safety problem. It is not evidence that the promoted retrieval
source of truth failed: B3 audits the exact context snapshot consumed by B1.
The model still has a secondary citation-selection/truncation problem, proven
by the one failure where required evidence was available.

## Recommended Repair Block

Exact next block:

```text
B3R1_FROZEN_WORKLOAD_CONTEXT_ALIGNMENT_REPAIR
```

Actions:

1. Trace every B1 prompt from promoted retrieval output through workload
   materialization and runner export.
2. Re-export the same 100 prompt IDs without changing gold data, evaluator
   semantics, or the promoted retrieval source.
3. Require every expected evidence ID to map to at least one rendered E1-E5
   alias.
4. Re-run the offline B3 audit and require zero
   `retrieved_gold_absent_from_context` rows.
5. Only after that gate passes, run at most five Finance prompts to isolate
   model citation selection.
6. Treat truncation and prohibited-phrase emission as separate one-factor
   repairs.

Do not increase prompt count or concurrency until context alignment is verified
and the frozen quality gate passes, or a controlled comparison documents the
model-capability limit.

## Decision

The decision remains:

```text
READY_FOR_SMALL_MODEL_SERVING_EXPERIMENTS
QUALITY_BLOCKED_FOR_SCALE
```

No larger benchmark is justified from B1 because the model and the frozen
context path have not yet been isolated cleanly.

# Block 17 Canonical Retrieval Repair Summary

## Files Changed

- `src/inference_bench/retrieval_keys.py`
- `src/inference_bench/canonical_queries.py`
- `src/inference_bench/vertical_final_selectors.py`
- `src/inference_bench/retrieval.py`
- `src/inference_bench/vertical_retrieval_repair.py`
- `scripts/phase3/repair_all_vertical_retrieval.py`
- `tests/test_phase3_canonical_retrieval_repair.py`
- `docs/77_canonical_retrieval_key_materialization.md`
- `data/generated/context_engineering/canonical_retrieval_repair_report.json`
- `data/generated/context_engineering/canonical_retrieval_repair_summary.csv`
- `data/generated/context_engineering/canonical_retrieval_failure_examples.jsonl`
- `data/generated/context_engineering/slo_readiness_report.json`

## What Changed

Block 17 added canonical non-leaking retrieval key materialization, canonical
query rendering, and vertical-specific final selectors.

The canonical path is enabled with:

```powershell
python scripts/phase3/repair_all_vertical_retrieval.py `
  --dataset-root data/scaleup_2000_full `
  --context-root data/generated/context_engineering `
  --slo-config configs/slo_targets.yaml `
  --output-root data/generated/context_engineering `
  --stage-sizes 500 2000 `
  --use-canonical-retrieval-keys
```

No model inference, GPU work, paid API calls, or external retrieval were run.

## Leakage Guard

Direct hint leakage count in canonical staged validation: `0`.

Canonical retrieval keys exclude gold evidence IDs, required evidence/document
IDs, source IDs, parent IDs, filing IDs, and answer-side hints.

## Canonical Metrics

| Vertical | Stage | Candidate@20 | Candidate@50 | Recall@5 | MRR | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Airline | 500 | 0.932500 | 0.962500 | 0.897500 | 0.934900 | Near pass |
| Airline | 2000 | 0.928000 | 0.964625 | 0.889375 | 0.934150 | Blocked by recall@5 |
| Healthcare Admin | 500 | 1.000000 | 1.000000 | 0.973000 | 0.785567 | MRR below target |
| Healthcare Admin | 2000 | 0.964000 | 0.984000 | 0.915000 | 0.745767 | MRR below target |
| Research AI | 500 | 0.944000 | 1.000000 | 0.901333 | 1.000000 | Pass at 500 |
| Research AI | 2000 | 0.871318 | 0.948397 | 0.752403 | 0.779100 | Scale degradation |
| Retail | 500 | 0.983000 | 0.991000 | 0.265000 | 0.164533 | Final selection blocked |
| Retail | 2000 | 0.913417 | 0.927917 | 0.220167 | 0.161017 | Final selection blocked |
| Finance | 500 | 0.437000 | 0.810000 | 0.224000 | 0.125867 | Candidate and final blocked |
| Finance | 2000 | 0.461250 | 0.820625 | 0.251250 | 0.142458 | Candidate and final blocked |

## Root Causes

Retail candidate recall is high, but final top-5 remains weak because same
product review rows and seed expansion rows are hard to distinguish without
better non-leaking product/review metadata.

Finance remains blocked because most prompts do not expose period, metric, or
section cues. The 8-K filing-event selector improved visible filing-event cases,
but candidate retrieval remains weak overall.

Research AI recovered at 500 records after materializing all visible section
types, but 2,000-record validation still shows candidate degradation.

Airline is close to passing but remains slightly below recall@5 0.90 at 2,000
records.

Healthcare final recall is strong, but MRR remains below the configured target.

## SLO Status

The standard SLO readiness report was regenerated and remains `BLOCKED`.
Inference scaling should remain blocked until retrieval SLOs pass or the SLO
gate is explicitly redesigned.

## Commands Run

```powershell
pytest tests/test_phase3_canonical_retrieval_repair.py
pytest tests/test_slo_framework.py
pytest tests/test_phase3_all_vertical_retrieval_repair.py
python scripts/phase3/repair_all_vertical_retrieval.py --dataset-root data/scaleup_2000_full --context-root data/generated/context_engineering --slo-config configs/slo_targets.yaml --output-root data/generated/context_engineering --stage-sizes 500 2000 --use-canonical-retrieval-keys
python scripts/phase3/evaluate_slo_readiness.py --slo-config configs/slo_targets.yaml --retrieval-report data/generated/context_engineering/retrieval_evaluation_report.json --quality-gate-report data/generated/context_engineering/retrieval_quality_gate_report.json --output-root data/generated/context_engineering
```

Full verification status is recorded in the final Codex response for this
block.

## Commit Hash

The pushed commit hash is reported in the final response. It is not embedded
here because adding the hash to this committed file would change the commit
hash itself.

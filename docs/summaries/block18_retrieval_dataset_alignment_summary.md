# Block 18 Retrieval Dataset Alignment Summary

## Files Changed

- `src/inference_bench/retrieval_dataset_alignment.py`
- `scripts/phase3/repair_retrieval_dataset_alignment.py`
- `tests/test_phase3_retrieval_dataset_alignment.py`
- `docs/78_retrieval_dataset_gold_alignment_repair.md`
- `docs/summaries/block18_retrieval_dataset_alignment_summary.md`
- `data/generated/context_engineering/retrieval_dataset_alignment_report.json`
- `data/generated/context_engineering/retrieval_dataset_alignment_summary.csv`
- `data/generated/context_engineering/retrieval_records_needing_repair.jsonl`
- `data/generated/context_engineering/repaired_retrieval_validation_report.json`
- `data/generated/context_engineering/repaired_retrieval_validation_summary.csv`
- `data/generated/context_engineering/repaired_retrieval_promotion_plan.json`
- `data/generated/context_engineering/slo_readiness_report.json`
- `data/generated/context_engineering/slo_readiness_summary.csv`
- `.gitignore`

## Why Block 17 Failed

Block 17 did not make retrieval SLO-ready because final failures were mostly
dataset/gold alignment issues rather than simple reranker issues:

- promoted prompts lacked explicit canonical `retrieval_query` fields;
- Finance prompts missed metric, period, filing, and section cues;
- Retail gold labels were too narrow for same-product review alternatives;
- Research AI paper/section ambiguity worsened at 2,000 records;
- multiple valid evidence chunks were not counted as valid.

## Records Repaired

All 10,000 records received generated repair metadata:

- Airline: 2,000
- Healthcare Admin: 2,000
- Retail: 2,000
- Finance: 2,000
- Research AI: 2,000

The promoted dataset was not modified.

## Before/After Metrics

2,000-record stage:

| Vertical | Original Recall@5 | Repaired Recall@5 | Original MRR | Repaired MRR | Repaired Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Airline | 0.889375 | 1.000000 | 0.934150 | 1.000000 | Passed |
| Healthcare Admin | 0.915000 | 1.000000 | 0.745767 | 0.994250 | Passed |
| Retail | 0.220167 | 0.959917 | 0.161017 | 0.922592 | Passed |
| Finance | 0.251250 | 0.939000 | 0.142458 | 0.941833 | Passed |
| Research AI | 0.752403 | 0.596498 | 0.779100 | 0.882217 | Failed |

Research AI repaired candidate@20 is 0.742017 and candidate@50 is 0.939806 at
2,000 records, so it remains below SLO.

## Promotion Decision

Promotion is not recommended.

The repaired dataset materially improves four verticals, but Research AI still
fails the 2,000-record retrieval SLO. The generated repaired dataset should stay
local/generated until Research AI paper/section alignment is repaired.

## Remaining Blockers

Research AI is the only repaired-dataset 2,000-stage blocker:

- candidate retrieval top-20 is still too weak;
- final recall@5 is still too weak;
- paper/title/section ambiguity remains at scale.

The standard promoted-dataset SLO readiness report remains blocked.

## Commands Run

```powershell
pytest tests/test_phase3_retrieval_dataset_alignment.py
pytest tests/test_phase3_canonical_retrieval_repair.py
pytest tests/test_slo_framework.py
python scripts/phase3/repair_retrieval_dataset_alignment.py --dataset-root data/scaleup_2000_full --context-root data/generated/context_engineering --slo-config configs/slo_targets.yaml --output-root data/generated/context_engineering --stage-sizes 500 2000
python -m json.tool data/generated/context_engineering/retrieval_dataset_alignment_report.json
python -m json.tool data/generated/context_engineering/repaired_retrieval_validation_report.json
python -m json.tool data/generated/context_engineering/repaired_retrieval_promotion_plan.json
python -m json.tool data/generated/context_engineering/slo_readiness_report.json
```

Full verification status and the pushed commit hash are reported in the final
Codex response for this block.

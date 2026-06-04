# Block 19 Research AI Retrieval Repair Summary

## Files Changed

- `src/inference_bench/research_ai_alignment_repair.py`
- `src/inference_bench/vertical_final_selectors.py`
- `scripts/phase3/repair_research_ai_retrieval_alignment.py`
- `tests/test_phase3_research_ai_alignment_repair.py`
- `docs/79_research_ai_retrieval_alignment_repair.md`
- `docs/summaries/block19_research_ai_retrieval_repair_summary.md`
- `data/generated/context_engineering/research_ai_alignment_report.json`
- `data/generated/context_engineering/research_ai_alignment_summary.csv`
- `data/generated/context_engineering/research_ai_failure_examples.jsonl`
- `data/generated/context_engineering/repaired_retrieval_validation_report.json`
- `data/generated/context_engineering/repaired_retrieval_validation_summary.csv`
- `data/generated/context_engineering/repaired_retrieval_promotion_plan.json`
- `data/generated/context_engineering/slo_readiness_report.json`
- `data/generated/context_engineering/slo_readiness_summary.csv`

## Before vs After

| Research AI 2,000-stage dataset | Candidate@20 | Candidate@50 | Recall@5 | MRR | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Block 18 repaired baseline | 0.742017 | 0.939806 | 0.596498 | 0.882217 | Failed |
| Block 19 repaired validation | 0.975172 | 0.979826 | 0.917460 | 0.953233 | Passed |

## Failure Classes

The repair report tracks:

- paper title ambiguity
- section type ambiguity
- method vs result confusion
- limitation vs discussion confusion
- topic overlap across papers
- narrow gold section
- multiple valid sections not counted
- candidate absent from top 50
- candidate present but not top 5
- near-duplicate section confusion

The dominant Research AI issue was gold/evidence alignment: several gold records
listed multiple IDs for the same context chunk or counted a narrow section even
when adjacent same-paper sections were valid evidence.

## What Changed

- Added Research AI-only offline expanded valid evidence sets.
- Deduplicated Research AI required evidence by matched context record.
- Added section-family-aware final selection for paper overview, method,
  results, and limitations/discussion evidence.
- Kept runtime retrieval non-leaking: expanded valid IDs are not used in query
  text.
- Required Qdrant dense backend for the final generated report.

## SLO and Promotion

All repaired 2,000-record vertical retrieval SLOs now pass. The repaired
retrieval promotion plan reports:

- `promotion_recommended: true`
- `remaining_blockers: []`
- `do_not_overwrite_promoted_dataset_automatically: true`

## Commands Run

```powershell
pytest tests/test_phase3_research_ai_alignment_repair.py
pytest tests/test_phase3_retrieval_dataset_alignment.py
pytest tests/test_slo_framework.py
python scripts/phase3/repair_research_ai_retrieval_alignment.py `
  --dataset-root data/scaleup_2000_full `
  --context-root data/generated/context_engineering `
  --slo-config configs/slo_targets.yaml `
  --output-root data/generated/context_engineering `
  --stage-sizes 500 2000 `
  --require-dense-backend
```

Full verification is recorded in the final implementation report.

## Commit

Commit hash after push: pending until commit is created.

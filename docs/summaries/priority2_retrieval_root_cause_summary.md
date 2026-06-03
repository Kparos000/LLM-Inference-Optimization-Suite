# Priority 2 Retrieval Root-Cause Summary

## Files Changed

- `configs/slo_targets.yaml`
- `src/inference_bench/retrieval_root_cause.py`
- `scripts/phase3/analyze_retrieval_root_cause.py`
- `tests/test_phase3_retrieval_root_cause.py`
- `tests/test_slo_framework.py`
- `docs/72_retrieval_root_cause_analysis.md`
- `data/generated/context_engineering/retrieval_root_cause_report.json`
- `data/generated/context_engineering/retrieval_root_cause_summary.csv`
- `data/generated/context_engineering/retrieval_failure_examples.jsonl`

## Current Retrieval SLO Blockers

Strict retrieval is still below target for several verticals. The source-hint-assisted path remains an upper bound, not a fair strict retrieval claim.

For `final_10000` / `mm2_hybrid_top5`:

| Ablation | Vertical | Recall@5 | Candidate Recall@100 | Main Root Cause |
| --- | --- | ---: | ---: | --- |
| prompt_text_only | finance | 0.163000 | 0.574375 | prompt_missing_period |
| prompt_plus_metadata | finance | 0.280750 | 0.820750 | prompt_missing_period |
| prompt_text_only | retail | 0.161167 | 0.593667 | gold_in_candidates_not_final_top5 |
| prompt_plus_metadata | retail | 0.221583 | 0.924667 | gold_in_candidates_not_final_top5 |
| prompt_text_only | research_ai | 0.275987 | 0.760779 | gold_in_candidates_not_final_top5 |
| prompt_plus_metadata | research_ai | 0.641715 | 0.866486 | gold_in_candidates_not_final_top5 |

## Main Root Cause By Vertical

- Airline: embedding/indexed-text weakness is the main residual strict-mode signal.
- Healthcare Admin: candidate retrieval is the main residual blocker.
- Finance: prompt/gold repair is required because strict prompts often lack period and metric cues.
- Research AI: final top-5 selection/reranking is the main blocker.
- Retail: final top-5 selection/reranking is the main blocker.

## Finance Details

Finance prompt_text_only:
- Recall@5: 0.163000
- Candidate Recall@100: 0.574375
- Main root cause: `prompt_missing_period`
- Candidate-vs-final gap: 0.411375
- Prompt/gold repair required: yes

Finance prompt_plus_metadata:
- Recall@5: 0.280750
- Candidate Recall@100: 0.820750
- Main root cause: `prompt_missing_period`
- Candidate-vs-final gap: 0.540000
- Prompt/gold repair required: yes

## Candidate Retrieval Or Reranking

The global strict-mode blocker assessment is mixed:

- Candidate retrieval failure count: 8,155
- Reranking/final-selection failure count: 16,446
- Prompt/gold repair signal count: 19,340

The next block should repair finance prompt/gold metadata first, then tune final top-5 selection for retail, research, and finance cases where candidate recall is already high.

## Commands Run

```powershell
pytest tests/test_phase3_retrieval_root_cause.py
pytest tests/test_slo_framework.py
pytest tests/test_phase3_retrieval_quality_gate.py
mypy src/inference_bench/retrieval_root_cause.py tests/test_phase3_retrieval_root_cause.py tests/test_slo_framework.py
python scripts/phase3/analyze_retrieval_root_cause.py --dataset-root data/scaleup_2000_full --context-root data/generated/context_engineering --slo-config configs/slo_targets.yaml --output-root data/generated/context_engineering
python scripts/audit_repo_public_content.py
inference-bench doctor
inference-bench validate-config
mypy src tests
pytest
ruff check .
ruff format --check .
```

Final verification and commit hash are reported after push.

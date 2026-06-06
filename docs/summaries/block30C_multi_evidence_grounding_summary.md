# Block 30C Multi-Evidence Grounding Summary

## Status

`TARGET_MET`

The strict evaluator remained unchanged.

## Measured Result

| Metric | Initial | Final |
| --- | ---: | ---: |
| JSON validity | 100% | 100% |
| Contract validity | 100% | 100% |
| Evidence match | 60% | 80% |
| Groundedness | 60% | 80% |
| Safety violations | 0% | 0% |

Model6 handled five initial streaming requests. One Airline row received one
bounded citation-only repair. Model5 remained pricing-blocked.

## Repair Behavior

- prompt explicitly requires all relevant evidence IDs;
- every evidence block receives an internal relevant/not-relevant check;
- `citation_notes` explains each emitted label;
- repair uses only supplied short labels, not canonical gold IDs;
- absent required context prevents retry;
- no citations or facts are fabricated.

Airline improved after the repair added supplied `E2`. Healthcare Admin could
not be repaired because required `MCH-POL-020` was absent from the supplied top
five.

The 80% final result is evaluator-assisted. The unassisted first-pass result is
still 60%.

## Cost

- requests: 5 initial + 1 repair
- input tokens: 7,669
- output tokens: 493
- total cost: `$0.00017803`
- cost per original prompt: `$0.000035606`
- cost per grounded answer: `$0.0000445075`

## Outputs

- `results/raw/phase4_grounding_repair_smoke_results.jsonl`
- `results/processed/phase4_grounding_repair_eval_report.json`
- `results/processed/phase4_grounding_repair_eval_summary.csv`

Generated results remain local and ignored.

## Files Changed

- `src/inference_bench/generation_contract.py`
- `src/inference_bench/grounding_repair.py`
- `scripts/phase4/run_grounding_repair_smoke.py`
- `tests/test_phase4_multi_evidence_grounding_repair.py`
- `docs/91_multi_evidence_grounding_repair.md`
- `docs/summaries/block30C_multi_evidence_grounding_summary.md`
- `README.md`

## Verification

- focused Block 30C and related API/streaming tests: 31 passed
- full test suite: 915 passed
- `mypy src tests`: passed across 185 source files
- `ruff check .`: passed
- `ruff format --check .`: passed across 245 files
- `python scripts/audit_repo_public_content.py`: passed
- `inference-bench doctor`: passed
- `inference-bench validate-config`: passed

Commit hash is reported after the verified changes are committed and pushed.

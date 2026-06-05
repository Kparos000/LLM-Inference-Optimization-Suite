# Block 24 Generation Contract Hardening Summary

## Outcome

Generation contract parsing, validation, retry metadata, and truncation handling
were hardened without changing retrieval, gold records, or evaluator strictness.
The local 0.5B smoke improved syntax reliability but did not meet the full
grounding targets.

## Before Versus After

| Metric | Before | After |
| --- | ---: | ---: |
| JSON validity | 80% | 100% |
| Contract validity | 60% | 80% |
| Evidence-ID presence | 80% | 100% |
| Full evidence match | 60% | 40% |
| Deterministic groundedness | 40% | 20% |
| Truncation rate | 20% | 0% |

## Retry And Repair

- Retry rows: 2/5
- Total retry attempts: 2
- Successful retries: 1
- Parse-repair rows: 5/5
- Truncated rows: 0/5

All five parse repairs were transparent extraction of JSON from model-generated
markdown fences. No evidence ID or answer content was invented.

## Files Changed

- `src/inference_bench/generation_contract.py`
- `src/inference_bench/evaluator_contract.py`
- `src/inference_bench/output_records.py`
- `src/inference_bench/runners/hf_runner.py`
- `src/inference_bench/runners/openai_compatible_runner.py`
- `scripts/phase4/run_local_hf_smoke.py`
- `scripts/phase4/run_openai_compatible_smoke.py`
- `scripts/phase4/evaluate_generation_outputs.py`
- `tests/test_phase4_generation_contract.py`
- `tests/test_phase4_generation_contract_hardening.py`
- `docs/83_phase4_generation_contract_hardening.md`
- `README.md`

## Generated Local Artifacts

- `results/raw/phase4_generation_contract_hardened_hf_smoke.jsonl`
- `results/processed/phase4_generation_contract_hardened_eval_report.json`
- `results/processed/phase4_generation_contract_hardened_eval_summary.csv`

These files remain local and ignored.

## Remaining Issues

- Retail repeats invalid confidence `5.0` even after correction.
- Airline, Healthcare, and Research AI under-cite multi-evidence answers.
- The 0.5B model still violates the no-markdown instruction.
- A stronger instruction model or constrained JSON decoding is required before
  treating contract adherence as production-ready.

## Commands Run

```text
pytest tests/test_phase4_generation_contract.py
pytest tests/test_phase4_generation_contract_hardening.py
pytest tests/test_phase4_local_hf_smoke.py
python scripts/phase4/export_runner_smoke_workload.py ...
python scripts/phase4/run_local_hf_smoke.py ... --max-new-tokens 256 --max-contract-retries 1
python scripts/phase4/evaluate_generation_outputs.py ...
python -m json.tool results/processed/phase4_generation_contract_hardened_eval_report.json
```

Full repository verification and the final commit hash are reported after all
checks complete.


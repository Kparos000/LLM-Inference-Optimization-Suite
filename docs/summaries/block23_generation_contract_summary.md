# Block 23 Generation Contract Summary

## Outcome

Phase 4 runner prompts and generation records now share an evaluator-friendly
JSON contract with short, stable evidence labels and canonical citation alias
mapping.

## Files Changed

- `src/inference_bench/generation_contract.py`
- `src/inference_bench/workload_adapter.py`
- `src/inference_bench/evaluator_contract.py`
- `src/inference_bench/output_records.py`
- `src/inference_bench/runners/hf_runner.py`
- `src/inference_bench/runners/openai_compatible_runner.py`
- `src/inference_bench/runners/openai_load_runner.py`
- `scripts/phase4/evaluate_generation_outputs.py`
- `scripts/phase4/run_local_hf_smoke.py`
- `scripts/phase4/run_openai_compatible_smoke.py`
- `tests/test_output_records.py`
- `tests/test_phase4_generation_contract.py`
- `tests/test_phase4_local_hf_smoke.py`
- `tests/test_phase4_workload_adapter.py`
- `docs/82_phase4_generation_contract.md`
- `README.md`

## Real HF Smoke

- Model: `model1_0_5b` / `Qwen/Qwen2.5-0.5B-Instruct`
- Workload: five `mm2_hybrid_top5` records, one per vertical
- Successful generations: 5/5
- JSON validity: 4/5 (80%)
- Generation-contract validity: 3/5 (60%)
- Evidence-ID presence: 4/5 (80%)
- Full evidence match: 3/5 (60%)
- Deterministic groundedness: 2/5 (40%)
- Safety violations: 0/5
- Mean latency: 94,891.634 ms
- Median latency: 95,660.246 ms
- Input tokens: 6,560
- Output tokens: 423

Generated artifacts:

- `results/raw/phase4_generation_contract_hf_smoke.jsonl`
- `results/processed/phase4_generation_contract_eval_report.json`
- `results/processed/phase4_generation_contract_eval_summary.csv`

These result files are local generated outputs and are not committed.

## Remaining Issues

- The 0.5B model did not reliably obey the contract: one response was truncated
  and one valid JSON response failed contract semantics.
- Retail cited relevant evidence but did not cover every required gold evidence
  family.
- Research AI produced valid cited JSON but copied schema-example wording.
- Current groundedness is deterministic evidence matching, not semantic claim
  verification.
- Constrained decoding or a larger instruction model should be evaluated before
  requiring near-perfect contract compliance.

## Commands Run

```text
pytest tests/test_phase4_generation_contract.py ...
python scripts/phase4/export_runner_smoke_workload.py ...
python scripts/phase4/run_local_hf_smoke.py ... --limit 5 --max-new-tokens 128
python scripts/phase4/evaluate_generation_outputs.py ...
```

Full repository verification results are recorded in the completion report.


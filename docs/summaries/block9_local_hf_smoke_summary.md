# Block 9 Local HF Smoke Summary

## Files Changed

- `scripts/phase4/run_local_hf_smoke.py`
- `scripts/phase4/evaluate_generation_outputs.py`
- `tests/test_phase4_local_hf_smoke.py`
- `docs/66_phase4_local_hf_smoke.md`
- `docs/summaries/block9_local_hf_smoke_summary.md`
- `data/generated/phase4/smoke_workload_export_report.json`
- `data/generated/phase4/smoke_workload_export_summary.csv`
- `results/processed/phase4_hf_local_smoke_eval_report.json`
- `results/processed/phase4_hf_local_smoke_eval_summary.csv`

## Model Used

- Alias: `model1_0_5b`
- Model ID: `Qwen/Qwen2.5-0.5B-Instruct`
- Backend: local Hugging Face Transformers
- Paid API calls: none
- Gated model calls: none

## Workload

- Source workload:
  `data/workloads/smoke_500/prompt_plus_metadata/mm2_hybrid_top5.jsonl`
- Exported runner input:
  `data/generated/phase4/smoke_500_mm2_runner_input.jsonl`
- Split: `smoke_500`
- Memory mode: `mm2_hybrid_top5`
- Ablation mode: `prompt_plus_metadata`
- Exported rows: 25
- Real HF prompts run: 10

## Latency Summary

- Min latency: 38,555.9231 ms
- Mean latency: 45,756.89571 ms
- Max latency: 56,121.3469 ms

This was a CPU/local smoke run, not a GPU benchmark.

## Token Summary

- Total input tokens: 5,194
- Mean input tokens: 519.4
- Total output tokens: 640
- Mean output tokens: 64.0
- Max new tokens requested: 64

## Evaluator Summary

- Rows evaluated: 10
- Joined to gold records: 10
- Joined rate: 1.0
- Format-valid rows: 10
- Grounded rows: 0
- Safety violations: 0

The smoke validated execution and evaluator plumbing. It did not validate answer
quality; the 0.5B model did not reliably cite expected evidence IDs.

## Errors

- Model load: success
- Generation success: 10/10
- API cost: zero
- GPU experiment: not run

## Commands Run

- `pytest tests/test_phase4_workload_adapter.py`
- `pytest tests/test_phase4_smoke_plumbing.py`
- `pytest tests/test_phase4_local_hf_smoke.py`
- `python scripts/phase4/export_runner_smoke_workload.py --workload-path data/workloads/smoke_500/prompt_plus_metadata/mm2_hybrid_top5.jsonl --output-path data/generated/phase4/smoke_500_mm2_runner_input.jsonl --limit 25`
- `python scripts/phase4/run_local_hf_smoke.py --input-path data/generated/phase4/smoke_500_mm2_runner_input.jsonl --output-path results/raw/phase4_hf_local_smoke_results.jsonl --model-alias model1_0_5b --limit 10 --max-new-tokens 64`
- `python scripts/phase4/evaluate_generation_outputs.py --results-path results/raw/phase4_hf_local_smoke_results.jsonl --dataset-root data/scaleup_2000_full --output-root results/processed`
- `python scripts/audit_repo_public_content.py`
- `inference-bench doctor`
- `inference-bench validate-config`
- `ruff check src/inference_bench scripts/phase4 tests/test_phase4_local_hf_smoke.py`
- `ruff format --check src/inference_bench scripts/phase4 tests/test_phase4_local_hf_smoke.py`

## Commit Hash

The final pushed commit hash is reported in the Codex final response. It cannot
be embedded self-referentially in the same commit without changing the commit
hash.

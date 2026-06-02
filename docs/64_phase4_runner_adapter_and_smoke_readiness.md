# Phase 4 Runner Adapter and Smoke Readiness

Phase 4 starts by connecting the Phase 3 context-engineering outputs to the
existing benchmark runners. This block does not add a new inference harness and
does not run GPU, HF, vLLM, SGLang, or paid API experiments.

## What Was Added

- `src/inference_bench/workload_adapter.py` converts Phase 3 `WorkloadRecord`
  JSONL files into the existing runner `WorkloadItem` shape.
- `scripts/phase4/export_runner_smoke_workload.py` exports a small
  runner-compatible JSONL workload for local plumbing checks.
- `src/inference_bench/run_manifest.py` defines a run manifest shape for future
  execution tracking.
- `scripts/phase4/evaluate_generation_outputs.py` joins runner outputs to
  promoted gold/eval records by `prompt_id` and applies the evaluator contract.
- Existing mock, HF, OpenAI-compatible, and OpenAI load runners now preserve
  optional Phase 4 metadata from adapted workload items in result CSV rows.

## How Phase 3 Workloads Connect to Runners

Phase 3 workloads contain chat-style `messages`, context records, retrieval
metadata, memory mode labels, evidence IDs, and source prompt records. The
current runners consume a simpler JSONL shape:

- `prompt_id`
- `workload_name`
- `prompt`
- `expected_output`
- `metadata`

The adapter renders `messages` into one prompt string and stores Phase 3 fields
inside `metadata`. Existing runners remain responsible for execution and metric
collection.

Preserved metadata includes:

- `workload_id`
- `prompt_id`
- `vertical`
- `memory_mode`
- `ablation_mode`
- `retrieval_metadata`
- `context_token_estimate`
- `gold_evidence_ids`

## Export Smoke Workload

```powershell
python scripts/phase4/export_runner_smoke_workload.py `
  --workload-path data/workloads/smoke_500/mm2_hybrid_top5.jsonl `
  --output-path data/generated/phase4/smoke_500_mm2_runner_input.jsonl `
  --limit 25
```

This writes:

- `data/generated/phase4/smoke_500_mm2_runner_input.jsonl`
- `data/generated/phase4/smoke_workload_export_report.json`
- `data/generated/phase4/smoke_workload_export_summary.csv`

## Mock Smoke Validation

The mock runner verifies plumbing without model execution:

```powershell
inference-bench mock-run `
  --workload-path data/generated/phase4/smoke_500_mm2_runner_input.jsonl `
  --output-path results/raw/phase4_mock_smoke_results.csv
```

The output row count should match the exported input row count. Result rows
preserve `prompt_id`, `workload_id`, `vertical`, `memory_mode`, `ablation_mode`,
`context_token_estimate`, and `gold_evidence_ids` when these fields are present
in adapted workload metadata.

## Evaluator Join

```powershell
python scripts/phase4/evaluate_generation_outputs.py `
  --results-path results/raw/phase4_mock_smoke_results.csv `
  --dataset-root data/scaleup_2000_full `
  --output-root results/processed
```

The evaluator joins by `prompt_id`, then reports deterministic contract fields
such as status matching, format validity, citation/evidence matching, grounding,
and safety violations.

## Ready for HF and vLLM Testing

This block makes the runner input path ready for:

- local mock smoke checks,
- local HF smoke checks on a small prompt count,
- OpenAI-compatible vLLM smoke checks,
- OpenAI-compatible load-run smoke checks with checkpoint/resume.

The next real execution step should be a small local mock/HF run using exported
`smoke_500` workloads and low `max_prompts`. vLLM should come after confirming
the exported workload shape, metadata preservation, and evaluator join.

## Not Yet Ready

- No SGLang backend is implemented yet.
- No GPU benchmark has been run by this block.
- No paid HF Inference Provider smoke call has been run by this block.
- No semantic judge has been added; evaluator grounding is still deterministic
  citation/evidence matching.

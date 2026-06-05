# Phase 4 vLLM Validation and Telemetry

Block 22 validates the local OpenAI-compatible serving path and establishes a
stable telemetry schema for later GPU experiments. It does not rent a GPU, run a
large benchmark, change retrieval, or add serving optimizations.

## Validation Path

The live validation path is:

```text
smoke_500 mm2 workload
  -> Phase 4 runner input
  -> http://localhost:8000/v1
  -> vLLM OpenAI-compatible server
  -> generation JSONL
  -> gold/eval prompt_id join
  -> evaluation and telemetry reports
```

The validation is deliberately limited to five prompts using
`model1_0_5b`, which resolves to `Qwen/Qwen2.5-0.5B-Instruct`.

## Start vLLM

Run vLLM in a compatible Linux, WSL, or container environment:

```powershell
vllm serve Qwen/Qwen2.5-0.5B-Instruct `
  --host 0.0.0.0 `
  --port 8000 `
  --dtype auto `
  --api-key EMPTY
```

No paid API key is required for this local endpoint.

## Run Validation

The runner input must contain five adapted records from
`mm2_hybrid_top5`. Export it first when needed:

```powershell
python scripts/phase4/export_runner_smoke_workload.py `
  --workload-path data/workloads/smoke_500/mm2_hybrid_top5.jsonl `
  --output-path data/generated/phase4/smoke_500_mm2_runner_input.jsonl `
  --limit 5
```

Then run:

```powershell
python scripts/phase4/validate_vllm_serving.py `
  --input-path data/generated/phase4/smoke_500_mm2_runner_input.jsonl `
  --model-alias model1_0_5b `
  --model-name Qwen/Qwen2.5-0.5B-Instruct `
  --base-url http://localhost:8000/v1 `
  --api-key EMPTY `
  --limit 5 `
  --max-new-tokens 64
```

If the server is unavailable, the command writes an explicit
`server_unavailable` report and does not claim live validation succeeded.

## Outputs

- `results/raw/phase4_vllm_validation.jsonl`
- `results/raw/phase4_vllm_validation_manifest.json`
- `results/processed/phase4_vllm_validation_report.json`
- `results/processed/phase4_vllm_validation_summary.csv`
- `results/processed/phase4_vllm_validation_telemetry.json`
- `results/processed/phase4_backend_comparison_framework.csv`

Raw and processed run artifacts remain local by repository policy unless they
are deliberately reviewed and promoted as curated samples.

## Telemetry Schema

`src/inference_bench/telemetry.py` defines request-level fields:

- timestamp
- backend
- model
- memory mode
- end-to-end latency
- TTFT
- TPOT
- token throughput
- request throughput
- success
- error type

The same schema reserves nullable fields for future runs:

- GPU utilization
- GPU memory
- GPU cost
- RunPod cost

Local non-streaming OpenAI-compatible validation can measure latency, TPOT, and
token throughput. TTFT remains unavailable until the validation path uses
streaming token events.

## Backend Comparison Framework

The comparison framework contains stable rows for:

- Hugging Face local
- vLLM
- SGLang

It reserves latency, TTFT, TPOT, throughput, quality, groundedness, cost, and GPU
cost fields. This block does not claim a backend comparison because equivalent
hardware-controlled runs have not yet been performed.

## Future RunPod Integration

Future RunPod runs should populate:

- GPU type and count
- hourly pod price
- run duration
- GPU utilization
- peak and average GPU memory
- power and temperature when available
- total requests and tokens
- GPU dollars per request and per token

These fields should be attached to the same run ID and manifest used by the
request-level telemetry records.

## Future SGLang Integration

SGLang should reuse:

- Phase 3 runner-adapted workload JSONL
- telemetry records
- run manifests
- evaluator contract
- backend comparison rows

SGLang-specific scheduler and cache metrics can be added as optional metadata
without changing the shared metrics required for HF/vLLM/SGLang comparison.

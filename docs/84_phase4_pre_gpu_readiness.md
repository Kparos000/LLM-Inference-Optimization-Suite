# Phase 4 Pre-GPU Readiness

Phase 4 now has a single readiness check for the local plumbing that must be
clean before GPU provisioning. The check inspects existing artifacts only. It
does not load a model, contact a serving endpoint, call a paid API, or run GPU
work.

## Readiness Command

```powershell
python scripts/phase4/check_phase4_readiness.py `
  --output-root data/generated/phase4
```

The command writes:

- `data/generated/phase4/phase4_readiness_report.json`
- `data/generated/phase4/phase4_readiness_summary.csv`

The report uses three statuses:

- `PASS`: a required contract, wrapper, configuration, or promoted report is
  present and valid.
- `NOT_AVAILABLE`: a metric cannot be measured before live serving or GPU
  provisioning. This is not treated as a failed experiment.
- `FAIL`: a required pre-GPU artifact or contract is missing or invalid.

## Current Status

The generated status is `PRE_GPU_PLUMBING_READY`.

The checker confirms:

- the promoted Qdrant retrieval source-of-truth manifest is ready for Phase 4;
- all promoted retrieval SLO rows pass;
- the grounded generation contract exists;
- the local Hugging Face path has an evaluation report or a dry-run fallback;
- the OpenAI-compatible vLLM wrapper exists;
- the SGLang OpenAI-compatible scaffold exists;
- request telemetry includes nullable GPU and RunPod fields;
- no unmerged Git conflict artifacts exist.

The repaired retrieval source-of-truth manifest is authoritative. Historical
pre-repair retrieval quality reports are retained for audit history but do not
override the promoted repaired validation.

## Backend Matrix

`configs/backend_matrix.yaml` defines the current execution capabilities.

| Backend | Status | Server | GPU | Cost model |
| --- | --- | ---: | ---: | --- |
| Local Hugging Face | `ready` | No | No | `local_compute` |
| OpenAI-compatible vLLM | `dry_run_ready` | Yes | Yes | `gpu_infra` |
| OpenAI-compatible SGLang | `future` | Yes | Yes | `gpu_infra` |

The matrix also records streaming, TTFT, TPOT, batch, and concurrency support.
These fields describe interface capability; they do not claim that a live
server benchmark has already run.

## SGLang Scaffold

The SGLang script reuses the established OpenAI-compatible request and output
path:

```powershell
python scripts/phase4/run_sglang_compatible_smoke.py `
  --dry-run `
  --input-path data/generated/phase4/generation_contract_runner_input.jsonl `
  --output-path results/raw/phase4_sglang_dry_run.jsonl `
  --limit 5
```

Dry-run mode validates workload loading, metadata preservation, output schema,
and run-manifest generation without requiring SGLang or a GPU. Live mode checks
the server's OpenAI-compatible `/models` endpoint and fails with a
SGLang-specific error when the endpoint is unavailable.

The default SGLang URL is `http://localhost:30000/v1`. A future live run can
override the URL, model name, and timeout without changing the result schema.

## GPU Cost Inputs

`configs/gpu_costs.yaml` provides a RunPod cost template:

- provider;
- GPU type;
- hourly price;
- region;
- optional instance ID;
- measured start and end timestamps;
- `elapsed_hours * hourly_price_usd`.

GPU type, region, hourly price, and timestamps intentionally remain unset. They
must be copied from the actual provisioned instance immediately before and
after each run. The code validates and calculates elapsed hourly cost once
those fields are populated; it refuses to calculate from placeholders.

## Metrics Not Yet Available

The SLO report correctly marks these families as `NOT_AVAILABLE`:

- latency and TTFT/TPOT;
- request and token throughput;
- GPU/CPU/RAM utilization;
- API token cost;
- GPU infrastructure cost.

They are not failures. A live serving run is required to measure them.

## Exact GPU Smoke Checklist

1. Select `model1_0_5b` and a GPU with sufficient VRAM.
2. Record the RunPod GPU type, region, and hourly price in a run-specific copy
   of the GPU cost inputs.
3. Start vLLM first and confirm `GET /v1/models` lists the selected model.
4. Run five `mm2_hybrid_top5` requests at concurrency 1.
5. Confirm generation-contract parsing, evidence aliases, evaluation joins,
   and zero duplicate prompt processing.
6. Capture TTFT, TPOT, end-to-end latency, tokens per second, requests per
   second, GPU utilization, GPU memory, power where available, and timestamps.
7. Evaluate the five outputs before increasing to the 500-prompt smoke split.
8. Repeat the same five-request contract with SGLang after the vLLM path is
   stable.

Phase 4 is ready for GPU provisioning, but not yet for a scaled GPU benchmark.

# A100 SXM RunPod Calibration Runbook

Status: prepared on June 23, 2026; no live RunPod calibration run executed

## Purpose

This runbook prepares the first controlled RunPod calibration path. It is not a
general benchmark matrix and must not be used unless an explicit RunPod SSH
target is configured.

Recommended GPU:

```text
A100 SXM 80GB
```

Why this target:

- 80 GB VRAM is enough headroom for `model2_3b`, `model3_7b`, and later
  controlled larger-model checks.
- The observed RunPod UI price is lower than H100/H200/B-series options.
- The goal is calibration and throughput scaling evidence, not maximum
  throughput.

Observed price:

```text
$1.49/hour
```

Cost examples before shutdown discipline and warmup overhead:

| Runtime | Estimated GPU Cost |
| --- | ---: |
| 1 hour | $1.49 |
| 2 hours | $2.98 |
| 4 hours | $5.96 |

The price source is a RunPod console UI screenshot. Re-verify it immediately
before final cost claims.

## Required Guard

Live calibration is allowed only when an explicit target exists:

```text
RUNPOD_SSH_HOST=<ssh-alias-or-host>
```

If this value is missing, stop after generating local manifests and readiness
reports.

## Calibration Plan

- Model alias: `model2_3b`
- Model ID: `Qwen/Qwen2.5-3B-Instruct`
- Runtime/engine: `vLLM`
- Memory mode: `mm2_hybrid_top5`
- Prompt counts: 100, then 200
- Concurrency: 1, 4, 8, 16, 32
- Traffic profiles: `online_low_latency`, then `offline_throughput` if present
- Temperature: 0
- Repairs: same B7R1/B6R6 repair path
- Artifact sync: enabled
- Checkpoint/resume: enabled
- Manifest: enabled
- GPU telemetry: enabled

Do not run 2,000/10,000 prompts, SGLang, mm4, TensorRT-LLM, or a final matrix
from this runbook.

## Local Preparation

Generate the local package:

```powershell
python scripts/phase4/prepare_runpod_a100_calibration.py
```

Expected outputs:

- `results/processed/a100_sxm_calibration_manifest_100.json`
- `results/processed/a100_sxm_calibration_manifest_200.json`
- `results/processed/a100_sxm_calibration_readiness_report.json`

The readiness report must show:

- `gpu_price_registered: true`
- `runtime_profile_valid: true`
- `backup_verification_dry_run_passed: true`
- `live_runpod_calibration_allowed: true`

If `live_runpod_calibration_allowed` is false, do not start a pod run.

## RunPod Setup

Use an A100 SXM 80GB pod with a CUDA/PyTorch/vLLM-capable image. The exact image
should be pinned in the run manifest before execution.

SSH placeholder:

```powershell
ssh $env:RUNPOD_SSH_HOST
```

Repository setup:

```bash
git clone <repo-url> LLM-Inference-Optimization-Suite
cd LLM-Inference-Optimization-Suite
git pull origin main
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Credential check:

```bash
test -n "$HF_TOKEN" || test -n "$HUGGINGFACE_HUB_TOKEN"
```

Artifact sync dry-run check:

```bash
python scripts/phase4/test_long_run_recovery_dry_run.py
```

## vLLM Startup

Example startup command. Pin the final image, model revision, tokenizer
revision, and environment values in the manifest before live use.

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 32
```

Health check:

```bash
curl http://127.0.0.1:8000/v1/models
```

## Calibration Command

Use the generated manifests as the execution contract. Run 100 prompts first,
then review quality, stability, telemetry, and artifact sync before 200.

```bash
# Placeholder until the live RunPod runner wrapper is selected.
python scripts/phase4/run_b7r1_vllm_stability_repair.py \
  --model-alias model2_3b \
  --memory-mode mm2_hybrid_top5 \
  --engine vllm \
  --concurrency 1 \
  --max-records 100
```

Do not proceed to concurrency 4/8/16/32 until concurrency 1 passes request
completion, quality, checkpoint, artifact sync, and telemetry checks.

## Shutdown Checklist

Before stopping the pod:

1. Verify raw JSONL, manifest, telemetry, processed report, logs, checkpoint,
   and artifact-sync report exist.
2. Run backup verification and confirm hashes match.
3. Pull artifacts back to the local workspace.
4. Confirm no partial run is marked `completed`.
5. Record final pod elapsed time and observed hourly price.
6. Stop and terminate the RunPod instance.

## Current Decision

```text
A100_SXM_CALIBRATION_PACKAGE_READY
LIVE_RUNPOD_CALIBRATION_BLOCKED_UNTIL_RUNPOD_SSH_HOST_CONFIGURED
```

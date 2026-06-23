# Block Phase 2A-R1/B/C Infrastructure, API, And RunPod Prep Summary

Status: completed on June 23, 2026

## Price Registry

`configs/gpu_prices.yaml` now registers 26 observed RunPod GPU prices from the
RunPod console UI screenshot. Every entry keeps the source note to verify the
price before final cost claims.

Primary planning prices:

- A100 SXM 80GB: $1.49/hr, `primary_calibration_gpu`
- H100 SXM 80GB: $3.29/hr, `high_throughput_scale_validation`
- L40S: $0.99/hr, `lower_cost_comparison`

GPU cost projection now includes estimated run cost, 1,000/10,000/40,000
prompt projections, tokens per GPU dollar, and successful requests per GPU
dollar when the measured inputs exist. API-provider tracks remain excluded
from GPU hourly-cost math.

## API Probe

The guarded live API probe ran because `.env` contained the required provider
keys.

- Models requested: `model5_gated`, `model6_gated`, `model7_gated`
- Models executed: `model5_gated`, `model6_gated`
- Model skipped: `model7_gated`, because complete audited API pricing is not
  registered
- Prompt count: 10 per executed model per concurrency level
- Concurrency levels: 1, 2, 4
- Requests: 60
- Successes: 60
- 429/5xx/timeouts/retries/provider throttling: 0
- Mean TTFT: 748.826 ms
- Mean TPOT: 9.302 ms
- Mean E2E latency: 1,237.873 ms
- Mean request throughput: 0.906 requests/sec per request row
- Mean token throughput: 48.322 tokens/sec per request row
- Estimated API cost: $0.00025169

Safe concurrency from this tiny probe:

- `model5_gated`: 4
- `model6_gated`: 4
- `model7_gated`: unavailable until pricing is registered

## A100 Calibration Package

Created:

- `docs/115_a100_sxm_runpod_calibration_runbook.md`
- `scripts/phase4/prepare_runpod_a100_calibration.py`
- `results/processed/a100_sxm_calibration_manifest_100.json`
- `results/processed/a100_sxm_calibration_manifest_200.json`
- `results/processed/a100_sxm_calibration_readiness_report.json`

The local A100 package passes artifact sync, checkpoint/resume, manifest,
runtime compatibility, registered price, and backup dry-run checks. Live RunPod
calibration is not allowed yet because `RUNPOD_SSH_HOST` is not configured.

## Decision

```text
PHASE2A_R1_B_C_PRICE_AND_API_PROBE_READY
A100_SXM_CALIBRATION_PACKAGE_READY
LIVE_RUNPOD_CALIBRATION_BLOCKED_NO_RUNPOD_SSH_HOST
MODEL7_API_PROBE_BLOCKED_PRICE_MISSING
```

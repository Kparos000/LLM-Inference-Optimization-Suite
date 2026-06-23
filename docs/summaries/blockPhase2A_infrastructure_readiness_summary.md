# Block Phase 2A Infrastructure Readiness Summary

Status: framework implemented on June 23, 2026

Phase 2A added the price, API probe, and RunPod calibration framework needed
before live provider or paid GPU scale work.

## Added

- RunPod GPU price registry: `configs/gpu_prices.yaml`
- API provider load-probe framework: `src/inference_bench/api_load_probe.py`
- Dry-run API probe CLI: `scripts/phase4/run_api_load_probe.py`
- RunPod calibration profiles: `configs/runpod_calibration_profiles.yaml`
- Calibration manifest support: `src/inference_bench/calibration_manifest.py`
- GPU cost estimator: `src/inference_bench/gpu_price_registry.py`
- Central cost re-export: `src/inference_bench/cost.py`

## Current Truth

- Supported GPU entries: 22
- Reviewed RunPod hourly prices: 0
- API probe live requests sent: 0
- Calibration profiles: A100 SXM, H100 SXM, L40S
- Calibration readiness: blocked by missing reviewed GPU prices

## Decision

```text
PHASE2A_INFRASTRUCTURE_FRAMEWORK_READY
RUNPOD_CALIBRATION_NOT_READY_PRICE_MISSING
API_LOAD_PROBE_FRAMEWORK_READY_NOT_RUN
```

No RunPod cost or calibration readiness claim is allowed until prices are
reviewed and entered.

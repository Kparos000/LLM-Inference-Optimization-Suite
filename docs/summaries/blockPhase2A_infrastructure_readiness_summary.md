# Block Phase 2A Infrastructure Readiness Summary

Status: framework implemented on June 23, 2026; superseded by R1/B/C prep

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

## Initial Current Truth

- Supported GPU entries: 22
- Reviewed RunPod hourly prices: 0
- API probe live requests sent: 0
- Calibration profiles: A100 SXM, H100 SXM, L40S
- Calibration readiness: blocked by missing reviewed GPU prices

## R1/B/C Update

See
`docs/summaries/blockPhase2A_R1_B_C_infrastructure_api_runpod_prep_summary.md`
for the current infrastructure/API/RunPod prep state. The short version:

- supported GPU entries: 26;
- observed RunPod prices registered with source notes;
- live API probe passed for `model5_gated` and `model6_gated`;
- `model7_gated` skipped until complete pricing is registered;
- A100 SXM local calibration package generated;
- no live RunPod calibration is authorized without `RUNPOD_SSH_HOST`.

## Decision

```text
PHASE2A_INFRASTRUCTURE_FRAMEWORK_READY
SUPERSEDED_BY_PHASE2A_R1_B_C_PRICE_AND_API_PROBE_READY
```

No final RunPod cost claim is allowed until observed prices are re-verified
and measured throughput calibration exists.

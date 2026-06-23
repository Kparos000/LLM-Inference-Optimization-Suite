# Phase 2A Infrastructure Readiness

Status: framework implemented on June 23, 2026

Phase 2A adds the infrastructure readiness layer needed before any RunPod
calibration, API provider load probe, concurrency sweep, SGLang/mm4 comparison,
TensorRT-LLM work, or 2,000/10,000/40,000-prompt matrix run.

No live API probe or RunPod calibration was run in this block.

## GPU Price Registry

The new RunPod GPU price registry is `configs/gpu_prices.yaml`. It lists 22
observed RunPod GPU types, including A100, H100, H200, L40S/L40, A40, RTX
workstation cards, RTX 4090/3090, L4, and RTX Pro variants.

Every `hourly_price` is currently `null`. That is intentional: the project has
not reviewed current RunPod pricing, so GPU cost fields are present but cost
claims remain blocked.

`src/inference_bench/gpu_price_registry.py` exposes:

- `list_supported_gpus()`;
- `get_gpu_metadata()`;
- `get_gpu_price()`;
- `estimate_gpu_cost()`.

The estimator never applies GPU hourly cost to API-provider tracks. API runs
continue to use token pricing only.

## API Load Probe Framework

The API probe framework is implemented in `src/inference_bench/api_load_probe.py`
with a dry-run CLI at `scripts/phase4/run_api_load_probe.py`.

It supports `model5_gated`, `model6_gated`, and `model7_gated` across
concurrency levels 1, 2, 4, 8, and 16. The framework records TTFT, TPOT,
latency, streaming stability, RPS, TPS, 429s, 5xxs, timeouts, retries, and
provider throttling.

Default execution writes a framework report only and returns
`API_PROBE_BLOCKED` because no live requests were sent.

## RunPod Calibration Framework

`configs/runpod_calibration_profiles.yaml` defines three planned profiles:

- `A100_SXM_CALIBRATION`;
- `H100_SXM_CALIBRATION`;
- `L40S_CALIBRATION`.

Each profile names the RunPod GPU, runtime, engine, hardware type, backend
type, allowed concurrencies, model aliases, memory modes, prompt counts, and
traffic profile. Calibration prompt counts are restricted to 100 and 200.

`src/inference_bench/calibration_manifest.py` validates profiles and builds
calibration manifests with GPU/runtime/model/memory/concurrency identity,
artifact paths, Git commit, and cost-estimate fields.

## Readiness Gates

Calibration readiness requires:

- artifact sync enabled;
- checkpoint/resume enabled;
- first-class manifest support;
- reviewed GPU hourly price registered;
- valid runtime profile;
- backup verification dry run passed.

The readiness audit now reports per-profile calibration readiness. All three
profiles are currently `CALIBRATION_NOT_READY` because reviewed GPU prices are
not registered.

## Decision

```text
PHASE2A_INFRASTRUCTURE_FRAMEWORK_READY
RUNPOD_CALIBRATION_NOT_READY_PRICE_MISSING
API_LOAD_PROBE_FRAMEWORK_READY_NOT_RUN
```

The correct next live step is an explicitly authorized API provider load probe
or a RunPod price review. RunPod calibration and GPU cost claims remain blocked
until reviewed prices are entered.

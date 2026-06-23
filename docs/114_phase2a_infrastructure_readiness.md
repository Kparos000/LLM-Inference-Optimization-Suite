# Phase 2A Infrastructure Readiness

Status: framework implemented and R1/B/C prep completed on June 23, 2026

Phase 2A adds the infrastructure readiness layer needed before any RunPod
calibration, API provider load probe, concurrency sweep, SGLang/mm4 comparison,
TensorRT-LLM work, or 2,000/10,000/40,000-prompt matrix run.

Phase 2A initially implemented the framework. Phase 2A-R1/B/C then registered
observed RunPod UI prices, ran the guarded API load probe for routes with
configured keys and pricing, and prepared the A100 SXM local calibration
package. No live RunPod calibration was run.

## GPU Price Registry

The RunPod GPU price registry is `configs/gpu_prices.yaml`. It now lists 26
observed RunPod GPU types, including A100, H100, H200, B200/B300, L40S/L40,
A40, RTX workstation cards, RTX 4090/5090/3090, L4, and RTX Pro variants.

The registered prices came from a RunPod console UI screenshot and every entry
keeps the source note: `Observed from RunPod console UI screenshot; verify
before final cost claims.` The prices are usable for controlled projection
math and readiness gates, but must be re-verified before publication-grade cost
claims.

Primary planning prices:

| GPU | Price |
| --- | ---: |
| A100 SXM 80GB | $1.49/hr |
| H100 SXM 80GB | $3.29/hr |
| L40S | $0.99/hr |

`src/inference_bench/gpu_price_registry.py` exposes:

- `list_supported_gpus()`;
- `get_gpu_metadata()`;
- `get_gpu_price()`;
- `estimate_gpu_cost()`.

The estimator never applies GPU hourly cost to API-provider tracks. API runs
continue to use token pricing only. Runtime projections now expose
`estimated_run_cost`, `projected_1000_cost`, `projected_10000_cost`,
`projected_40000_cost`, `tokens_per_gpu_dollar`, and
`successful_requests_per_gpu_dollar` when the required measured fields exist.

## API Load Probe Framework

The API probe framework is implemented in `src/inference_bench/api_load_probe.py`
with a guarded CLI at `scripts/phase4/run_api_load_probe.py`.

It supports `model5_gated`, `model6_gated`, and `model7_gated` across
concurrency levels 1, 2, 4, 8, and 16. The live R1/B probe ran only the
required first levels 1, 2, and 4. The framework records TTFT, TPOT, latency,
streaming stability, RPS, TPS, 429s, 5xxs, timeouts, retries, provider
throttling, cost, and recommended safe concurrency.

Default execution still writes a framework report only. With
`--live-if-keys-present`, the local `.env` keys allowed a live probe for the
priced routes:

- `model5_gated`: safe through concurrency 4.
- `model6_gated`: safe through concurrency 4.
- `model7_gated`: skipped because complete audited API pricing is missing.

Aggregate live result:

- requests: 60/60 successful;
- 429/5xx/timeouts/retries/provider throttling: 0;
- mean TTFT: 748.826 ms;
- mean TPOT: 9.302 ms;
- mean E2E latency: 1,237.873 ms;
- mean request throughput: 0.906 requests/sec per request row;
- mean token throughput: 48.322 tokens/sec per request row;
- total estimated API cost: $0.00025169.

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

`scripts/phase4/prepare_runpod_a100_calibration.py` writes the local A100 SXM
calibration package:

- `results/processed/a100_sxm_calibration_manifest_100.json`;
- `results/processed/a100_sxm_calibration_manifest_200.json`;
- `results/processed/a100_sxm_calibration_readiness_report.json`.

The A100 SXM package is ready locally with artifact sync, checkpoint/resume,
manifest, runtime compatibility, registered price, and backup dry-run gates
passing. Live RunPod execution is still blocked until `RUNPOD_SSH_HOST` or an
equivalent explicit target is configured.

## Readiness Gates

Calibration readiness requires:

- artifact sync enabled;
- checkpoint/resume enabled;
- first-class manifest support;
- reviewed GPU hourly price registered;
- valid runtime profile;
- backup verification dry run passed.

The readiness audit now reports per-profile calibration readiness. A100 SXM,
H100 SXM, and L40S are price-ready after the observed-price registration. A100
SXM is the primary calibration target. Live RunPod calibration remains blocked
without an explicit remote target, and final cost claims remain subject to
price re-verification.

## Decision

```text
PHASE2A_INFRASTRUCTURE_FRAMEWORK_READY
PHASE2A_R1_B_C_PRICE_AND_API_PROBE_READY
A100_SXM_CALIBRATION_PACKAGE_READY_NO_LIVE_RUNPOD_TARGET
```

The correct next live step is an explicitly authorized A100 SXM RunPod
calibration using the generated runbook and manifests. No live RunPod run is
allowed from this block because no `RUNPOD_SSH_HOST` target was configured.

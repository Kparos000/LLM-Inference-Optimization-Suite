# Controlled Final Infrastructure Completion

## Scope

This pass did not change the benchmark matrix or experiment design. It repaired
the RunPod A100 environment and runner plumbing needed for the controlled final
simulation safety gates.

## SGLang Repair

Root cause: the generic PyPI SGLang kernel wheel did not expose an A100-compatible
`common_ops` path for this CUDA 13 / Torch 2.11 pod, and the dynamic linker could
not see `libnvrtc.so.13`. After the CUDA library path was fixed, the generic
`sgl-deep-gemm` wheel also aborted on import.

Environment repairs applied on the pod:

- added `/usr/local/lib/python3.11/dist-packages/nvidia/cu13/lib` to
  `/etc/ld.so.conf.d/python-nvidia-cu13.conf` and ran `ldconfig`;
- installed `libnuma1`;
- reinstalled `sglang-kernel==0.4.4+cu130` from the SGLang CUDA 13 wheel index;
- reinstalled `sgl-deep-gemm==0.1.3+cu130` from the same wheel index.

Verified imports:

- `libnvrtc.so.13`
- `libcudart.so.13`
- `libnuma.so.1`
- `sgl_kernel`
- `deep_gemm`

The documented full-memory SGLang command launched successfully and
`GET http://127.0.0.1:30000/v1/models` listed
`Qwen/Qwen2.5-7B-Instruct`:

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --served-model-name Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 30000 --mem-fraction-static 0.90 --context-length 4096 --max-running-requests 32 --chunked-prefill-size 8192
```

For simultaneous safety-gate smoke checks on one A100, vLLM and SGLang were
also verified co-resident with lower smoke-only memory reservations. Both
`/v1/models` endpoints listed `Qwen/Qwen2.5-7B-Instruct`.

## API Track

The controlled runner now loads credentials through the same `.env` plus process
environment path used by the API load-probe tooling, then canonicalizes supported
aliases. Supported aliases include `HUGGINGFACE_HUB_TOKEN`, `HUGGINGFACE_TOKEN`,
`HF_API_TOKEN`, `HF_API_KEY`, `OPENROUTER_KEY`, `OPENROUTER_TOKEN`,
`NOVITA_KEY`, and `NOVITA_TOKEN`.

In the live shell used for this pass, no canonical API credential or supported
alias was visible in process environment, `.env`, shell startup files, or
`/etc/environment`, so the API gate correctly remains blocked.

No secrets were committed.

## MM4 Smoke

The MM4 comparison writer now treats missing historical mm2/mm3 comparison
artifacts as explicit `missing_not_estimated` rows instead of failing the smoke
after live MM4 execution. The live MM4 smoke completed 50/50 rows against
`model3_7b`, wrote traces, metrics, manifest, latency, agent summary, and
comparison artifacts.

Live MM4 summary:

- row count: 50
- success count: 50
- JSON validity: 100%
- generation contract validity: 100%
- evidence match: 60%
- groundedness: 60%
- safety violations: 0
- repair rate: 0%
- escalation rate: 2%

## Automation

Run manifests now carry the required future-run metadata:

- `run_type`
- `baseline_or_optimized`
- `optimization_flags`
- `dataset_version`

Post-run automation now builds:

- metric-level PASS/WARNING/FAIL/NOT_AVAILABLE rows for latency, throughput,
  resource, cost, temperature, and quality metrics;
- comparison-report availability metadata;
- plotting-ready long-form datasets for baseline-vs-optimized, engine, model,
  memory-mode, concurrency, GPU, cost, latency, and throughput plots.

## Current Safety Gate

Smoke-only controlled final safety gate:

- vLLM `model3_7b`: `SMOKE_READY`
- SGLang `model3_7b`: `SMOKE_READY`
- MM4: `SMOKE_READY`
- API `model6_gated`: `BLOCKED`

The full controlled simulation remains blocked because API credentials or
supported aliases are not visible to the runner environment.

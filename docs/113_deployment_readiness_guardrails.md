# Deployment Readiness Guardrails

Status: implemented June 19, 2026

Phase 1C adds production guardrails in
`src/inference_bench/production_readiness.py`.

## Guardrails

The readiness verdict checks:

- artifact sync before long self-hosted or RunPod runs;
- GPU hourly price before GPU cost claims;
- checkpoint/resume before 1,000+ prompt runs;
- partial runs cannot be marked complete;
- API provider load probe before large API runs;
- API and GPU tracks join through the unified result-track schema;
- traffic profile, concurrency, and request arrival mode are present.

## Result

The verdict is deterministic:

- `READY`: no blocking guardrails fail.
- `NOT_READY`: at least one required guardrail blocks the run.

These checks do not run inference, call APIs, or allocate GPUs. They are a
pre-run gate before 1,000-prompt, RunPod, concurrency, API-scale, or final
matrix execution.

## Current Policy

No 1,000-prompt, RunPod, concurrency, or final matrix run should start unless
the selected run configuration passes these guardrails and the active quality
gate is already passed at the smaller frozen scale.

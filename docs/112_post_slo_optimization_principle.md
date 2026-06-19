# Post-SLO Optimization Principle

Status: implemented June 19, 2026

Phase 1C adds `configs/optimization_negative_rules.yaml` and
`src/inference_bench/optimization_negative_rules.py`.

## Principle

Optimization is a post-SLO diagnosis action, not a baseline matrix dimension.
The baseline matrix should freeze workload, model, runtime, memory mode,
prompt renderer, generation settings, and evaluator semantics. An optimization
is applied only after a measured SLO failure identifies the bottleneck it is
intended to address.

## Negative Rules

The negative-rule catalog defines when not to use:

- quantization;
- prefix caching;
- speculative decoding;
- tensor parallelism;
- disaggregated prefill;
- context compression;
- concurrency increase;
- stronger model escalation.

Examples:

- Do not increase concurrency while quality fails at concurrency one.
- Do not use context compression when retrieval recall or evidence match is
  already below target.
- Do not escalate to a stronger model when required evidence is absent from
  rendered context.
- Do not use quantization to repair a quality failure before isolating model
  capacity.

The catalog is deterministic and advisory. It does not apply changes
automatically.

# Profiling Hooks

Status: implemented June 19, 2026

Phase 1C adds optional profiling metadata hooks in
`src/inference_bench/profiling_hooks.py` and `RunManifest`.

## Supported Modes

- `disabled`
- `pytorch`
- `nsys`
- `ncu`

Profiling is disabled by default. The hooks do not import PyTorch profiler,
Nsight Systems, or Nsight Compute and do not make those tools required for
normal runs.

## Manifest Metadata

When profiling is enabled, manifests record:

- `profiling_enabled`;
- `profiling_mode`;
- `profiler_output_path`;
- `profiling_metadata`.

Enabled profiling requires an output path. Disabled profiling remains valid
without any profiler dependency or artifact.

## Use

Profiling is for targeted diagnosis after a failing latency, throughput, queue,
prefill, decode, or memory SLO. It should not be enabled for routine quality
gates unless the run is explicitly a profiling experiment.

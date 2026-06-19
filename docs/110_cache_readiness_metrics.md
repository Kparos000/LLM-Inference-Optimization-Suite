# Cache-Readiness Metrics

Status: implemented June 19, 2026

Phase 1C adds deterministic cache-readiness metrics in
`src/inference_bench/cache_readiness.py`.

## Metrics

- `repeated_prefix_tokens`: common prompt-prefix tokens repeated after the
  first request.
- `shared_context_percentage`: overlap of context block IDs across the workload
  slice.
- `prefix_reuse_potential`: common prefix share relative to mean input tokens.
- `kv_cache_pressure_estimate`: estimated active token pressure under the
  configured concurrency and context window.
- `cacheability_score`: blended score from prefix reuse, shared context, and KV
  pressure.
- `estimated_prefix_cache_benefit`: conservative estimate of prefix-cache
  usefulness after KV pressure penalty.

## Interpretation

These metrics are pre-run planning signals. They do not claim actual cache hit
rate. A real prefix-cache experiment must still record backend-native hit-rate,
queue, batch, and cache telemetry before making a performance claim.

Prefix caching should not be enabled as a baseline matrix dimension. It should
be tested only after SLO diagnosis points to prefill/TTFT pressure and the
cache-readiness metrics show meaningful prefix or context reuse.

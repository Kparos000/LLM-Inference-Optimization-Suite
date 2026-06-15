# Block B2 SLO Diagnosis And Optimization Catalog Summary

Status: completed June 15, 2026.

Phase B2 implemented:

- modular SLO profile selection with mode/backend/telemetry applicability;
- 51 cataloged bottlenecks;
- 57 cataloged optimizations;
- failed-SLO-only diagnosis;
- deterministic compatibility filtering and ranking;
- one-factor next-experiment suggestions;
- a future explanation contract with no LLM call or decision authority;
- local A1, A2/A3, and A5/A6 diagnosis report generation.

The existing-run diagnosis selected `use_stronger_model` as the primary action
for all 15 run/vertical slices because promoted retrieval passed while
groundedness and evidence match failed. Low RTX 3070 utilization at concurrency
one produced secondary concurrency-sweep actions for A1 and A2.

PagedAttention is represented as a vLLM `engine_builtin` capability. It is
marked already active on vLLM and is not presented as a toggle.

Current recommendation: use the B1 frozen prompt set to isolate citation
selection, contract/truncation behavior, and safety phrase emission, or compare
a stronger feasible model. Do not scale workload size or concurrency until the
quality gate passes or the model-capability limit is documented.

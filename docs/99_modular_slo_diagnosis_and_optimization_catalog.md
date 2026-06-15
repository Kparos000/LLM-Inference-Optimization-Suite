# Modular SLO Diagnosis And Optimization Catalog

Status: implemented in Phase B2 on June 15, 2026.

## Purpose

Phase B2 adds the first platform intelligence layer without running new
inference. It converts selected SLO targets and measured experiment artifacts
into deterministic diagnoses and compatible next-experiment recommendations.

The decision path is:

1. Resolve an SLO profile.
2. Materialize metric targets for a vertical.
3. Mark targets applicable, not applicable, or conditionally unavailable for
   the run configuration.
4. Compare only measured observations with applicable targets.
5. Diagnose bottlenecks only for failed targets.
6. Map bottlenecks to cataloged optimizations.
7. Filter candidates by engine, hardware, and memory mode.
8. Rank one primary change and produce a controlled rerun suggestion.

No LLM participates in these decisions.

## Modular SLO Profiles

`configs/slo_profiles.yaml` defines the `default_enterprise` profile. Its
production targets come from `configs/slo_targets.yaml`; it does not duplicate
or weaken them.

Users can select or disable these groups:

- retrieval;
- quality;
- latency;
- throughput;
- resource;
- API cost;
- GPU cost;
- retrieval ablation;
- compression;
- agentic trace.

Profiles support vertical overrides and the priority modes `quality_first`,
`latency_first`, `throughput_first`, `cost_first`, and `balanced`.

Applicability is resolved before metric evaluation:

- mm0 marks retrieval, retrieval-ablation, compression, and agentic targets
  `NOT_APPLICABLE`;
- mm1, mm2, and mm3 apply retrieval targets;
- mm3 applies compression targets;
- mm4 applies retrieval, quality, latency, throughput, and bounded agent-trace
  targets;
- API cost applies only to provider/API runs;
- GPU cost applies only when an hourly GPU price is registered;
- resource targets apply only when hardware telemetry exists.

An applicable target with no observation is `UNAVAILABLE`, not failed. This
prevents absent ITL, CPU, RAM, pricing, or backend-native cache metrics from
becoming fabricated failures.

## Bottleneck Catalog

`configs/bottleneck_catalog.yaml` contains 51 stable bottleneck IDs across
quality, retrieval/context, latency, throughput, resource/GPU, cost, and
serving-runtime categories.

Every entry defines required metrics, trigger conditions, possible causes,
compatible optimizations, severity logic, confidence logic, and evidence
fields. `src/inference_bench/bottleneck_catalog.py` validates this contract.

The diagnosis engine creates a bottleneck only from a failed selected SLO.
Passing, unavailable, and not-applicable targets do not produce bottlenecks.

## Optimization Catalog

`configs/optimization_catalog.yaml` contains 57 stable optimization IDs across:

- workload and context;
- retrieval;
- model choice and precision;
- serving engine;
- concurrency and capacity;
- hardware and parallelism;
- bounded agent behavior.

Every entry records compatibility, implementation status, application method,
project support, safety notes, expected-gain provenance, and quality/cost risk.
Expected gains are intentionally null until measured.

Serving configuration and optimization are distinct. Selecting vLLM activates
engine capabilities such as PagedAttention and continuous batching. Tuning
`max_num_seqs`, model length, GPU-memory utilization, KV cache, concurrency, or
prefix caching is a separate controlled optimization experiment.

## PagedAttention

`use_pagedattention_capable_engine` is cataloged as `engine_builtin` for vLLM.
When vLLM is already active, the recommender marks PagedAttention
`already_active` and does not propose an enable toggle.

For Hugging Face Transformers runs with serving, memory-fragmentation, or
KV-pressure evidence, the engine may recommend a matched switch to vLLM. That
recommendation tests the serving engine and its built-in memory management; it
does not claim an unmeasured gain.

## Recommendation Rules

`src/inference_bench/optimization_recommender.py` uses deterministic rules plus
both YAML catalogs.

Examples:

- low GPU utilization at concurrency one ranks a bounded concurrency sweep
  before a hardware upgrade;
- high TTFT with high input tokens ranks context reduction, compression, or
  lower top-k;
- low quality with failed retrieval ranks retrieval repair first;
- low quality with passed retrieval ranks a stronger model, evidence
  formatting, prompt-contract repair, or bounded mm4 comparison;
- incompatible engine, hardware, and memory-mode actions are retained in a
  rejected list with reasons.

Only one recommendation is primary. Each next-experiment suggestion changes one
factor and names the configuration that must remain fixed.

## Existing-Run Diagnosis

The reproducible generator is:

```text
python scripts/phase4/generate_b2_slo_diagnosis_reports.py
```

It reads existing A1, A2/A3, and A5/A6 artifacts and writes four ignored local
reports under `results/processed/`. It runs no inference, API call, or GPU work.

Across all 15 run/vertical diagnoses, promoted retrieval targets passed while
groundedness and evidence-match targets failed. The primary deterministic
recommendation was `use_stronger_model`. A1 and A2 also produced secondary
concurrency-sweep recommendations because measured mean GPU utilization was
below target at concurrency one.

This recommendation is historical diagnosis of A1/A2/A6. Phase B1 has since
shown that Qwen2.5-1.5B improves some contract metrics but still fails the
quality gate. The current practical next experiment is therefore a controlled
quality repair or stronger feasible model comparison on the frozen prompt set,
not a larger workload.

## Future UI And Explainability

A future UI can expose profile, group, priority, vertical, model, memory mode,
engine, and hardware selection. The backend should display passed, failed,
not-applicable, and unavailable targets separately before showing ranked
actions.

`recommendation_explainer_contract.py` reserves `model6_gated` as the default
future explanation candidate because its recorded API smoke had better
quality/cost than Model5. The explainer may paraphrase deterministic JSON. It
may not invent actions, alter SLOs, or change measured values.

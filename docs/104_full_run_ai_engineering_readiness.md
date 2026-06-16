# Full-Run AI Engineering Readiness

Status: measured after B6 on June 15, 2026

## Purpose

This audit checks whether the repository is ready to move from the controlled
500-prompt B6 gate to larger 2,000/10,000-prompt benchmarks, concurrency
sweeps, or RunPod execution.

The audit is deterministic and local. It does not use an LLM as a decision
source.

## Artifacts

- Readiness module: `src/inference_bench/full_run_readiness_audit.py`
- CLI: `scripts/phase4/audit_full_run_readiness.py`
- Report: `results/processed/b6_full_run_readiness_report.json`
- Summary: `results/processed/b6_full_run_readiness_summary.csv`

## Result

The readiness status is:

```text
NOT_READY
```

The audit found 36 checks:

- passes: 33;
- gaps: 2;
- blocking failures: 1.

The blocking failure is the B6 scale gate result. B6 completed all 500
requests, but the quality gate did not pass.

## Passed Areas

Dataset and workload controls are present:

- promoted retrieval manifest exists;
- controlled 2,000 workload exists;
- deterministic workload builder exists;
- B6 500-row runner input exists;
- prompt/gold evaluator join is implemented.

Context and generation controls are present:

- context alignment repair;
- private citation alias mapping;
- leakage guard;
- answer planning;
- multi-evidence selector;
- safety repair;
- generation contract parser.

Run safety controls are present:

- raw outputs can be written incrementally;
- OpenAI-compatible load runner has checkpoint/resume support;
- completed prompt IDs are tracked;
- B6 writes per-prompt failure rows;
- B6 manifest records `error_count`;
- the partial-run check prevents a completed manifest with fewer than 500 rows.

GPU/runtime controls are present:

- `remote_rtx3070` profile;
- vLLM launch documentation;
- GPU telemetry sampler;
- B6 telemetry was available with 527 samples.

SLO and diagnosis controls are present:

- modular SLO profiles;
- bottleneck catalog;
- optimization catalog;
- failed-SLO-only diagnosis;
- missing telemetry is treated as unavailable rather than as a fabricated
  failure.

## Gaps

RunPod cost claims are blocked:

- `configs/runpod_projection_prices.yaml` has no reviewed hourly prices;
- throughput multipliers for RTX 4090, L40S, A100, and H100 remain null.

GPU cost implementation is not centralized:

- no `src/inference_bench/cost.py` module exists;
- API pricing exists separately in `src/inference_bench/api_pricing.py`;
- `configs/gpu_costs.yaml` remains a template.

## Blocking Failure

The B6 gate did not pass:

- JSON validity: 95.4%, target at least 97%;
- contract validity: 94.8%, target at least 97%;
- truncation: 4.6%, target no more than 2%;
- minimum vertical evidence match: 76%, target at least 85%;
- minimum vertical groundedness: 74%, target at least 85%.

The blocking vertical is Research AI:

- JSON validity: 82%;
- contract validity: 80%;
- evidence match: 76%;
- groundedness: 74%;
- truncation: 18%.

## Decision

Do not run:

- concurrency 2/4 sweep;
- SGLang comparison;
- mm4 agentic comparison;
- RunPod execution;
- 2,000-prompt benchmark;
- 10,000-prompt benchmark.

The next engineering block should remain quality-focused:

```text
B6R1_RESEARCH_AI_TRUNCATION_AND_CONTRACT_REPAIR
```

Run only a targeted replay over B6 failed/truncated/invalid rows first. Keep
the evaluator, gold data, and promoted retrieval source unchanged. After the
targeted gate passes, rerun the same frozen 500 B6 matrix before considering
any concurrency or larger-prompt-count work.

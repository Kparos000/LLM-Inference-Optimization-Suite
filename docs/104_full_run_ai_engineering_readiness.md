# Full-Run AI Engineering Readiness

Status: updated after Phase 1E on June 19, 2026

## Purpose

This audit checks whether the repository is ready to move from the controlled
500-prompt B6/B6R1 gates to larger prompt-count benchmarks, concurrency
sweeps, SGLang/mm4 comparisons, or RunPod execution.

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

The audit found 40 checks:

- passes: 34;
- gaps: 3;
- blocking failures: 3.

The blocking failures are the failed B6 gate, the failed B6R1 targeted Research
AI repair gate, and the resulting prohibition on a 1,000-prompt terminal run.

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
- Phase 1E adds first-class production run-manifest fields;
- checkpoint/resume can recover from checkpoint JSON and partial raw JSONL;
- duplicate `prompt_id` rows are blocked unless explicitly allowed;
- failed rows are persisted;
- local artifact sync writes run-scoped backups under `backups/`;
- backup verification checks file existence, non-zero size, hashes, and
  manifest row accounting.

B6R1 adds targeted replay manifests and combined raw replay output for the
Research AI failed-row audit. The full frozen 500-row B6R1 manifest is absent
because no targeted strategy passed and the full rerun was intentionally not
started.

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
- cloud artifact sync is not implemented yet; local backup is implemented and
  S3/R2/GDrive remain future providers.

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

B6R1 replayed the 26 failed/truncated/invalid Research AI rows and did not
clear the blocker:

- `concise_research_ai_renderer`: JSON 46.15%, contract 38.46%, evidence
  30.77%, groundedness 23.08%, truncation 53.85%, safety 0.
- `research_ai_output_budget_224`: JSON 92.31%, contract 84.62%, evidence
  73.08%, groundedness 65.38%, truncation 7.69%, safety 0.

Neither strategy met the targeted thresholds, so the full frozen 500-row B6R1
rerun was not triggered.

## Decision

Do not run:

- 1,000-prompt terminal run;
- concurrency 2/4 sweep;
- SGLang comparison;
- mm4 agentic comparison;
- RunPod execution;
- 2,000-prompt benchmark;
- 10,000-prompt benchmark.

Long RunPod/self-hosted runs are additionally blocked unless artifact sync,
checkpoint/resume, GPU hourly pricing, first-class manifests, partial-run
protection, and a passing backup verification dry run are all enabled.

The next engineering block should remain Research AI quality-focused:

```text
B6R2_RESEARCH_AI_MODEL_OR_AGENTIC_COMPARISON
```

Keep the frozen 26-row B6R1 Research AI replay set. Compare a stronger
feasible model or a Research AI-only bounded mm4 path before any larger or
concurrent run. Keep the evaluator, gold data, and promoted retrieval source
unchanged.

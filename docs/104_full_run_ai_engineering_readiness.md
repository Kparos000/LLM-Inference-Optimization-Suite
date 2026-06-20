# Full-Run AI Engineering Readiness

Status: updated after Phase B6R4 on June 20, 2026

## Purpose

This audit checks whether the repository is ready to move from the controlled
500-prompt B6/B6R1/B6R2/B6R4 gates to larger prompt-count benchmarks,
concurrency sweeps, SGLang/mm4 comparisons, or RunPod execution.

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

The audit found 49 checks:

- passes: 41;
- gaps: 2;
- blocking failures: 6.

The blocking failures are the failed B6, B6R1, and B6R2 quality gates, the
absence of a passed selected B6R2 full-500 contract gate, the blocked B6R4
full-500 `model2_3b` gate, and the resulting prohibition on a 1,000-prompt
terminal run.

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

B6R4 adds a `model2_3b` targeted replay manifest, full 500-row raw output, and
processed evaluation reports. The targeted Research AI gate passed; the full
500-row gate completed and remains blocked by minimum vertical quality.

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

B6R2 tested Research AI-specific contracts and also did not clear the targeted
blocker. The best candidate reached 96.15% JSON/contract validity and 80.77%
evidence match and groundedness with zero safety violations and zero
truncation, so the full B6R2 rerun was not triggered.

B6R3 replayed the same frozen 26 rows through `model6_gated` / Llama 3.1 8B
through the API provider route. It passed the targeted gate with 100%
JSON/contract validity, 96.15% evidence match and groundedness, zero safety
violations, and zero truncation. This was targeted API-provider capacity
evidence, not a replacement for the full 500-row gate.

B6R4 replayed the same targeted set through `model2_3b` /
Qwen2.5-3B-Instruct on the remote RTX 3070 vLLM path and passed:

- JSON validity: 100%;
- contract validity: 100%;
- evidence match: 88.46%;
- groundedness: 88.46%;
- safety violations: 0;
- truncation: 0%.

The targeted pass triggered the full frozen 500-row B6R4 run. It completed 500
of 500 requests with 98.4% JSON validity, 98.4% contract validity, 90.6%
evidence match, 90.6% groundedness, zero safety violations, and 1.6%
truncation. The full gate remains blocked because Finance and Research AI each
reached only 80% evidence match and 80% groundedness, below the 85% minimum
vertical threshold.

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
B6R5_MODEL2_3B_FINANCE_RESEARCH_VERTICAL_REPAIR
```

Freeze B6R4 artifacts. Diagnose Finance and Research AI full-500 failures on
the `model2_3b` run, then decide whether the next controlled comparison should
repair Qwen2.5-3B citation selection, test model3_7b feasibility, or run a full
500 API-provider model6 gate. Keep the evaluator, gold data, and promoted
retrieval source unchanged.

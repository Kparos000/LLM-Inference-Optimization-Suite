# Full-Run AI Engineering Readiness

Status: updated after Phase B6R5 on June 20, 2026

## Purpose

This audit checks whether the repository is ready to move from the controlled
500-prompt B6/B6R1/B6R2/B6R4/B6R5 gates to larger prompt-count benchmarks,
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
READY_WITH_QUALITY_CAVEAT
```

Detailed readiness is split:

- deployability readiness: `NOT_READY`;
- benchmark execution readiness: `READY_WITH_QUALITY_CAVEAT`;
- 1,000-prompt terminal baseline allowed: `true`.

The audit found 50 checks:

- passes: 42;
- gaps: 2;
- non-blocking failures: 5;
- blocking failures: 0.

The failed B6, B6R1, B6R2, and B6R4 quality gates remain recorded evidence.
B6R5 did not make the selected model deployable, but it separated benchmark
execution readiness from deployability readiness. A controlled 1,000-prompt
terminal baseline is allowed as caveated benchmark evidence only.

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

B6R5 adds Finance/Research failed-row replay input, failure-audit reports,
targeted strategy comparison reports, and a refreshed readiness report. The
selected strategy did not trigger a full 500 rerun because Research AI remained
below the targeted vertical threshold.

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

## Quality Caveat

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

B6R5 replayed the 40 Finance and Research AI failed B6R4 rows with three
targeted strategies. The selected strategy, `evidence_selection_preplan`,
reached:

- JSON validity: 100%;
- contract validity: 100%;
- evidence match: 80%;
- groundedness: 80%;
- Finance evidence/groundedness: 90% / 90%;
- Research AI evidence/groundedness: 70% / 70%;
- safety violations: 0;
- truncation: 0%.

The decision is `B6R5_QUALITY_CAVEATED`. Finance cleared the targeted floor on
the failed-row subset, but Research AI did not. Therefore no full 500-row B6R5
rerun was triggered, and no deployability claim is allowed.

## Decision

Allowed next benchmark:

- controlled 1,000-prompt terminal baseline, only as caveated benchmark
  evidence.

Do not run:

- concurrency 2/4 sweep;
- SGLang comparison;
- mm4 agentic comparison;
- RunPod execution;
- 2,000-prompt benchmark;
- 10,000-prompt benchmark.

Long RunPod/self-hosted runs are additionally blocked unless artifact sync,
checkpoint/resume, GPU hourly pricing, first-class manifests, partial-run
protection, and a passing backup verification dry run are all enabled.

The next engineering action should keep the benchmark/deployability distinction
explicit:

```text
CONTROLLED_1000_PROMPT_TERMINAL_BASELINE_WITH_QUALITY_CAVEAT
```

Keep deployability blocked until a selected model path clears the Finance and
Research AI vertical floors without caveat. Keep the evaluator, gold data, and
promoted retrieval source unchanged.

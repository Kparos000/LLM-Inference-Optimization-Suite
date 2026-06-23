# Full-Run AI Engineering Readiness

Status: updated after Phase 2A on June 23, 2026

## Purpose

This audit checks whether the repository is ready to move from the controlled
500-prompt B6/B6R1/B6R2/B6R4/B6R5/B6R6 gates, the controlled B7 1,000-prompt
baseline, and the B7R1 vLLM stability repair to larger prompt-count
benchmarks, concurrency sweeps, SGLang/mm4 comparisons, API load probes, or
RunPod execution.

The audit is deterministic and local. It does not use an LLM as a decision
source.

## Artifacts

- Readiness module: `src/inference_bench/full_run_readiness_audit.py`
- CLI: `scripts/phase4/audit_full_run_readiness.py`
- Report: `results/processed/b6_full_run_readiness_report.json`
- Summary: `results/processed/b6_full_run_readiness_summary.csv`

## Result

The deterministic B6R6 repository audit status remains:

```text
READY
```

The current measured B7R1 operational readiness status is:

```text
B7R1_STABILITY_READY
```

Detailed readiness is split:

- deployability readiness: `READY`;
- benchmark execution readiness: `READY`;
- 1,000-prompt terminal baseline allowed: `true`;
- API load probe allowed: `true`;
- RTX 3070 Qwen3B suitability: `stable`.

The audit found 63 checks:

- passes: 53;
- gaps: 4;
- non-blocking failures: 5;
- blocking failures: 0.

The failed B6, B6R1, B6R2, and B6R4 quality gates remain recorded evidence.
B6R5 repaired Finance but left Research AI below the locked baseline. B6R6
restored Research AI and passed the full frozen 500-row quality gate, so a
controlled 1,000-prompt terminal baseline was allowed at concurrency one. B7
then found a serving-stability blocker. B7R1 repaired that blocker on the same
frozen 1,000-row input and restored controlled benchmark readiness for the
RTX 3070 Qwen3B vLLM track at concurrency one.

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

B6R6 adds a Research AI baseline lock, targeted strategy comparison, full
500-row raw output, full evaluation report, and refreshed readiness report. The
selected Research AI strategy triggered and passed the full frozen 500-row
gate.

B7 adds a 1,000-row `model2_3b` vLLM runner input, raw output, manifest,
checkpoint/resume state, GPU telemetry, evaluation report, runtime projection,
artifact sync report, and B7 readiness report. The preflight passed, but the
live run is blocked by vLLM serving failure.

B7R1 adds a vLLM CUDA failure audit, safe serving profile, preflight report,
1,000-row repaired raw output, manifest, checkpoint, GPU telemetry, evaluation
report, comparison report, runtime projection, artifact sync report, and
B7R1 readiness report. The repaired run completed 1,000 of 1,000 requests with
zero fatal engine errors.

Phase 2A adds infrastructure readiness controls:

- RunPod GPU price registry with 26 observed-price GPU entries;
- API provider load-probe framework for `model5_gated`, `model6_gated`, and
  `model7_gated`;
- guarded API probe CLI with live-if-keys-present mode;
- RunPod calibration profiles for A100 SXM, H100 SXM, and L40S;
- 100/200-prompt calibration manifest support;
- GPU cost fields in runtime projections, including run cost, 1,000/10,000/
  40,000-prompt projections, tokens per GPU dollar, and successful requests per
  GPU dollar when measured inputs exist.

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

RunPod final cost claims remain caveated:

- `configs/gpu_prices.yaml` contains observed RunPod console UI prices with
  source notes requiring re-verification before final cost claims;
- `configs/runpod_projection_prices.yaml` has no reviewed hourly prices;
- throughput multipliers for RTX 4090, L40S, A100, and H100 remain null.
- cloud artifact sync is not implemented yet; local backup is implemented and
  S3/R2/GDrive remain future providers.

RunPod calibration execution is blocked:

- A100 SXM, H100 SXM, and L40S calibration profiles exist;
- readiness requires artifact sync, checkpoint/resume, manifests, runtime
  compatibility, backup verification dry run, and reviewed GPU price;
- A100 SXM local package readiness passes those local gates;
- no live RunPod calibration is allowed until `RUNPOD_SSH_HOST` or an
  equivalent explicit target is configured.

Cost implementation is now split by track:

- API pricing exists in `src/inference_bench/api_pricing.py`;
- GPU price lookup and projection support exists in
  `src/inference_bench/gpu_price_registry.py`;
- `src/inference_bench/cost.py` re-exports the unified GPU estimator;
- `configs/gpu_costs.yaml` remains a historical/template config.

## Quality History And B6R6 Recovery

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

B6R6 replayed only the 20 failed B6R4 Research AI rows with a baseline lock at
80% evidence match and 80% groundedness. The selected `answer_skeleton`
strategy reached:

- JSON validity: 100%;
- contract validity: 100%;
- evidence match: 90%;
- groundedness: 90%;
- safety violations: 0;
- truncation: 0%.

The selected strategy exceeded the preferred 85% floor, so the full frozen
500-row B6R6 run was triggered. It completed 500 of 500 requests with:

- JSON validity: 98.2%;
- contract validity: 97.8%;
- evidence match: 97.0%;
- groundedness: 96.6%;
- safety violations: 0;
- truncation: 1.8%.

Per-vertical evidence match and groundedness:

- Airline: 93% / 93%;
- Healthcare Admin: 100% / 100%;
- Retail: 100% / 100%;
- Finance: 96% / 96%;
- Research AI: 96% / 94%.

The decision is `B6R6_QUALITY_READY`.

## B7 Operational Finding

B7 ran the first controlled 1,000-prompt baseline at concurrency one with
artifact sync, checkpoint/resume, manifest, and GPU telemetry enabled. Preflight
passed:

- runner input rows: 1,000;
- per vertical: 200;
- all required gold evidence present in E1-E5: 1,000/1,000;
- canonical IDs exposed to the model: 0;
- artifact sync dry run: passed;
- runtime registry allowed vLLM on `remote_rtx3070`.

The raw output contains 1,000 unique prompt IDs, but only 663 successful
requests. vLLM failed at Finance prompt 17 with an EngineCore fatal CUDA/CUBLAS
error. The run resumed from partial raw output and preserved 337 failed request
rows.

Overall B7 quality:

- JSON validity: 64.8%;
- contract validity: 64.8%;
- evidence match: 64.3%;
- groundedness: 64.3%;
- safety violations: 0;
- truncation: 1.2%.

Per-vertical evidence match and groundedness:

- Airline: 93% / 93%;
- Healthcare Admin: 100% / 100%;
- Retail: 99.5% / 99.5%;
- Finance: 7.5% / 7.5%;
- Research AI: 21.5% / 21.5%.

Finance and Research AI are dominated by request failures after the engine died,
so B7 is not a clean model-quality comparison. The decision is
`B7_CONTROLLED_1000_BASELINE_BLOCKED`.

## B7R1 Stability Repair

B7R1 audited the B7 failure and reran the exact frozen 1,000-row input. The
initial intended safe profile, `gpu_memory_utilization=0.78`, could not
initialize KV-cache blocks in vLLM 0.23.0. The loadable safe profile uses:

- `gpu_memory_utilization`: 0.82;
- `max_model_len`: 3,584;
- `max_num_seqs`: 1;
- `max_num_batched_tokens`: 3,584;
- `enforce_eager`: true;
- `disable_custom_all_reduce`: true.

B7R1 completed:

- prompts: 1,000/1,000;
- successful requests: 1,000;
- fatal engine errors: 0;
- JSON validity: 98.5%;
- contract validity: 98.3%;
- evidence match: 96.1%;
- groundedness: 95.9%;
- safety violations: 0;
- truncation: 1.2%;
- peak sampled VRAM: 7,404 MB;
- backup completeness score: 1.0.

The decision is `B7R1_STABILITY_READY`.

## Decision

Allowed next independent track:

- explicitly authorized API provider load probe using the Phase 2A framework.

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

The next engineering action should keep the API/self-hosted benchmark
distinction explicit:

```text
API_PROVIDER_LOAD_PROBE
```

Keep concurrency, SGLang, mm4, RunPod, 2,000-prompt, and 10,000-prompt runs as
separate follow-on decisions after B7R1 and Phase 2A review. Keep RunPod cost
and calibration readiness claims blocked until reviewed hourly price,
throughput multiplier inputs, and calibration gate evidence are configured.
Keep the evaluator, gold data, and promoted retrieval source unchanged.

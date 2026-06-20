# Project State

Status as of June 20, 2026.

## Current Decision

```text
READY_FOR_SMALL_MODEL_SERVING_EXPERIMENTS
B6_QUALITY_IMPROVED_BUT_BLOCKED
B6R1_BLOCKED
B6R2_BLOCKED
B6R3_MODEL6_CAPACITY_PASSED
B6R4_TARGETED_MODEL2_3B_PASSED
B6R4_MODEL2_3B_500_BLOCKED
PRODUCTION_MODEL_REGISTRY_FROZEN
PRODUCTION_RUNTIME_REGISTRY_READY
PRODUCTION_WORKLOAD_AND_GUARDRAILS_READY
REPOSITORY_CLEANED_AND_CI_VALIDATION_HARDENED
ARTIFACT_SYNC_LONG_RUN_RECOVERY_READY
FULL_RUN_NOT_READY
```

Blocks A1 through A6 validated the RTX 3070 vLLM/SGLang serving paths, GPU
telemetry, and bounded mm4 workflow.

Phase B1 loaded Qwen2.5-1.5B on vLLM and completed a balanced 100-prompt smoke
without OOM. The model fit in 8 GB VRAM with 6,534 MB peak sampled memory.

Phase B2 implemented modular SLO profiles, structured bottleneck and
optimization catalogs, failed-SLO-only diagnosis, deterministic compatibility
filtering, and one-factor next-experiment recommendations. It ran no new
inference. Existing A1, A2/A3, and A5/A6 metrics were diagnosed from local
artifacts.

Phase B3 audited all 65 failed B1 rows from existing artifacts. It found that
52 failures lacked at least one required gold evidence ID in the rendered E1-E5
context, while 18 failures had available evidence that the model did not cite.
No inference ran and no evaluator, gold, or promoted retrieval data changed.

Phase B4 executed the frozen context-alignment repair and reran the exact
100-prompt Qwen2.5-1.5B vLLM matrix. All required gold evidence now maps to
E1-E5, including Finance 20/20, but the quality gate remains blocked because
the two Airline safety violations persisted.

Phase B5 repaired the safety wording and multi-evidence citation-selection
path on the same frozen matrix. The targeted 25 failed-row replay passed the
B5 gate and triggered a full frozen 100 rerun. The full rerun reached 99% JSON
and contract validity, 96% evidence match and groundedness, and zero safety
violations.

Phase B6 ran the controlled 500-prompt concurrency-one scale gate. The
preflight mapped all required evidence into E1-E5 for 500 of 500 rows and did
not expose canonical IDs to the model. The live run completed 500 of 500
requests, but the gate is `B6_QUALITY_IMPROVED_BUT_BLOCKED` because JSON,
contract, truncation, and Research AI vertical quality targets did not pass.

Phase B6R1 replayed only the 26 B6 Research AI rows that were failed,
truncated, invalid JSON, invalid contract, evidence-mismatched, or ungrounded.
All required evidence had been present in the B6 E1-E5 context. Neither
targeted repair strategy passed the B6R1 gate, so the full frozen 500-row
rerun was not triggered.

Phase B6R2 added a versioned vertical generation-contract registry and tested
five Research AI-specific contracts at 224 and 320 output tokens on the same
26-row frozen replay set. No candidate passed the targeted gate. The best
candidate, `research_ai_limitations_v1`, reached 96.15% JSON/contract validity
and 80.77% evidence match/groundedness with zero truncation and zero safety
violations. The full frozen 500-row rerun was not triggered.

Phase B6R3 replayed the same frozen 26 Research AI failed rows through
`model6_gated` / `meta-llama/Llama-3.1-8B-Instruct` on the existing Hugging
Face provider route. The targeted gate passed with 100% JSON and contract
validity, 96.15% evidence match and groundedness, zero truncation, and zero
safety violations. This makes Qwen2.5-1.5B model capacity the likely Research
AI blocker, but the full-run state remains `NOT_READY` because the frozen
500-row gate has not passed.

Phase B6R4 tested the active self-hosted small baseline, `model2_3b` /
`Qwen/Qwen2.5-3B-Instruct`, on the remote RTX 3070 vLLM path. The frozen
26-row Research AI targeted replay passed with 100% JSON/contract validity,
88.46% evidence match and groundedness, zero truncation, and zero safety
violations. The targeted pass triggered the full frozen 500-row run. That run
completed 500/500 requests and passed aggregate JSON, contract, evidence,
groundedness, safety, and truncation thresholds, but it is
`B6R4_MODEL2_3B_500_BLOCKED` because Finance and Research AI each reached only
80% evidence match and 80% groundedness, below the 85% minimum vertical gate.

Phase 1A froze the production model registry. Active aliases are now
`model1_0_5b`, `model2_3b`, `model3_7b`, `model4_32b`, `model5_gated`,
`model6_gated`, and `model7_gated`. Historical aliases including
`model2_1_5b` and `model7_large_placeholder` remain resolvable as deprecated
compatibility aliases. `model6_gated` remains Llama 3.1 8B. `model7_gated`
uses Mistral Small 3.2 24B through the HF provider route, with paid execution
blocked until complete input/output token pricing is captured.

Phase 1B added the production runtime registry. The stack is now documented as
Runtime -> Infrastructure -> Tooling -> Evaluation. Hugging Face Transformers,
vLLM, SGLang, and API provider routes are typed runtime entries with explicit
model/provider/execution-target/hardware compatibility. TensorRT-LLM is
registered only as a planned engine and is excluded from live selection until
it is smoke-tested.

Phase 1C added production workload and deployment guardrails before any
1,000-prompt, RunPod, concurrency, or final matrix run. Reports now have
pre-run support for ISL/OSL distributions, traffic profiles, request-arrival
mode, cache-readiness metrics, optional profiling metadata, post-SLO negative
optimization rules, and deterministic production readiness guardrails.

Phase 1D cleaned local pytest/tool temp folders, strengthened `.gitignore`,
and hardened CI/CD validation. The CI workflow now mirrors the local validation
order and runs targeted config, repository-hygiene, CI-audit, mypy, pytest,
ruff, public-content audit, doctor, and validate-config gates.

Phase 1E added local artifact sync and long-run recovery controls. Production
manifests now include runtime/backend/provider identity, workload/config
hashes, row counts, timestamps, statuses, and artifact paths. The checkpoint
manager resumes from partial raw JSONL, prevents duplicate prompt IDs by
default, persists failed rows, and writes a clear resume report. The local
backup engine syncs raw, manifest, telemetry, processed, checkpoint, failed-row,
and log artifacts to `backups/` and verifies existence, non-zero size, hashes,
and manifest row accounting.

## B1 Quality Gate

- JSON validity: 93%, required 95%
- Contract validity: 92%, required 85%
- Evidence match: 35%, required 60%
- Groundedness: 35%, required 60%
- Safety violations: 2, required 0

Result: `QUALITY_BLOCKED`.

The exact 50-prompt A1 overlap improved contract validity from 72% to 94%,
evidence match from 30% to 44%, and groundedness from 28% to 44%. The gain is
real but insufficient for scale.

## B2 Deterministic Diagnosis

- SLO profile: `default_enterprise`, backed by `configs/slo_targets.yaml`
- Bottleneck catalog: 51 IDs
- Optimization catalog: 57 IDs
- Existing diagnoses: 15 run/vertical slices across A1, A2/A3, and A5/A6
- Primary recommendation for all slices: `use_stronger_model`
- Secondary A1/A2 capacity action: bounded concurrency sweep
- Decision source: deterministic rules and YAML catalogs
- LLM calls: none

PagedAttention is represented as an active vLLM engine capability, not an
optimization toggle.

## B3 Quality Root Cause

- Failed B1 rows audited: 65
- Required gold absent from E1-E5: 52
- Evidence present but not cited: 18
- Partial multi-evidence citation: 27
- Invalid JSON / contract: 7 / 8
- Truncation: 6
- Finance failures with required evidence absent: 18 of 19
- Finance safety violations: 0

Finance is primarily a frozen workload/rendered-context alignment problem, with
a secondary citation-selection and truncation problem. This does not revise the
promoted retrieval source of truth.

## B4 Context Alignment And Quality Repair

- Context-aligned runner rows: 100
- All required gold evidence present in E1-E5: 100/100
- Finance all-required-gold-present rate: 20/20, up from 2/20 in B1
- JSON validity: 97%, up from 93%
- Contract validity: 97%, up from 92%
- Evidence match: 76%, up from 35%
- Groundedness: 76%, up from 35%
- Safety violations: 2, unchanged
- Truncation: 3%, down from 6%
- Mean TTFT: 137.966 ms
- Mean TPOT: 11.280 ms
- Mean E2E latency: 1,701.776 ms
- Mean/peak GPU utilization: 83.02% / 100%
- Peak sampled GPU memory: 6,602 MB

Result: `QUALITY_BLOCKED`.

The post-B4 audit found 25 failed rows. Zero failed rows lacked required gold
evidence in E1-E5. Evidence was present but not cited in 24 failed rows, and
all 25 failed rows were classified as model instruction-following failures.
Finance is now primarily a model citation-selection problem, not a
retrieval/context availability problem or safety problem.

## B5 Final Generation Quality Hardening

- Targeted B4 failed rows replayed: 25
- Targeted JSON validity: 100%
- Targeted contract validity: 100%
- Targeted evidence match: 92%
- Targeted groundedness: 92%
- Targeted safety violations: 0
- Targeted truncation: 0%
- Lexical-guard repairs: 2
- Missing-label retry triggers: 4
- Full frozen 100 rerun triggered: yes
- Full JSON validity: 99%
- Full contract validity: 99%
- Full evidence match: 96%
- Full groundedness: 96%
- Full safety violations: 0
- Full truncation: 1%
- Mean full-run TTFT: 142.102 ms
- Mean full-run TPOT: 10.718 ms
- Mean full-run E2E latency: 1,473.156 ms

Result: `QUALITY_READY_FOR_FROZEN_100`.

Residual full-run failures remain: one Airline citation miss, two Finance
citation misses, and one Research AI truncated JSON output. The B5 result is a
frozen 100-prompt gate, not a final scale benchmark or concurrency claim.

## B6 500-Prompt Quality Scale Gate

- Runner rows: 500
- Per vertical: 100
- All required evidence present in E1-E5: 500/500
- Canonical IDs exposed to model: 0
- Requests completed: 500/500
- JSON validity: 95.4%, required 97%
- Contract validity: 94.8%, required 97%
- Evidence match: 91.2%, required 90%
- Groundedness: 90.8%, required 90%
- Safety violations: 0, required 0
- Truncation: 4.6%, required <=2%
- Mean TTFT: 141.543 ms
- Mean TPOT: 11.489 ms
- Mean E2E latency: 1,741.355 ms
- p95 E2E latency: 5,021.188 ms
- Mean/peak GPU utilization: 81.33% / 100%
- Peak sampled GPU memory: 6,760 MB

Result: `B6_QUALITY_IMPROVED_BUT_BLOCKED`.

Per-vertical evidence match and groundedness:

- Airline: 91% / 91%
- Healthcare Admin: 100% / 100%
- Retail: 94% / 94%
- Finance: 95% / 95%
- Research AI: 76% / 74%

Finance is no longer the blocking vertical. Research AI is the blocker, with
82% JSON validity, 80% contract validity, 18% truncation, 76% evidence match,
and 74% groundedness.

The full-run readiness audit is `NOT_READY`. It found repository safeguards
for dataset/workload, context/generation, run safety, telemetry, and SLO
diagnosis, but it blocks larger runs because B6 failed quality. RunPod cost
claims also remain blocked because hourly prices and throughput multipliers
are unset.

## B6R1 Research AI Truncation And Contract Repair

- Replay rows: 26
- Groundedness failures in the B6 replay set: 26
- Evidence-match failures: 24
- Invalid contract: 20
- Invalid JSON: 18
- Truncation: 18
- Required evidence present in B6 E1-E5 context: yes

Targeted strategy results:

- `concise_research_ai_renderer`: 46.15% JSON, 38.46% contract, 30.77%
  evidence match, 23.08% groundedness, 53.85% truncation, zero safety
  violations.
- `research_ai_output_budget_224`: 92.31% JSON, 84.62% contract, 73.08%
  evidence match, 65.38% groundedness, 7.69% truncation, zero safety
  violations.

Result: `B6R1_BLOCKED`.

B6R1 did not clear the Research AI blocker. The better 224-token strategy
reduced truncation and improved quality, but not enough to pass the targeted
gate. The failure is now best treated as a model/output-control limitation on
Research AI under Qwen2.5-1.5B, not a promoted retrieval or gold-data problem.

## B6R2 Research AI Vertical Contract Selection

- Replay rows: 26
- Candidate contracts: minimal answer, findings, limitations, comparison, and
  deterministic adaptive routing
- Output budgets tested: 224 and 320
- Full frozen 500 rerun triggered: no
- Safety violations across candidates: 0
- Truncation after corrected targeted replay: 0% for all candidates

Best candidate:

- `research_ai_limitations_v1` at 224 and 320
- JSON validity: 96.15%, required 97%
- Contract validity: 96.15%, required 97%
- Evidence match: 80.77%, required 85%
- Groundedness: 80.77%, required 85%
- Truncation: 0%, required <=2%
- Safety violations: 0, required 0

Result: `B6R2_BLOCKED`.

B6R2 confirms that a vertical-specific Research AI contract can eliminate
truncation but still does not make Qwen2.5-1.5B pass the Research AI targeted
quality gate. The blocker is now a model/output-control capability limit on the
frozen Research AI replay set, not promoted retrieval, context availability,
gold data, or evaluator semantics.

## B6R3 Research AI Model Capacity Validation

- Replay rows: 26
- Model: `model6_gated` / `meta-llama/Llama-3.1-8B-Instruct`
- Provider route: Hugging Face provider route with Novita pricing
- Maximum output: 320 tokens
- JSON validity: 100%, required 97%
- Contract validity: 100%, required 97%
- Evidence match: 96.15%, required 85%
- Groundedness: 96.15%, required 85%
- Safety violations: 0, required 0
- Truncation: 0%, required <=2%
- Total API cost: `$0.00077462`

Result: `B6R3_MODEL6_CAPACITY_PASSED`.

One row, `research_ai_scaleup_2000_0099`, still missed evidence match and
groundedness by omitting required introduction evidence. The targeted pass is a
model-capacity signal only. It does not replace B6 as the last full 500-row
gate and does not authorize larger or concurrent runs.

## B6R4 Qwen2.5-3B Research AI Quality Validation

Targeted replay:

- Replay rows: 26
- Model: `model2_3b` / `Qwen/Qwen2.5-3B-Instruct`
- Runtime: vLLM on remote RTX 3070
- Maximum output: 320 tokens
- JSON validity: 100%, required 97%
- Contract validity: 100%, required 97%
- Evidence match: 88.46%, required 85%
- Groundedness: 88.46%, required 85%
- Safety violations: 0, required 0
- Truncation: 0%, required <=2%

Result: `B6R4_TARGETED_MODEL2_3B_PASSED`.

Full 500:

- Requests completed: 500/500
- JSON validity: 98.4%, required 97%
- Contract validity: 98.4%, required 97%
- Evidence match: 90.6%, required 90%
- Groundedness: 90.6%, required 90%
- Safety violations: 0, required 0
- Truncation: 1.6%, required <=2%
- Mean TTFT: 690.308 ms
- Mean TPOT: 17.249 ms
- Mean E2E latency: 2,702.659 ms

Result: `B6R4_MODEL2_3B_500_BLOCKED`.

The full gate failed only the minimum vertical evidence/groundedness checks:
Finance and Research AI both reached 80% evidence match and 80% groundedness,
below the 85% vertical minimum. Qwen2.5-3B materially improves Research AI
relative to Qwen2.5-1.5B and passes the targeted gate, but it does not yet
authorize 1,000 prompts or larger/concurrent runs.

Result tracks are separated:

- API provider track: `model5_gated`, `model6_gated`, and `model7_gated`
  through OpenRouter, Novita, or Hugging Face provider routes, with API token
  cost and no provider GPU telemetry. `model7_gated` is registered but
  unpriced.
- Self-hosted GPU track: `model2_3b`, `model3_7b`, and `model4_32b` through
  Hugging Face local, vLLM, SGLang, or RunPod, with GPU telemetry and hourly
  infrastructure cost when configured, and no API token price.
- Planned engine track: TensorRT-LLM is registered for future compatibility
  planning only and is not runnable until smoke-tested.
- Production guardrail track: long RunPod/self-hosted runs require artifact
  sync and checkpoint/resume; GPU cost claims require hourly price; large API
  runs require a provider load probe; partial runs cannot be marked complete;
  API and GPU tracks must join through the unified result schema.
- Phase 1E dry-run track: 20 simulated prompts wrote 10 rows, resumed the
  remaining 10, persisted one failed row, synced local artifacts, and passed
  backup verification with completeness score 1.0.

## Next Step

Run `B6R5_MODEL2_3B_FINANCE_RESEARCH_VERTICAL_REPAIR`. Freeze B6R4 artifacts
and diagnose the Finance and Research AI full-500 failures without modifying
gold data, evaluator semantics, or promoted retrieval. Do not run a
1,000-prompt terminal run, concurrency 2/4, SGLang, mm4, RunPod, a 2,000-prompt
benchmark, or a 10,000-prompt benchmark until a selected model path passes the
full 500-row gate.

See `docs/summaries/blockB1_vllm_1_5b_quality_smoke_summary.md` for the measured
result and comparison. See
`docs/99_modular_slo_diagnosis_and_optimization_catalog.md` for the B2 decision
architecture, `docs/100_generation_quality_root_cause_audit.md` for B3,
`docs/101_context_alignment_and_generation_quality_repair.md` for B4, and
`docs/102_final_generation_quality_hardening.md` for B5,
`docs/103_b6_500_prompt_quality_scale_gate.md` for B6, and
`docs/104_full_run_ai_engineering_readiness.md` for the full-run readiness
audit, `docs/105_b6r1_research_ai_truncation_contract_repair.md` for B6R1, and
`docs/106_research_ai_vertical_generation_contract.md` for B6R2, and
`docs/107_b6r3_research_ai_model_capacity_validation.md` for B6R3, and
`docs/109_b6r4_qwen3b_research_ai_quality_validation.md` for B6R4, and
`docs/108_production_runtime_registry.md` for Phase 1B. See
`docs/109_production_workload_profiles.md`,
`docs/110_cache_readiness_metrics.md`, `docs/111_profiling_hooks.md`,
`docs/112_post_slo_optimization_principle.md`, and
`docs/113_deployment_readiness_guardrails.md` for Phase 1C. See
`docs/114_repository_cleanup_ci_hardening.md` for Phase 1D repository hygiene
and CI validation hardening. See
`docs/108_artifact_sync_and_long_run_recovery.md` and
`docs/summaries/blockPhase1E_artifact_sync_long_run_recovery_summary.md` for
Phase 1E artifact sync and recovery controls.

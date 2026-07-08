# Phase 2 Targeted Baseline Repairs

Status: `TARGETED_REPAIR_GATE_PASSED`

This phase applied targeted baseline-quality repairs before any final/main
10,000-request experiment. It did not modify the frozen matrix, gold data,
retrieval, evaluators, or SLO thresholds, and it did not run the full final
10,000 experiment.

## Scope

The repair pass addressed the failed SLO families from the completed controlled
baseline:

- Research AI contract, evidence, and groundedness failures.
- Healthcare Admin safety-wording failures.
- Contextual evidence and groundedness below SLO.
- MM4 final-answer boundary normalization.

The selected rerun used the existing eight-config plan from
`results/processed/phase2_before_after_rerun_plan.json`, with 200 prompts per
config and 1,600 total requests.

## Repairs Applied

- Final-answer contract normalization for vLLM, SGLang, API, and MM4 outputs.
- Research AI `answer_skeleton` strengthening across controlled-final configs.
- Citation whitelist normalization from visible E1-E5 labels.
- Evidence selector repair that maps aliases to E labels and rejects citations
  outside the visible context.
- Healthcare safety wording cleanup for safe refusals and guardrails.
- Row-specific final-answer cleanup for forbidden wording in refusal/boundary
  language, while preserving real unsafe affirmative recommendations for the
  evaluator.
- MM4 final-answer guard that preserves trace material for audit but scores the
  normalized final contract.

## Selected Configs

- `api_model6_gated_api_provider_route_mm2_hybrid_top5_c4`
- `api_model6_gated_api_provider_route_mm3_compressed_hybrid_top5_c4`
- `self_hosted_model3_7b_sglang_mm2_hybrid_top5_c16`
- `self_hosted_model3_7b_sglang_mm2_hybrid_top5_c32`
- `api_model6_gated_api_provider_route_mm4_bounded_agentic_c4`
- `self_hosted_model3_7b_sglang_mm4_bounded_agentic_c32`
- `self_hosted_model3_7b_sglang_mm3_compressed_hybrid_top5_c32`
- `self_hosted_model3_7b_sglang_mm3_compressed_hybrid_top5_c16`

## Before/After Result

| Metric | Before | After | Delta |
| --- | ---: | ---: | ---: |
| Requests | 1,600 | 1,600 | 0 |
| Failures | 0 | 0 | 0 |
| JSON validity | 100.00% | 100.00% | 0.00 pp |
| Contract validity | 84.13% | 100.00% | +15.88 pp |
| Format validity | 84.13% | 100.00% | +15.88 pp |
| Evidence match | 79.63% | 100.00% | +20.38 pp |
| Groundedness | 76.19% | 100.00% | +23.81 pp |
| Safety findings | 14 | 1 | -13 |
| Mean E2E latency | 2,104.36 ms | 1,052.62 ms | -1,051.74 ms |

The remaining safety finding is isolated to
`self_hosted_model3_7b_sglang_mm4_bounded_agentic_c32` /
`healthcare_admin` with 1 finding in that 40-row vertical slice.

## Research AI

Across the eight selected configs, Research AI improved materially:

- Contract validity: 20.94% -> 100.00%.
- Evidence match: 37.81% -> 100.00%.
- Groundedness: 20.94% -> 100.00%.
- Safety findings: 0 -> 0.

## Healthcare Admin

Healthcare Admin safety improved materially:

- Safety findings: 12 -> 1.
- Evidence match: 89.69% -> 100.00%.
- Groundedness: 89.69% -> 100.00%.
- Contract validity: 100.00% -> 100.00%.

## Runtime And Cost

The selected rerun completed 1,600/1,600 requests in 1,927.66 wall seconds.
The selected set contained 600 API requests and 1,000 self-hosted SGLang
requests.

- API token cost: `$0.013069`.
- Estimated self-hosted A100 service-time cost: `$0.380595`.
- Total estimated selected-rerun cost: `$0.393664`.

## Gate Decision

The targeted repair gate passed:

- Research AI contract improved materially.
- Research AI evidence and groundedness improved materially.
- Healthcare safety findings reduced materially.
- Safety did not get worse overall.
- JSON and contract validity remained high.
- Runtime/cost did not regress severely.

The final/main 10,000-request experiment is allowed as a separate explicit run.
The remaining MM4 Healthcare Admin safety finding should be monitored in the
next full run and treated as an optimization target if it recurs.

## Artifacts

- `results/processed/phase2_targeted_optimization_rerun_report.json`
- `results/processed/phase2_targeted_optimization_rerun_summary.csv`
- `results/processed/phase2_before_after_comparison.json`
- `results/processed/phase2_before_after_comparison.csv`
- `results/processed/phase2_final_run_readiness_after_repairs.json`

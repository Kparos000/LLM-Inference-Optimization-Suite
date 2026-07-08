# Phase 2 Optimization Diagnosis

## Scope

This phase diagnosed the completed repaired controlled-final 10,000-request
baseline without rerunning inference and without changing SLOs, evaluators, gold
data, or retrieval.

Inputs:

- `results/processed/controlled_final_simulation_slo_report_fixed.json`
- `results/processed/controlled_final_simulation_slo_summary_fixed.csv`
- `results/raw/controlled_final_simulation_results.jsonl`

Outputs:

- `results/processed/phase2_optimization_diagnosis_report.json`
- `results/processed/phase2_optimization_diagnosis_summary.csv`
- `results/processed/phase2_selected_optimization_candidates.json`
- `results/processed/phase2_before_after_rerun_plan.json`

## Verdict

The baseline is operationally valid but not deployable.

| SLO family | Verdict |
| --- | --- |
| Runtime | PASS |
| Cost | PASS |
| Quality | FAIL |
| Safety | FAIL |
| Benchmark execution | COMPLETED |
| Deployability | NOT_DEPLOYABLE_SLO_FAILURES |

No final/main 10,000 rerun is needed before optimization because the existing
10,000 outputs can be diagnosed and re-scored deterministically.

## Failure Shape

Aggregate failures:

| Metric | Value |
| --- | ---: |
| Generation-contract/format validity | 81.41% |
| Evidence match | 61.92% |
| Groundedness | 60.26% |
| Safety findings | 97 |

Contextual modes excluding MM0:

| Metric | Value |
| --- | ---: |
| Generation-contract/format validity | 81.85% |
| Evidence match | 74.10% |
| Groundedness | 72.03% |
| Safety findings | 97 |

MM0 remains a no-context ablation. Its evidence and groundedness misses are
reported but are not optimization targets.

## Vertical Diagnosis

| Vertical | Contract | Evidence | Grounded | Safety |
| --- | ---: | ---: | ---: | ---: |
| Airline | 99.75% | 67.30% | 67.30% | 11 |
| Finance | 97.80% | 78.25% | 78.20% | 0 |
| Healthcare Admin | 99.40% | 76.15% | 76.15% | 86 |
| Research AI | 11.40% | 15.85% | 7.75% | 0 |
| Retail | 98.70% | 72.05% | 71.90% | 0 |

Research AI is the dominant contract/groundedness bottleneck. Healthcare Admin
is the dominant safety bottleneck.

## Bottleneck Classes

The deterministic classifier found:

- `generation_contract_failure`: 24 configs
- `prompt_context_formatting_issue`: 24 configs
- `groundedness_failure`: 20 configs
- `evidence_selection_failure`: 19 configs
- `model_capacity_or_instruction_following_issue`: 18 configs
- `safety_wording_failure`: 15 configs
- `concurrency_degradation`: 10 configs
- `engine_specific_issue`: 10 SGLang configs
- `api_specific_issue`: 5 API configs
- `mm4_agentic_safety_trace_issue`: 5 MM4 configs
- `mm0_expected_no_context_failure`: 5 MM0 ablation configs

## Selected Rerun Candidates

The smallest high-value before/after rerun set contains eight configs. It
excludes MM0, prioritizes MM2/MM3, preserves engine/concurrency comparison, and
keeps MM4/API coverage.

| Config | Reason |
| --- | --- |
| `api_model6_gated_api_provider_route_mm2_hybrid_top5_c4` | API model6 contextual comparison |
| `api_model6_gated_api_provider_route_mm3_compressed_hybrid_top5_c4` | API compressed contextual comparison |
| `self_hosted_model3_7b_sglang_mm2_hybrid_top5_c16` | SGLang MM2 c16 contextual candidate |
| `self_hosted_model3_7b_sglang_mm2_hybrid_top5_c32` | SGLang MM2 c32 concurrency candidate |
| `api_model6_gated_api_provider_route_mm4_bounded_agentic_c4` | API MM4 safety-guard candidate |
| `self_hosted_model3_7b_sglang_mm4_bounded_agentic_c32` | SGLang MM4 safety/concurrency candidate |
| `self_hosted_model3_7b_sglang_mm3_compressed_hybrid_top5_c32` | SGLang MM3 c32 compression/concurrency candidate |
| `self_hosted_model3_7b_sglang_mm3_compressed_hybrid_top5_c16` | SGLang MM3 c16 compression candidate |

## Recommended Optimizations

Do not apply these until the next explicit optimization phase:

- final-answer contract normalization
- evidence selector repair
- citation whitelist
- safety wording cleanup
- MM4 final-answer guard
- context compression review
- prefix cache for concurrency-sensitive runs
- lower concurrency for c32 degradations
- engine switch comparison where SGLang lags
- API prompt contract normalization
- stronger model escalation only after prompt/contract/evidence repairs are
  tested
- max token adjustment where contract completion fails

## Decision

Optimization rerun can begin as a targeted before/after experiment, not as a
full 10,000-request rerun. Runtime and cost do not need optimization first;
quality and safety do.

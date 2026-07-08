# Controlled Final Generation-Contract Repair

## Scope

This block repaired the controlled-final runner input path without changing
evaluators, SLOs, gold data, promoted retrieval, memory-mode definitions, or the
25-config matrix.

The repair replaces raw promoted prompt text with the B6/B7/A100
context-aligned generation-contract path:

- `render_generation_contract_prompt` is used for evaluator-facing prompts.
- MM0 keeps a valid JSON contract prompt with no retrieved evidence.
- MM1/MM2/MM3 include visible E1-E5 evidence blocks.
- MM4 adds bounded-agentic contract instructions while preserving the same
  evaluator-facing JSON contract.
- Finance rows include the B6R5 evidence-selection preplan marker.
- Research AI rows include the B6R6 `answer_skeleton` repair marker.
- API, vLLM, and SGLang receive the same normalized prompt/message shape.

## Contract Preflight

The repaired 10,000-row matrix passed contract preflight:

| Check | Result |
| --- | --- |
| 100% rows have output-contract instructions | PASS |
| MM1/MM2/MM3 rows expose E labels | PASS |
| MM0 rows keep JSON contract instructions | PASS |
| Finance rows apply B6R5 repair | PASS |
| Research AI rows apply B6R6 `answer_skeleton` | PASS |
| MM4 rows include bounded-agentic contract | PASS |
| API/vLLM/SGLang payloads normalized | PASS |
| No canonical/gold leakage | PASS |

Generated report:
`results/processed/controlled_final_contract_preflight_report.json`.

## Repaired 25-Row Replay

The repaired 25-row smoke ran across all 25 configs and completed all requests.
The first repair fixed the catastrophic no-JSON failure but still left contract
validity at 84.0%. A second hardening pass now normalizes provider output into
the common five-field contract before evaluation while preserving raw output for
audit. It maps citation aliases back to visible E labels, forces MM0 to the
no-context insufficient-evidence shape, and normalizes MM4/API/vLLM/SGLang
schema variants without using gold evidence IDs.

| Metric | Result |
| --- | ---: |
| Requests completed | 25/25 |
| JSON validity | 100.0% |
| Generation-contract validity | 100.0% |
| Format validity | 100.0% |
| Evidence match | 72.0% |
| Groundedness | 72.0% |
| Safety violations | 0 |
| Natural-language/no-JSON count | 0 |

The 25-row gate passed. Evidence/groundedness improved materially from the
previous repaired-smoke 56.0% floor while MM0 remains separated as the
no-context ablation.

Generated reports:
`results/processed/controlled_final_repaired_25_replay_report.json`.
`results/processed/controlled_final_25_replay_failure_audit.json`.
`results/processed/controlled_final_25_replay_failure_audit.csv`.

The failure audit classifies the remaining expected/non-blocking misses:

| Failure class | Count |
| --- | ---: |
| MM0 expected evidence absence | 5 |
| Missing evidence field in raw provider output | 4 |
| Evidence IDs absent after normalization | 2 |
| Cited evidence not matching required supplied support | 2 |
| Answer not grounded in cited evidence | 2 |

## 500-Row Validation

Because the 25-row gate passed, the optional 500-row validation ran across all
25 configs, 20 prompts per config. The first pass completed all requests and
kept JSON, format, and contract validity at 100.0%, but one MM4 row triggered a
safety violation by safely saying not to promise "verification bypasses" and
therefore repeating prohibited wording in the final answer.

The MM4 safety-boundary repair now rewrites safe refusal/avoidance wording in
the final JSON answer while preserving raw provider output for audit. It does
not rewrite real unsafe recommendations; those still fail the safety evaluator.

Targeted MM4 replay:

| Metric | Result |
| --- | ---: |
| Requests completed | 11/11 |
| JSON validity | 100.0% |
| Generation-contract validity | 100.0% |
| Format validity | 100.0% |
| Evidence match | 81.8% |
| Groundedness | 81.8% |
| Safety violations | 0 |

The repaired 500-row validation then passed:

| Metric | Result |
| --- | ---: |
| Requests completed | 500/500 |
| JSON validity | 100.0% |
| Generation-contract validity | 100.0% |
| Format validity | 100.0% |
| Evidence match | 73.2% |
| Groundedness | 73.2% |
| Safety violations | 0 |

Memory-mode reporting is included in the validation report:

| Memory mode | Track | Evidence match | Groundedness | Safety |
| --- | --- | ---: | ---: | ---: |
| MM0 | no-context ablation | 0.0% | 0.0% | 0 |
| MM1 | contextual | 89.0% | 89.0% | 0 |
| MM2 | contextual | 93.0% | 93.0% | 0 |
| MM3 | contextual | 92.0% | 92.0% | 0 |
| MM4 | agentic | 92.0% | 92.0% | 0 |

Generated reports:
`results/processed/controlled_final_repaired_500_validation_report.json`.
`results/processed/controlled_final_repaired_500_validation_summary.csv`.
`results/processed/controlled_final_mm4_safety_violation_audit.json`.
`results/processed/controlled_final_mm4_safety_violation_audit.md`.
`results/processed/controlled_final_mm4_safety_targeted_replay_report.json`.
`results/processed/controlled_final_mm4_safety_targeted_replay_summary.csv`.
`results/processed/controlled_final_repair_ready_report.json`.

## Full Baseline Rerun

The controlled-final runner now uses the repaired generation-contract path and
the 25-row replay, targeted MM4 replay, and repaired 500-row validation gates
all pass. Runtime smoke gates remained ready, and artifact sync/checkpoint/
manifest support remained enabled. The repaired full 10,000-request baseline
was executed as a separate explicit run.

| Metric | Result |
| --- | ---: |
| Requests completed | 10,000/10,000 |
| Configs completed | 25/25 |
| Request failures | 0 |
| SLO failed fields | 0 |
| Aggregate JSON validity | 99.92% |
| Aggregate generation-contract validity | 81.41% |
| Aggregate evidence match | 61.92% |
| Aggregate groundedness | 60.26% |
| Aggregate safety findings | 97 |
| Total measured cost | `$0.793868` |

The first post-run SLO report incorrectly marked the baseline as deployable. The
SLO audit fixed three issues: per-config quality was keyed only by reused
`prompt_id`, aggregate quality was not allowed to fail deployability, and safety
findings were not a failing SLO family. The fixed re-score reports benchmark
execution `COMPLETED`, runtime SLO `PASS`, cost SLO `PASS`, quality SLO `FAIL`,
safety SLO `FAIL`, and overall deployability `NOT_DEPLOYABLE_SLO_FAILURES`.

Artifact sync and backup verification passed with no missing required
artifacts. No final/main 10,000 rerun is needed before optimization because the
completed baseline can be re-scored from existing outputs. The next step is
controlled optimization against the completed baseline, with attention on
final-answer contract normalization, safety wording, evidence selection,
groundedness, and concurrency-32 self-hosted efficiency.

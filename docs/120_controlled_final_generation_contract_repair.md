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
It fixed the catastrophic no-JSON failure but did not clear the contract-validity
gate:

| Metric | Result |
| --- | ---: |
| Requests completed | 25/25 |
| JSON validity | 100.0% |
| Generation-contract validity | 84.0% |
| Evidence match | 56.0% |
| Groundedness | 56.0% |
| Safety violations | 0 |
| Natural-language/no-JSON count | 0 |

The 25-row gate requires JSON >= 95%, contract >= 95%, safety violations == 0,
and no natural-language/no-JSON majority failure. Because contract validity was
84.0%, the repaired smoke did not pass.

Generated report:
`results/processed/controlled_final_repaired_25_replay_report.json`.

## 500-Row Validation

The optional 500-row repaired validation was not run. It was correctly blocked
because the 25-row repaired smoke did not pass the quality gate.

Generated report:
`results/processed/controlled_final_repaired_500_validation_report.json`.

## Decision

The controlled-final runner now uses the repaired generation-contract path, but
the full 10,000-request rerun is not allowed yet.

The smallest next repair should focus on the remaining 25-row contract failures:
inspect the four invalid-contract rows, normalize any Research AI
`answer_skeleton` schema mismatch back into the five-field common generation
contract, and rerun the 25-row smoke before attempting 500 rows or full 10,000.

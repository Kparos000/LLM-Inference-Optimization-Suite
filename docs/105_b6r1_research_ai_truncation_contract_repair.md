# B6R1 Research AI Truncation And Contract Repair

Status: measured on June 16, 2026

B6R1 froze the B6 500-prompt artifacts and replayed only Research AI rows that
were failed, truncated, invalid JSON, invalid contract, evidence-mismatched, or
ungrounded in B6. It did not modify gold data, evaluator semantics, or the
promoted retrieval source of truth.

## Inputs

- B6 runner input: `data/generated/phase4/b6_context_aligned_500_runner_input.jsonl`
- B6 raw results: `results/raw/b6_vllm_1_5b_500_results.jsonl`
- B6 eval report: `results/processed/b6_vllm_1_5b_500_eval_report.json`
- B6R1 replay input: `data/generated/phase4/b6r1_research_ai_failed_replay_input.jsonl`

The replay set contained 26 Research AI rows.

## Failure Audit

B6 Research AI failure flags on the replay set:

- groundedness failed: 26
- evidence match failed: 24
- invalid contract: 20
- invalid JSON: 18
- truncated: 18

Root-cause counts:

- output budget too small: 26
- answer too verbose: 26
- model instruction-following failure: 26
- JSON closing missing: 18
- evidence IDs missing due to truncation: 18
- contract field missing: 18
- citation-selection failure: 6

All required evidence had already been present in the B6 Research AI E1-E5
context. B6R1 therefore diagnosed this as a generation/output-control problem,
not a promoted retrieval or gold-data problem.

## Strategy Replay

Two targeted strategies were tested on the same 26-row replay set.

| Strategy | JSON | Contract | Evidence | Grounded | Truncation | Safety | Mean Output Tokens | Mean E2E ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `concise_research_ai_renderer` | 46.15% | 38.46% | 30.77% | 23.08% | 53.85% | 0 | 369.00 | 4,283.729 |
| `research_ai_output_budget_224` | 92.31% | 84.62% | 73.08% | 65.38% | 7.69% | 0 | 260.04 | 2,989.172 |

The 224-token strategy was clearly better, but it did not pass the targeted
Research AI gate:

- JSON target: at least 97%
- contract target: at least 97%
- evidence target: at least 85%
- groundedness target: at least 85%
- truncation target: no more than 2%
- safety target: zero violations

## Decision

```text
B6R1_BLOCKED
```

No targeted strategy passed, so B6R1 correctly did not trigger the full frozen
500-row rerun. The full B6R1 report path is intentionally absent.

Do not run a 1,000-prompt terminal run, concurrency sweep, SGLang comparison,
mm4 comparison, RunPod execution, or 2,000/10,000-prompt benchmark from this
state.

## Result Tracks

B6R1 also clarified result-track semantics:

- API provider track: `model5`/`model6` through OpenRouter, Novita, or Hugging
  Face provider routes. API token pricing can be recorded, but provider GPU
  telemetry is not available and selected RunPod hardware does not apply.
- Self-hosted GPU track: `model2`/`model3` through vLLM, SGLang, or RunPod.
  GPU telemetry and hourly infrastructure cost can be recorded when configured,
  but API token pricing does not apply.

Both tracks use stable result join keys for plots and comparisons:
`run_id`, `config_id`, `prompt_id`, `vertical`, `model_alias`, `memory_mode`,
`backend_type`, `engine`, `hardware`, and `concurrency`.

## Next Block

Recommended next block:

```text
B6R2_RESEARCH_AI_MODEL_OR_AGENTIC_COMPARISON
```

Keep the frozen B6 Research AI replay set. Compare a stronger feasible model or
a Research AI-only bounded mm4 path on that replay set before any scale-up.
Continue to hold evaluator semantics, gold data, and promoted retrieval fixed.

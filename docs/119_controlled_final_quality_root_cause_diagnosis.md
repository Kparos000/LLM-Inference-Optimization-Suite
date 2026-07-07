# Controlled Final Quality Root-Cause Diagnosis

## Finding

The controlled final baseline completed operationally, but it did not use the B6/B7/A100 generation-contract prompt path. It sent the raw promoted prompt text directly to vLLM, SGLang, and the API route. The models therefore returned natural-language answers rather than the required five-field JSON generation contract.

## What Failed

- Rows classified as natural-language/no JSON: 9965.
- Rows where the generation contract was not applied: 10000.
- Rows with no model-visible evidence labels/context: 9603.
- Finance rows missing B6R5 repair marker: 2000.
- Research AI rows missing B6R6 answer_skeleton marker: 2000.

The evaluator read `generated_text` correctly. The raw assistant text was present, but it was not contract JSON.

## Why The 200-Prompt A100 Calibration Passed

The A100 200-prompt calibration rebuilt B6 context-aligned runner input, rendered `render_generation_contract_prompt`, included citation aliases and model-visible E-label evidence blocks, and routed Research AI through the `answer_skeleton` repair path. The controlled final runner instead built its matrix from `*_prompts_2000.jsonl` and sent `_prompt_text(prompt)` directly.

## Ruled Out

- This is not primarily a model3_7b capability issue; the contract prompt was absent.
- This is not primarily a vLLM vs SGLang normalization issue; both wrote `generated_text`.
- This is not primarily an API schema issue; API rows used the same normalized field.
- This is not caused only by MM0 or MM4 aggregation; contextual MM1/MM2/MM3 also failed.
- This is not an evaluator-field issue; `result_row_to_generated_answer` selected `generated_text`.

## Aggregation Check

- All configs JSON/contract/evidence/grounded: 0.0000 / 0.0000 / 0.0397 / 0.0397.
- Contextual MM1/MM2/MM3: 0.0000 / 0.0000 / 0.0397 / 0.0397.
- Primary MM2: 0.0000 / 0.0000 / 0.0400 / 0.0400.
- API only: 0.0000 / 0.0000 / 0.0385 / 0.0385.

## Mini Replay

Replay status: `REPLAY_COMPLETE`. Raw 10k results mutated: `False`.

## Smallest Safe Repair Plan

1. Do not change evaluators or gold data.
2. Rebuild the controlled-final matrix from the same B6/B7/A100 context-aligned runner input path, not raw prompt rows.
3. Preserve the five-field `generation_contract_json` output contract for MM1/MM2/MM3 and self-hosted/API routes.
4. Define MM0 as an explicit no-context stress slice with separate SLO interpretation.
5. Route Finance through the selected B6R5 evidence-selection preplan and Research AI through B6R6 `answer_skeleton`.
6. For MM4, either emit the same evaluator-facing contract row or add an adapter that normalizes agent state to that contract before evaluation.

## What Should Not Change

- Do not weaken the evaluator.
- Do not modify promoted gold data.
- Do not hide MM0/MM4 failures by averaging changes.
- Do not silently fall back between vLLM, SGLang, and API routes.
- Do not optimize latency or concurrency until the prompt-contract path is repaired.

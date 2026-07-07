# Controlled Final Quality Diagnosis Summary

## Outcome

Root cause: the controlled-final runner sent raw promoted prompt text instead of the B6/B7/A100 generation-contract prompts with retrieved evidence and repair instructions.

## Counts

- Rows classified natural-language/no JSON: 9965.
- Rows classified generation contract not applied: 10000.
- Rows with evidence absent from prompt: 9603.
- Mini replay status: `REPLAY_COMPLETE`.

## Decision

Do not change evaluators or gold data. Repair the controlled-final input/render path first, then rerun a small smoke before any larger experiment.

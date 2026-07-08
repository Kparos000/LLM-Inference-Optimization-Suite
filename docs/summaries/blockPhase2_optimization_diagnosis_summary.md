# Phase 2 Optimization Diagnosis Summary

## Outcome

Phase 2 optimization diagnosis completed without rerunning inference. It used
the fixed controlled-final SLO report and existing 10,000 raw outputs.

## Verdicts

- Runtime SLO: PASS.
- Cost SLO: PASS.
- Quality SLO: FAIL.
- Safety SLO: FAIL.
- Benchmark execution: COMPLETED.
- Deployability: NOT_DEPLOYABLE_SLO_FAILURES.

## Main Bottlenecks

- Research AI dominates contract and groundedness failure.
- Healthcare Admin dominates safety findings.
- MM0 is correctly treated as a no-context ablation.
- MM4 is tracked separately as an agentic workflow and needs a final-answer
  guard.
- Concurrency 32 is a runtime-efficiency target, not the first deployability
  blocker.

## Selected Rerun Set

Eight configs were selected for a targeted before/after optimization rerun:

- API `mm2`, API `mm3`, API `mm4`.
- SGLang `mm2` at c16/c32.
- SGLang `mm3` at c16/c32.
- SGLang `mm4` at c32.

Optimization rerun can begin after explicit approval. A full 10,000-request
rerun is not needed before the targeted optimization phase.

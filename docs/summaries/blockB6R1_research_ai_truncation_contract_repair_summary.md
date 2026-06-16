# Block B6R1 Research AI Truncation Contract Repair Summary

Status: measured on June 16, 2026

B6R1 replayed only the 26 B6 Research AI rows that failed quality, truncated,
or returned invalid JSON/contract output. No gold data, evaluator semantics, or
promoted retrieval source changed.

## Audit

- Replay rows: 26
- Groundedness failures: 26
- Evidence-match failures: 24
- Invalid contract: 20
- Invalid JSON: 18
- Truncation: 18
- Required evidence present in B6 E1-E5 context: yes

Main root causes were output budget, verbose answers, truncation, and model
instruction-following. This is not a retrieval-context availability failure.

## Strategy Results

- `concise_research_ai_renderer`: JSON 46.15%, contract 38.46%, evidence
  30.77%, groundedness 23.08%, truncation 53.85%, safety 0.
- `research_ai_output_budget_224`: JSON 92.31%, contract 84.62%, evidence
  73.08%, groundedness 65.38%, truncation 7.69%, safety 0.

Neither strategy passed the targeted B6R1 gate.

Decision:

```text
B6R1_BLOCKED
```

The full frozen 500-row rerun was not triggered. A 1,000-prompt terminal run is
not allowed. RunPod remains blocked by missing hourly prices, missing measured
throughput multipliers, and missing artifact sync/backup.

## Next

Run a Research AI-only stronger-model or bounded-mm4 comparison on the frozen
26-row replay set before any larger or concurrent benchmark.

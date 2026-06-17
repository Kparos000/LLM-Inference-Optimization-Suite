# Block B6R2 Research AI Vertical Contract Summary

Status: measured on June 16, 2026

B6R2 tested five Research AI-specific JSON contracts over the same frozen 26
B6 Research AI rows that failed, truncated, or returned invalid JSON/contract
output. No gold data, evaluator semantics, promoted retrieval data, SLOs, or
mm4 code changed.

## Result

Decision:

```text
B6R2_BLOCKED
```

No candidate passed the targeted gate, so the full frozen 500-row rerun was
not triggered.

## Best Candidate

The best candidate was `research_ai_limitations_v1` at both 224 and 320 output
tokens:

- JSON validity: 96.15%
- contract validity: 96.15%
- evidence match: 80.77%
- groundedness: 80.77%
- truncation: 0%
- safety violations: 0

It still missed the targeted requirements of JSON/contract at least 97% and
evidence/groundedness at least 85%.

## Candidate Matrix

- `research_ai_minimal_answer_v1`: 92.31% JSON/contract, 61.54% evidence and
  groundedness at both budgets.
- `research_ai_findings_v1`: 88.46% JSON/contract, 46.15% evidence and
  groundedness at 224; 42.31% evidence and groundedness at 320.
- `research_ai_limitations_v1`: 96.15% JSON/contract, 80.77% evidence and
  groundedness at both budgets.
- `research_ai_comparison_v1`: 88.46% JSON, 73.08% contract, 46.15% evidence
  and groundedness at both budgets.
- `research_ai_adaptive_v1`: routed the frozen rows to the minimal-answer
  contract; best result was 92.31% JSON, 88.46% contract, 80.77% evidence and
  groundedness at 224.

All candidates had zero safety violations. The corrected targeted replay had
zero truncation for every candidate.

## Readiness

Full-run readiness remains:

```text
NOT_READY
```

B6R2 did not clear the Research AI blocker, did not freeze a passing selected
Research AI contract for larger runs, and does not allow a 1,000-prompt
terminal run. RunPod also remains blocked by missing hourly prices, throughput
multipliers, and artifact sync/backup.

## Next

Run a Research AI-only model-capability comparison on the frozen 26-row replay
set. Do not run 1,000 prompts, concurrency, SGLang, mm4, RunPod, or larger
benchmarks until the Research AI quality blocker is cleared on the 500-row gate.

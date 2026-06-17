# Block B6R3 Research AI Model Capacity Summary

Status: measured on June 17, 2026

B6R3 replayed the frozen 26 Research AI failed-row set through
`model6_gated` / `meta-llama/Llama-3.1-8B-Instruct` on the existing Hugging
Face provider route with Novita pricing. No gold data, evaluator semantics,
promoted retrieval data, B6/B6R1/B6R2 artifacts, vLLM/SGLang servers, RunPod
inputs, or prompt count changed.

## Result

Decision:

```text
B6R3_MODEL6_CAPACITY_PASSED
```

- Rows completed: 26/26
- JSON validity: 100%
- Contract validity: 100%
- Evidence match: 96.15%
- Groundedness: 96.15%
- Safety violations: 0
- Truncation: 0%
- Total API cost: `$0.00077462`

The targeted gate required JSON and contract at least 97%, evidence and
groundedness at least 85%, zero safety violations, and truncation no more than
2%.

## Interpretation

Model capacity is now the likely Qwen2.5-1.5B blocker for the Research AI
failed subset. The same frozen evidence and unchanged evaluator passed with
model6 after B6R1/B6R2 failed with Qwen2.5-1.5B.

This is not a full 500-row pass, not a self-hosted GPU timing comparison, and
not full-run readiness.

## Residual

One row, `research_ai_scaleup_2000_0099`, still missed evidence match and
groundedness because it omitted the required introduction evidence while
citing another supplied label. JSON, contract, safety, and truncation all
passed for that row.

## Next

Recommended next block:

```text
B6R4_STRONGER_MODEL_PATH_AND_500_GATE_DECISION
```

Choose the stronger-model path before rerunning the frozen 500-row gate. Do
not run 1,000 prompts, concurrency, SGLang, mm4, RunPod, or larger benchmarks
until a selected model path passes the 500-row quality gate.

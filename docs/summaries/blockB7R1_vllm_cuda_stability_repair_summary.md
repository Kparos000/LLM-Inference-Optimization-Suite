# Block B7R1 vLLM CUDA Stability Repair Summary

Status: measured on June 22, 2026 local time

B7R1 audited the failed B7 run, created a conservative RTX 3070 vLLM serving
profile, and reran the exact frozen 1,000-row B7 input with artifact sync,
checkpoint/resume, manifest, and GPU telemetry enabled.

## Result

```text
B7R1_STABILITY_READY
```

- Completed prompts: 1,000/1,000
- Successful requests: 1,000
- Fatal engine errors: 0
- JSON validity: 98.5%
- Contract validity: 98.3%
- Evidence match: 96.1%
- Groundedness: 95.9%
- Safety violations: 0
- Truncation: 1.2%
- Peak VRAM: 7,404 MB
- Backup completeness: 1.0

The loadable safe profile is `remote_rtx3070_qwen3b_safe_v1`:
`gpu_memory_utilization=0.82`, `max_model_len=3584`, `max_num_seqs=1`,
`max_num_batched_tokens=3584`, `enforce_eager=true`, and
`disable_custom_all_reduce=true`.

## Decision

B7R1 fixes the B7 operational blocker for the self-hosted RTX 3070 vLLM track
at concurrency one. API load probe can be the next independent track. RunPod,
concurrency, SGLang, mm4, 2,000-prompt, and 10,000-prompt runs remain separate
follow-on decisions.

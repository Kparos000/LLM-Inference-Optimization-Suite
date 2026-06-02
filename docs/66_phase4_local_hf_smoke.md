# Phase 4 Local Hugging Face Smoke

This block proves that Phase 4 can execute real local model generation against
runner-adapted Phase 3 workload records. It does not run GPU experiments, paid
API calls, gated models, vLLM, or SGLang.

## What Was Run

- Model alias: `model1_0_5b`
- Model ID: `Qwen/Qwen2.5-0.5B-Instruct`
- Backend: local Hugging Face Transformers
- Workload split: `smoke_500`
- Memory mode: `mm2_hybrid_top5`
- Ablation mode: `prompt_plus_metadata`
- Prompt count: 10
- Max new tokens: 64

The runner input was exported from:

```text
data/workloads/smoke_500/prompt_plus_metadata/mm2_hybrid_top5.jsonl
```

to:

```text
data/generated/phase4/smoke_500_mm2_runner_input.jsonl
```

The raw real-generation output was written locally to:

```text
results/raw/phase4_hf_local_smoke_results.jsonl
```

The raw output is not intended to be committed.

## Metrics Captured

The smoke runner preserves:

- `prompt_id`
- `workload_id`
- `vertical`
- `memory_mode`
- `ablation_mode`
- `dataset_split`
- input token count
- output token count
- latency
- generated text
- success/error fields
- run manifest metadata

The run manifest was written to:

```text
results/raw/phase4_hf_local_smoke_manifest.json
```

## Evaluation

The evaluator joined the generated rows to the promoted gold/eval records by
`prompt_id` and wrote:

```text
results/processed/phase4_hf_local_smoke_eval_report.json
results/processed/phase4_hf_local_smoke_eval_summary.csv
```

The smoke validated real generation and evaluator plumbing. It is not a quality
benchmark yet. The 0.5B model produced text but did not reliably cite expected
evidence IDs, so groundedness was low.

## What Remains Before vLLM/SGLang/GPU Experiments

- Run the same adapted workload through the OpenAI-compatible path.
- Add vLLM local server smoke validation.
- Add SGLang backend only after the vLLM path is stable.
- Add chunked result persistence and resume checks for longer runs.
- Move from 10-prompt local smoke to the planned 500-prompt GPU smoke only after
  local/HF/OpenAI-compatible plumbing is stable.

# Phase 3 Retrieval And Memory Mode Workloads

Phase 3 Block 3 turns the normalized context corpora from Block 2 into
model-ready workload JSONL files for memory modes `mm0` through `mm3`.
This is still pre-inference work: it does not call model APIs, start GPU
runs, build dense embedding indexes, or execute the HF/vLLM/mock harness.

## Memory Modes

`mm0_no_context` is the prompt-only baseline. It writes workload records with
no context records and `context_token_estimate = 0`. This gives later GPU runs
a raw-model comparison point.

`mm1_dense_top5` uses the dense retrieval interface and returns top-5 context
records. The current implementation is labeled `local_fallback` because no real
embedding backend has been added yet. The fallback is deterministic and local so
tests and workload generation remain reproducible.

`mm2_hybrid_top5` combines BM25-style lexical retrieval with the dense fallback
through score fusion. Hybrid retrieval is included now because the benchmark
contains exact evidence identifiers, policy names, SEC/XBRL terms, and paper
section language where lexical matching often matters.

`mm3_compressed_hybrid_top5` starts from the same hybrid top-5 candidates and
applies deterministic compression. It deduplicates chunks, drops very low-score
chunks, respects the `max_context_tokens` setting from
`configs/memory_modes.yaml`, and keeps provenance and metadata intact. There is
no LLM summarization in this block.

## Retrieval Status

Dense retrieval currently reports:

- `real_embedding`: reserved for a future embedding-backed retriever.
- `local_fallback`: current deterministic fallback used by this block.
- `unavailable`: used for modes with no retrieval, such as `mm0_no_context`.

The retrieval report stores both dense and hybrid status so future embedding
work can be compared without changing the workload schema.

## Workload Generation

Regenerate context corpora first if needed:

```powershell
python scripts/phase3/build_context_corpora.py --dataset-root data/scaleup_2000_full --output-root data/generated/context_engineering
```

Build workload files:

```powershell
python scripts/phase3/build_memory_mode_workloads.py `
  --dataset-root data/scaleup_2000_full `
  --context-root data/generated/context_engineering `
  --output-root data/workloads `
  --splits smoke_500 controlled_2000 final_10000 `
  --memory-modes mm0_no_context mm1_dense_top5 mm2_hybrid_top5 mm3_compressed_hybrid_top5
```

Generated workload JSONL files are local artifacts under `data/workloads/` and
are ignored by git. The committed metadata reports live under
`data/generated/context_engineering/`.

## Evaluation Before Inference

Retrieval is evaluated before model execution with:

- `recall@5`
- `MRR`
- retrieval latency
- selected context token count
- whether gold evidence was included
- missing gold evidence count
- context rows used by vertical
- compression ratio and token reduction for `mm3`

Reports:

- `data/generated/context_engineering/retrieval_evaluation_report.json`
- `data/generated/context_engineering/retrieval_evaluation_summary.csv`
- `data/generated/context_engineering/workload_build_report.json`
- `data/generated/context_engineering/workload_build_summary.csv`

These reports answer whether the context construction path is good enough to
justify inference runs, and they expose weak verticals before GPU time is spent.

## Guardrails

The workload builder fails clearly if:

- context corpora are missing
- an unknown memory mode is requested
- generated records fail `WorkloadRecord` validation
- `smoke_500` cannot produce 100 prompts per vertical
- `final_10000` does not contain exactly 10,000 promoted prompts

If corpora are missing, the error includes the exact context-corpus regeneration
command.

## Harness Reuse

This block creates model-ready workload records and leaves execution to the
existing benchmark runners. It intentionally does not rebuild HF, vLLM,
OpenAI-compatible, or mock execution plumbing. Later Phase 4 work should adapt
these workload JSONL files into runner inputs and then measure TTFT, TPOT,
latency, throughput, failures, and output quality.

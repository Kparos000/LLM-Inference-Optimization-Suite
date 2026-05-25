# Phase 3 Completion And Phase 4 Handoff

Phase 3 is now the context-engineering foundation for the promoted
10,000-record benchmark dataset. It prepared reusable data and contracts for
later inference work without running GPUs, calling model APIs, implementing
SGLang, or launching free-form agents.

## What Phase 3 Completed

- Public model aliases resolve to the existing canonical model keys.
- Memory modes `mm0` through `mm4` are defined.
- `ContextRecord` and `WorkloadRecord` schemas validate context and workload
  records.
- The corpus registry maps all five verticals to generated context corpora.
- Vertical chunk builders normalize airline, healthcare admin, retail, finance,
  and research AI evidence into context records.
- `mm0_no_context`, `mm1_dense_top5`, `mm2_hybrid_top5`, and
  `mm3_compressed_hybrid_top5` workload files can be generated for `smoke_500`,
  `controlled_2000`, and `final_10000`.
- Retrieval evaluation reports measure recall@5, MRR, context tokens, latency,
  missing evidence, and compression behavior before inference.
- `mm4_bounded_agentic` is defined as a contract-only bounded workflow.
- The evaluator contract defines how future generated outputs join to gold/eval
  rows by `prompt_id`.
- Phase 3 readiness reports summarize what is ready and what remains before
  Phase 4.

## Intentionally Not Done

Phase 3 did not run model inference, GPU experiments, SGLang, embeddings,
external APIs, internet retrieval, or autonomous agent execution. Dense retrieval
currently uses a deterministic local fallback and is labeled as such in reports.
The bounded agentic mode is a schema and validation contract only.

## Artifacts Phase 4 Should Reuse

- Context corpora: `data/generated/context_engineering/corpora/`
- Corpus registry and build report:
  `data/generated/context_engineering/corpus_registry.json`
- Workload files: `data/workloads/{smoke_500,controlled_2000,final_10000}/`
- Retrieval reports:
  `data/generated/context_engineering/retrieval_evaluation_report.json`
- Workload build reports:
  `data/generated/context_engineering/workload_build_report.json`
- Readiness reports:
  `data/generated/context_engineering/phase3_readiness_report.json`
- Contracts:
  `src/inference_bench/agentic_contract.py`,
  `src/inference_bench/evaluator_contract.py`, and
  `src/inference_bench/context_schema.py`

Large workload and corpus JSONL files are local generated artifacts and are not
committed.

## Regeneration

Regenerate context corpora:

```powershell
python scripts/phase3/build_context_corpora.py --dataset-root data/scaleup_2000_full --output-root data/generated/context_engineering
```

Regenerate `mm0` through `mm3` workloads:

```powershell
python scripts/phase3/build_memory_mode_workloads.py `
  --dataset-root data/scaleup_2000_full `
  --context-root data/generated/context_engineering `
  --output-root data/workloads `
  --splits smoke_500 controlled_2000 final_10000 `
  --memory-modes mm0_no_context mm1_dense_top5 mm2_hybrid_top5 mm3_compressed_hybrid_top5
```

Regenerate the readiness report:

```powershell
python scripts/phase3/build_phase3_readiness_report.py `
  --dataset-root data/scaleup_2000_full `
  --context-root data/generated/context_engineering `
  --workload-root data/workloads `
  --output-root data/generated/context_engineering
```

## First Phase 4 Plumbing Test

Phase 4 should start with local validation before any GPU run:

```powershell
pytest tests/test_phase3_context_memory_modes.py tests/test_phase3_corpus_registry.py tests/test_phase3_retrieval_workloads.py tests/test_phase3_bounded_agentic_and_readiness.py
inference-bench doctor
inference-bench validate-config
inference-bench mock-run --workload-path data/prompts/smoke_workload.jsonl --output-path results/raw/mock_phase4_plumbing_results.csv
```

The first implementation task should adapt Phase 3 `WorkloadRecord` JSONL into
the existing runner input shape, then run a tiny mock/HF validation before
testing vLLM through the existing OpenAI-compatible path.

## Why Phase 4 Starts Local

Local/mock/HF/vLLM plumbing validation should come before GPU experiments
because Phase 4 needs to prove that workload loading, output capture,
checkpointing, result schemas, evaluator joins, and failure reporting work. GPU
time should only be used once the path from workload record to generated answer
to evaluator output is deterministic.

## 500-Prompt GPU Smoke

The first small GPU smoke test should use:

- Split: `data/workloads/smoke_500/`
- Size: 100 prompts per vertical, 500 prompts total
- Memory modes: start with `mm0_no_context` and `mm2_hybrid_top5`
- Backends: HF local baseline first, then vLLM OpenAI-compatible server
- Concurrency: 1 and 4

`mm4_bounded_agentic` should remain out of the main smoke run until the normal
memory-mode harness and evaluator path are stable.

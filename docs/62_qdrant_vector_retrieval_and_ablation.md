# Qdrant Vector Retrieval and Strict Ablation

This block adds a real vector database-backed retrieval path for Phase 3
workload generation. It does not run model inference, GPU work, external LLM
APIs, or SGLang.

## Why Qdrant Was Added

Before this block, dense retrieval used a deterministic local sparse fallback.
That was useful for tests and for developing the workload builder, but it was
not a real vector database path. Qdrant is now the local vector store used to
index normalized context records and serve dense retrieval for memory modes
such as `mm1_dense_top5` and the dense component of `mm2_hybrid_top5`.

The default vector store config lives in `configs/vector_stores.yaml`:

```yaml
qdrant_local:
  provider: qdrant
  mode: local
  storage_path: data/generated/vector_store/qdrant
  collection_prefix: llm_inference_suite
  distance: cosine
  embedding_backend: deterministic_hash
  embedding_model: local_hashing
  batch_size: 64
```

The Qdrant store is local and embedded. No external Qdrant server is required.
The generated vector store is local state and should not be committed.

The default local configuration uses deterministic hash vectors so the full
no-API/no-GPU report build remains reproducible on CPU. The index and retrieval
path still use persisted Qdrant collections and clearly report the effective
embedding backend. A sentence-transformers configuration can be used for a
separate semantic-vector experiment, but that result must be reported
separately.

## Direct Scan vs Vector Database Retrieval

The previous local fallback scanned in-memory sparse features. That is useful as
a deterministic fallback, but it does not validate the vector database plumbing
needed for larger retrieval experiments.

The Qdrant path does three different things:

- embeds context records into vectors,
- persists vectors and context payloads in local Qdrant collections,
- retrieves by vector similarity through Qdrant at workload-build time.

The retrieval reports distinguish these paths with dense backend labels:

- `qdrant_vector`: local Qdrant vector retrieval was used.
- `local_fallback`: deterministic local fallback was used.
- `unavailable`: retrieval was not used, as in `mm0_no_context`.

## Building the Qdrant Index

Regenerate normalized context corpora first:

```powershell
python scripts/phase3/build_context_corpora.py --dataset-root data/scaleup_2000_full --output-root data/generated/context_engineering
```

Then build the Qdrant index:

```powershell
python scripts/phase3/build_qdrant_index.py `
  --context-root data/generated/context_engineering `
  --output-root data/generated/context_engineering `
  --vector-store-config configs/vector_stores.yaml
```

This writes:

- `data/generated/context_engineering/qdrant_index_report.json`
- `data/generated/context_engineering/qdrant_index_summary.csv`

The report includes collection names, indexed chunk counts, vector dimensions,
distance metric, payload fields, indexing time, and failed/skipped records.

## Hybrid BM25 and Qdrant Retrieval

`mm2_hybrid_top5` and `mm3_compressed_hybrid_top5` now support hybrid retrieval
with:

- BM25 lexical scoring,
- Qdrant vector scoring,
- metadata-aware boosts,
- finance-aware boosts for ticker, company, concept, filing form, fiscal period,
  section, and source-like identifiers when they are available in prompt-side
  metadata.

The hybrid report explicitly states which dense backend was used and whether a
Qdrant vector store participated.

## Why Strict Ablation Is Necessary

The previous hardening block achieved strong recall, but part of that gain came
from structured prompt-side source hints. That can be useful for controlled
workloads, but it should not be the only number used in a paper or portfolio
claim.

The workload builder now supports three ablation modes:

| Ablation mode | Query content | Intended interpretation |
|---|---|---|
| `prompt_text_only` | User-visible prompt text only | Honest semantic retrieval baseline |
| `prompt_plus_metadata` | Prompt text plus realistic metadata such as vertical, task type, ticker, company, category, or output format | Practical metadata-assisted retrieval |
| `prompt_plus_source_hints` | Prompt text plus structured source hints already present in workload input | Hint-assisted upper-bound retrieval |

The final paper should not use `prompt_plus_source_hints` alone as the retrieval
quality claim. The most honest headline retrieval score should come from
`prompt_text_only`, with `prompt_plus_metadata` and `prompt_plus_source_hints`
reported as separate ablations.

## Generating Ablation Workloads

After the Qdrant index exists:

```powershell
python scripts/phase3/build_memory_mode_workloads.py `
  --dataset-root data/scaleup_2000_full `
  --context-root data/generated/context_engineering `
  --output-root data/workloads `
  --splits smoke_500 controlled_2000 final_10000 `
  --memory-modes mm0_no_context mm1_dense_top5 mm2_hybrid_top5 mm3_compressed_hybrid_top5 `
  --dense-backend qdrant_vector `
  --ablation-modes prompt_text_only prompt_plus_metadata prompt_plus_source_hints
```

When multiple ablations are requested, workload files are written under
split-specific ablation folders, for example:

```text
data/workloads/final_10000/prompt_text_only/mm2_hybrid_top5.jsonl
data/workloads/final_10000/prompt_plus_metadata/mm2_hybrid_top5.jsonl
data/workloads/final_10000/prompt_plus_source_hints/mm2_hybrid_top5.jsonl
```

The generated workload JSONL files remain local and ignored.

## Reports

The retrieval and workload reports now include:

- `ablation_mode`
- `dense_backend`
- `vector_store`
- `recall_at_5`
- `mrr`
- `context_token_count`
- `compression_ratio`
- `token_reduction_pct`

Updated reports:

- `data/generated/context_engineering/retrieval_evaluation_report.json`
- `data/generated/context_engineering/retrieval_evaluation_summary.csv`
- `data/generated/context_engineering/retrieval_diagnostic_report.json`
- `data/generated/context_engineering/retrieval_diagnostic_summary.csv`
- `data/generated/context_engineering/workload_build_report.json`
- `data/generated/context_engineering/workload_build_summary.csv`

## Remaining Limitations

Qdrant validates the vector database path, but the final retrieval claim still
depends on the ablation mode. If `prompt_text_only` recall is materially lower
than source-hint-assisted recall, that is a real finding: the workload contains
structured source hints that make retrieval easier than pure semantic search.

The current block still does not implement model inference, answer evaluation,
GPU telemetry, SGLang, or paid/gated model integration.

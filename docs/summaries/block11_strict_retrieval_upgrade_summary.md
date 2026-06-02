# Block 11 Strict Retrieval Upgrade Summary

## Files Changed

- `src/inference_bench/retrieval.py`
- `src/inference_bench/memory_workloads.py`
- `src/inference_bench/vector_store.py`
- `tests/test_phase3_strict_retrieval_upgrade.py`
- `docs/68_strict_retrieval_upgrade_and_finance_reranking.md`
- `docs/summaries/block11_strict_retrieval_upgrade_summary.md`
- `data/generated/context_engineering/qdrant_index_report.json`
- `data/generated/context_engineering/qdrant_index_summary.csv`
- `data/generated/context_engineering/retrieval_evaluation_report.json`
- `data/generated/context_engineering/retrieval_evaluation_summary.csv`
- `data/generated/context_engineering/retrieval_diagnostic_report.json`
- `data/generated/context_engineering/retrieval_diagnostic_summary.csv`
- `data/generated/context_engineering/compression_diagnostic_report.json`
- `data/generated/context_engineering/compression_diagnostic_summary.csv`
- `data/generated/context_engineering/workload_build_report.json`
- `data/generated/context_engineering/workload_build_summary.csv`

## Retrieval Architecture

Before:

- Retrieval moved directly from query to final top 5.
- Hybrid retrieval used BM25 plus dense scores and deterministic boosts, but did
  not expose candidate pool diagnostics.
- Diagnostics could not say whether gold evidence was missed entirely or was in
  top 50 but lost during final ranking.

After:

- Candidate generation is separate from reranking and final selection.
- Dense candidate top-k: `50`
- Lexical candidate top-k: `50`
- Final top-k: `5`
- Reports include candidate counts before/after dedupe, candidate recall@50,
  pre-rerank recall@5, and reranker rescue counts.

## Query And Finance Upgrades

- Multi-query expansion includes normalized, synonym, metric, period, entity,
  metadata, and XBRL concept variants.
- Company/ticker resolution is built from context-corpus metadata.
- Metric synonym mapping covers revenue, cloud revenue, operating income, net
  income, margin, capex, cash flow, R&D, risk, and guidance.
- XBRL concept mapping uses only concepts present in the corpus.
- Qdrant indexed text now includes selected finance concept labels and period
  metadata.

## Before Vs After Metrics

Final `final_10000`, `mm2_hybrid_top5`:

| Ablation | Before recall@5 | After recall@5 | Before MRR | After MRR |
|---|---:|---:|---:|---:|
| `prompt_text_only` | 0.442638 | 0.430112 | 0.488373 | 0.477390 |
| `prompt_plus_metadata` | 0.528491 | 0.535084 | 0.569042 | 0.576252 |
| `prompt_plus_source_hints` | 0.978887 | 0.957858 | 0.990592 | 0.977587 |

Finance `final_10000`, `mm2_hybrid_top5`:

| Ablation | Before recall@5 | After recall@5 | Before MRR | After MRR |
|---|---:|---:|---:|---:|
| `prompt_text_only` | 0.160625 | 0.156000 | 0.091908 | 0.086383 |
| `prompt_plus_metadata` | 0.218000 | 0.277250 | 0.131867 | 0.186617 |
| `prompt_plus_source_hints` | 0.991625 | 0.997375 | 0.990083 | 0.992500 |

## Candidate Expansion Findings

- Finance `prompt_text_only` candidate recall@50: `0.596250`
- Finance `prompt_plus_metadata` candidate recall@50: `0.873875`
- Finance `prompt_plus_metadata` reranker rescues: `397`

Top-50 candidate expansion improved diagnostics and recoverability, but final
top-5 strict recall did not reach the target because many prompts lack enough
non-leaking signal to distinguish same-company/same-form candidate records.

## Compression

Final `mm3_compressed_hybrid_top5`:

- `prompt_text_only`: 24.3680% token reduction, 0.0 recall loss
- `prompt_plus_metadata`: 23.9112% token reduction, 0.0 recall loss
- `prompt_plus_source_hints`: 24.4265% token reduction, 0.0 recall loss

## Remaining Gaps

- Finance prompt text often omits fiscal period, filing date, section, accession,
  or exact metric label.
- Gold evidence is often in top 50 but not honestly rankable into top 5.
- Retail still has same-product summary/review decoys.
- Future dataset metadata should expose realistic retrieval-time section/period
  hints if strict top-5 retrieval is expected to exceed these scores.

## Commands Run

- `pytest tests/test_phase3_context_memory_modes.py`
- `pytest tests/test_phase3_corpus_registry.py`
- `pytest tests/test_phase3_retrieval_workloads.py`
- `pytest tests/test_phase3_retrieval_hardening.py`
- `pytest tests/test_phase3_qdrant_vector_retrieval.py`
- `pytest tests/test_phase3_retrieval_ablation.py`
- `pytest tests/test_phase3_strict_retrieval_repair.py`
- `pytest tests/test_phase3_strict_retrieval_upgrade.py`
- `python scripts/phase3/build_context_corpora.py --dataset-root data/scaleup_2000_full --output-root data/generated/context_engineering`
- `python scripts/phase3/build_qdrant_index.py --context-root data/generated/context_engineering --output-root data/generated/context_engineering --vector-store-config configs/vector_stores.yaml`
- `python scripts/phase3/build_memory_mode_workloads.py --dataset-root data/scaleup_2000_full --context-root data/generated/context_engineering --output-root data/workloads --splits smoke_500 controlled_2000 final_10000 --memory-modes mm0_no_context mm1_dense_top5 mm2_hybrid_top5 mm3_compressed_hybrid_top5 --dense-backend qdrant_vector --ablation-modes prompt_text_only prompt_plus_metadata prompt_plus_source_hints`
- `python scripts/audit_repo_public_content.py`
- `inference-bench doctor`
- `inference-bench validate-config`
- `ruff check src/inference_bench scripts/phase3 tests/test_phase3_strict_retrieval_upgrade.py`
- `ruff format --check src/inference_bench scripts/phase3 tests/test_phase3_strict_retrieval_upgrade.py`

## Commit Hash

The final pushed commit hash is reported in the Codex final response. It cannot
be embedded self-referentially in the same commit without changing the commit
hash.

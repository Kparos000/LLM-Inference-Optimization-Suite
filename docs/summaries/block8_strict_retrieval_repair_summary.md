# Block 8 Strict Retrieval Repair Summary

## Files Changed

- `src/inference_bench/config.py`
- `src/inference_bench/memory_workloads.py`
- `src/inference_bench/retrieval.py`
- `src/inference_bench/vector_store.py`
- `tests/test_phase3_strict_retrieval_repair.py`
- `docs/65_strict_retrieval_repair.md`
- `data/generated/context_engineering/*retrieval*`
- `data/generated/context_engineering/*compression*`
- `data/generated/context_engineering/workload_build_*`

## Retrieval Logic Changes

- Strict ablations now scrub generated evidence/document IDs from query text.
- Query enrichment adds allowed visible-text synonyms and finance aliases.
- Hybrid reranking uses precomputed query features, finance metadata matches, and
  title/section overlap.
- Qdrant indexed text includes title, body text, selected metadata, and the
  indexed-text strategy.
- Workload generation uses a Qdrant vector snapshot for local high-volume
  scoring after the Qdrant index is built.

## Before vs After Metrics

Earlier reported final_10000 values:

- `prompt_text_only` hybrid recall@5: 0.483159.
- `prompt_plus_metadata` hybrid recall@5: 0.569423.
- Finance `prompt_text_only` hybrid recall@5: 0.328250.
- Finance `prompt_plus_metadata` hybrid recall@5: 0.353375.

Final strict values after leakage guard:

- `prompt_text_only` hybrid recall@5: 0.442638, MRR: 0.488373.
- `prompt_plus_metadata` hybrid recall@5: 0.528491, MRR: 0.569042.
- Finance `prompt_text_only` recall@5: 0.160625, MRR: 0.091908.
- Finance `prompt_plus_metadata` recall@5: 0.218000, MRR: 0.131867.
- `prompt_plus_source_hints` recall@5: 0.978887, MRR: 0.990592.

The strict target was not reached. The earlier strict baseline included leakage
from generated identifiers embedded in prompt text, so the lower post-guard
score is more honest.

## Compression Results

- `prompt_text_only`: 26.3149% token reduction, 0.0 recall loss.
- `prompt_plus_metadata`: 26.0301% token reduction, 0.0 recall loss.
- `prompt_plus_source_hints`: 26.3504% token reduction, 0.0 recall loss.

## Remaining Gaps

- Finance prompts often lack enough no-hint specificity to identify exact filing
  accession, section, or XBRL fact.
- Retail has same-product evidence decoys; global evidence-type boosts regressed
  the full dataset and were reverted.
- Source-hint mode remains an assisted upper bound and must not be used as the
  only final claim.

## Commands Run

- `pytest tests/test_phase3_context_memory_modes.py`
- `pytest tests/test_phase3_corpus_registry.py`
- `pytest tests/test_phase3_retrieval_workloads.py`
- `pytest tests/test_phase3_retrieval_hardening.py`
- `pytest tests/test_phase3_qdrant_vector_retrieval.py`
- `pytest tests/test_phase3_retrieval_ablation.py`
- `pytest tests/test_phase3_strict_retrieval_repair.py`
- `python scripts/phase3/build_context_corpora.py --dataset-root data/scaleup_2000_full --output-root data/generated/context_engineering`
- `python scripts/phase3/build_qdrant_index.py --context-root data/generated/context_engineering --output-root data/generated/context_engineering --vector-store-config configs/vector_stores.yaml`
- `python scripts/phase3/build_memory_mode_workloads.py --dataset-root data/scaleup_2000_full --context-root data/generated/context_engineering --output-root data/workloads --splits smoke_500 controlled_2000 final_10000 --memory-modes mm0_no_context mm1_dense_top5 mm2_hybrid_top5 mm3_compressed_hybrid_top5 --dense-backend qdrant_vector --ablation-modes prompt_text_only prompt_plus_metadata prompt_plus_source_hints`

## Commit Hash

The final pushed commit hash is reported in the Codex final response. It cannot
be embedded self-referentially in the same commit without changing the commit
hash.

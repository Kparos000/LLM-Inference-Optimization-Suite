# Retrieval Root-Cause Analysis

This analysis pauses retrieval optimization before another scoring change. The goal is to explain why strict retrieval misses the current SLO targets without running inference, GPU work, paid API calls, or another retrieval pass.

The generated report reads existing Phase 3 artifacts:

- `data/generated/context_engineering/retrieval_evaluation_report.json`
- `data/generated/context_engineering/retrieval_diagnostic_report.json`
- `data/generated/context_engineering/gold_evidence_audit_report.json`
- `data/generated/context_engineering/evidence_selection_report.json`
- `data/generated/context_engineering/reranker_calibration_report.json`
- `data/generated/context_engineering/corpus_registry.json`
- `data/generated/context_engineering/corpus_build_report.json`
- `data/scaleup_2000_full/`

Outputs:

- `data/generated/context_engineering/retrieval_root_cause_report.json`
- `data/generated/context_engineering/retrieval_root_cause_summary.csv`
- `data/generated/context_engineering/retrieval_failure_examples.jsonl`

## Classification Logic

Failures are classified into root-cause labels such as:

- prompt-visible signal gaps: `prompt_missing_entity`, `prompt_missing_metric`, `prompt_missing_period`
- metadata gaps: `metadata_missing_entity`, `metadata_missing_metric`, `metadata_missing_period`
- corpus alignment gaps: `gold_not_in_corpus`, `evidence_label_too_narrow`
- candidate generation gaps: `gold_absent_from_candidate_pool`, `weak_dense_similarity`, `weak_lexical_match`
- final selection gaps: `gold_in_candidates_not_final_top5`, `reranker_miscalibrated`
- chunking gaps: `chunk_too_broad`, `chunk_too_narrow`

Aggregate counts are diagnostic and not mutually exclusive. For example, a finance failure can simultaneously show missing prompt period cues and a final top-5 selection miss.

## Current Pattern

The current strict retrieval blocker is mostly not a Qdrant availability problem. Source-hint-assisted retrieval remains strong, which means the corpus can often support the answer when direct evidence hints are available.

Strict retrieval shows three different failure patterns:

- Finance: prompt/gold metadata repair is required. Strict prompts often lack explicit metric and period cues, and final top-5 selection still drops gold evidence even when candidate recall is much higher than final recall.
- Retail and Research AI: candidate recall is materially higher than final recall, so reranking/final evidence selection is the main next target.
- Airline and Healthcare Admin: failures are smaller and more candidate-retrieval/indexed-text oriented.

Finance examples also track ticker/company, metric, period, XBRL concept, form/section detection, and whether gold appears in top 10, 20, or 50 candidate windows.

## Recommended Next Block

Repair prompt/gold metadata for finance period and metric cues, then tune final top-5 evidence selection where candidate recall already exceeds final recall. Do not claim source-hint-assisted scores as strict retrieval quality.

## Regeneration

```powershell
python scripts/phase3/analyze_retrieval_root_cause.py `
  --dataset-root data/scaleup_2000_full `
  --context-root data/generated/context_engineering `
  --slo-config configs/slo_targets.yaml `
  --output-root data/generated/context_engineering
```

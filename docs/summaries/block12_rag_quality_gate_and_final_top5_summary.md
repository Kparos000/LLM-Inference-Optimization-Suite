# Block 12 RAG Quality Gate And Final Top-5 Summary

## Files Changed

- `src/inference_bench/retrieval.py`
- `src/inference_bench/memory_workloads.py`
- `src/inference_bench/retrieval_quality_gate.py`
- `src/inference_bench/gold_evidence_audit.py`
- `src/inference_bench/reranker_calibration.py`
- `src/inference_bench/evidence_contract.py`
- `tests/test_phase3_retrieval_quality_gate.py`
- `tests/test_phase3_gold_evidence_audit.py`
- `tests/test_phase3_reranker_calibration.py`
- `tests/test_phase3_evidence_contract.py`
- `docs/70_rag_quality_gate_and_final_top5_retrieval.md`
- `docs/summaries/block12_rag_quality_gate_and_final_top5_summary.md`
- generated context-engineering reports under
  `data/generated/context_engineering/`

## Architecture Before Vs After

Before:

- Candidate generation and final top-5 selection were reported mostly through
  recall@5 and candidate recall@50.
- There was no hard pass/fail retrieval quality gate.
- There was no standalone gold/corpus alignment audit.
- Evidence selected for generation did not have a normalized contract.

After:

- Candidate diagnostics include recall@10, recall@20, recall@50, recall@100,
  and recall@200.
- The retrieval quality gate writes machine-readable `PASSED` or `BLOCKED`
  status.
- Gold evidence is audited against the normalized context corpus.
- Reranker calibration reports deterministic train/dev/test splits and
  forbidden-feature checks.
- The final selected top-5 evidence records are exposed through a structured
  evidence contract without gold labels.

## Quality Gate Targets

- Overall `prompt_plus_metadata` hybrid recall@5 >= 0.80
- Finance `prompt_plus_metadata` hybrid recall@5 >= 0.80
- Overall `prompt_text_only` hybrid recall@5 >= 0.70 if achievable
- Finance `prompt_text_only` hybrid recall@5 >= 0.65 if achievable
- Source-hint assisted hybrid recall@5 >= 0.95
- `mm3` compression token reduction >= 20%
- `mm3` compression recall loss <= 5 percentage points

## Results

Quality gate result: **BLOCKED**

Final `final_10000`, `mm2_hybrid_top5`:

| Ablation | Before recall@5 | After recall@5 | Before MRR | After MRR |
|---|---:|---:|---:|---:|
| `prompt_text_only` | 0.430112 | 0.419831 | 0.477390 | 0.478437 |
| `prompt_plus_metadata` | 0.535084 | 0.528976 | 0.576252 | 0.582645 |
| `prompt_plus_source_hints` | 0.957858 | 0.969545 | 0.977587 | 0.981233 |

Finance `final_10000`, `mm2_hybrid_top5`:

| Ablation | Before recall@5 | After recall@5 | Before MRR | After MRR |
|---|---:|---:|---:|---:|
| `prompt_text_only` | 0.156000 | 0.163000 | 0.086383 | 0.099108 |
| `prompt_plus_metadata` | 0.277250 | 0.280750 | 0.186617 | 0.174333 |
| `prompt_plus_source_hints` | 0.997375 | 0.997375 | 0.992500 | 0.991833 |

Candidate diagnostics:

| Ablation | Overall recall@10 | recall@20 | recall@50 | Finance recall@10 | recall@20 | recall@50 |
|---|---:|---:|---:|---:|---:|---:|
| `prompt_text_only` | 0.472483 | 0.574350 | 0.699372 | 0.212125 | 0.301750 | 0.574375 |
| `prompt_plus_metadata` | 0.684055 | 0.737770 | 0.837856 | 0.360875 | 0.489625 | 0.820750 |
| `prompt_plus_source_hints` | 0.898371 | 0.973294 | 0.985950 | 0.997500 | 0.997500 | 0.997500 |

The practical full-build candidate depth remains 50. Recall@100 and
recall@200 fields are present, but the reports mark them as not feasible for
this run and bound them by the available top-50 candidate pool. Two attempted
full top-200 runs timed out before report generation.

## Gold Evidence Audit

- All verticals have zero gold evidence IDs missing from the context corpus.
- Research AI has 40 gold records without explicit gold evidence IDs.
- Finance has 1,835 prompts without an exposed metric term and 2,000 prompts
  without an exposed period term.
- Finance has 1,940 strict retrieval rows where gold evidence is in top 50 but
  not final top 5.
- Finance has 1,116 strict retrieval rows where gold evidence is absent from
  top 50.

Candidate retrieval and reranking are both blockers, but final selection is the
largest metadata-assisted finance blocker: candidate recall@50 is `0.820750`
while final recall@5 is `0.280750`.

## Reranker And Evidence Selector

- Reranker backend label: `calibrated_linear`
- Strict reranker features exclude gold IDs and direct source identifiers.
- Source-hint features are enabled only for `prompt_plus_source_hints`.
- Finance selector strategy: `finance_calibrated_top5`
- General selector strategy: `calibrated_top5`
- Oracle selection remains diagnostic-only and is not used for final retrieval.
- Selected evidence is validated against the retrieval-to-generation evidence
  contract without gold labels.

## Compression

- `prompt_text_only`: 27.0512% token reduction, 0.0 recall loss
- `prompt_plus_metadata`: 26.8253% token reduction, 0.0 recall loss
- `prompt_plus_source_hints`: 26.8134% token reduction, 0.0 recall loss

## Remaining Blockers

- The 80% metadata-assisted recall@5 target is not met.
- Finance prompts often omit a visible metric and always omit an explicit
  period, making exact top-5 ranking under strict no-hint rules ambiguous.
- Retail and Research AI also have large top-50-to-top-5 selection gaps.
- The dominant report labels are `poor_scoring` and
  `gold_in_top50_not_top5`; finance additionally shows `missed_finance_metric`
  and `missed_period`.
- A stronger trained reranker needs candidate-level feature/label matrices,
  not only aggregate report calibration.

The final metrics were regenerated by:

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

## Leakage Guard

Strict modes still block direct gold/source identifiers. Gold labels are used
only for offline diagnostics, quality-gate scoring, and calibration reporting.

## Commands Run

- All required Phase 3 retrieval test files
- Context corpus regeneration
- Qdrant index regeneration
- Full memory-mode workload and retrieval report generation
- JSON/CSV report inspection
- Repository audit, doctor, config validation, Ruff checks, and git status

## Commit Hash

The final pushed commit hash is reported in the Codex final response. It cannot
be embedded self-referentially in the same commit without changing the commit
hash.

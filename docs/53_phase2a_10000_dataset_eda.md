# 10,000-Record Dataset EDA

This document explains the public-facing 10,000-record dataset EDA layer for
the promoted 10,000-record benchmark dataset under `data/scaleup_2000_full/`.

Internally, this corresponds to Phase 2A data preparation. The script remains
under `scripts/phase2/` because it belongs to that data-preparation stage, but
the generated dashboard, reports, and filenames use public-facing
`dataset_10000_eda` naming.

The promoted 10,000-record benchmark dataset contains:

- 10,000 prompts
- 10,000 gold/eval records
- 4,740 promoted benchmark KB rows
- five verticals: airline, healthcare_admin, retail, finance, and research_ai

Run:

```powershell
python scripts/phase2/explore_phase2a_promoted_dataset.py --dataset-root data/scaleup_2000_full --write-report
```

Generated EDA artifacts are written under `data/generated/dataset_10000/`.
Finance-specific EDA is mirrored under `data/generated/finance/` so reviewers
can open the finance view directly without navigating through the global
cross-vertical folder.
To remove known old generated EDA artifacts from the previous internal path,
run the same command with `--cleanup-legacy-eda`.

## What Gets Generated

- `dashboard/dataset_10000_eda_overview.html`: interactive Plotly overview dashboard.
- `dashboard/dataset_10000_eda_overview.md`: short dashboard companion summary.
- `dataset_10000_eda_inventory.json`: inventory and manifest count validation.
- `dataset_10000_eda_summary.csv`: compact per-vertical summary table.
- `dataset_10000_eda_prompt_profile.json`: prompt length, task, status, output-format, and duplicate-template profile.
- `dataset_10000_eda_kb_profile.json`: KB size, document-type, duplication, and referenced/unused evidence profile.
- `dataset_10000_eda_gold_profile.json`: reference-answer, must-include, must-not-include, and evidence-count profile.
- `dataset_10000_eda_alignment_report.json`: prompt/gold alignment checks.
- `dataset_10000_eda_evidence_reuse_report.json`: evidence coverage, reuse concentration, and unused KB share.
- `dataset_10000_eda_safety_report.json`: safety and domain-boundary scan.
- `dataset_10000_eda_workload_shape_report.json`: estimated prompt, KB, and expected-output token pressure.
- `dataset_10000_eda_summary.md`: human-readable top-level summary.
- `interactive/*.html`: standalone Plotly charts for inventory, status, task mix, output format, prompt/gold/KB length, workload shape, evidence reuse, and heatmaps.
- `plots/*.png`: static paper-ready PNG charts.
- `term_visuals/*_top_terms_bar.html`: interactive clean-term bar charts.
- `term_visuals/*_term_treemap.html`: interactive clean-term treemaps.
- `word_clouds/*_wordcloud.png`: one word cloud style image per vertical.
- `word_views/*_clean_terms.txt`: cleaned terms with filler and boilerplate removed, plus a separate domain-term view.
- `word_views/*_domain_terms.txt`: less aggressive domain vocabulary views.
- `word_views/*_tfidf_terms.txt`: TF-IDF-style vertical-distinctive terms.
- `verticals/*/*_eda.html`: per-vertical EDA pages for domain-specific review.

## How To Read The Dashboard

The dashboard starts with dataset cards for prompts, gold/evals, KB rows,
vertical count, critical issues, and warnings. The grouped inventory chart
confirms that prompts and gold/evals are balanced while KB volume differs by
vertical. Status, output-format, and task-type charts show what later benchmark
runs will ask models to do. Length charts show prompt, reference answer, and KB
row shape. Evidence reuse and workload-shape charts identify where retrieval and
context assembly may become costly later.

## Research AI KB Boundary

The promoted Research AI benchmark KB is the committed evidence used by the
10,000-record benchmark. If available, the EDA also reads the optional full
Research AI retrieval corpus under `data/generated/phase2a/retrieval_corpus/`
and reports its row count. That corpus is broader future retrieval material; it
is not the same thing as the promoted benchmark KB.

If the optional retrieval corpus is missing, the EDA still runs and clearly
reports that the comparison was skipped.

## Why This Runs Before Inference

This EDA checks inventory, alignment, evidence coverage, text shape, safety
boundaries, workload pressure, and vertical-specific data quality before any
model benchmark is run. It is intended for technical paper figures, GitHub
documentation, Streamlit-ready analytics, portfolio screenshots, and demo
materials.

This phase does not run RAG, inference, embeddings, vector indexes, model APIs,
or GPU experiments.

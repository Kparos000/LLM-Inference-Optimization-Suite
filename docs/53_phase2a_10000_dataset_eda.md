# Phase 2A-16R Promoted Dataset EDA

Phase 2A-16R rebuilds the EDA layer for the promoted Phase 2A benchmark under
`data/scaleup_2000_full/`.

The promoted dataset contains:

- 10,000 prompts
- 10,000 gold/eval records
- 4,740 promoted benchmark KB rows
- five verticals: airline, healthcare_admin, retail, finance, and research_ai

Run:

```powershell
python scripts/phase2/explore_phase2a_promoted_dataset.py --dataset-root data/scaleup_2000_full --write-report
```

Generated EDA artifacts are written under `data/generated/phase2a/eda/`.

## What Gets Generated

- `dashboard/phase2a_10000_overview.html`: interactive Plotly overview dashboard.
- `dashboard/phase2a_10000_overview.md`: short dashboard companion summary.
- `interactive/*.html`: standalone Plotly charts for inventory, status, task mix,
  output format, prompt/gold/KB length, workload shape, evidence reuse, and
  heatmaps.
- `plots/*.png`: static paper-ready PNG charts.
- `word_clouds/*_wordcloud.png`: one word cloud style image per vertical.
- `word_views/*_clean_terms.txt`: cleaned terms with filler and boilerplate
  removed, plus a separate domain-term view.
- `word_views/*_tfidf_terms.txt`: TF-IDF-style vertical-distinctive terms.
- `verticals/*/*_eda.html`: per-vertical EDA pages for domain-specific review.
- JSON reports for inventory, prompt profile, KB profile, gold/eval profile,
  alignment, evidence reuse, safety/domain boundary checks, and workload shape.

## How To Read The Dashboard

The dashboard starts with dataset cards for prompts, gold/evals, KB rows,
vertical count, critical issues, and warnings. The grouped inventory chart
confirms that prompts and gold/evals are balanced while KB volume differs by
vertical. Status, output-format, and task-type charts show what the later
benchmark will ask models to do. Length charts show prompt, reference answer,
and KB row shape. Evidence reuse and workload-shape charts identify where
retrieval and context assembly may become costly later.

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

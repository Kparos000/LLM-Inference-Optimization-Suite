# Legacy 10,000-Record Dataset EDA Location

This directory is the old internal generated-output path. The current
public-facing 10,000-record dataset EDA is written to:

```text
data/generated/eda/dataset_10000/
```

The legacy path can be cleaned with:

```powershell
python scripts/phase2/explore_phase2a_promoted_dataset.py --dataset-root data/scaleup_2000_full --write-report --cleanup-legacy-eda
```

## Historical Layout

Older generated files in this directory were created by:

```powershell
python scripts/phase2/explore_phase2a_promoted_dataset.py --dataset-root data/scaleup_2000_full --write-report
```

Layout:

- `dashboard/` contains the executive Plotly overview HTML and markdown summary.
- `interactive/` contains one standalone Plotly HTML file per major chart.
- `plots/` contains static PNG figures for papers and documentation.
- `word_clouds/` contains one generated word cloud style PNG per vertical.
- `word_views/` contains cleaned and TF-IDF-style term tables.
- `verticals/` contains one interactive HTML EDA page per vertical.
- `dataset_10000_eda_*_profile.json` and `dataset_10000_eda_*_report.json`
  contain machine-readable EDA.

This is an analytics layer for the promoted 10,000-record benchmark dataset. It
does not run inference, build RAG, create embeddings, call model APIs, or create
vector indexes. Generated artifacts are local outputs and are not the benchmark
source data.

Internally, this belongs to the Phase 2A data-preparation stage. Public-facing
generated filenames and dashboard titles use `dataset_10000_eda` naming under
the current public output path.

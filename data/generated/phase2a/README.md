# Phase 2A Generated Outputs

This directory contains local generated outputs for Phase 2A data preparation,
QA, retrieval-corpus export, and legacy/internal analytics artifacts.

Committed benchmark source data for the promoted 10,000-record checkpoint lives
under `data/scaleup_2000_full/`. Generated reports under this directory are
derived artifacts for review, documentation, and future app/reporting layers.

The current public-facing 10,000-record dataset EDA output location is:

```text
data/generated/dataset_10000/
```

Finance-specific EDA is mirrored under:

```text
data/generated/finance/
```

The historical `data/generated/phase2a/eda/` path is a legacy internal location
and should not be used for public review. It can be cleaned with:

```powershell
python scripts/phase2/explore_phase2a_promoted_dataset.py --dataset-root data/scaleup_2000_full --write-report --cleanup-legacy-eda
```

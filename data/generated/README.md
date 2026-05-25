# Generated Data Outputs

`data/generated/` is for local artifacts produced by repository scripts. These
files are derived from committed inputs or external acquisition steps and should
not be treated as canonical benchmark source data unless a promotion document
explicitly says so.

Most generated data is ignored by Git. Small README files are kept so future
runs have a documented layout.

The 10,000-record dataset EDA outputs live under:

```text
data/generated/eda/dataset_10000/
```

Internally, the generator script is stored under `scripts/phase2/` because the
dataset was prepared during the Phase 2A data-preparation stage.

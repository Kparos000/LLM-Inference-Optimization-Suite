# Generated Data Outputs

`data/generated/` is for local artifacts produced by repository scripts. These
files are derived from committed inputs or external acquisition steps and should
not be treated as canonical benchmark source data unless a promotion document
explicitly says so.

Most generated data is ignored by Git. Small README files are kept so future
runs have a documented layout.

The 10,000-record dataset EDA outputs live under:

```text
data/generated/dataset_10000/
```

Finance-specific EDA is mirrored under:

```text
data/generated/finance/
```

Internally, the generator script is stored under `scripts/phase2/` because the
dataset was prepared during the Phase 2A data-preparation stage. The old
internal `data/generated/phase2a/eda/` path and former
`data/generated/eda/dataset_10000/` wrapper are legacy locations and should not
be used for public review.

# Raw Results

`results/raw/` is for raw inference/benchmark outputs written directly by
runner commands. These files may include per-prompt rows, generation traces,
metadata, and system information from local smoke-test or benchmark runs.

Current files in this folder may include smoke-test artifacts. They are useful
for harness validation, but they are not final benchmark findings unless a
matching experiment log or promoted result table says so.

Dataset EDA outputs do not belong here. The 10,000-record dataset EDA lives at:

```text
data/generated/eda/dataset_10000/
```

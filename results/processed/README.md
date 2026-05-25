# Processed Results

`results/processed/` is for processed metrics derived from raw
inference/benchmark outputs. Typical contents include comparison tables,
aggregated latency/throughput summaries, and cleaned result CSVs.

Current processed files may include smoke-test or early harness artifacts. Treat
them as benchmark plumbing checks unless they are tied to a documented
experiment run.

Dataset EDA outputs do not belong here. The 10,000-record dataset EDA lives at:

```text
data/generated/eda/dataset_10000/
```

# Results Directory

`results/` contains inference/benchmark outputs, not dataset EDA artifacts.

The current 10,000-record dataset EDA lives under:

```text
data/generated/phase2a/eda/
```

Some existing cost, latency, and throughput figures in `results/` may be
smoke-test or early harness artifacts. If a source CSV contains only the
placeholder optimization value `none`, treat the derived plot as a smoke-test
artifact, not as a final benchmark result.

Real inference result plots will be generated later after the benchmark data,
context construction, and execution plans are ready.

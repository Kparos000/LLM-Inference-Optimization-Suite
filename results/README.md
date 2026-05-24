# Results Directory

`results/` contains experiment outputs, not Phase 2A data curation artifacts.

The current Phase 2A promoted-dataset EDA lives under:

```text
data/generated/phase2a/eda/
```

Some existing cost, latency, and throughput files in `results/` are smoke-test
or early harness artifacts. If a source CSV contains only the placeholder
optimization value `none`, treat the derived plot as a smoke-test artifact, not
as a final benchmark result.

Final inference experiment outputs will be generated later after the benchmark
data, context construction, and execution plans are ready.

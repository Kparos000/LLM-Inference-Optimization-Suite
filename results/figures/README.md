# Results Figures

This folder is for figures generated from experiment results.

The existing `cost_by_optimization.png`, `latency_by_optimization.png`, and
`throughput_by_optimization.png` files may reflect early smoke-test data rather
than final benchmark findings. In particular, plots sourced from CSV rows where
the only optimization is `none` should be read as harness validation artifacts.

Do not use this folder for dataset EDA. The current 10,000-record dataset EDA
dashboard, interactive charts, static PNG plots, word clouds, and reports are
generated under:

```text
data/generated/phase2a/eda/
```

Real inference result plots will be generated here later.

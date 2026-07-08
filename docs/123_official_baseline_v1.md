# Official Baseline V1

Status: `BASELINE_FROZEN`

The official production baseline inference experiment completed on the RunPod
A100 SXM 80GB pod and is frozen under `experiments/baseline_v1/`.

This run is the reference baseline for subsequent optimization comparisons. No
inference optimization, scheduler tuning, batching tuning, prefix-cache tuning,
KV-cache tuning, concurrency tuning, MM4 tuning, prompt tuning, retrieval
changes, evaluator changes, or SLO-threshold changes were applied during the
run.

## Matrix

- Dataset: 10,000 prompts, 2,000 per vertical.
- Verticals: Airline, Healthcare Admin, Retail, Finance, Research AI.
- Self-hosted model: `model3_7b` / `Qwen/Qwen2.5-7B-Instruct`.
- API model: `model6_gated` / `meta-llama/Llama-3.1-8B-Instruct`.
- Self-hosted engines: vLLM and SGLang.
- API runtime: provider route.
- Memory modes: MM0, MM1, MM2, MM3, MM4.
- Self-hosted concurrency: 16 and 32.
- API concurrency: 4.
- Total configs: 25.
- Total requests: 10,000.

## Result

- Requests attempted: 10,000.
- Requests completed: 10,000.
- Failed requests: 0.
- Runtime: 1,891.030 seconds.
- Total cost: `$0.817373`.
- GPU cost: `$0.782676`.
- API cost: `$0.034697`.

## Runtime Metrics

- Mean TTFT: 439.893 ms.
- Mean TPOT: 48.010 ms.
- Mean E2E latency: 1,914.616 ms.
- P50 E2E latency: 1,760.566 ms.
- P95 E2E latency: 3,599.813 ms.
- P99 E2E latency: 4,405.132 ms.
- Mean total tokens/sec: 473.726.

## Quality And Safety

- JSON validity: 99.93%.
- Contract validity: 81.54%.
- Evidence match: 62.30%.
- Groundedness: 60.73%.
- Safety findings: 103.
- Truncation: 0.00%.

## SLO Verdict

- Benchmark execution: `COMPLETED`.
- Runtime SLO: `PASS`.
- Cost SLO: `PASS`.
- Quality SLO: `FAIL`.
- Safety SLO: `FAIL`.
- Deployability: `NOT_DEPLOYABLE_SLO_FAILURES`.

The baseline is operationally complete and reproducible, but it is not
deployable by SLO. Subsequent optimization should target quality and safety
while preserving the runtime and cost passes.

## Archive

The frozen archive is `experiments/baseline_v1/`.

It contains compressed raw results, GPU telemetry, manifest, checkpoint,
evaluation report and summary, runtime report, cost report, SLO report,
comparison reports, plotting CSV/JSON, progress log, artifact sync report,
metadata, and SHA256 checksums.

Checksum verification:

```bash
cd experiments/baseline_v1
sha256sum -c SHA256SUMS.txt
```

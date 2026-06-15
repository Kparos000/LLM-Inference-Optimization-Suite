# Block B1 Qwen2.5-1.5B vLLM Quality Smoke

Status: `QUALITY_BLOCKED`

Date: June 15, 2026

## Decision

Qwen2.5-1.5B loaded and completed 100 of 100 requests on the remote RTX 3070
without OOM. It improved contract validity, evidence match, and groundedness
over the 0.5B A1 baseline, but it did not pass the frozen quality gate.

Do not scale the workload or run the optional concurrency sweep. Repair and
isolate grounded-output quality first.

## Frozen Matrix

- Hardware: `remote_rtx3070`, NVIDIA GeForce RTX 3070, 8 GB VRAM
- Engine: vLLM 0.23.0
- Model: `model2_1_5b`, `Qwen/Qwen2.5-1.5B-Instruct`
- Memory mode: `mm2_hybrid_top5`
- Retrieval source: promoted baseline, unchanged
- Records: 100 total, 20 per vertical
- Concurrency: 1
- Streaming: enabled
- Temperature: 0
- Maximum output: 128 tokens
- Generation contract: enabled

The server used the pinned A1 image digest and these conservative settings:

```text
--dtype half
--max-model-len 4096
--gpu-memory-utilization 0.75
--max-num-seqs 4
--enforce-eager
```

## Quality Gate

| Metric | Required | Observed | Result |
| --- | ---: | ---: | --- |
| JSON validity | >= 95% | 93% | Fail |
| Contract validity | >= 85% | 92% | Pass |
| Evidence match | >= 60% | 35% | Fail |
| Groundedness | >= 60% | 35% | Fail |
| Safety violations | 0 | 2 | Fail |

Evidence presence was 92%. Six outputs were truncated, one had invalid JSON,
and one had an invalid contract. All 100 rows used provider-reported token
counts.

## Per-Vertical Quality

| Vertical | JSON | Contract | Evidence presence | Evidence match | Grounded | Safety violations | Truncated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Airline | 95% | 95% | 95% | 25% | 25% | 2 | 1 |
| Healthcare Admin | 100% | 100% | 100% | 55% | 55% | 0 | 0 |
| Retail | 90% | 85% | 85% | 35% | 35% | 0 | 2 |
| Finance | 95% | 95% | 95% | 5% | 5% | 0 | 1 |
| Research AI | 85% | 85% | 85% | 55% | 55% | 0 | 2 |

Finance remains the dominant evidence-coverage failure. Of 65 evidence
mismatches, 27 matched only part of the required evidence and 38 matched none
of the required evidence.

Both safety failures were Airline outputs that emitted the prohibited phrase
`verification bypass` while describing a cautionary policy. The evaluator was
not changed and correctly marked the literal prohibited phrase.

## Runtime And GPU

- Mean TTFT: 185.529 ms
- Mean TPOT: 11.341 ms
- Mean ITL p50/p95/p99: 10.722 / 18.625 / 30.253 ms
- Mean E2E: 1,269.874 ms
- E2E p50/p95/p99: 1,257.905 / 1,624.989 / 1,933.648 ms
- Mean observed total-token throughput: 1,259.139 tokens/s
- Mean/peak GPU utilization: 77.87% / 100%
- Mean/peak GPU memory: 6,419.23 / 6,534 MB
- Mean/peak power: 125.37 / 145.66 W
- Mean/peak temperature: 63.36 / 69 C

The model fit within 8 GB VRAM. Peak sampled memory was 6,534 MB, leaving about
1,658 MB of board memory at the sample peak.

## A1 Comparison

The aggregate B1 comparison uses 100 prompts versus A1's 50 prompts. The first
10 prompts per vertical are also an exact 50-prompt overlap.

On the exact overlap:

| Metric | A1 0.5B | B1 1.5B | Delta |
| --- | ---: | ---: | ---: |
| JSON validity | 98% | 94% | -4 pp |
| Contract validity | 72% | 94% | +22 pp |
| Evidence presence | 78% | 94% | +16 pp |
| Evidence match | 30% | 44% | +14 pp |
| Groundedness | 28% | 44% | +16 pp |
| Safety violations | 2 | 1 | -1 |
| Mean TTFT | 147.859 ms | 203.805 ms | +37.8% |
| Mean TPOT | 22.002 ms | 11.303 ms | -48.6% |
| Mean E2E | 880.496 ms | 1,295.228 ms | +47.1% |

The observed token-throughput delta is retained in the JSON report, but it is
not tokenizer-matched: A1 used the legacy whitespace estimate and B1 used vLLM
provider usage.

For the full runs, B1 peak memory was 162 MB higher than A1. Mean utilization,
power, and temperature were materially higher, consistent with the larger
model doing more GPU work.

## Runtime Projection

Linear concurrency-one estimates from the measured 0.7854 requests/s:

| Work | Estimated time |
| --- | ---: |
| 500 prompts | 10.61 minutes |
| 2,500 prompts | 53.05 minutes |
| 5,000 prompts | 1.77 hours |
| 10,000 prompts | 3.54 hours |
| Eight 10,000-prompt configs | 28.29 hours |

These are planning estimates, not guarantees.

RTX 4090, L40S, A100, and H100 RunPod fields remain `null`. The repository does
not have current hourly prices or measured throughput multipliers for this B1
workload, so no RunPod time or cost was fabricated. Configure both fields in
`configs/runpod_projection_prices.yaml` before producing those estimates.

## Root Cause And Recommendation

The 1.5B model improved structured-output adherence but did not reliably select
all required evidence. Finance remained near-zero despite 95% evidence
presence, which indicates a generation/citation-selection failure under the
current contract rather than a request-serving or VRAM failure. The 128-token
limit also caused six truncations and contributed to the JSON gate failure.

Recommendation: **repair quality first**. Keep the RTX 3070 for bounded
diagnostics, but do not scale request count or run concurrency 2/4 from this
block. A later stronger-model or RunPod decision should use the same frozen
prompts and evaluator after the citation-selection and truncation causes are
isolated.

## Artifacts

- `configs/experiments/b1_remote_rtx3070_vllm_1_5b_quality_smoke.yaml`
- `configs/runpod_projection_prices.yaml`
- `results/raw/b1_remote_rtx3070_vllm_1_5b_results.jsonl`
- `results/raw/b1_remote_rtx3070_vllm_1_5b_manifest.json`
- `results/processed/b1_vllm_1_5b_quality_report.json`
- `results/processed/b1_vllm_1_5b_quality_summary.csv`
- `results/processed/b1_vllm_1_5b_vs_0_5b_comparison.json`
- `results/processed/b1_vllm_1_5b_runtime_projection.json`
- `results/processed/b1_vllm_1_5b_latency_summary.csv`
- `results/processed/b1_vllm_1_5b_gpu_telemetry.csv`
- `results/processed/b1_vllm_1_5b_gpu_telemetry_summary.json`

Raw and processed run artifacts remain ignored by repository policy. This
reviewed summary records the durable public claims.

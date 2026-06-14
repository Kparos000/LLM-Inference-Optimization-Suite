# Block A1 Remote RTX 3070 vLLM Smoke Summary

Date: 2026-06-14

Status: **SERVING PASS, QUALITY FAIL**

## Scope

Block A1 ran the first current live vLLM GPU validation on the remote RTX 3070.
It used 50 existing `smoke_500` records, exactly 10 from each vertical, with
`mm2_hybrid_top5`, `prompt_plus_metadata`, streaming, temperature zero, and a
128-token output cap.

No retrieval, dataset, gold, evaluator, SGLang, mm4, RunPod, or large-model
work was performed.

## Hardware And Server

- SSH alias: `zeever-gpu`
- OS: Ubuntu 22.04.5 LTS
- GPU: NVIDIA GeForce RTX 3070, 8 GB
- Driver/CUDA reported: 580.159.03 / 13.0
- vLLM: 0.23.0
- Image digest:
  `sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- `/v1/models`: PASS

An existing Ollama 7B model occupied about 4.9 GB. It was unloaded before A1.
The Ollama service remained active. The model was not reloaded after A1 because
the block prohibited running a large model.

## Execution

- Requested: 50
- Completed: 50
- Successful HTTP/model generations: 50
- Wall time: 48.547 seconds
- Measured aggregate request rate: 1.0299 requests/second
- Estimated input tokens: 31,879
- Estimated output tokens: 1,779

Token counts are the existing runner's whitespace estimates. They are not
provider-native tokenizer counts. vLLM logs reported approximately 94.6 to
96.6 generation tokens/second during active intervals.

## Latency

| Metric | Mean | p50 | p95 | p99 |
| --- | ---: | ---: | ---: | ---: |
| TTFT | 147.859 ms | 78.076 ms | 105.378 ms | 1,852.096 ms |
| TPOT | 22.002 ms | 21.103 ms | 29.669 ms | 35.515 ms |
| E2E | 880.496 ms | 781.589 ms | 1,299.408 ms | 2,803.140 ms |

The first Airline request produced the dominant TTFT warm-up/outlier effect.
The other vertical mean TTFT values ranged from 68.7 to 90.7 ms.

## Quality

| Metric | Result |
| --- | ---: |
| JSON validity | 49/50 (98%) |
| Generation-contract validity | 36/50 (72%) |
| Evidence-ID presence | 39/50 (78%) |
| Full evidence match | 15/50 (30%) |
| Deterministic groundedness | 14/50 (28%) |
| Safety violations | 2/50 (4%) |
| Truncation | 1/50 (2%) |

Per-vertical evidence match / groundedness:

- Airline: 30% / 20%
- Healthcare Admin: 50% / 50%
- Retail: 30% / 30%
- Finance: 0% / 0%
- Research AI: 40% / 40%

The main blocker is model output behavior, especially under-citation and
invalid or incomplete contract fields. Retrieval was not changed, and all
generated failures remain visible in the evaluator report.

## GPU Telemetry

Twenty-seven remote nvidia-smi samples were captured during the 48.5-second
run. SSH/process-query overhead prevented a true one-sample-per-second rate.

| Metric | Minimum | Mean | Maximum |
| --- | ---: | ---: | ---: |
| GPU utilization | 0% | 37.15% | 74% |
| Memory used | 6,232 MB | 6,303 MB | 6,372 MB |
| Power draw | 18.12 W | 68.31 W | 81.39 W |
| Temperature | 42 C | 47.81 C | 51 C |

The vLLM engine process used approximately 6.06 to 6.20 GB. Peak board memory
left about 1.82 GB of nominal headroom.

## Runtime Estimates

These are concurrency-one linear estimates, not guarantees.

| Prompts | Measured-throughput estimate | Mean-latency estimate | p50 estimate | p95 estimate |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 8.09 min | 7.34 min | 6.51 min | 10.83 min |
| 2,500 | 40.46 min | 36.69 min | 32.57 min | 54.14 min |
| 5,000 | 80.91 min | 73.37 min | 65.13 min | 108.28 min |
| 10,000 | 161.82 min | 146.75 min | 130.26 min | 216.57 min |

## Outputs

- `results/raw/a1_remote_rtx3070_vllm_smoke_results.jsonl`
- `results/raw/a1_remote_rtx3070_vllm_smoke_manifest.json`
- `results/processed/a1_remote_rtx3070_vllm_eval_report.json`
- `results/processed/a1_remote_rtx3070_vllm_eval_summary.csv`
- `results/processed/a1_remote_rtx3070_vllm_latency_summary.csv`
- `results/processed/a1_remote_rtx3070_vllm_runtime_projection.json`
- `results/processed/a1_remote_rtx3070_gpu_telemetry.csv`
- `results/processed/a1_remote_rtx3070_gpu_telemetry_summary.json`

Raw and processed result artifacts remain local and ignored under repository
policy.

## Files Changed

- `configs/hardware/remote_rtx3070.yaml`
- `configs/experiments/a1_remote_rtx3070_vllm_smoke.yaml`
- `src/inference_bench/gpu_telemetry.py`
- `scripts/phase4/sample_gpu_telemetry.py`
- `scripts/phase4/run_remote_vllm_smoke.py`
- `tests/test_gpu_telemetry.py`
- `tests/test_a1_vllm_smoke_config.py`
- `docs/96_remote_rtx3070_vllm_smoke.md`
- `docs/summaries/blockA1_remote_rtx3070_vllm_smoke_summary.md`
- `docs/95_definitive_technical_briefing.md`
- `README.md`

Document 96 is used for the A1 runbook because document 95 already names the
authoritative technical briefing.

## Recommendation

The RTX 3070 is sufficient for the next small-model serving experiment:
controlled concurrency 2/4 using the same frozen input and model. It is not
ready evidence for a 7B workload or the full benchmark.

Quality scaling remains blocked for Qwen 0.5B. Before using this model for
larger quality claims, the project needs either a stronger model that fits the
hardware or an explicitly bounded contract/citation repair comparison. The
strict evaluator must remain unchanged.

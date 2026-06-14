# mm4 Agentic Smoke

Status: completed on June 14, 2026.

## Frozen Matrix

- Hardware: remote NVIDIA RTX 3070, 8 GB.
- Engine: vLLM 0.23.0.
- Framework: LangGraph.
- Model: `Qwen/Qwen2.5-0.5B-Instruct`.
- Memory mode: `mm4_bounded_agentic`.
- Context source: frozen promoted `mm2_hybrid_top5` workload records.
- Prompts: 50, with 10 from each vertical.
- Concurrency: 1.
- Streaming: enabled.
- Temperature: 0.
- Maximum new tokens: 128.

The prompt IDs exactly match the A1 mm2 baseline and the measured A6 mm3
baseline.

## Commands

Start the documented A1 vLLM container:

```powershell
ssh zeever-gpu 'docker rm -f llm-suite-a1-vllm >/dev/null 2>&1 || true; docker run -d --gpus all --ipc=host --name llm-suite-a1-vllm -p 8000:8000 -v $HOME/.cache/huggingface:/root/.cache/huggingface vllm/vllm-openai@sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f --model Qwen/Qwen2.5-0.5B-Instruct --served-model-name Qwen/Qwen2.5-0.5B-Instruct --dtype half --max-model-len 4096 --gpu-memory-utilization 0.75 --max-num-seqs 4 --enforce-eager --host 0.0.0.0 --port 8000'
```

Create the local tunnel and run the smoke:

```powershell
ssh -N -L 8000:127.0.0.1:8000 zeever-gpu
python scripts/phase4/run_mm4_agentic_smoke.py `
  --run-mm3-baseline `
  --base-url http://127.0.0.1:8000/v1 `
  --api-key EMPTY `
  --timeout-seconds 180 `
  --max-new-tokens 128
```

Stop the server:

```powershell
ssh zeever-gpu "docker stop llm-suite-a1-vllm"
```

## Operational Result

- Server start: pass.
- Model load and `/v1/models`: pass.
- mm4 rows completed: 50 of 50.
- Unique prompt IDs: 50.
- Prompt-ID parity with mm2/mm3: pass.
- Request failures: 0.
- Final graph statuses: 47 answer, 3 escalate.
- Maximum observed limits: one retrieval round, two generation attempts, one
  repair, and three action-tool calls.

## mm4 Metrics

| Metric | Result |
| --- | ---: |
| JSON validity | 98% |
| Contract validity | 94% |
| Evidence match | 44% |
| Deterministic groundedness | 42% |
| Safety violations | 4% |
| Repair rate | 6% |
| Escalation rate | 6% |
| Mean TTFT | 181.903 ms |
| Mean E2E latency | 1,022.239 ms |
| p95 E2E latency | 1,417.007 ms |
| p99 E2E latency | 2,351.349 ms |
| Mean TPOT | 9.065 ms |

Raw provider usage for mm4 was 83,540 input and 4,533 output tokens. Cross-mode
comparison uses explicit whitespace-normalized counts because the historical
mm2/mm3 OpenAI-compatible runner stored whitespace estimates rather than
provider tokenizer usage.

## mm2/mm3/mm4 Comparison

| Metric | mm2 | mm3 | mm4 |
| --- | ---: | ---: | ---: |
| Groundedness | 28% | 26% | 42% |
| Evidence match | 30% | 26% | 44% |
| Contract validity | 72% | 66% | 94% |
| Safety violations | 4% | 2% | 4% |
| Mean TTFT | 147.859 ms | 114.117 ms | 181.903 ms |
| Mean E2E | 880.496 ms | 770.627 ms | 1,022.239 ms |
| Normalized input tokens | 31,879 | 28,356 | 37,123 |
| Normalized output tokens | 1,779 | 1,653 | 2,012 |
| Repair rate | 0% | 0% | 6% |
| Escalation rate | 0% | 0% | 6% |

Against mm2, mm4 gained 14 percentage points of groundedness and 22 points of
contract validity, while mean E2E increased by 141.743 ms and normalized total
tokens increased by 5,477.

Against mm3, mm4 gained 16 percentage points of groundedness and 28 points of
contract validity, while mean E2E increased by 251.612 ms and normalized total
tokens increased by 9,126.

## Node Latency

Mean node timings were:

- classify: 0.016 ms;
- plan: 0.004 ms;
- retrieve: 0.015 ms;
- assemble: 0.337 ms;
- first generation: 950.098 ms;
- validate: 0.354 ms;
- finalize/escalate: 0.005 ms.

For the three repaired rows, the repair node averaged 976.909 ms. Model
generation dominates graph overhead.

## Cost

No GPU hourly price is registered for the remote development server. GPU cost,
cost per request, and cost per grounded answer are therefore unavailable and
were not estimated.

## Decision

Keep mm4 in the controlled benchmark matrix as an opt-in quality/validation
mode. Do not make it the default RTX 3070 path and do not scale beyond the
50-prompt gate yet. The quality uplift is measurable, but 42% groundedness
still fails the project SLO and latency/token use regress relative to mm2/mm3.

## Artifacts

Raw and processed outputs are generated locally under:

```text
results/raw/a6_mm4_agentic_smoke_results.jsonl
results/raw/a6_mm4_agentic_smoke_traces.jsonl
results/raw/a6_mm4_agentic_smoke_manifest.json
results/processed/a6_mm4_agentic_eval_report.json
results/processed/a6_mm4_agentic_eval_summary.csv
results/processed/a6_mm4_agentic_latency_summary.csv
results/processed/a6_mm4_agentic_trace_summary.csv
results/processed/a6_mm4_vs_mm2_mm3_comparison_report.json
results/processed/a6_mm4_vs_mm2_mm3_comparison_summary.csv
```

These directories remain ignored by repository policy. This document and the
block summary preserve the reviewed measurements.

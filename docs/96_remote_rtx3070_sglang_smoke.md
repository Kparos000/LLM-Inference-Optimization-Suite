# Remote RTX 3070 SGLang Smoke

Block A2/A3 validates SGLang 0.5.13 on the `remote_rtx3070` hardware profile
using the same 50 prompts, model, generation settings, evaluator, and telemetry
schema as the A1 vLLM smoke.

## Frozen Matrix

- Hardware: `remote_rtx3070`
- Model: `model1_0_5b` / `Qwen/Qwen2.5-0.5B-Instruct`
- Engine: SGLang 0.5.13
- Memory mode: `mm2_hybrid_top5`
- Retrieval ablation: `prompt_plus_metadata`
- Records: 10 per vertical, 50 total
- Concurrency: 1
- Streaming: enabled
- Temperature: 0
- Maximum output: 128 tokens
- Generation contract: enabled

The checked-in matrix is
`configs/experiments/a2_remote_rtx3070_sglang_smoke.yaml`.

## Start SGLang

The exact successful command was:

```powershell
ssh zeever-gpu 'docker rm -f llm-suite-a2-sglang >/dev/null 2>&1 || true; docker run -d --gpus all --ipc=host --name llm-suite-a2-sglang -p 30000:30000 -v $HOME/.cache/huggingface:/root/.cache/huggingface lmsysorg/sglang@sha256:952ebb195c41b10dc01fa63c41c9bfc14f2ee02ffe8da71e11aeab5f3f7c7772 python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct --served-model-name Qwen/Qwen2.5-0.5B-Instruct --dtype half --context-length 4096 --mem-fraction-static 0.70 --disable-cuda-graph --host 0.0.0.0 --port 30000'
```

Port `30000` serves the OpenAI-compatible API. SGLang recommends `sglang serve`
as the newer entrypoint, but the measured container accepted
`python3 -m sglang.launch_server`.

## Health Check

```powershell
ssh zeever-gpu "curl -fsS http://127.0.0.1:30000/v1/models"
```

The response must contain `Qwen/Qwen2.5-0.5B-Instruct`.

For a local SSH tunnel:

```powershell
ssh -N -L 30000:127.0.0.1:30000 zeever-gpu
```

## Run A2

```powershell
python scripts/phase4/run_remote_sglang_smoke.py `
  --base-url http://127.0.0.1:30000/v1 `
  --telemetry-ssh-host zeever-gpu `
  --telemetry-interval-seconds 1 `
  --telemetry-duration-seconds 600 `
  --timeout-seconds 180
```

The active Python environment must include the project `openai` optional
dependency.

## Stop SGLang

```powershell
ssh zeever-gpu "docker stop llm-suite-a2-sglang"
```

The measured container and local tunnel were stopped after artifact
generation.

## RTX 3070 Limitations

- The board has 8 GB VRAM.
- SGLang used up to 6,551 MB in this run, 179 MB more than vLLM A1.
- The 0.5B model validates serving behavior, not final answer quality.
- No infrastructure hourly price is registered, so this run makes no GPU cost
  claim.
- SGLang still captured piecewise CUDA graphs during startup despite the
  general `--disable-cuda-graph` flag; the measured command and logs must be
  retained when interpreting memory use.
- Backend-native queue, batch, radix-cache, and KV-cache time series were not
  collected.

Expected failure modes include insufficient VRAM during model/KV-cache
allocation, incompatible CUDA or attention kernels, model-download failure,
port conflicts, a missing OpenAI client dependency, and health-check timeout.

## Result

SGLang started, loaded the model, passed `/v1/models`, and completed 50 of 50
requests.

| Metric | SGLang | vLLM | SGLang minus vLLM |
| --- | ---: | ---: | ---: |
| Mean TTFT | 135.971 ms | 147.859 ms | -11.887 ms |
| Mean TPOT | 24.202 ms | 22.002 ms | +2.200 ms |
| Mean E2E | 1,066.357 ms | 880.496 ms | +185.861 ms |
| Mean throughput | 673.905 tok/s | 880.575 tok/s | -206.670 tok/s |
| Contract validity | 58% | 72% | -14 points |
| Evidence match | 36% | 30% | +6 points |
| Groundedness | 24% | 28% | -4 points |
| Safety violations | 4% | 4% | no change |
| Mean GPU utilization | 33.38% | 37.15% | -3.77 points |
| Peak GPU memory | 6,551 MB | 6,372 MB | +179 MB |

SGLang remains in the controlled comparison matrix as a secondary engine. vLLM
remains the default RTX 3070 backend because it delivered lower E2E latency,
lower TPOT, higher throughput, lower peak memory, and better contract and
grounding rates on the matched run.

## Artifacts

- `results/raw/a2_remote_rtx3070_sglang_smoke_results.jsonl`
- `results/raw/a2_remote_rtx3070_sglang_smoke_manifest.json`
- `results/processed/a2_remote_rtx3070_sglang_eval_report.json`
- `results/processed/a2_remote_rtx3070_sglang_eval_summary.csv`
- `results/processed/a2_remote_rtx3070_sglang_latency_summary.csv`
- `results/processed/a2_remote_rtx3070_sglang_gpu_telemetry.csv`
- `results/processed/a2_remote_rtx3070_sglang_gpu_telemetry_summary.json`
- `results/processed/a2_vllm_vs_sglang_comparison_report.json`
- `results/processed/a2_vllm_vs_sglang_comparison_summary.csv`

These raw and processed smoke artifacts remain local under the repository
output policy.

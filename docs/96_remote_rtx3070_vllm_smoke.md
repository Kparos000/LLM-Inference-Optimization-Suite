# Remote RTX 3070 vLLM Smoke

Block A1 validates the current OpenAI-compatible generation path against a
real vLLM server on the `remote_rtx3070` hardware profile.

The requested filename was `docs/95_remote_rtx3070_vllm_smoke.md`. This
repository already uses document 95 for the authoritative technical briefing,
so this runbook uses document 96 to preserve both documents.

## Frozen Matrix

- Hardware: `remote_rtx3070`
- Model: `model1_0_5b` / `Qwen/Qwen2.5-0.5B-Instruct`
- Engine: vLLM 0.23.0
- Memory mode: `mm2_hybrid_top5`
- Retrieval ablation: `prompt_plus_metadata`
- Records: 10 per vertical, 50 total
- Concurrency: 1
- Streaming: enabled
- Temperature: 0
- Maximum output: 128 tokens
- Generation contract: enabled

## Connect

```powershell
ssh zeever-gpu
```

The SSH alias resolves over Tailscale. No host address, username, or credential
is stored in the repository.

## Free GPU Memory

The host had an Ollama 7B model resident before A1. Unload it without stopping
the Ollama service:

```powershell
ssh zeever-gpu "ollama stop qwen2.5:7b-instruct"
```

Confirm available memory:

```powershell
ssh zeever-gpu "nvidia-smi"
```

## Start vLLM

The exact A1 command was:

```powershell
ssh zeever-gpu 'docker rm -f llm-suite-a1-vllm >/dev/null 2>&1 || true; docker run -d --gpus all --ipc=host --name llm-suite-a1-vllm -p 8000:8000 -v $HOME/.cache/huggingface:/root/.cache/huggingface vllm/vllm-openai:latest --model Qwen/Qwen2.5-0.5B-Instruct --served-model-name Qwen/Qwen2.5-0.5B-Instruct --dtype half --max-model-len 4096 --gpu-memory-utilization 0.75 --max-num-seqs 4 --enforce-eager --host 0.0.0.0 --port 8000'
```

The pulled image resolved to:

```text
vLLM version: 0.23.0
image digest: sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f
```

Port `8000` serves the OpenAI-compatible API.

## Health Check

On the remote host:

```powershell
ssh zeever-gpu "curl -fsS http://localhost:8000/v1/models"
```

From a Tailscale-connected client, use the host from `ssh -G zeever-gpu`:

```powershell
Invoke-RestMethod http://<tailscale-host>:8000/v1/models
```

The expected model ID is `Qwen/Qwen2.5-0.5B-Instruct`.

## Run A1

```powershell
python scripts/phase4/run_remote_vllm_smoke.py `
  --base-url http://<tailscale-host>:8000/v1 `
  --telemetry-ssh-host zeever-gpu `
  --telemetry-interval-seconds 1 `
  --timeout-seconds 180
```

The script exports the frozen balanced input, calls the existing streaming
OpenAI-compatible runner, runs the unchanged evaluator, and writes latency,
telemetry, manifest, and projection reports.

## Inspect GPU Memory

```powershell
ssh zeever-gpu "nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits"
```

Process memory:

```powershell
ssh zeever-gpu "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits"
```

## Stop vLLM

```powershell
ssh zeever-gpu "docker stop llm-suite-a1-vllm"
```

The A1 run stopped the container after report collection. The Ollama service
remained active, but its 7B model was not reloaded because A1 explicitly
prohibits running a large model.

## Why This Model And Concurrency

Qwen2.5-0.5B is the smallest open model in the registry and was already
validated through local Transformers. It isolates remote serving, streaming,
telemetry, and evaluation plumbing before spending memory on a stronger model.

Concurrency starts at one so TTFT, TPOT, memory, and quality can be interpreted
without scheduler contention. The measured mean GPU utilization was only
37.15%, so a later controlled small-model block can test concurrency 2 and 4
without changing the A1 baseline.

## Result

The model loaded, `/v1/models` passed, and all 50 requests completed. Serving
reliability passed, but final answer quality did not:

- JSON validity: 98%
- generation-contract validity: 72%
- evidence match: 30%
- deterministic groundedness: 28%
- safety violations: 2 of 50

The RTX 3070 is suitable for continued small-model serving validation. It is
not evidence that the 8 GB card can run the registered 7B or 32B benchmark
roles under the required context and concurrency settings.

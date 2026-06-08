# Controlled Inference Readiness Audit

Block 31B audits the controls already present before the first small GPU vLLM
smoke. The audit does not contact a model, serving backend, GPU, or API.

## Decision

Status: `NOT_READY`

The inference stack is implemented and locally validated, but the live GPU
execution inputs are not frozen:

- RunPod GPU type, region, and hourly price are unset.
- The exact GPU/model selection is not confirmed.
- The five-prompt GPU smoke matrix is not recorded as a reviewed run input.

These are operational blockers, not retrieval or runner implementation
failures.

## Controls Already Built

| Category | Status | Existing control |
| --- | --- | --- |
| Dataset readiness | PASS | Promoted retrieval source-of-truth and passing retrieval SLOs |
| Workload control | PASS | `smoke_500`, `controlled_2000`, and `final_10000` definitions and local materialization |
| Memory modes | PASS | `mm0` through `mm3`; `mm4` explicitly contract-only |
| Context engineering | PASS | Vertical chunking, canonical keys, Qdrant path, compression reports |
| Run safety | PASS | Chunking, checkpoint, resume, timeout, failure rows |
| Observability | PASS | Run manifest, per-request metadata, telemetry schema, `error_type` |
| Cost schema | PASS | API pricing, GPU cost formula, explicit paid-call and request-count guards |
| Serving paths | PASS | Local HF, OpenRouter, HF/Novita; vLLM and SGLang dry-run adapters |
| SLO definitions | PASS | Retrieval, quality, latency, throughput, resource, API cost, GPU cost |
| Live GPU cost inputs | FAIL | RunPod values remain intentionally unset |
| Frozen GPU smoke inputs | FAIL | GPU/model and reviewed matrix are not confirmed |

Ignored workload and result files are treated separately from implementation
readiness. The audit reports whether each split is materialized locally while
also recognizing the checked-in deterministic workload builder. This avoids
making clean-checkout CI depend on local generated JSONL files.

## Run Safety Detail

The OpenAI-compatible load runner already supports:

- chunked execution;
- checkpoint JSON with completed prompt IDs;
- resume without duplicate prompt processing;
- append-mode result persistence;
- request timeout handling;
- per-request success and error fields.

Before a main GPU sweep, live GPU telemetry, structured append-only run logs,
OOM/timeout retry classification, and resume parity across every selected
backend still need validation.

## Exact Next Block

Block 32A should:

1. Select the GPU and model for the first smoke.
2. Record the actual RunPod region and hourly price.
3. Freeze a five-prompt, `mm2_hybrid_top5`, concurrency-1 vLLM matrix.
4. Start vLLM and verify `/v1/models`.
5. Run the guarded five-request smoke.
6. Capture TTFT, TPOT, throughput, GPU utilization, memory, power, temperature,
   and elapsed infrastructure cost.

No concurrency sweep or 500-prompt run should start until this five-request
gate passes.

## Artifacts

- `data/generated/phase4/controlled_inference_readiness_report.json`
- `data/generated/phase4/controlled_inference_readiness_summary.csv`

Regenerate with:

```powershell
python scripts/phase4/audit_controlled_inference_readiness.py `
  --output-root data/generated/phase4
```


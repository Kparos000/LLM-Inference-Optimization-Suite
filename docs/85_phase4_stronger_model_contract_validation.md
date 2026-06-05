# Phase 4 Stronger-Model Contract Validation

Block 26 adds a guarded comparison path for determining whether the generation
contract failures observed with `model1_0_5b` are primarily model-capability
failures or pipeline failures.

## Experiment Contract

The comparison uses exactly five records:

- one prompt per vertical;
- `mm2_hybrid_top5`;
- `prompt_plus_metadata`;
- `qdrant_vector`;
- `qdrant_local`;
- no source hints;
- the same generation contract and strict evaluator used in Block 24.

The runner input is rebuilt from:

```text
data/workloads/smoke_500/prompt_plus_metadata/mm2_hybrid_top5.jsonl
```

This prevents the comparison from accidentally using the older pre-promotion
runner input.

## Model Selection

The default decision order is:

1. Run `model2_1_5b` (`Qwen/Qwen2.5-1.5B-Instruct`) locally only when its full
   model snapshot is already cached.
2. Otherwise select `model5_gated`
   (`meta-llama/Llama-3.2-3B-Instruct`) as an API-route dry-run.
3. A paid Hugging Face Inference Provider call is possible only when the user
   explicitly passes `--allow-paid-api-call`, supplies `HF_TOKEN`, accepts the
   model license, and has a real pricing snapshot.

The local path passes `local_files_only=True` to Transformers. It does not
silently download model weights.

## Current Run Result

Status: `DRY_RUN_ONLY`

The local 1.5B model was not cached. At execution time the machine had roughly
3.09 GB of free physical memory, so downloading and attempting an uncached
1.5B CPU model was not treated as a safe implicit action.

The gated fallback ran in dry-run mode:

- model alias: `model5_gated`;
- records: 5;
- paid API calls: 0;
- GPU work: none;
- vLLM work: none;
- generated model answers: none.

The dry-run proves workload selection, model routing, metadata preservation,
output schema, and paid-call guards. It does not prove stronger-model quality.

## Block 24 Comparison

Because no stronger model generated answers, stronger-model metrics remain
null rather than being reported as zero.

| Metric | Block 24 0.5B | Stronger model |
| --- | ---: | ---: |
| JSON validity | 100% | Not measured |
| Contract validity | 80% | Not measured |
| Evidence-ID presence | 100% | Not measured |
| Evidence match | 40% | Not measured |
| Groundedness | 20% | Not measured |
| Mean latency | 146.223 seconds | Not measured |
| Median latency | 172.321 seconds | Not measured |
| Input tokens | 7,456 | Not measured |
| Output tokens | 574 | Not measured |

The causal question remains open: Block 26 has not yet shown whether the 0.5B
model caused the contract and grounding failures.

## Commands

Safe automatic decision path:

```powershell
python scripts/phase4/run_stronger_model_contract_smoke.py `
  --output-path results/raw/phase4_stronger_model_contract_smoke.jsonl

python scripts/phase4/evaluate_stronger_model_contract.py `
  --results-path results/raw/phase4_stronger_model_contract_smoke.jsonl `
  --dataset-root data/scaleup_2000_full `
  --report-path results/processed/phase4_stronger_model_contract_eval_report.json `
  --summary-path results/processed/phase4_stronger_model_contract_eval_summary.csv
```

Local 1.5B validation after the model is deliberately cached and sufficient
RAM is available:

```powershell
python scripts/phase4/run_stronger_model_contract_smoke.py `
  --execution-mode local_hf `
  --local-model-alias model2_1_5b `
  --output-path results/raw/phase4_stronger_model_contract_smoke.jsonl
```

Explicit paid API validation requires all safety prerequisites:

```powershell
python scripts/phase3/snapshot_hf_inference_pricing.py `
  --models model5_gated `
  --output configs/api_pricing.yaml `
  --report data/generated/context_engineering/hf_api_pricing_snapshot_report.json

python scripts/phase4/run_stronger_model_contract_smoke.py `
  --execution-mode hf_api `
  --fallback-model-alias model5_gated `
  --allow-paid-api-call `
  --output-path results/raw/phase4_stronger_model_contract_smoke.jsonl
```

The paid command still refuses to run without `HF_TOKEN` and pricing.

## Output Policy

Generated raw and processed results remain local and ignored:

- `results/raw/phase4_stronger_model_contract_smoke.jsonl`
- `results/processed/phase4_stronger_model_contract_eval_report.json`
- `results/processed/phase4_stronger_model_contract_eval_summary.csv`

The evaluator was not weakened, no evidence IDs were fabricated, retrieval was
not modified, and no GPU or vLLM path was used.

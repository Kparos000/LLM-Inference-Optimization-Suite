# Controlled Final Simulation Summary

## What Ran

- Added `scripts/phase4/run_controlled_final_simulation.py`.
- Built the 30-config, 15,000-request controlled final-simulation matrix.
- Ran safety-gate checks before any full-model/provider execution.
- Wrote blocked-run evaluation, comparison, SLO, cost, manifest, and artifact-sync reports.

## Outcome

The full controlled simulation did not run. The safety gates blocked execution
before any model/provider request:

- vLLM `model3_7b`: smoke-ready because `/v1/models` at
  `http://localhost:8000/v1` listed `Qwen/Qwen2.5-7B-Instruct`.
- SGLang `model3_7b`: runtime registry now allows SGLang for `model3_7b` on
  `a100_sxm_80gb`, and the package is importable, but
  `http://localhost:30000/v1/models` refused the health-check connection.
- API `model6_gated`: blocked because `HF_TOKEN` and provider API credentials
  were absent.
- MM4: bounded LangGraph runner is importable, but no MM4 matrix run was
  attempted because the required track smokes were blocked.

## Request Counts

- Planned requests: 15,000.
- Attempted requests: 0.
- Completed configs: 0.
- Not-run configs: 30.

## Decision

The final 10,000-prompt experiment is not allowed yet. The next step is to make
the vLLM, SGLang, API, and MM4 smoke gates pass without fallback or skipped
configs.

Exact SGLang startup command:

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --served-model-name Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 30000 --mem-fraction-static 0.90 --context-length 4096 --max-running-requests 32 --chunked-prefill-size 8192
```

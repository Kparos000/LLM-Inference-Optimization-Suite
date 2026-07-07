# Controlled Final Simulation Summary

## What Ran

- Added `scripts/phase4/run_controlled_final_simulation.py`.
- Built the 30-config, 15,000-request controlled final-simulation matrix.
- Ran safety-gate checks before any full-model/provider execution.
- Wrote blocked-run evaluation, comparison, SLO, cost, manifest, and artifact-sync reports.

## Outcome

The full controlled simulation did not run. The safety gates blocked execution
before any model/provider request:

- vLLM `model3_7b`: blocked because no local vLLM server was serving
  `Qwen/Qwen2.5-7B-Instruct`.
- SGLang `model3_7b`: blocked because SGLang is not installed/importable and the
  runtime registry does not allow SGLang for `model3_7b` on `a100_sxm_80gb`.
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

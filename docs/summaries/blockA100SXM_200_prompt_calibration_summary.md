# A100 SXM 200-Prompt Calibration Summary

## What Changed

- Added `scripts/phase4/run_a100_sxm_calibration.py`.
- Added A100 runner/output tests.
- Registered `a100_sxm_80gb` as a vLLM-supported hardware target.
- Ran the live 200-prompt A100 SXM calibration with `model2_3b`, vLLM, `mm2_hybrid_top5`, concurrency 1, artifact sync, checkpoint/resume, manifest, and GPU telemetry.
- Documented the run in `docs/116_a100_sxm_200_prompt_calibration.md`.

## Live Result

- Completed prompts: 200/200.
- Successful requests: 200.
- Failed requests: 0.
- Overall quality: 99.0% JSON valid, 98.5% contract valid, 97.5% evidence match, 97.0% grounded, 0 safety violations.
- Runtime: 138.158 s wall time, 1.448 requests/sec, 2,189.29 aggregate tokens/sec.
- Latency: 52.49 ms mean TTFT, 6.23 ms mean TPOT, 675.71 ms mean E2E.
- GPU telemetry: 128 samples, 95.80% mean GPU utilization, 74,247 MB max VRAM, 266.52 W mean power, 53 C max temperature.
- Cost estimate: `$0.0572` at `$1.49/hr`.
- Artifact sync: passed local backup verification.

## Per-Vertical Quality

| Vertical | Evidence match | Grounded |
| --- | ---: | ---: |
| airline | 100.0% | 100.0% |
| healthcare_admin | 100.0% | 97.5% |
| retail | 97.5% | 97.5% |
| finance | 97.5% | 97.5% |
| research_ai | 92.5% | 92.5% |

## Decision

The 1,000-prompt A100 baseline is allowed by the calibration rule because the 200-prompt run completed, passed quality gates, verified artifact sync, and captured GPU telemetry. It is not run automatically by this block.

Generated runner input, raw results, telemetry, processed reports, checkpoints, backups, and derived workload/context artifacts remain uncommitted outputs.

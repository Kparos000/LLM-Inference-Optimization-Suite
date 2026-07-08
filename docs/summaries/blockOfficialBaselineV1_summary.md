# Block Official Baseline V1 Summary

Status: `BASELINE_FROZEN`

The official baseline inference experiment completed the frozen 25-config,
10,000-request matrix on RunPod A100 SXM 80GB.

- Requests: 10,000 attempted, 10,000 completed, 0 failed.
- Runtime: 1,891.030 seconds.
- Total cost: `$0.817373`.
- Mean E2E latency: 1,914.616 ms.
- JSON validity: 99.93%.
- Contract validity: 81.54%.
- Evidence match: 62.30%.
- Groundedness: 60.73%.
- Safety findings: 103.

SLO verdict:

- Runtime: `PASS`.
- Cost: `PASS`.
- Quality: `FAIL`.
- Safety: `FAIL`.
- Deployability: `NOT_DEPLOYABLE_SLO_FAILURES`.

The baseline is frozen in `experiments/baseline_v1/` with checksums. It is the
reference for all subsequent optimization comparisons.

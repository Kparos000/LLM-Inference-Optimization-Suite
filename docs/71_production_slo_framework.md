# Production SLO Framework

The production SLO framework is the central pass/fail gate for future retrieval,
inference, quality, cost, throughput, and resource experiments. It gives every
vertical the same metric families so results can be compared like-for-like across
the promoted 10,000-record benchmark dataset.

This framework does not run inference, GPU experiments, or paid API calls. It
evaluates existing reports and marks unavailable future metrics explicitly.

## Why SLOs Matter

The benchmark is meant to support engineering decisions, not just produce
interesting charts. SLOs define what is good enough before a run can be used for
larger scaling, portfolio claims, or paper results.

Retrieval SLOs are especially important before inference scaling. If the context
layer cannot retrieve the right evidence, a faster or larger model will still be
answering from weak context. In that case, inference scaling should be blocked
until retrieval is repaired.

## Metric Families

Each vertical defines the same seven metric families in
`configs/slo_targets.yaml`:

- `retrieval_slo`
- `quality_slo`
- `latency_slo`
- `throughput_slo`
- `resource_slo`
- `api_cost_slo`
- `gpu_cost_slo`

Only thresholds differ by vertical.

## Latency Metrics

Latency is not secondary for any vertical. Every vertical tracks:

- `TTFT`: time to first token. This captures queueing, prefill, scheduling, and
  first-token responsiveness.
- `ITL`: inter-token latency. This captures token-to-token decode smoothness.
- `TPOT`: time per output token. This captures decode throughput from the user
  request perspective.
- `E2E latency`: end-to-end request latency from submission to completed answer.

Each latency family includes p50, p95, and p99 thresholds.

## Initial Targets

Initial targets are intentionally strict enough to block weak runs, but they can
be adjusted after empirical Phase 4 and Phase 5 smoke results.

| Vertical | Final Recall@5 | MRR | TTFT p95 max | E2E p95 max |
| --- | ---: | ---: | ---: | ---: |
| airline | 0.90 | 0.85 | 1,000 ms | 4,000 ms |
| retail | 0.90 | 0.85 | 1,000 ms | 4,000 ms |
| healthcare_admin | 0.90 | 0.85 | 1,500 ms | 6,000 ms |
| finance | 0.90 | 0.90 | 2,500 ms | 10,000 ms |
| research_ai | 0.90 | 0.90 | 3,000 ms | 12,000 ms |

Healthcare, Finance, and Research AI carry stricter quality requirements because
grounding, citation accuracy, and evidence match matter more in those workflows.

## Cost SLOs

API token cost and self-hosted GPU infrastructure cost are tracked separately.

API-priced models use token pricing:

- input tokens per request
- output tokens per request
- total tokens per request
- API cost per request
- API cost per 1,000 requests
- API cost per successful answer
- API cost per grounded successful answer

Self-hosted GPU runs use infrastructure pricing:

- GPU hourly price availability
- GPU cost per request
- GPU cost per 1,000 requests
- GPU cost per successful answer
- GPU cost per grounded successful answer
- tokens per GPU dollar

This avoids mixing API token economics with self-hosted GPU economics.

## Readiness Evaluation

Run:

```powershell
python scripts/phase3/evaluate_slo_readiness.py `
  --slo-config configs/slo_targets.yaml `
  --retrieval-report data/generated/context_engineering/retrieval_evaluation_report.json `
  --quality-gate-report data/generated/context_engineering/retrieval_quality_gate_report.json `
  --output-root data/generated/context_engineering
```

Outputs:

- `data/generated/context_engineering/slo_readiness_report.json`
- `data/generated/context_engineering/slo_readiness_summary.csv`

Current readiness status is `BLOCKED` because retrieval SLOs are not yet met.
Inference, latency, cost, throughput, and telemetry metrics are marked
`NOT_AVAILABLE` until Phase 4 and Phase 5 experiments produce those reports.

## Future Charts

The SLO framework is designed to support paper and dashboard charts such as:

- TTFT p95 by vertical vs SLO
- ITL p95 by vertical vs SLO
- E2E p95 by vertical vs SLO
- Recall@5 by vertical vs SLO
- Groundedness by vertical vs SLO
- Cost per successful answer vs SLO

## Optimization Loop

The intended loop is:

1. Run retrieval or inference experiment.
2. Evaluate SLO readiness.
3. Repair the first blocked metric family.
4. Rerun the experiment.
5. Promote only results that meet the relevant SLO gate.

This keeps future optimization work tied to measurable engineering outcomes.


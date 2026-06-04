# Canonical Retrieval Key Materialization

Block 17 adds a canonical retrieval path before Phase 4 inference work. It does
not run model inference, GPU jobs, paid API calls, or external retrieval.

## What Changed

The retrieval repair now has three explicit layers:

- `src/inference_bench/retrieval_keys.py` derives non-leaking retrieval keys from
  prompt-visible text and realistic prompt metadata.
- `src/inference_bench/canonical_queries.py` renders raw, normalized, enriched,
  compact keyword, and Qdrant-oriented query forms.
- `src/inference_bench/vertical_final_selectors.py` applies deterministic
  vertical-specific final top-5 selection after hybrid candidate generation.

The existing all-vertical repair command remains backward-compatible. Passing
`--use-canonical-retrieval-keys` switches staged validation to the canonical
query path and writes canonical report filenames.

## Leakage Policy

Canonical retrieval keys must not include:

- gold evidence IDs
- required evidence, document, chunk, or policy IDs
- direct hidden source IDs
- parent IDs
- filing IDs
- answer-side hints

Gold IDs are still used only for offline recall and MRR measurement.

## Vertical Key Strategy

Airline keys use support type, route, policy issue terms, baggage/refund/delay
signals, and escalation signals.

Healthcare Admin keys use support type, department, safety boundary, admin
procedure terms, privacy signals, and identity signals. The selector is
conservative because the existing healthcare path is already relatively strong.

Retail keys use category, product title, product-title terms, support intent,
review issue terms, and policy context. Final selection boosts real review and
summary evidence and suppresses seed expansion rows when they crowd the final
top-5.

Finance keys use company, ticker, filing type, period, fiscal quarter/year,
metric family, filing section, and XBRL concept family when those cues are
available. The selector now boosts 8-K filing-event chunks when the prompt asks
for an 8-K filing point without a specific metric or period.

Research AI keys use topic, paper title, all visible section types, topic terms,
method signals, and results signals. The Qdrant query no longer injects generic
method/results/limitations terms unless those are visible from the prompt or
metadata.

## Reports

Canonical staged validation writes:

- `data/generated/context_engineering/canonical_retrieval_repair_report.json`
- `data/generated/context_engineering/canonical_retrieval_repair_summary.csv`
- `data/generated/context_engineering/canonical_retrieval_failure_examples.jsonl`

The standard SLO readiness command still writes:

- `data/generated/context_engineering/slo_readiness_report.json`
- `data/generated/context_engineering/slo_readiness_summary.csv`

## Latest Canonical Results

The 500 and 2,000 staged validation used Qdrant hybrid retrieval with
`prompt_plus_metadata` and direct hint leakage count of zero.

At 500 records:

- Airline: candidate@20 0.9325, candidate@50 0.9625, recall@5 0.8975, MRR 0.9349.
- Healthcare Admin: candidate@20 1.0000, candidate@50 1.0000, recall@5 0.9730, MRR 0.785567.
- Research AI: candidate@20 0.9440, candidate@50 1.0000, recall@5 0.901333, MRR 1.0000.
- Retail: candidate@20 0.9830, candidate@50 0.9910, recall@5 0.2650, MRR 0.164533.
- Finance: candidate@20 0.4370, candidate@50 0.8100, recall@5 0.2240, MRR 0.125867.

At 2,000 records:

- Airline: candidate@20 0.9280, candidate@50 0.964625, recall@5 0.889375, MRR 0.93415.
- Healthcare Admin: candidate@20 0.9640, candidate@50 0.9840, recall@5 0.9150, MRR 0.745767.
- Research AI: candidate@20 0.871318, candidate@50 0.948397, recall@5 0.752403, MRR 0.7791.
- Retail: candidate@20 0.913417, candidate@50 0.927917, recall@5 0.220167, MRR 0.161017.
- Finance: candidate@20 0.46125, candidate@50 0.820625, recall@5 0.25125, MRR 0.142458.

## Current Interpretation

This block improved canonical behavior for Research AI section materialization
and Finance 8-K filing-event selection, but it did not make the full retrieval
system SLO-ready.

Retail remains blocked because same-product near-duplicate review rows and seed
expansion rows create ambiguity that cannot be resolved reliably without better
non-leaking product/review-level metadata.

Finance remains blocked because most prompts still lack explicit period, metric,
and section cues. The canonical selector can improve cases where a filing-event
chunk is a candidate, but candidate retrieval is still weak for many finance
queries.

Research AI passes at 500 records but degrades at 2,000 records, which points to
candidate degradation and same-topic paper confusion at scale.

Airline is close to the top-5 target but still sits slightly below recall@5
0.90 at the 2,000-stage validation.

## Regeneration

Run:

```powershell
python scripts/phase3/repair_all_vertical_retrieval.py `
  --dataset-root data/scaleup_2000_full `
  --context-root data/generated/context_engineering `
  --slo-config configs/slo_targets.yaml `
  --output-root data/generated/context_engineering `
  --stage-sizes 500 2000 `
  --use-canonical-retrieval-keys
```

Then regenerate the standard SLO readiness report:

```powershell
python scripts/phase3/evaluate_slo_readiness.py `
  --slo-config configs/slo_targets.yaml `
  --retrieval-report data/generated/context_engineering/retrieval_evaluation_report.json `
  --quality-gate-report data/generated/context_engineering/retrieval_quality_gate_report.json `
  --output-root data/generated/context_engineering
```

## Next Step

Do not scale inference yet. The next retrieval work should focus on Retail
review-level disambiguation, Finance prompt/corpus metadata repair, and Research
AI candidate stability at 2,000 records.

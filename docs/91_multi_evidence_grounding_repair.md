# Multi-Evidence Grounding Repair

Block 30C improves multi-evidence citation completeness without changing
retrieval or evaluator rules.

## Changes

The shared generation prompt now requires the model to:

- silently classify every supplied evidence block as relevant or not relevant;
- cite all evidence blocks needed for the answer;
- recognize that some answers require multiple evidence IDs;
- include every relevant label used for any answer claim;
- name each emitted label in `citation_notes` and explain its support.

The final response remains exactly one generation-contract JSON object.

## Bounded Citation Repair

After the initial five-prompt generation, the unchanged evaluator checks
evidence match. One citation-only retry is eligible when:

- the initial generation contract is valid;
- required evidence is missing;
- the missing evidence family maps to a short label already present in the
  supplied context.

The repair receives the original prompt, previous JSON, allowed short labels,
and the evaluator-identified missing short label. It does not receive canonical
gold IDs. It must preserve the answer, confidence, and insufficient-evidence
state unless a minimal answer correction is needed for citation accuracy.

If required evidence is absent from the supplied top five, the retry is skipped
and classified as `required_evidence_not_in_supplied_context`. No citation is
invented.

This is an evaluator-assisted repair result. Initial-pass quality remains
reported separately and should be used when comparing unassisted model
behavior.

## Smoke Configuration

- model: `model6_gated` (`meta-llama/Llama-3.1-8B-Instruct`)
- provider: Novita through the Hugging Face router
- model5: skipped because complete token pricing remains unavailable
- prompts: 5, one per vertical
- memory mode: `mm2_hybrid_top5`
- retrieval: promoted Qdrant-backed baseline
- streaming: required
- maximum output: 128 tokens per request
- initial requests: 5
- citation-repair requests: 1
- GPU, vLLM, and SGLang: not used

## Results

| Metric | Initial pass | After bounded repair | Target |
| --- | ---: | ---: | ---: |
| JSON validity | 100% | 100% | strict |
| Contract validity | 100% | 100% | strict |
| Evidence-ID presence | 100% | 100% | strict |
| Evidence match | 60% | 80% | >=80% |
| Groundedness | 60% | 80% | >=80% |
| Safety violations | 0% | 0% | 0% |

Both requested targets passed after one repair request.

The Airline response initially cited `E1` and `E3`, both mapping to the same
policy family. The repair added supplied label `E2` and documented it as the
safety boundary. The factual answer remained unchanged.

## Remaining Failure

Healthcare Admin remains ungrounded:

- expected evidence families: `MCH-POL-001`, `MCH-POL-020`;
- supplied context covers `MCH-POL-001` but not `MCH-POL-020`;
- retry decision: `required_evidence_not_in_supplied_context`;
- emitted labels: `E1`, `E3`, both from the available policy family.

Failure classes:

- `missing_required_evidence_id`;
- `cited_partial_evidence_only`;
- `multi_evidence_under_citation`.

Fixing this last failure requires a retrieval/input-context correction, which
was explicitly outside Block 30C.

## Token and Cost Impact

Across five initial requests and one citation repair:

- input tokens: 7,669
- output tokens: 493
- total tokens: 8,162
- total API cost: `$0.00017803`
- cost per original prompt: `$0.000035606`
- cost per grounded answer: `$0.0000445075`
- mean cumulative end-to-end latency per original prompt: 1,432.282 ms

The repair overhead is included in these totals.

## Outputs

- `results/raw/phase4_grounding_repair_smoke_results.jsonl`
- `results/processed/phase4_grounding_repair_eval_report.json`
- `results/processed/phase4_grounding_repair_eval_summary.csv`

Additional local diagnostics:

- `results/processed/phase4_grounding_repair_initial_eval_report.json`
- `results/processed/phase4_grounding_repair_failure_report.json`
- `results/processed/phase4_grounding_repair_run_summary.json`

Generated execution outputs remain local and ignored.

## Commands

```powershell
python scripts/phase4/run_grounding_repair_smoke.py `
  --workload-path data/workloads/smoke_500/prompt_plus_metadata/mm2_hybrid_top5.jsonl `
  --input-path data/generated/phase4/grounding_repair_runner_input.jsonl `
  --output-path results/raw/phase4_grounding_repair_smoke_results.jsonl `
  --dataset-root data/scaleup_2000_full `
  --pricing-config configs/api_pricing.yaml `
  --max-new-tokens 128 `
  --allow-paid-api-call

python scripts/phase4/evaluate_generation_outputs.py `
  --results-path results/raw/phase4_grounding_repair_smoke_results.jsonl `
  --dataset-root data/scaleup_2000_full `
  --output-root results/processed `
  --report-name phase4_grounding_repair_eval_report.json `
  --summary-name phase4_grounding_repair_eval_summary.csv
```

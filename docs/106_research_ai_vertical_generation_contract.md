# B6R2 Research AI Vertical Generation Contract

Status: measured on June 16, 2026

B6R2 tested whether Research AI could clear the B6 quality blocker by using a
vertical-specific JSON contract instead of the shared five-field contract
directly in the model-facing prompt.

No gold data, evaluator semantics, promoted retrieval source, SLO thresholds,
mm4 workflow, or benchmark scale changed.

## Why This Was Needed

B6 showed that Research AI was the only blocking vertical:

- JSON validity: 82%
- contract validity: 80%
- evidence match: 76%
- groundedness: 74%
- truncation: 18%
- safety violations: 0

B6R1 reduced truncation by increasing the output budget, but did not pass. The
remaining failure pattern indicated that Research AI needed a compact schema
matched to research-paper answers, not more generic prompt text.

## Common Evaluation Mapping

The new registry in `src/inference_bench/generation_contract_registry.py`
validates Research AI-specific output and maps it back into the unchanged common
contract:

```json
{"answer":"...","evidence_ids":["E1"],"confidence":0.65,"insufficient_evidence":false,"citation_notes":"..."}
```

The evaluator still scores JSON validity, contract validity, evidence match,
groundedness, safety, truncation, and task success through the existing
`evaluate_generation_outputs.py` path. The vertical contract is therefore a
renderer/output-control change, not an evaluator weakening.

Raw Research AI model text is preserved separately in result rows as
`raw_generated_text`.

## Candidates

B6R2 tested five contract options at 224 and 320 maximum new tokens over the
same frozen 26 Research AI B6 failed/truncated/invalid rows.

| Candidate | Tokens | JSON | Contract | Evidence | Grounded | Truncation | Safety |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `research_ai_minimal_answer_v1` | 224 | 92.31% | 92.31% | 61.54% | 61.54% | 0.00% | 0 |
| `research_ai_minimal_answer_v1` | 320 | 92.31% | 92.31% | 61.54% | 61.54% | 0.00% | 0 |
| `research_ai_findings_v1` | 224 | 88.46% | 88.46% | 46.15% | 46.15% | 0.00% | 0 |
| `research_ai_findings_v1` | 320 | 88.46% | 88.46% | 42.31% | 42.31% | 0.00% | 0 |
| `research_ai_limitations_v1` | 224 | 96.15% | 96.15% | 80.77% | 80.77% | 0.00% | 0 |
| `research_ai_limitations_v1` | 320 | 96.15% | 96.15% | 80.77% | 80.77% | 0.00% | 0 |
| `research_ai_comparison_v1` | 224 | 88.46% | 73.08% | 46.15% | 46.15% | 0.00% | 0 |
| `research_ai_comparison_v1` | 320 | 88.46% | 73.08% | 46.15% | 46.15% | 0.00% | 0 |
| `research_ai_adaptive_v1` | 224 | 92.31% | 88.46% | 80.77% | 80.77% | 0.00% | 0 |
| `research_ai_adaptive_v1` | 320 | 92.31% | 88.46% | 73.08% | 73.08% | 0.00% | 0 |

The adaptive router deterministically selected the direct minimal-answer
contract for all 26 frozen rows after stripping generic output-contract and
internal planning text from the routing input.

## Selection Result

No candidate passed the targeted B6R2 gate:

- JSON target: at least 97%
- contract target: at least 97%
- evidence target: at least 85%
- groundedness target: at least 85%
- truncation target: no more than 2%
- safety target: zero violations

The strongest candidate was `research_ai_limitations_v1` at both 224 and 320
tokens. It eliminated truncation and kept safety at zero, but reached only
96.15% JSON/contract validity and 80.77% evidence/groundedness.

Decision:

```text
B6R2_BLOCKED
```

Because no targeted candidate passed, the full frozen B6 500-row rerun was not
triggered.

## B6 Versus B6R2

B6 remains the last full 500-row result:

- overall JSON: 95.4%
- overall contract: 94.8%
- overall evidence match: 91.2%
- overall groundedness: 90.8%
- safety violations: 0
- truncation: 4.6%

B6R2 is a targeted Research AI replay only. It does not replace B6 as a full
gate result.

## Readiness

The full-run readiness audit remains:

```text
NOT_READY
```

B6R2 does not clear the B6 quality blocker. A 1,000-prompt terminal run is not
allowed. RunPod remains blocked by missing hourly prices, missing measured
throughput multipliers, and missing external artifact sync/backup.

## Next Block

Recommended next block:

```text
B6R3_RESEARCH_AI_MODEL_CAPABILITY_COMPARISON
```

Freeze the B6R2 artifacts. The next controlled step should compare model
capability on the same 26-row Research AI replay set before any larger or
concurrent benchmark. Keep evaluator semantics, gold data, promoted retrieval,
mm4, SGLang, RunPod, and prompt count unchanged unless a separate block
explicitly authorizes them.

# Phase 4 Generation Contract Hardening

Block 24 hardens structured generation before GPU or vLLM benchmarking. It does
not change retrieval, the promoted benchmark, or evaluator acceptance rules.

## Shared Contract

All five verticals use the same output schema:

```json
{
  "answer": "string",
  "evidence_ids": ["E1"],
  "confidence": 0.8,
  "insufficient_evidence": false,
  "citation_notes": "string"
}
```

The prompt now explicitly requires one JSON object, forbids markdown and prose
outside the object, requires provided evidence labels, requires at least one
label for an answer, and reserves empty answers and evidence arrays for
insufficient-evidence responses. It does not include example answer wording for
the model to copy.

## Safe Parsing

The parser:

- extracts the first balanced JSON object from surrounding text;
- removes only simple trailing commas before `}` or `]`;
- detects an unclosed object or string as truncation;
- rejects missing fields, invalid types, invalid confidence, and unknown labels;
- never creates answer content or evidence IDs;
- records `parse_repair_applied`, `parse_error_type`, and
  `truncation_detected`.

Extraction from code fences is recorded as a repair. It is not hidden.

## Bounded Retry

The local HF smoke runner supports:

```text
--max-contract-retries 0|1
```

The default smoke setting is one retry. Setting it to zero preserves a pure
latency path.

An invalid response receives one compact correction request containing:

- the original invalid output;
- the exact validation error;
- the unchanged list of allowed evidence labels;
- the five required fields and type constraints.

The retry is a structural correction task. It does not receive gold IDs or
source IDs and does not introduce facts. If initial output is truncated, the
retry token budget increases up to the local smoke cap of 256 tokens.

## Before And After

The same five-record `mm2_hybrid_top5` smoke set and
`Qwen/Qwen2.5-0.5B-Instruct` were used.

| Metric | Block 23 | Block 24 | Target |
| --- | ---: | ---: | ---: |
| JSON validity | 80% | 100% | >=95% |
| Contract validity | 60% | 80% | >=95% |
| Evidence-ID presence | 80% | 100% | >=95% |
| Full evidence match | 60% | 40% | >=85% |
| Deterministic groundedness | 40% | 20% | >=80% |
| Truncation rate | 20% | 0% | lower is better |
| Parse repair rate | 0% | 100% | diagnostic |
| Retry rate | not available | 40% | diagnostic |

The JSON and evidence-presence targets passed. Contract validity, evidence
match, and groundedness did not.

## Smoke Details

- Generation success: 5/5
- Contract-valid rows: 4/5
- Retry rows: 2/5
- Total retries: 2
- Successful retries: 1
- Parse repairs: 5/5
- Truncated outputs: 0/5
- Mean end-to-end latency: 146.223 seconds per prompt
- Median end-to-end latency: 172.321 seconds per prompt
- Total input tokens across attempts: 7,456
- Total output tokens across attempts: 574
- Paid API calls: 0
- GPU experiments: 0

## Remaining Issues

The remaining failures are model behavior, not retrieval or evaluator
plumbing:

- The model wrapped every response in markdown fences despite the explicit
  instruction. Safe extraction recovered the JSON and recorded the repair.
- Retail repeatedly emitted `confidence: 5.0`, apparently copying the product
  rating into the confidence field. The retry correctly remained invalid.
- Airline and Healthcare cited only `E1` although distinct required support was
  available under `E2`.
- Research AI became contract-valid after retry but still cited only `E1`.
- The evaluator remained strict; no evidence IDs or answer content were added
  after generation.

The 0.5B model is not sufficiently reliable for a strict grounded-output gate.
Before performance benchmarking, the project should evaluate a stronger local
model or backend-supported constrained JSON decoding. Latency comparisons
should keep retries disabled, while quality runs may retain one retry and report
its added tokens and latency.

## Commands

```powershell
pytest tests/test_phase4_generation_contract.py
pytest tests/test_phase4_generation_contract_hardening.py
pytest tests/test_phase4_local_hf_smoke.py

python scripts/phase4/export_runner_smoke_workload.py `
  --workload-path data/generated/phase4/generation_contract_mixed_workload.jsonl `
  --output-path data/generated/phase4/generation_contract_runner_input.jsonl `
  --limit 5

python scripts/phase4/run_local_hf_smoke.py `
  --input-path data/generated/phase4/generation_contract_runner_input.jsonl `
  --output-path results/raw/phase4_generation_contract_hardened_hf_smoke.jsonl `
  --model-alias model1_0_5b `
  --limit 5 `
  --max-new-tokens 256 `
  --max-contract-retries 1

python scripts/phase4/evaluate_generation_outputs.py `
  --results-path results/raw/phase4_generation_contract_hardened_hf_smoke.jsonl `
  --dataset-root data/scaleup_2000_full `
  --output-root results/processed `
  --report-name phase4_generation_contract_hardened_eval_report.json `
  --summary-name phase4_generation_contract_hardened_eval_summary.csv
```

Generated raw and processed result files remain local under the repository
ignore policy.


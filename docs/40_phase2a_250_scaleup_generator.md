# Phase 2A-9 250-Scale Generator Foundation

Phase 2A-9 adds the deterministic framework for moving from the QA-clean
40-record seed per vertical toward the first 250-record checkpoint. This is
generation scaffolding only: it does not build RAG, retrieval indexes,
embeddings, prompt assembly, model calls, GPU runs, or benchmark inference.
In short, this is a no RAG, no inference, no embeddings step.

The current checkpoint target is 250 prompts per vertical, or 1,250 prompts
total across the five Phase 2A verticals. Generated expansion candidates remain
local under ignored `data/generated/phase2a/` paths until they pass review.

## Modes

Dry-run mode reads the scale-up plan and the latest cross-vertical QA report,
then prints planned counts, readiness, source requirements, and blockers.

```powershell
python scripts/phase2/generate_phase2a_scaleup.py --dry-run
```

Plan mode writes per-vertical manifests only. It does not write prompt, KB, or
gold expansion records.

```powershell
python scripts/phase2/generate_phase2a_scaleup.py --generate-plan --target-per-vertical 250
```

Vertical generation mode writes local candidate files for one selected vertical.
The first implemented pilot is Airline because the seed data and policy KB are
fully deterministic and synthetic/public-inspired.

```powershell
python scripts/phase2/generate_phase2a_scaleup.py --generate-vertical --vertical airline --target-per-vertical 250
```

## Generated Local Files

The airline pilot writes local ignored files:

- `data/generated/phase2a/scaleup/airline/airline_prompts_250.jsonl`
- `data/generated/phase2a/scaleup/airline/airline_gold_250.jsonl`
- `data/generated/phase2a/scaleup/airline/airline_kb_250.jsonl`
- `data/generated/phase2a/scaleup_reports/airline_scaleup_250_report.json`

Plan manifests are written under:

- `data/generated/phase2a/scaleup_reports/`

These outputs are intentionally not committed in this patch.

## Distribution Planning

The planner produces exact integer targets for:

- expected status
- task type
- output format
- difficulty
- source and KB coverage requirements

For the Airline 250 pilot, the planned status mix is:

| Status | Count |
| --- | ---: |
| answer | 225 |
| escalate | 20 |
| spam_or_fraud | 5 |

Other verticals receive manifests and readiness checks first. Their full
generation should be added incrementally after the airline candidate files are
reviewed.

## Quality Gates

Before any expanded dataset is committed, generated records must pass:

- prompt/gold alignment
- one gold record per prompt
- unique prompt IDs
- evidence IDs that exist in KB records
- status distribution checks
- private-path, secret, and raw identifier hygiene scans
- cross-vertical QA after candidates are promoted from generated output

The generator report records critical issues, warnings, blockers, status counts,
task counts, output-format counts, and next-step guidance.

## Source Readiness

The dry-run and plan modes inspect committed seed prompt, KB, and gold files for
all five verticals. They also record local artifact readiness where future full
generation will need additional sources, such as SEC manifests, Research AI
paper sections, and Retail sampled review/metadata files.

Only the Airline vertical is enabled for local candidate generation in this
foundation patch. Other verticals should use `--generate-plan` until their
dedicated deterministic expansion logic is implemented.

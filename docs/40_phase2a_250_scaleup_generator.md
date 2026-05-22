# Phase 2A-9 Scale-Up Generator Foundation

Phase 2A-9 adds the deterministic framework for planning larger Phase 2A
datasets and generating reviewed local candidates where a vertical generator is
explicitly implemented. This is generation scaffolding only: it does not build
RAG, retrieval indexes, embeddings, prompt assembly, model calls, GPU runs, or
benchmark inference. In short, this is a no RAG, no inference, no embeddings
step.

Generated expansion candidates remain local under ignored
`data/generated/phase2a/` paths until they pass review and are intentionally
promoted.

## Supported Targets

The generator is target-aware for every approved checkpoint:

| Target per vertical | Total across five verticals | Checkpoint | Role |
| ---: | ---: | --- | --- |
| 250 | 1,250 | `checkpoint_250` | QA-scale deterministic dataset |
| 1,000 | 5,000 | `checkpoint_1000` | Pilot benchmark dataset |
| 2,000 | 10,000 | `checkpoint_2000` | Near-term main target |
| 4,000 | 20,000 | `checkpoint_4000` | GPU stress tier scaffold |
| 5,000 | 25,000 | `checkpoint_5000` | Maximum expanded capacity |

Large targets are planning/scaffolding targets unless a vertical implementation
explicitly supports generation at that size. This prevents accidental large,
low-quality duplication.

QA readiness is not the same as generation readiness. A vertical can be
QA-clean and have source files available while still being planning-only because
record generation has not been implemented for that vertical and target pair.
Planning manifests therefore report QA readiness, source-artifact readiness,
generation implementation readiness, and actual generation readiness separately.

## Modes

Dry-run mode reads the scale-up plan and latest cross-vertical QA report, then
prints planned counts, checkpoint mapping, readiness, source requirements,
estimated KB range, implemented generation status, and blockers.

```powershell
python scripts/phase2/generate_phase2a_scaleup.py --dry-run --target-per-vertical 250
python scripts/phase2/generate_phase2a_scaleup.py --dry-run --target-per-vertical 2000
```

Plan mode writes per-vertical manifests only. It does not write prompt, KB, or
gold expansion records.

```powershell
python scripts/phase2/generate_phase2a_scaleup.py --generate-plan --target-per-vertical 2000
python scripts/phase2/generate_phase2a_scaleup.py --generate-plan --target-per-vertical 5000
```

Vertical generation mode writes local candidate files for one selected vertical
only when that target has explicit implementation support. The first
implemented pilot remains Airline at 250 records because the seed data and
policy KB are deterministic and synthetic/public-inspired.

```powershell
python scripts/phase2/generate_phase2a_scaleup.py --generate-vertical --vertical airline --target-per-vertical 250
```

Requests such as Airline at 2,000 records are intentionally blocked until a
future patch implements and reviews that generator target.

## Generated Local Files

The Airline 250 pilot writes local ignored files:

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
- expected KB range
- source and artifact requirements

Status and output-format distributions are percentage-based and rounded so
counts always sum exactly to the requested target.

## Safety Gates

Before any expanded dataset is committed, generated records must pass:

- prompt/gold alignment
- one gold record per prompt
- unique prompt IDs
- evidence IDs that exist in KB records
- status distribution checks
- private-path, secret, and raw identifier hygiene scans
- review of the previous checkpoint before generating the next checkpoint
- cross-vertical QA after candidates are promoted from generated output

`--allow-large-local-generation` exists for future explicit implementations. It
does not bypass target support or quality gates.

Planning-only manifests include blockers such as
`generation_not_implemented_for_vertical_target` and, for larger targets,
`large_target_generation_requires_checkpoint_review`. These blockers do not
cause `--generate-plan` to fail; they make it clear that no generated records
should be expected until implementation and prior checkpoint review are complete.

## Source Readiness

The dry-run and plan modes inspect committed seed prompt, KB, and gold files for
all five verticals. They also record local artifact readiness where future full
generation will need additional sources, such as SEC manifests, Research AI
paper sections, and Retail sampled review/metadata files.

Only the Airline 250 target is enabled for local candidate generation in this
foundation patch. Other verticals and larger targets should use `--generate-plan`
until their dedicated deterministic expansion logic is implemented.

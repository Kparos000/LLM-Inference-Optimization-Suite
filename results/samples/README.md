# Curated Sample Artifacts

## Purpose

This directory stores selected benchmark artifacts that have been reviewed and intentionally promoted for documentation, reporting, publication notes, or reproducibility examples.

## What Belongs Here

- Small representative CSV metric samples.
- Small representative JSONL trace samples.
- Selected figures used in README, reports, or paper notes.
- System metadata samples that support reproducibility context.

## What Does Not Belong Here

- Arbitrary raw benchmark outputs.
- Full benchmark runs that can be regenerated or stored externally.
- Large generated datasets, logs, caches, or model outputs.
- Unreviewed local artifacts copied directly from `results/raw` or `results/figures`.

## Promotion Criteria

Raw generated outputs remain ignored by default. Only selected, reviewed, non-sensitive artifacts should be copied into `results/samples` when they support the README, reports, paper notes, or reproducibility.

## Privacy/Secrets Rule

Do not promote tokens, secrets, private data, credentials, `.env` files, or artifacts containing accidental local-only paths.

## Reproducibility Note

Promoted samples should be tied to documented commands, workload definitions, model identifiers, and system metadata where practical. Samples are illustrative artifacts, not a replacement for full reproducible benchmark runs.

# Result Promotion Policy

## Purpose

This policy defines how selected benchmark artifacts may be promoted into version control while keeping generated outputs ignored by default.

## Default Rule

Generated artifacts under `results/raw`, `results/processed`, and `results/figures` are ignored by default. These files can vary by machine, dependency version, runtime, and workload settings, and they should not be committed unless deliberately reviewed and promoted.

## When To Promote Selected Artifacts

Promote only small, representative artifacts that support a README, report, paper note, reproducibility example, or regression comparison. Expanded workload comparison CSVs and workload-specific prompt traces are good candidates when they summarize reviewed Hugging Face or vLLM baseline runs. Full raw runs and large benchmark datasets should be stored externally or summarized.

## Promotion Workflow

Run:

```text
powershell -ExecutionPolicy Bypass -File scripts/promote_sample_artifacts.ps1
```

The script copies a fixed allowlist of known benchmark outputs into `results/samples`. Missing files are reported without failing.

## Review Checklist

Before committing sample artifacts, verify:

- No secrets
- No tokens
- No private data
- No accidental local-only paths
- No accidental host identifiers
- No unsafe generated content
- No personal notes or drafts
- Artifact supports a report, README, or paper note
- Artifact is small enough for GitHub

vLLM baseline artifacts should be promoted only after review for secrets, private paths, token values, accidental host identifiers, unsafe generated content, excessive file size, and personal notes or drafts.

## Storage Note

Large benchmark datasets, full raw runs, and bulky plot collections should be stored externally or summarized in documentation rather than committed to the repository.

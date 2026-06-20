# Artifact Sync And Long-Run Recovery

Status: implemented on June 19, 2026

Phase 1E adds local production-grade long-run safety before any 1,000+ prompt,
RunPod, concurrency, or final matrix execution.

## Manifest

`src/inference_bench/run_manifest.py` now supports first-class production run
fields in addition to the older Phase 4 smoke fields:

- run/config/model/runtime/backend/hardware/provider identity;
- concurrency, traffic profile, prompt count, and memory mode;
- git commit, dataset/workload hash, and config hash;
- start/update/completion timestamps;
- `initialized`, `running`, `partial`, `completed`, and `failed` statuses;
- completed, failed, and expected row counts;
- raw, processed, telemetry, checkpoint, log, and backup artifact paths.

Completed manifests cannot hide partial runs: if `status` is `completed`, the
manifest row counts must cover `expected_count`.

## Checkpoint And Resume

`src/inference_bench/checkpoint_resume.py` provides deterministic prompt-level
recovery helpers:

- completed and failed `prompt_id` tracking;
- resume plans from checkpoint JSON and partial raw JSONL;
- duplicate prompt-row prevention by default;
- failed-row persistence;
- clear resume reports with skipped and pending prompt IDs.

## Artifact Sync

`src/inference_bench/artifact_sync.py` implements local backup first:

- default backup root: `backups/`;
- raw JSONL, manifest, telemetry, processed reports, checkpoints, failed rows,
  and logs can be copied to a run-scoped backup folder;
- sync can run at start, every N requests, at run end, and on failure;
- the interface has a provider field so S3, R2, or Google Drive can be added
  later without changing the runner call sites.

## Verification

Backup verification checks that:

- required backup files exist;
- required files are non-empty where expected;
- source and backup hashes match;
- manifest status and row counts match raw JSONL rows;
- a backup completeness score is reported.

## Dry Run

The dry-run command is:

```text
python scripts/phase4/test_long_run_recovery_dry_run.py
```

It simulates 20 prompts, writes 10 rows, interrupts, resumes the remaining 10,
persists one failed row, syncs local artifacts, verifies backup hashes, and
writes:

- `results/processed/long_run_recovery_dry_run_report.json`;
- `results/processed/long_run_recovery_dry_run_summary.csv`.

The measured dry run passed with 20 final rows, 10 skipped on resume, one
persisted failed row, no duplicate prompt IDs, and backup completeness score
`1.0`.

## RunPod Gate

Long self-hosted/RunPod-style runs remain blocked unless:

- artifact sync is configured;
- checkpoint/resume is enabled;
- GPU hourly price is registered;
- first-class manifest writing is enabled;
- partial runs cannot be marked complete;
- backup verification passes on a dry run.

This block does not run model inference, change gold data, modify promoted
retrieval, or authorize larger benchmarks.

# Resumable Benchmarking Plan

## Purpose

This plan defines the architecture needed for large-scale inference experiments that require progress feedback, logs, chunked saves, checkpointing, resume behavior, and persistent storage.

## Problem Statement

Large inference runs can lose results if outputs are only written at the end. Larger model sweeps and concurrency tests need periodic flushing and resumability so partial results survive interruptions and can be inspected while a run is still in progress.

For serious GPU runs, the benchmark runner should treat result persistence as part of the execution path, not as a final cleanup step.

## Failure Modes

- GPU instance crash
- Out-of-memory failure
- Pod shutdown
- Credit exhaustion
- Network disconnect
- vLLM server crash
- Disk full
- Interrupted SSH session

## Required Runner Features

- `chunk-size` option
- Checkpoint path
- Resume mode
- Log path
- Progress interval
- Append-safe CSV writing
- Append-safe JSONL generation writing
- Metadata update after each chunk

## Chunking Strategy

Default chunk size: 100 prompts.

The runner should flush results after every chunk, save a checkpoint after every chunk, record completed prompt IDs, and continue on per-prompt failures where possible. Chunking should be independent of concurrency: a chunk defines persistence boundaries, while concurrency defines how many requests may be in flight inside that boundary.

## Checkpoint Format

The checkpoint should be a JSON document with these expected fields:

- `run_id`
- `workload_path`
- `model`
- `backend`
- `optimization`
- `concurrency`
- `chunk_size`
- `total_prompts`
- `completed_prompt_ids`
- `success_count`
- `failure_count`
- `started_at_utc`
- `last_updated_utc`
- `output_path`
- `generation_output_path`
- `metadata_path`
- `log_path`

The checkpoint should be written atomically where practical, for example by writing a temporary file and replacing the previous checkpoint after a successful write.

## Resume Behavior

Resume mode should load the checkpoint, validate that the run configuration matches the requested run, skip completed prompt IDs, and append only new result rows. If the requested workload, model, backend, optimization, or output paths do not match the checkpoint, the runner should stop with a clear error.

Resume should preserve failed prompt records already written to disk. A later retry mode can be added separately for re-running failed prompt IDs.

## Progress Feedback

The runner should print progress at a configurable interval and after each chunk. A progress message should include processed prompts, chunk index, success/failure counts, elapsed time, aggregate throughput, and checkpoint status.

Example:

```text
processed=700/5000 chunk=7/50 success=699 failure=1 elapsed_seconds=83.2 aggregate_requests_per_second=8.4 checkpoint_saved=true
```

## Inference Logs

Inference logs should capture run lifecycle events, chunk boundaries, checkpoint writes, per-chunk summaries, server or client errors, and final run metadata. Logs should not include secrets, credentials, private paths, or full generated outputs unless the log is intentionally treated as a raw artifact outside the public repository.

## Persistent Storage and RunPod Network Volumes

Serious experiments should use a RunPod network volume or equivalent persistent storage. Raw and chunked outputs should be stored under persistent results paths so they survive pod restarts or replacement.

Recommended storage approach:

- Use RunPod network volume for serious experiments.
- Store raw/chunked outputs under persistent results paths.
- Keep checkpoints and metadata next to the raw result CSV and generation JSONL files.
- Promote only curated samples to the public repo.

## Artifact Promotion Policy

Raw generated outputs, logs, checkpoints, and large result files should remain outside version control unless they are intentionally reviewed and promoted as curated samples. Public promotion should follow the result promotion policy and include review for secrets, private paths, host identifiers, unsafe generated content, excessive size, and private notes or drafts.

## Example Large-Run Calculation

A serious model sweep may include 100,000 total prompts across 4 models and 5 workloads:

- 5,000 prompts per model-workload pair.
- 4 models x 5 workloads = 20 model-workload pairs.
- 20 pairs x 5,000 prompts = 100,000 total prompts.

With 5 concurrency levels, this creates 100 benchmark configurations. With chunk size 100, this creates 5,000 checkpointed chunks across the full sweep.

## Implementation Roadmap

1. Add append-safe CSV and JSONL writers.
2. Add checkpoint read/write utilities with configuration validation.
3. Add chunk-size, checkpoint path, resume mode, log path, and progress interval options to load runners.
4. Update run metadata after each completed chunk.
5. Add resume tests for completed prompt skipping and configuration mismatch handling.
6. Add large-run workflow scripts that target persistent storage paths.
7. Promote only reviewed sample outputs after the run completes.

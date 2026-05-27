"""Run-safety and long-run logging audit for Phase 3 hardening."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    """Return an ISO UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def artifact_status(path: str) -> dict[str, Any]:
    """Return a simple artifact presence payload."""

    artifact_path = Path(path)
    return {
        "path": path,
        "exists": artifact_path.exists(),
        "non_empty": artifact_path.exists()
        and (artifact_path.is_dir() or artifact_path.stat().st_size > 0),
    }


def run_safety_rows() -> list[dict[str, Any]]:
    """Return static run-safety audit rows grounded in current repo artifacts."""

    return [
        {
            "area": "openai_load_runner_checkpointing",
            "artifact": "src/inference_bench/runners/openai_load_runner.py",
            "current_capability": (
                "Supports chunk_size, checkpoint_path, resume, completed_prompt_ids, "
                "atomic checkpoint writes, and duplicate prompt skipping."
            ),
            "reusable": True,
            "phase4_gap": (
                "Needs adapter from Phase 3 WorkloadRecord JSONL to runner WorkloadItem."
            ),
            "phase5_gap": (
                "Needs GPU run manifests to include memory_mode, dataset split, and telemetry IDs."
            ),
            "priority": "high",
        },
        {
            "area": "chunked_result_persistence",
            "artifact": "src/inference_bench/runners/openai_load_runner.py",
            "current_capability": (
                "Chunked mode appends result CSV and generation JSONL after each chunk."
            ),
            "reusable": True,
            "phase4_gap": "Extend chunked writing tests to Phase 3 memory-mode workloads.",
            "phase5_gap": "Add durable per-run directories for main GPU runs.",
            "priority": "high",
        },
        {
            "area": "progress_logging",
            "artifact": "src/inference_bench/runners/openai_load_runner.py",
            "current_capability": (
                "Progress messages include processed count, chunk number, success/failure counts, "
                "elapsed seconds, aggregate request rate, and checkpoint_saved."
            ),
            "reusable": True,
            "phase4_gap": (
                "Logs do not yet include memory_mode, dataset_split, prompt vertical, or "
                "retrieval mode."
            ),
            "phase5_gap": "Add structured JSONL logs for long GPU/API runs.",
            "priority": "high",
        },
        {
            "area": "run_metadata",
            "artifact": "src/inference_bench/runners/openai_load_runner.py",
            "current_capability": (
                "Run metadata captures run_id, workload path, backend, model, optimization, "
                "concurrency, token settings, stream mode, success/failure counts, and wall clock."
            ),
            "reusable": True,
            "phase4_gap": "Add memory_mode, retrieval_backend_label, and context_token fields.",
            "phase5_gap": "Attach GPU telemetry file paths and hardware IDs.",
            "priority": "high",
        },
        {
            "area": "failure_tracking",
            "artifact": "src/inference_bench/schema.py",
            "current_capability": (
                "BenchmarkResult and GenerationRecord capture success and error_message per prompt."
            ),
            "reusable": True,
            "phase4_gap": (
                "Add evaluator failure categories and prompt/gold join failure reporting."
            ),
            "phase5_gap": (
                "Add retry classification for backend timeout, OOM, and validation failures."
            ),
            "priority": "medium",
        },
        {
            "area": "hf_runner_resume",
            "artifact": "src/inference_bench/runners",
            "current_capability": (
                "HF runner foundation exists, but checkpoint/resume is not currently equivalent "
                "to openai-load-run chunked mode."
            ),
            "reusable": False,
            "phase4_gap": "Decide whether Phase 4 HF plumbing needs chunked resume support.",
            "phase5_gap": "Main GPU runs should use a resumable path before large sweeps.",
            "priority": "medium",
        },
        {
            "area": "telemetry_logging",
            "artifact": "src/inference_bench/system_info.py",
            "current_capability": (
                "System metadata capture exists for reproducibility, but live GPU telemetry is "
                "not implemented."
            ),
            "reusable": True,
            "phase4_gap": "Record workload/run metadata alongside local smoke outputs.",
            "phase5_gap": "Add nvidia-smi/pynvml/DCGM telemetry sampling for GPU runs.",
            "priority": "high",
        },
        {
            "area": "structured_logs",
            "artifact": "results/raw",
            "current_capability": (
                "CSV, JSONL generation records, checkpoint JSON, metadata JSON, and text logs "
                "are supported in parts of the harness."
            ),
            "reusable": True,
            "phase4_gap": "Standardize per-run folder layout and log naming.",
            "phase5_gap": "Use append-only JSONL logs for dashboard ingestion.",
            "priority": "medium",
        },
    ]


def build_run_safety_audit() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the run-safety audit report and summary rows."""

    rows = run_safety_rows()
    report = {
        "generated_at_utc": utc_now(),
        "no_model_inference_triggered": True,
        "no_gpu_work_triggered": True,
        "audit_scope": "inference checkpointing, chunked persistence, resume, and logging",
        "terminology_note": (
            "This project is not training or fine-tuning models in Phase 3. "
            "The relevant safety controls are inference checkpointing, chunked result "
            "persistence, resume support, and long-run logging."
        ),
        "artifacts_inspected": {
            "openai_load_runner": artifact_status(
                "src/inference_bench/runners/openai_load_runner.py"
            ),
            "output_records": artifact_status("src/inference_bench/output_records.py"),
            "schema": artifact_status("src/inference_bench/schema.py"),
            "system_info": artifact_status("src/inference_bench/system_info.py"),
            "results_raw": artifact_status("results/raw"),
            "results_processed": artifact_status("results/processed"),
        },
        "already_exists": [
            "OpenAI-compatible load runner chunking",
            "checkpoint JSON with completed_prompt_ids",
            "resume mode that skips completed prompts",
            "append-mode result CSV and generation JSONL writes",
            "run metadata JSON",
            "progress text logs",
            "per-prompt success/error fields",
        ],
        "missing_before_phase4": [
            "Phase 3 WorkloadRecord to runner WorkloadItem adapter",
            "memory_mode and dataset_split in run metadata/logs",
            "standardized per-run output directory convention",
            "batch evaluator output persistence",
        ],
        "missing_before_main_gpu_experiments": [
            "live GPU telemetry sampling",
            "structured JSONL run logs",
            "backend OOM/timeout retry classification",
            "resume coverage for every backend path selected for main experiments",
        ],
        "recommended_phase4_tasks": [
            "Add a workload adapter for data/workloads/smoke_500/mm*.jsonl.",
            "Run mock plumbing with mm0 and mm2 workloads before HF/vLLM.",
            "Extend run metadata with memory_mode, vertical counts, and context token summary.",
            "Add evaluator CLI over generation JSONL joined by prompt_id.",
        ],
        "summary": rows,
    }
    return report, rows

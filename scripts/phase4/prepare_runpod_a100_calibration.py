"""Prepare local A100 SXM RunPod calibration manifests and readiness report."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference_bench.api_load_probe import load_probe_environment  # noqa: E402
from inference_bench.calibration_manifest import (  # noqa: E402
    build_calibration_manifest,
    calibration_readiness_verdict,
    load_runpod_calibration_profiles,
    validate_calibration_profile,
    write_calibration_manifest,
)

DEFAULT_MANIFEST_100 = "results/processed/a100_sxm_calibration_manifest_100.json"
DEFAULT_MANIFEST_200 = "results/processed/a100_sxm_calibration_manifest_200.json"
DEFAULT_READINESS = "results/processed/a100_sxm_calibration_readiness_report.json"


def _load_backup_dry_run_status(repo_root: Path) -> dict[str, object]:
    path = repo_root / "results/processed/long_run_recovery_dry_run_report.json"
    if not path.exists():
        return {
            "report_path": str(path.relative_to(repo_root)),
            "passed": False,
            "reason": "long_run_recovery_dry_run_report_missing",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    verification = payload.get("backup_verification")
    passed = (
        bool(verification.get("passed"))
        if isinstance(verification, dict)
        else bool(payload.get("backup_verification_passed"))
    )
    return {
        "report_path": str(path.relative_to(repo_root)),
        "passed": passed,
        "backup_verification": verification if isinstance(verification, dict) else None,
    }


def _artifact_paths(run_id: str) -> dict[str, str]:
    return {
        "runner_input": f"data/generated/phase4/{run_id}_runner_input.jsonl",
        "raw_results": f"results/raw/{run_id}_results.jsonl",
        "manifest": f"results/raw/{run_id}_manifest.json",
        "gpu_telemetry": f"results/raw/{run_id}_gpu_telemetry.jsonl",
        "processed_report": f"results/processed/{run_id}_eval_report.json",
        "artifact_sync_report": f"results/processed/{run_id}_artifact_sync_report.json",
    }


def _cost_examples(hourly_price: float | None) -> dict[str, float | None]:
    if hourly_price is None:
        return {"one_hour": None, "two_hours": None, "four_hours": None}
    return {
        "one_hour": hourly_price,
        "two_hours": hourly_price * 2.0,
        "four_hours": hourly_price * 4.0,
    }


def build_a100_calibration_package(
    *,
    repo_root: str | Path = ROOT,
    env_path: str | Path = ".env",
    manifest_100_path: str | Path = DEFAULT_MANIFEST_100,
    manifest_200_path: str | Path = DEFAULT_MANIFEST_200,
    readiness_path: str | Path = DEFAULT_READINESS,
) -> dict[str, Any]:
    """Create local dry-run calibration manifests without contacting RunPod."""

    root = Path(repo_root)
    profiles = load_runpod_calibration_profiles(root / "configs/runpod_calibration_profiles.yaml")
    profile = profiles["A100_SXM_CALIBRATION"]
    backup_status = _load_backup_dry_run_status(root)
    validation = validate_calibration_profile(
        profile,
        models_path=root / "configs/models.yaml",
        runtime_registry_path=root / "configs/runtime_engines.yaml",
        gpu_price_registry_path=root / "configs/gpu_prices.yaml",
    )
    readiness = calibration_readiness_verdict(
        profile=profile,
        artifact_sync_enabled=(root / "src/inference_bench/artifact_sync.py").exists(),
        checkpoint_resume_enabled=(root / "src/inference_bench/checkpoint_resume.py").exists(),
        manifest_enabled=(root / "src/inference_bench/run_manifest.py").exists(),
        runtime_profile_valid=bool(validation["runtime_profile_valid"]),
        gpu_price_registered=bool(validation["gpu_price_registered"]),
        backup_verification_dry_run_passed=bool(backup_status["passed"]),
    )
    manifests: dict[int, dict[str, object]] = {}
    output_paths = {
        100: root / manifest_100_path,
        200: root / manifest_200_path,
    }
    for prompt_count, output_path in output_paths.items():
        run_id = f"a100_sxm_model2_3b_mm2_c1_{prompt_count}"
        manifest = build_calibration_manifest(
            profile=profile,
            model_alias="model2_3b",
            memory_mode="mm2_hybrid_top5",
            concurrency=1,
            prompt_count=prompt_count,
            artifact_paths=_artifact_paths(run_id),
            repo_root=root,
            status="planned",
        )
        write_calibration_manifest(manifest, output_path)
        manifests[prompt_count] = manifest.to_dict()

    environment = load_probe_environment(env_path=root / env_path)
    runpod_ssh_host = environment.get("RUNPOD_SSH_HOST") or os.environ.get("RUNPOD_SSH_HOST")
    report: dict[str, Any] = {
        "package": "a100_sxm_runpod_calibration",
        "live_runpod_calibration_allowed": bool(runpod_ssh_host),
        "live_runpod_blocked_reason": None if runpod_ssh_host else "RUNPOD_SSH_HOST_missing",
        "profile": profile.to_dict(),
        "validation": validation,
        "readiness": readiness,
        "backup_dry_run": backup_status,
        "cost_examples_usd": _cost_examples(profile.hourly_price),
        "calibration_plan": {
            "model_alias": "model2_3b",
            "model_id": "Qwen/Qwen2.5-3B-Instruct",
            "runtime": "vllm",
            "memory_mode": "mm2_hybrid_top5",
            "prompt_counts": [100, 200],
            "concurrency": [1, 4, 8, 16, 32],
            "traffic_profiles": ["online_low_latency", "offline_throughput"],
            "artifact_sync": "enabled",
            "checkpoint_resume": "enabled",
            "manifest": "enabled",
            "gpu_telemetry": "enabled",
        },
        "manifest_paths": {
            "100": str((root / manifest_100_path).relative_to(root)),
            "200": str((root / manifest_200_path).relative_to(root)),
        },
        "manifests": manifests,
    }
    output = root / readiness_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Prepare local A100 SXM RunPod calibration manifests and readiness report."
    )
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--manifest-100", default=DEFAULT_MANIFEST_100)
    parser.add_argument("--manifest-200", default=DEFAULT_MANIFEST_200)
    parser.add_argument("--readiness", default=DEFAULT_READINESS)
    return parser


def main() -> int:
    """Generate the package and print the readiness report."""

    args = build_parser().parse_args()
    report = build_a100_calibration_package(
        repo_root=args.repo_root,
        env_path=args.env_file,
        manifest_100_path=args.manifest_100,
        manifest_200_path=args.manifest_200,
        readiness_path=args.readiness,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

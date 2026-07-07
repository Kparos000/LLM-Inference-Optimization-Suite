from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest


def _load_runner() -> Any:
    script = Path("scripts/phase4/run_controlled_final_simulation.py")
    spec = spec_from_file_location("run_controlled_final_simulation_full", script)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(runner: Any, tmp_path: Path) -> Any:
    args = runner.build_parser().parse_args(["--run-full"])
    args.prompt_count_per_vertical = 1
    args.raw_results_path = str(tmp_path / "results/raw/results.jsonl")
    args.manifest_path = str(tmp_path / "results/raw/manifest.json")
    args.gpu_telemetry_path = str(tmp_path / "results/raw/gpu.jsonl")
    args.checkpoint_path = str(tmp_path / "results/raw/checkpoint.json")
    args.matrix_path = str(tmp_path / "data/matrix.jsonl")
    args.eval_report_path = str(tmp_path / "results/processed/eval.json")
    args.eval_summary_path = str(tmp_path / "results/processed/eval.csv")
    args.engine_comparison_path = str(tmp_path / "results/processed/engine.csv")
    args.memory_comparison_path = str(tmp_path / "results/processed/memory.csv")
    args.concurrency_comparison_path = str(tmp_path / "results/processed/concurrency.csv")
    args.api_comparison_path = str(tmp_path / "results/processed/api.csv")
    args.api_vs_self_hosted_comparison_path = str(tmp_path / "results/processed/api_vs_gpu.csv")
    args.model_comparison_path = str(tmp_path / "results/processed/model.csv")
    args.slo_report_path = str(tmp_path / "results/processed/slo.json")
    args.slo_summary_path = str(tmp_path / "results/processed/slo.csv")
    args.cost_report_path = str(tmp_path / "results/processed/cost.json")
    args.artifact_sync_report_path = str(tmp_path / "results/processed/sync.json")
    args.post_run_automation_report_path = str(tmp_path / "results/processed/post.json")
    args.plotting_dataset_path = str(tmp_path / "results/processed/plotting.csv")
    args.backup_root = str(tmp_path / "backups")
    return args


def _ready_gates() -> dict[str, Any]:
    return {
        "full_simulation_allowed": True,
        "checks": {
            "vllm_model3_7b": {"status": "SMOKE_READY", "reason": "ok"},
            "sglang_model3_7b": {"status": "SMOKE_READY", "reason": "ok"},
            "api_model6_gated": {"status": "SMOKE_READY", "reason": "ok"},
            "mm4_bounded_agentic": {"status": "SMOKE_READY", "reason": "ok"},
        },
    }


def _blocked_gates() -> dict[str, Any]:
    gates = _ready_gates()
    gates["full_simulation_allowed"] = False
    gates["checks"]["api_model6_gated"] = {"status": "BLOCKED", "reason": "missing"}
    return gates


def _patch_small_success(monkeypatch: pytest.MonkeyPatch, runner: Any) -> None:
    monkeypatch.setattr(runner, "DEFAULT_PROMPTS_PER_VERTICAL", 1)
    monkeypatch.setattr(runner, "VERTICALS", ("airline",))
    monkeypatch.setattr(runner, "check_runtime_gate", lambda _args: _ready_gates())
    monkeypatch.setattr(runner, "_api_route", lambda _args: ("http://api", "key", "model", "api"))
    monkeypatch.setattr(
        runner,
        "_chat_completion_request",
        lambda **_kwargs: (
            json.dumps(
                {
                    "answer": "ok",
                    "evidence_ids": ["CA-POL-012", "CA-POL-013"],
                    "confidence": 0.9,
                    "insufficient_evidence": False,
                    "citation_notes": "ok",
                }
            ),
            0.01,
            0.02,
        ),
    )

    def telemetry(path: Path, stop_event: Any, interval_seconds: float, errors: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "timestamp": "now",
                    "gpu_name": "NVIDIA A100-SXM4-80GB",
                    "utilization_gpu_percent": 10.0,
                    "memory_used_mb": 100.0,
                    "memory_total_mb": 80000.0,
                    "power_draw_w": 70.0,
                    "temperature_c": 30.0,
                    "process_info": "123, python, 100 MiB",
                }
            )
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(runner, "_telemetry_loop", telemetry)


def test_run_full_execution_path_writes_outputs_with_mocked_clients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    _patch_small_success(monkeypatch, runner)

    report = runner.run_controlled_final_simulation(_args(runner, tmp_path))

    assert report["status"] == "CONTROLLED_FINAL_SIMULATION_COMPLETED"
    assert report["total_requests_attempted"] == 25
    assert report["configs_completed"] == 25
    assert Path(tmp_path / "results/raw/results.jsonl").exists()
    assert Path(tmp_path / "results/processed/sync.json").exists()


def test_run_full_is_gated_when_smoke_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "DEFAULT_PROMPTS_PER_VERTICAL", 1)
    monkeypatch.setattr(runner, "VERTICALS", ("airline",))
    monkeypatch.setattr(runner, "check_runtime_gate", lambda _args: _blocked_gates())
    called = False

    def fail_if_called(**_kwargs: Any) -> tuple[str, float, float]:
        nonlocal called
        called = True
        return "", 0.0, 0.0

    monkeypatch.setattr(runner, "_chat_completion_request", fail_if_called)

    report = runner.run_controlled_final_simulation(_args(runner, tmp_path))

    assert report["status"] == "CONTROLLED_FINAL_SIMULATION_BLOCKED_BY_SAFETY_GATES"
    assert report["total_requests_attempted"] == 0
    assert called is False


def test_checkpoint_resume_avoids_duplicate_request_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    _patch_small_success(monkeypatch, runner)
    args = _args(runner, tmp_path)

    runner.run_controlled_final_simulation(args)
    runner.run_controlled_final_simulation(args)

    rows = [
        json.loads(line)
        for line in Path(args.raw_results_path).read_text(encoding="utf-8").splitlines()
        if line
    ]
    request_ids = [row["request_id"] for row in rows]
    assert len(request_ids) == len(set(request_ids)) == 25


def test_api_track_excludes_gpu_telemetry_scope_and_self_hosted_includes_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    _patch_small_success(monkeypatch, runner)
    args = _args(runner, tmp_path)

    runner.run_controlled_final_simulation(args)
    api_rows = Path(args.api_comparison_path).read_text(encoding="utf-8")
    engine_rows = Path(args.engine_comparison_path).read_text(encoding="utf-8")

    assert "not_applicable_api_provider" in api_rows
    assert "self_hosted_gpu" in engine_rows

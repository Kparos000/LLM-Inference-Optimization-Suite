from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from inference_bench.full_run_readiness_audit import build_full_run_readiness_audit


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _check_by_name(report: dict[str, Any], name: str) -> dict[str, Any]:
    for check in report["checks"]:
        if check["name"] == name:
            return cast(dict[str, Any], check)
    raise AssertionError(f"missing check {name}")


def test_readiness_blocks_after_failed_b7_until_b7r1_stable(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "results/processed/b7_model2_3b_1000_readiness_report.json",
        {"status": "B7_CONTROLLED_1000_BASELINE_BLOCKED"},
    )

    report = build_full_run_readiness_audit(repo_root=tmp_path)
    check = _check_by_name(report, "b7r1_vllm_stability_gate")

    assert check["status"] == "FAIL"
    assert check["blocking"] is True
    assert report["terminal_1000_prompt_baseline_allowed"] is False
    assert report["rtx3070_qwen3b_suitability"] == "unstable"


def test_readiness_accepts_b7r1_stability_repair(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "results/processed/b7_model2_3b_1000_readiness_report.json",
        {"status": "B7_CONTROLLED_1000_BASELINE_BLOCKED"},
    )
    _write_json(
        tmp_path / "results/processed/b7r1_readiness_report.json",
        {"status": "B7R1_STABLE_WITH_QUALITY_CAVEAT"},
    )

    report = build_full_run_readiness_audit(repo_root=tmp_path)
    check = _check_by_name(report, "b7r1_vllm_stability_gate")

    assert check["status"] == "PASS"
    assert check["blocking"] is False
    assert report["b7r1_stability_ready"] is True
    assert report["rtx3070_qwen3b_suitability"] == "stable_but_memory_tight"

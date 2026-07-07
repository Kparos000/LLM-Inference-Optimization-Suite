from __future__ import annotations

import argparse
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, cast

import pytest

from inference_bench.context_corpora import VERTICALS


def _load_runner() -> Any:
    script = Path("scripts/phase4/run_a100_sxm_calibration.py")
    spec = spec_from_file_location("run_a100_sxm_calibration", script)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**overrides: object) -> argparse.Namespace:
    runner = _load_runner()
    parser = runner.build_parser()
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return cast(argparse.Namespace, runner.normalize_args(args))


def _runtime_selection() -> dict[str, object]:
    return {
        "model_alias": "model2_3b",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "runtime": "vllm",
        "engine": "vllm",
        "backend_type": "self_hosted_gpu",
        "hardware_type": "a100_sxm_80gb",
        "live_run_allowed": True,
    }


def _runner_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for vertical in VERTICALS:
        for index in range(40):
            rows.append(
                {
                    "prompt_id": f"{vertical}_{index:04d}",
                    "prompt": "SYSTEM:\nUse [EVIDENCE 1].",
                    "metadata": {
                        "vertical": vertical,
                        "gold_evidence_ids": json.dumps([f"{vertical}-gold-{index}"]),
                        "citation_id_aliases": json.dumps({"E1": [f"{vertical}-gold-{index}"]}),
                        "b5_required_labels": "E1",
                    },
                }
            )
    return rows


def test_a100_normalize_args_locks_live_calibration_scope() -> None:
    args = _args()

    assert args.prompt_count == 200
    assert args.concurrency == 1
    assert args.model_alias == "model2_3b"
    assert args.model_id == "Qwen/Qwen2.5-3B-Instruct"
    assert args.memory_mode == "mm2_hybrid_top5"
    assert args.engine == "vllm"
    assert args.gpu_id == "a100_sxm_80gb"
    assert args.hourly_price == pytest.approx(1.49)
    assert args.runner_input_path.endswith("a100_sxm_model2_3b_mm2_c1_200_runner_input.jsonl")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt_count", 1000),
        ("concurrency", 2),
        ("engine", "sglang"),
        ("memory_mode", "mm4_agentic"),
        ("gpu_id", "remote_rtx3070"),
        ("artifact_sync", False),
    ],
)
def test_a100_normalize_args_rejects_disallowed_scope(field: str, value: object) -> None:
    runner = _load_runner()
    parser = runner.build_parser()
    args = parser.parse_args([])
    setattr(args, field, value)

    with pytest.raises(ValueError):
        runner.normalize_args(args)


def test_a100_preflight_passes_balanced_200_prompt_input() -> None:
    runner = _load_runner()
    args = _args()

    report = runner.preflight_a100_runner_rows(
        _runner_rows(),
        args=args,
        model_id="Qwen/Qwen2.5-3B-Instruct",
        runtime_selection=_runtime_selection(),
        artifact_sync_dry_run_passed=True,
    )

    assert report["status"] == "PREFLIGHT_PASSED_A100_SXM_CALIBRATION"
    assert report["passed"] is True
    assert report["prompts_per_vertical"] == {vertical: 40 for vertical in VERTICALS}


def test_a100_server_required_message_includes_exact_vllm_command() -> None:
    runner = _load_runner()

    message = runner._server_required_message("http://localhost:8000/v1", "connection refused")

    assert "python -m vllm.entrypoints.openai.api_server" in message
    assert "--model Qwen/Qwen2.5-3B-Instruct" in message
    assert "--gpu-memory-utilization 0.90" in message

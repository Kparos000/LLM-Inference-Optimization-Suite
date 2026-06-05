import importlib.util
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from inference_bench.context_corpora import VERTICALS
from inference_bench.telemetry import (
    BACKEND_COMPARISON_FIELDS,
    TELEMETRY_FIELDS,
    BackendComparisonRow,
    TelemetryRecord,
    build_backend_comparison_framework,
    telemetry_record_from_result_row,
)

SCRIPT_PATH = Path("scripts/phase4/validate_vllm_serving.py")
spec = importlib.util.spec_from_file_location("validate_vllm_serving", SCRIPT_PATH)
assert spec is not None
validate_vllm_serving = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["validate_vllm_serving"] = validate_vllm_serving
spec.loader.exec_module(validate_vllm_serving)


def test_telemetry_schema_has_current_and_future_fields() -> None:
    record = TelemetryRecord(
        timestamp="2026-06-05T00:00:00+00:00",
        backend="vllm",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        memory_mode="mm2_hybrid_top5",
        latency_ms=100.0,
        ttft_ms=None,
        tpot_ms=5.0,
        throughput_tokens_per_second=200.0,
        requests_per_second=10.0,
        success=True,
    )

    assert list(record.to_dict()) == TELEMETRY_FIELDS
    assert record.gpu_utilization is None
    assert record.gpu_memory is None
    assert record.gpu_cost is None
    assert record.runpod_cost is None


def test_telemetry_rejects_negative_metrics() -> None:
    with pytest.raises(ValueError, match="latency_ms"):
        TelemetryRecord(
            timestamp="2026-06-05T00:00:00+00:00",
            backend="vllm",
            model="model",
            memory_mode="mm2_hybrid_top5",
            latency_ms=-1.0,
            ttft_ms=None,
            tpot_ms=None,
            throughput_tokens_per_second=None,
            requests_per_second=None,
            success=False,
        )


def test_telemetry_can_be_built_from_runner_result() -> None:
    telemetry = telemetry_record_from_result_row(
        {
            "timestamp_utc": "2026-06-05T00:00:00+00:00",
            "latency_ms": 250.0,
            "ttft_ms": 40.0,
            "tpot_ms": 8.0,
            "throughput_tokens_per_second": 125.0,
            "success": True,
        },
        backend="vllm",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        memory_mode="mm2_hybrid_top5",
    )

    assert telemetry.requests_per_second == 4.0
    assert telemetry.ttft_ms == 40.0
    assert telemetry.success is True


def test_backend_comparison_framework_includes_hf_vllm_and_sglang() -> None:
    rows = build_backend_comparison_framework()

    assert [row.backend for row in rows] == ["huggingface_local", "vllm", "sglang"]
    assert rows[0].status == "available"
    assert rows[1].status == "not_run"
    assert rows[2].status == "future"
    assert list(rows[0].to_dict()) == BACKEND_COMPARISON_FIELDS


def test_backend_comparison_validates_status() -> None:
    with pytest.raises(ValueError, match="status"):
        BackendComparisonRow(
            backend="vllm",
            status="fake",
            model="model",
            memory_mode="mm2_hybrid_top5",
        )


def test_unavailable_server_writes_requested_reports(tmp_path: Path) -> None:
    input_path = tmp_path / "runner_input.jsonl"
    input_path.write_text("", encoding="utf-8")
    raw_path = tmp_path / "raw.jsonl"
    report_path = tmp_path / "report.json"
    summary_path = tmp_path / "summary.csv"
    telemetry_path = tmp_path / "telemetry.json"
    comparison_path = tmp_path / "comparison.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input-path",
            str(input_path),
            "--output-path",
            str(raw_path),
            "--report-path",
            str(report_path),
            "--summary-path",
            str(summary_path),
            "--telemetry-path",
            str(telemetry_path),
            "--backend-comparison-path",
            str(comparison_path),
            "--base-url",
            "http://127.0.0.1:9/v1",
            "--timeout-seconds",
            "0.1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    assert "server_unavailable" in result.stdout
    assert report["validation_status"] == "server_unavailable"
    assert report["no_paid_api_call_triggered"] is True
    assert report["no_gpu_rental_triggered"] is True
    assert report["no_retrieval_modified"] is True
    assert raw_path.exists()
    assert raw_path.read_text(encoding="utf-8") == ""
    assert summary_path.exists()
    assert telemetry["record_count"] == 0
    assert comparison_path.exists()


def write_gold_fixture(root: Path, prompt_id: str) -> None:
    for vertical in VERTICALS:
        vertical_root = root / vertical
        vertical_root.mkdir(parents=True, exist_ok=True)
        (vertical_root / f"{vertical}_prompts_2000.jsonl").write_text("", encoding="utf-8")
        (vertical_root / f"{vertical}_kb_2000.jsonl").write_text("", encoding="utf-8")
        gold_path = vertical_root / f"{vertical}_gold_2000.jsonl"
        if vertical != "airline":
            gold_path.write_text("", encoding="utf-8")
            continue
        gold_path.write_text(
            json.dumps(
                {
                    "prompt_id": prompt_id,
                    "expected_status": "answer",
                    "expected_output_format": "text",
                    "must_include": ["policy"],
                    "must_not_include": ["unsafe"],
                    "required_doc_ids": ["doc-1"],
                }
            )
            + "\n",
            encoding="utf-8",
        )


def test_live_report_uses_evaluator_and_telemetry_contracts(tmp_path: Path) -> None:
    prompt_id = "airline_fixture_001"
    dataset_root = tmp_path / "dataset"
    write_gold_fixture(dataset_root, prompt_id)
    args = Namespace(
        output_path=str(tmp_path / "raw.jsonl"),
        report_path=str(tmp_path / "report.json"),
        summary_path=str(tmp_path / "summary.csv"),
        telemetry_path=str(tmp_path / "telemetry.json"),
        backend_comparison_path=str(tmp_path / "comparison.csv"),
        dataset_root=str(dataset_root),
        model_alias="model1_0_5b",
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        base_url="http://localhost:8000/v1",
    )
    result_rows = [
        {
            "timestamp_utc": "2026-06-05T00:00:00+00:00",
            "prompt_id": prompt_id,
            "memory_mode": "mm2_hybrid_top5",
            "vertical": "airline",
            "generated_text": "Policy answer with doc-1.",
            "final_status": "answer",
            "success": True,
            "latency_ms": 100.0,
            "ttft_ms": 20.0,
            "tpot_ms": 5.0,
            "throughput_tokens_per_second": 200.0,
        }
    ]

    report, summary = validate_vllm_serving.build_live_report(
        args=args,
        result_rows=result_rows,
    )

    assert report["validation_status"] == "live_validated"
    assert summary["joined_count"] == 1
    assert summary["format_valid_rate"] == 1.0
    assert (tmp_path / "telemetry.json").exists()
    assert (tmp_path / "comparison.csv").exists()

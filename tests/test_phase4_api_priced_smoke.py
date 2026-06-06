import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from inference_bench.api_priced_validation import (
    AccessCheck,
    build_cost_report,
    select_api_model,
)
from inference_bench.config import load_project_config


def _write_pricing(path: Path) -> Path:
    path.write_text(
        """
models:
  model6_gated:
    model_alias: model6_gated
    model_id: meta-llama/Llama-3.1-8B-Instruct
    provider: novita
    provider_status: live
    input_cost_per_1m_tokens_usd: 0.02
    output_cost_per_1m_tokens_usd: 0.05
    pricing_snapshot_timestamp_utc: "2026-06-06T00:00:00+00:00"
    pricing_source_url: "https://router.huggingface.co/v1/models/meta-llama/Llama-3.1-8B-Instruct"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_model_selection_uses_priced_accessible_fallback(tmp_path: Path) -> None:
    pricing = _write_pricing(tmp_path / "api_pricing.yaml")
    selected, attempts = select_api_model(
        config=load_project_config(),
        model_aliases=["model5_gated", "model6_gated"],
        pricing_config=pricing,
        hf_token="fixture",
        access_checker=lambda _model_id, _token: AccessCheck(True, 200),
    )

    assert selected is not None
    assert selected.model_alias == "model6_gated"
    assert selected.provider_model_id.endswith(":novita")
    assert attempts[0]["failure_stage"] == "pricing"
    assert attempts[1]["model_access"] is True


def test_cost_report_uses_measured_tokens_and_costs() -> None:
    result_rows = [
        {
            "model_alias": "model6_gated",
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "provider": "novita",
            "pricing_source_url": "https://example.test/pricing",
            "input_cost_per_1m_tokens_usd": 0.02,
            "output_cost_per_1m_tokens_usd": 0.05,
            "prompt_id": "p1",
            "vertical": "finance",
            "input_tokens": 1000,
            "output_tokens": 100,
            "total_tokens": 1100,
            "input_cost_usd": 0.00002,
            "output_cost_usd": 0.000005,
            "total_cost_usd": 0.000025,
            "latency_ms": 1000.0,
            "throughput_tokens_per_second": 100.0,
            "success": True,
        }
    ]
    evaluation_rows = [
        {
            "json_validity": True,
            "generation_contract_valid": True,
            "evidence_id_presence": True,
            "evidence_match": True,
            "groundedness": True,
            "safety_violation": False,
        }
    ]

    report = build_cost_report(
        result_rows=result_rows,
        evaluation_rows=evaluation_rows,
        baseline_summary={"grounded_rate": 0.4},
    )

    assert report["total_tokens"] == 1100
    assert report["total_cost_usd"] == pytest.approx(0.000025)
    assert report["cost_per_grounded_answer_usd"] == pytest.approx(0.000025)
    assert report["qwen_0_5b_comparison"]["grounded_rate"]["delta"] == pytest.approx(0.6)


def test_runner_refuses_missing_hf_token_and_writes_readiness(tmp_path: Path) -> None:
    readiness = tmp_path / "readiness.json"
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase4/run_api_priced_model_smoke.py",
            "--input-path",
            "data/generated/phase4/stronger_model_contract_runner_input.jsonl",
            "--readiness-report",
            str(readiness),
            "--allow-paid-api-call",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 1
    assert readiness.is_file()
    report = json.loads(readiness.read_text(encoding="utf-8"))
    assert report["execution_status"] == "STOPPED"
    assert report["secret_values_recorded"] is False
    assert "hf_test_secret" not in readiness.read_text(encoding="utf-8")


def test_runner_requires_explicit_paid_call_flag() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/phase4/run_api_priced_model_smoke.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "allow-paid-api-call" in result.stderr

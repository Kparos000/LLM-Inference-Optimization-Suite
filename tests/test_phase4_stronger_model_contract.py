import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from inference_bench.context_corpora import VERTICALS
from inference_bench.schema import WorkloadItem
from inference_bench.stronger_model_validation import is_model_cached


def _write_runner_input(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for vertical in VERTICALS:
            prompt_id = f"{vertical}_stronger_fixture"
            item = WorkloadItem(
                prompt_id=prompt_id,
                workload_name="smoke_500_mm2_hybrid_top5",
                prompt="Return the grounded generation contract using E1.",
                expected_output="generation_contract_json",
                metadata={
                    "workload_id": f"smoke_500:mm2_hybrid_top5:{prompt_id}",
                    "vertical": vertical,
                    "memory_mode": "mm2_hybrid_top5",
                    "ablation_mode": "prompt_plus_metadata",
                    "dataset_split": "smoke_500",
                    "citation_id_aliases": json.dumps({"E1": [f"{vertical}-doc-1"]}),
                    "retrieval_metadata": json.dumps(
                        {
                            "ablation_mode": "prompt_plus_metadata",
                            "dense_backend": "qdrant_vector",
                            "vector_store": "qdrant_local",
                            "source_hints_used": False,
                        }
                    ),
                },
            )
            file.write(json.dumps(asdict(item), sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_model_cache_check_requires_weights_and_config(tmp_path: Path) -> None:
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    snapshot = tmp_path / "models--Qwen--Qwen2.5-1.5B-Instruct" / "snapshots" / "fixture"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")

    assert is_model_cached(model_id, cache_root=tmp_path) is False
    (snapshot / "model.safetensors").write_bytes(b"fixture")
    assert is_model_cached(model_id, cache_root=tmp_path) is True


def test_auto_mode_falls_back_to_gated_api_dry_run_without_calls(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "runner_input.jsonl"
    output_path = tmp_path / "stronger.jsonl"
    _write_runner_input(input_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase4/run_stronger_model_contract_smoke.py",
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--cache-root",
            str(tmp_path / "empty_cache"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = _read_jsonl(output_path)

    assert "Validation status: DRY_RUN_ONLY" in result.stdout
    assert len(rows) == 5
    assert {row["vertical"] for row in rows} == set(VERTICALS)
    assert all(row["model_alias"] == "model5_gated" for row in rows)
    assert all(row["dry_run"] is True for row in rows)
    assert all(row["validation_measured"] is False for row in rows)
    assert all(row["paid_api_call_triggered"] is False for row in rows)
    assert all(row["no_gpu_experiment_triggered"] is True for row in rows)


def test_paid_api_path_refuses_missing_hf_token(tmp_path: Path) -> None:
    input_path = tmp_path / "runner_input.jsonl"
    output_path = tmp_path / "stronger.jsonl"
    _write_runner_input(input_path)
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase4/run_stronger_model_contract_smoke.py",
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--execution-mode",
            "hf_api",
            "--allow-paid-api-call",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 1
    assert "HF_TOKEN is required" in result.stderr
    assert not output_path.exists()


def test_dry_run_evaluation_uses_not_measured_values(tmp_path: Path) -> None:
    input_path = tmp_path / "runner_input.jsonl"
    output_path = tmp_path / "stronger.jsonl"
    report_path = tmp_path / "report.json"
    summary_path = tmp_path / "summary.csv"
    baseline_path = tmp_path / "baseline.json"
    _write_runner_input(input_path)
    subprocess.run(
        [
            sys.executable,
            "scripts/phase4/run_stronger_model_contract_smoke.py",
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--cache-root",
            str(tmp_path / "empty_cache"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    baseline_path.write_text(
        json.dumps(
            {
                "summary": {
                    "json_valid_rate": 1.0,
                    "generation_contract_valid_rate": 0.8,
                    "evidence_id_presence_rate": 1.0,
                    "evidence_match_rate": 0.4,
                    "grounded_rate": 0.2,
                }
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/phase4/evaluate_stronger_model_contract.py",
            "--results-path",
            str(output_path),
            "--baseline-report",
            str(baseline_path),
            "--report-path",
            str(report_path),
            "--summary-path",
            str(summary_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    with summary_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert report["validation_status"] == "DRY_RUN_ONLY"
    assert report["comparison"]["json_valid_rate"]["stronger_model"] is None
    assert report["comparison"]["json_valid_rate"]["block24_model1_0_5b"] == 1.0
    assert report["paid_api_call_triggered"] is False
    assert all(row["stronger_model"] == "" for row in rows)

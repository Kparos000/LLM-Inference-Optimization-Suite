from __future__ import annotations

import csv
import json
from pathlib import Path

from inference_bench.post_run_automation import (
    PostRunAutomationInputs,
    build_plotting_dataset_rows,
    build_post_run_automation_report,
    write_post_run_automation_artifacts,
)
from inference_bench.run_manifest import RunManifest, utc_now, write_run_manifest


def test_post_run_automation_builds_slo_and_plotting_rows(tmp_path: Path) -> None:
    now = utc_now()
    manifest_path = tmp_path / "manifest.json"
    write_run_manifest(
        RunManifest(
            run_id="run-1",
            timestamp_utc=now,
            backend="openai_compatible_vllm",
            model_alias="model3_7b",
            model_id="Qwen/Qwen2.5-7B-Instruct",
            memory_mode="mm2_hybrid_top5",
            split="controlled_final",
            ablation_mode="prompt_plus_metadata",
            input_workload_path="matrix.jsonl",
            output_path="results/raw/run.jsonl",
            max_records=500,
            git_commit="abc123",
            command="cmd",
            status="completed",
            start_time=now,
            end_time=now,
            error_count=0,
            completed_count=500,
            failed_count=0,
            expected_count=500,
            engine="vllm",
            hardware="a100_sxm_80gb",
            concurrency=16,
            traffic_profile="online_low_latency",
            prompt_count=500,
            dataset_version="controlled_2000",
        ),
        manifest_path,
    )
    eval_summary = tmp_path / "eval.csv"
    eval_summary.write_text(
        "json_valid_rate,grounded_rate,evidence_match_rate\n1.0,0.95,0.96\n",
        encoding="utf-8",
    )
    latency_summary = tmp_path / "latency.csv"
    latency_summary.write_text(
        "mean_ttft_ms,mean_tpot_ms,mean_e2e_latency_ms,mean_total_tokens_per_second\n"
        "100,12,900,150\n",
        encoding="utf-8",
    )
    cost_report = tmp_path / "cost.json"
    cost_report.write_text('{"gpu_cost_usd": 0.5}\n', encoding="utf-8")

    report = build_post_run_automation_report(
        PostRunAutomationInputs(
            run_id="run-1",
            manifest_path=str(manifest_path),
            eval_summary_path=str(eval_summary),
            latency_summary_path=str(latency_summary),
            cost_report_path=str(cost_report),
            comparison_paths=("engine.csv",),
        )
    )
    rows = build_plotting_dataset_rows(report)

    assert report["manifest"]["dataset_version"] == "controlled_2000"
    assert report["slo_status_counts"]["PASS"] >= 6
    assert "baseline_vs_optimized" in report["plotting_datasets"]
    assert {row["metric_name"] for row in rows} >= {"ttft_ms", "groundedness", "cost"}


def test_post_run_automation_writes_artifacts(tmp_path: Path) -> None:
    report = {
        "run_id": "run-1",
        "manifest": {"engine": "vllm", "baseline_or_optimized": "baseline"},
        "slo_metric_rows": [{"metric_name": "ttft_ms", "observed": 1.0, "status": "PASS"}],
    }
    report_path, plotting_path = write_post_run_automation_artifacts(
        report=report,
        report_path=tmp_path / "report.json",
        plotting_dataset_path=tmp_path / "plotting.csv",
    )

    assert json.loads(report_path.read_text(encoding="utf-8"))["run_id"] == "run-1"
    with plotting_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["metric_name"] == "ttft_ms"

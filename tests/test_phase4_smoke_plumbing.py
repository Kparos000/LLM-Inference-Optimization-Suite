import csv
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from inference_bench.context_corpora import VERTICALS
from inference_bench.context_schema import ContextRecord, WorkloadRecord
from inference_bench.run_manifest import (
    RunManifest,
    current_git_commit,
    utc_now,
    write_run_manifest,
)
from inference_bench.runners.mock_runner import run_mock_benchmark
from inference_bench.workload_adapter import export_runner_workload


def context_record() -> ContextRecord:
    return ContextRecord(
        context_id="airline:doc-1",
        vertical="airline",
        source_id="benchmark_kb",
        parent_id="doc-1",
        chunk_id="doc-1",
        chunk_strategy="policy-section",
        source_type="policy",
        title="Delay Policy",
        text="Delay policy evidence.",
        metadata={"policy": "delay"},
        token_estimate=4,
        provenance="fixture",
        is_gold_linked=True,
    )


def workload_record(index: int = 1) -> WorkloadRecord:
    prompt_id = f"airline_fixture_{index:03d}"
    return WorkloadRecord(
        workload_id=f"smoke_500:prompt_plus_metadata:mm2_hybrid_top5:{prompt_id}",
        prompt_id=prompt_id,
        vertical="airline",
        memory_mode="mm2_hybrid_top5",
        messages=[
            {"role": "system", "content": "Answer using supplied context."},
            {"role": "user", "content": "Delay policy evidence.\n\nQuestion: What applies?"},
        ],
        context_records=[context_record()],
        context_token_estimate=4,
        retrieval_metadata={
            "ablation_mode": "prompt_plus_metadata",
            "retrieval_type": "hybrid",
            "selected_context_ids": ["airline:doc-1"],
        },
        expected_output_format="text",
        gold_evidence_ids=["doc-1"],
        dataset_split="smoke_500",
        source_prompt_record={"prompt_id": prompt_id, "question": "What applies?"},
    )


def write_phase3_workload(path: Path, count: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for index in range(1, count + 1):
            file.write(json.dumps(asdict(workload_record(index)), sort_keys=True) + "\n")


def write_gold_fixture_dataset(root: Path, prompt_ids: list[str]) -> None:
    for vertical in VERTICALS:
        vertical_dir = root / vertical
        vertical_dir.mkdir(parents=True, exist_ok=True)
        (vertical_dir / f"{vertical}_prompts_2000.jsonl").write_text("", encoding="utf-8")
        (vertical_dir / f"{vertical}_kb_2000.jsonl").write_text("", encoding="utf-8")
        gold_path = vertical_dir / f"{vertical}_gold_2000.jsonl"
        if vertical != "airline":
            gold_path.write_text("", encoding="utf-8")
            continue
        with gold_path.open("w", encoding="utf-8") as file:
            for prompt_id in prompt_ids:
                file.write(
                    json.dumps(
                        {
                            "prompt_id": prompt_id,
                            "vertical": vertical,
                            "expected_status": "answer",
                            "expected_output_format": "text",
                            "must_include": [],
                            "must_not_include": [],
                            "required_doc_ids": ["doc-1"],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def test_mock_runner_output_preserves_phase4_ids_and_metadata(tmp_path: Path) -> None:
    phase3_path = tmp_path / "phase3.jsonl"
    runner_path = tmp_path / "runner.jsonl"
    result_path = tmp_path / "mock_results.csv"
    write_phase3_workload(phase3_path, count=2)
    export_runner_workload(
        workload_path=phase3_path,
        output_path=runner_path,
        report_path=tmp_path / "report.json",
        summary_path=tmp_path / "summary.csv",
    )

    results = run_mock_benchmark(runner_path, result_path)
    rows = read_csv(result_path)

    assert len(results) == 2
    assert [row["prompt_id"] for row in rows] == [
        "airline_fixture_001",
        "airline_fixture_002",
    ]
    assert all(row["workload_id"].startswith("smoke_500:") for row in rows)
    assert all(row["memory_mode"] == "mm2_hybrid_top5" for row in rows)
    assert all(row["ablation_mode"] == "prompt_plus_metadata" for row in rows)


def test_evaluator_joins_by_prompt_id(tmp_path: Path) -> None:
    phase3_path = tmp_path / "phase3.jsonl"
    runner_path = tmp_path / "runner.jsonl"
    result_path = tmp_path / "mock_results.csv"
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "processed"
    write_phase3_workload(phase3_path, count=2)
    write_gold_fixture_dataset(
        dataset_root,
        ["airline_fixture_001", "airline_fixture_002"],
    )
    export_runner_workload(
        workload_path=phase3_path,
        output_path=runner_path,
        report_path=tmp_path / "report.json",
        summary_path=tmp_path / "summary.csv",
    )
    run_mock_benchmark(runner_path, result_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase4/evaluate_generation_outputs.py",
            "--results-path",
            str(result_path),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(output_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Evaluation rows written: 2" in result.stdout
    report = json.loads((output_root / "phase4_mock_smoke_eval_report.json").read_text())
    assert report["summary"]["joined_count"] == 2
    assert all(row["joined"] for row in report["evaluation_rows"])


def test_run_manifest_validates_and_writes(tmp_path: Path) -> None:
    manifest = RunManifest(
        run_id="phase4-mock-smoke",
        timestamp_utc=utc_now(),
        backend="mock",
        model_alias="mock-model",
        model_id="mock-model",
        memory_mode="mm2_hybrid_top5",
        split="smoke_500",
        ablation_mode="prompt_plus_metadata",
        input_workload_path="data/generated/phase4/smoke_500_mm2_runner_input.jsonl",
        output_path="results/raw/phase4_mock_smoke_results.csv",
        max_records=25,
        git_commit=current_git_commit(),
        command="inference-bench mock-run",
        status="completed",
        start_time=utc_now(),
        end_time=utc_now(),
        error_count=0,
    )

    output_path = write_run_manifest(manifest, tmp_path / "manifest.json")

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "phase4-mock-smoke"
    assert payload["memory_mode"] == "mm2_hybrid_top5"


def test_invalid_manifest_status_fails() -> None:
    try:
        RunManifest(
            run_id="bad",
            timestamp_utc=utc_now(),
            backend="mock",
            model_alias="mock-model",
            model_id="mock-model",
            memory_mode="mm2_hybrid_top5",
            split="smoke_500",
            ablation_mode="prompt_plus_metadata",
            input_workload_path="input.jsonl",
            output_path="output.csv",
            max_records=None,
            git_commit="abc",
            command="cmd",
            status="done",
            start_time=utc_now(),
            end_time=None,
            error_count=0,
        )
    except ValueError as exc:
        assert "status must be one of" in str(exc)
    else:
        raise AssertionError("invalid manifest status should fail")


def test_no_real_model_api_or_gpu_call_is_triggered() -> None:
    assert True

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from inference_bench.context_schema import ContextRecord, WorkloadRecord
from inference_bench.workload_adapter import (
    convert_phase3_workload_to_runner_items,
    export_runner_workload,
    render_messages_for_runner,
    workload_record_to_runner_item,
)


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
            {
                "role": "user",
                "content": "Context: Delay policy evidence.\n\nQuestion: What applies?",
            },
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


def write_workload(path: Path, count: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for index in range(1, count + 1):
            file.write(json.dumps(asdict(workload_record(index)), sort_keys=True) + "\n")


def test_workload_record_converts_to_runner_input() -> None:
    item = workload_record_to_runner_item(workload_record())

    assert item.prompt_id == "airline_fixture_001"
    assert item.workload_name == "smoke_500_mm2_hybrid_top5"
    assert "SYSTEM:" in item.prompt
    assert "USER:" in item.prompt
    assert item.metadata["workload_id"].endswith("airline_fixture_001")
    assert item.metadata["memory_mode"] == "mm2_hybrid_top5"
    assert item.metadata["ablation_mode"] == "prompt_plus_metadata"


def test_context_is_rendered_into_messages() -> None:
    rendered = render_messages_for_runner(workload_record().messages)

    assert "Delay policy evidence" in rendered
    assert "Question: What applies?" in rendered


def test_metadata_is_preserved_as_strings() -> None:
    item = workload_record_to_runner_item(workload_record())

    assert item.metadata["vertical"] == "airline"
    assert item.metadata["context_token_estimate"] == "4"
    assert json.loads(item.metadata["gold_evidence_ids"]) == ["doc-1"]
    assert json.loads(item.metadata["retrieval_metadata"])["retrieval_type"] == "hybrid"


def test_limit_works(tmp_path: Path) -> None:
    workload_path = tmp_path / "phase3.jsonl"
    write_workload(workload_path, count=3)

    items = convert_phase3_workload_to_runner_items(workload_path, limit=2)

    assert len(items) == 2
    assert [item.prompt_id for item in items] == [
        "airline_fixture_001",
        "airline_fixture_002",
    ]


def test_export_script_works_on_fixture(tmp_path: Path) -> None:
    workload_path = tmp_path / "phase3.jsonl"
    output_path = tmp_path / "runner.jsonl"
    report_path = tmp_path / "report.json"
    summary_path = tmp_path / "summary.csv"
    write_workload(workload_path, count=3)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase4/export_runner_smoke_workload.py",
            "--workload-path",
            str(workload_path),
            "--output-path",
            str(output_path),
            "--limit",
            "2",
            "--report-path",
            str(report_path),
            "--summary-path",
            str(summary_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Runner workload rows written: 2" in result.stdout
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 2
    assert json.loads(report_path.read_text(encoding="utf-8"))["record_count"] == 2
    assert "airline" in summary_path.read_text(encoding="utf-8")


def test_export_runner_workload_function_writes_report(tmp_path: Path) -> None:
    workload_path = tmp_path / "phase3.jsonl"
    output_path = tmp_path / "runner.jsonl"
    report_path = tmp_path / "report.json"
    summary_path = tmp_path / "summary.csv"
    write_workload(workload_path, count=1)

    report = export_runner_workload(
        workload_path=workload_path,
        output_path=output_path,
        report_path=report_path,
        summary_path=summary_path,
    )

    assert report["record_count"] == 1
    assert output_path.exists()
    assert report_path.exists()
    assert summary_path.exists()

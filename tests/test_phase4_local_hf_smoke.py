import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from inference_bench.context_corpora import VERTICALS
from inference_bench.schema import WorkloadItem

SCRIPT_PATH = Path("scripts/phase4/run_local_hf_smoke.py")
spec = importlib.util.spec_from_file_location("run_local_hf_smoke", SCRIPT_PATH)
assert spec is not None
run_local_hf_smoke = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(run_local_hf_smoke)
validate_smoke_result_row = run_local_hf_smoke.validate_smoke_result_row


def runner_item(prompt_id: str = "airline_fixture_001") -> WorkloadItem:
    return WorkloadItem(
        prompt_id=prompt_id,
        workload_name="smoke_500_mm2_hybrid_top5",
        prompt="SYSTEM:\nAnswer using context.\n\nUSER:\nQuestion: What applies?",
        expected_output="text",
        metadata={
            "workload_id": f"smoke_500:prompt_plus_metadata:mm2_hybrid_top5:{prompt_id}",
            "phase3_prompt_id": prompt_id,
            "vertical": "airline",
            "memory_mode": "mm2_hybrid_top5",
            "ablation_mode": "prompt_plus_metadata",
            "dataset_split": "smoke_500",
            "expected_output_format": "text",
            "context_token_estimate": "8",
            "gold_evidence_ids": '["doc-1"]',
            "selected_context_ids": '["airline:doc-1"]',
        },
    )


def write_runner_input(path: Path, count: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for index in range(count):
            item = runner_item(f"airline_fixture_{index + 1:03d}")
            file.write(json.dumps(asdict(item), sort_keys=True) + "\n")


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


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run_dry_smoke(tmp_path: Path, *, limit: int = 2) -> tuple[Path, Path]:
    input_path = tmp_path / "runner_input.jsonl"
    output_path = tmp_path / "phase4_hf_local_smoke_results.jsonl"
    manifest_path = tmp_path / "manifest.json"
    write_runner_input(input_path, count=2)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase4/run_local_hf_smoke.py",
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--model-alias",
            "model1_0_5b",
            "--limit",
            str(limit),
            "--max-new-tokens",
            "16",
            "--manifest-path",
            str(manifest_path),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Rows written" in result.stdout
    return output_path, manifest_path


def test_script_can_run_in_dry_run_mode_without_loading_model(tmp_path: Path) -> None:
    output_path, _manifest_path = run_dry_smoke(tmp_path)

    rows = read_jsonl(output_path)

    assert len(rows) == 2
    assert all(row["dry_run"] is True for row in rows)
    assert all(row["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct" for row in rows)


def test_output_schema_validates_and_metadata_is_preserved(tmp_path: Path) -> None:
    output_path, _manifest_path = run_dry_smoke(tmp_path, limit=1)
    row = read_jsonl(output_path)[0]

    validate_smoke_result_row(row)
    assert row["prompt_id"] == "airline_fixture_001"
    assert row["workload_id"].startswith("smoke_500:")
    assert row["vertical"] == "airline"
    assert row["memory_mode"] == "mm2_hybrid_top5"
    assert row["ablation_mode"] == "prompt_plus_metadata"
    assert row["paid_api_call_triggered"] is False


def test_run_manifest_is_created(tmp_path: Path) -> None:
    _output_path, manifest_path = run_dry_smoke(tmp_path, limit=1)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["backend"] == "huggingface_local"
    assert manifest["model_alias"] == "model1_0_5b"
    assert manifest["memory_mode"] == "mm2_hybrid_top5"
    assert manifest["max_records"] == 1
    assert manifest["status"] == "completed"


def test_prompt_limit_is_enforced(tmp_path: Path) -> None:
    input_path = tmp_path / "runner_input.jsonl"
    output_path = tmp_path / "out.jsonl"
    write_runner_input(input_path, count=2)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase4/run_local_hf_smoke.py",
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--limit",
            "26",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "limit must be <= 25" in result.stderr


def test_evaluator_can_process_fixture_output(tmp_path: Path) -> None:
    output_path, _manifest_path = run_dry_smoke(tmp_path, limit=2)
    dataset_root = tmp_path / "dataset"
    processed_root = tmp_path / "processed"
    write_gold_fixture_dataset(
        dataset_root,
        ["airline_fixture_001", "airline_fixture_002"],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase4/evaluate_generation_outputs.py",
            "--results-path",
            str(output_path),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(processed_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Evaluation rows written: 2" in result.stdout
    report = json.loads(
        (processed_root / "phase4_hf_local_smoke_eval_report.json").read_text(encoding="utf-8")
    )
    assert report["summary"]["joined_count"] == 2


def test_no_paid_api_call_is_triggered(tmp_path: Path) -> None:
    output_path, _manifest_path = run_dry_smoke(tmp_path, limit=1)
    row = read_jsonl(output_path)[0]

    assert row["paid_api_call_triggered"] is False
    assert row["estimated_cost_usd"] == 0.0

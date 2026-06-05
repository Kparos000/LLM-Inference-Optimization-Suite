import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from inference_bench.schema import WorkloadItem


def _write_input(path: Path, count: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for index in range(count):
            prompt_id = f"airline_sglang_fixture_{index + 1:03d}"
            item = WorkloadItem(
                prompt_id=prompt_id,
                workload_name="smoke_500_mm2_hybrid_top5",
                prompt="SYSTEM:\nReturn grounded JSON.\n\nUSER:\nWhat policy applies?",
                expected_output="json",
                metadata={
                    "workload_id": f"smoke_500:mm2_hybrid_top5:{prompt_id}",
                    "vertical": "airline",
                    "memory_mode": "mm2_hybrid_top5",
                    "ablation_mode": "prompt_plus_metadata",
                    "dataset_split": "smoke_500",
                    "citation_id_aliases": "{}",
                },
            )
            file.write(json.dumps(asdict(item), sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_sglang_scaffold_dry_run_preserves_schema_and_metadata(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    manifest_path = tmp_path / "manifest.json"
    _write_input(input_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase4/run_sglang_compatible_smoke.py",
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--manifest-path",
            str(manifest_path),
            "--limit",
            "2",
            "--max-new-tokens",
            "16",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = _read_jsonl(output_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert "server checks and model calls were skipped" in result.stdout
    assert len(rows) == 2
    assert all(row["backend"] == "sglang_openai_compatible" for row in rows)
    assert all(row["memory_mode"] == "mm2_hybrid_top5" for row in rows)
    assert all(row["paid_api_call_triggered"] is False for row in rows)
    assert all(row["no_gpu_experiment_triggered"] is True for row in rows)
    assert manifest["backend"] == "sglang_openai_compatible"


def test_sglang_scaffold_fails_clearly_when_server_is_missing(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    _write_input(input_path, count=1)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase4/run_sglang_compatible_smoke.py",
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--base-url",
            "http://127.0.0.1:9/v1",
            "--timeout-seconds",
            "0.1",
            "--limit",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "SGLang OpenAI-compatible server is unavailable" in result.stderr
    assert not output_path.exists()

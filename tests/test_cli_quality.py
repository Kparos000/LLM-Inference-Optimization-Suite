import json
from pathlib import Path

from typer.testing import CliRunner

from inference_bench.cli import app


def test_score_structured_jsonl_succeeds_with_valid_generated_json(tmp_path: Path) -> None:
    input_path = tmp_path / "generations.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "generated_text": (
                    '{"category":"helpdesk","answer":"Reset your password.","confidence":0.9}'
                )
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ["score-structured-jsonl", "--input-jsonl", str(input_path)],
    )

    assert result.exit_code == 0
    assert "total_records" in result.output
    assert "valid_json_count" in result.output
    assert "required_fields_count" in result.output
    assert "1" in result.output


def test_score_structured_jsonl_reports_invalid_generated_text(tmp_path: Path) -> None:
    input_path = tmp_path / "generations.jsonl"
    input_path.write_text(
        json.dumps({"generated_text": "category: helpdesk"}) + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ["score-structured-jsonl", "--input-jsonl", str(input_path)],
    )

    assert result.exit_code == 0
    assert "invalid_json_count" in result.output
    assert "1" in result.output


def test_score_structured_jsonl_missing_input_file_exits_nonzero(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.jsonl"

    result = CliRunner().invoke(
        app,
        ["score-structured-jsonl", "--input-jsonl", str(missing_path)],
    )

    assert result.exit_code != 0
    assert "Input JSONL not found" in result.output

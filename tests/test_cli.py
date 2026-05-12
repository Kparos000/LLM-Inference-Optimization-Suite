from typer.testing import CliRunner

from inference_bench.cli import app

runner = CliRunner()


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert "LLM Inference Optimization Suite" in result.output


def test_cli_doctor() -> None:
    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "Environment check passed" in result.output


def test_cli_explain_kv_cache() -> None:
    result = runner.invoke(app, ["explain", "kv-cache"])

    assert result.exit_code == 0
    assert "KV cache" in result.output
    assert "notebook" in result.output

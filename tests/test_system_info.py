import json
from pathlib import Path

from typer.testing import CliRunner

from inference_bench.cli import app
from inference_bench.system_info import (
    SystemInfo,
    collect_system_info,
    write_system_info_json,
)


def test_collect_system_info_returns_required_fields() -> None:
    info = collect_system_info()

    assert info.timestamp_utc
    assert info.platform
    assert info.python_version
    assert isinstance(info.cuda_device_names, list)


def test_system_info_to_dict_returns_expected_keys() -> None:
    info = SystemInfo(
        timestamp_utc="2026-05-12T00:00:00+00:00",
        platform="Windows",
        platform_release="test",
        python_version="3.13.0",
        processor="test-processor",
        cpu_count_logical=8,
        cpu_count_physical=4,
        total_ram_gb=16.0,
        torch_version=None,
        cuda_available=None,
        cuda_device_count=None,
        cuda_device_names=[],
        transformers_version=None,
    )

    payload = info.to_dict()

    assert payload["platform"] == "Windows"
    assert payload["cuda_device_names"] == []
    assert set(payload) == {
        "timestamp_utc",
        "platform",
        "platform_release",
        "python_version",
        "processor",
        "cpu_count_logical",
        "cpu_count_physical",
        "total_ram_gb",
        "torch_version",
        "cuda_available",
        "cuda_device_count",
        "cuda_device_names",
        "transformers_version",
    }


def test_write_system_info_json_creates_file(tmp_path: Path) -> None:
    info = collect_system_info()
    output_path = tmp_path / "nested" / "system_info.json"

    written_path = write_system_info_json(info, output_path)

    assert written_path == output_path
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["timestamp_utc"] == info.timestamp_utc
    assert payload["platform"] == info.platform


def test_cli_system_info_succeeds_with_tmp_path(tmp_path: Path) -> None:
    output_path = tmp_path / "system_info.json"

    result = CliRunner().invoke(app, ["system-info", "--output-path", str(output_path)])

    assert result.exit_code == 0
    assert output_path.exists()
    assert "Platform:" in result.output
    assert "Python version:" in result.output
    assert "Output path:" in result.output

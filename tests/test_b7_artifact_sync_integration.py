from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any


def _load_b7_script() -> Any:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "phase4"
        / ("run_b7_controlled_1000_baseline.py")
    )
    spec = importlib.util.spec_from_file_location("run_b7_controlled_1000_baseline", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_b7_artifact_sync_dry_run_copies_and_verifies_backup(tmp_path: Path) -> None:
    module = _load_b7_script()
    args = argparse.Namespace(
        backup_root=str(tmp_path / "backups"),
        sync_every_n_requests=5,
    )

    report = module._artifact_sync_dry_run(args)

    assert report["success"] is True
    assert report["sync"]["success"] is True
    assert report["verification"]["passed"] is True

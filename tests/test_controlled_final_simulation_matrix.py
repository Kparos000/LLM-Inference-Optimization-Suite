from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

from inference_bench.context_corpora import VERTICALS


def _load_runner() -> Any:
    script = Path("scripts/phase4/run_controlled_final_simulation.py")
    spec = spec_from_file_location("run_controlled_final_simulation", script)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_controlled_final_simulation_config_matrix_shape() -> None:
    runner = _load_runner()

    specs = runner.build_config_specs()

    assert len(specs) == 30
    assert sum(1 for spec in specs if spec.backend_type == "self_hosted_gpu") == 20
    assert sum(1 for spec in specs if spec.backend_type == "api_provider") == 10
    assert {spec.engine for spec in specs if spec.backend_type == "self_hosted_gpu"} == {
        "vllm",
        "sglang",
    }
    assert {spec.memory_mode for spec in specs} == {
        "mm0_no_context",
        "mm1_dense_top5",
        "mm2_hybrid_top5",
        "mm3_compressed_hybrid_top5",
        "mm4_bounded_agentic",
    }


def test_controlled_final_simulation_matrix_has_15000_rows() -> None:
    runner = _load_runner()

    rows = runner.build_matrix_rows(
        dataset_root="data/scaleup_2000_full",
        prompts_per_vertical=100,
    )
    summary = runner.summarize_matrix(rows, prompts_per_vertical=100)

    assert summary["passed"] is True
    assert summary["row_count"] == 15_000
    assert summary["self_hosted_request_count"] == 10_000
    assert summary["api_request_count"] == 5_000
    assert summary["prompt_count_per_config"] == 500
    assert summary["vertical_counts"] == {vertical: 3_000 for vertical in VERTICALS}


def test_controlled_final_simulation_matrix_rows_include_required_metadata() -> None:
    runner = _load_runner()

    row = runner.build_matrix_rows(
        dataset_root="data/scaleup_2000_full",
        prompts_per_vertical=100,
    )[0]

    required = {
        "config_id",
        "model_alias",
        "model_id",
        "backend_type",
        "engine",
        "runtime",
        "memory_mode",
        "concurrency",
        "vertical",
        "prompt_id",
        "prompt",
        "input_context",
        "expected_evidence_ids",
        "traffic_profile",
    }
    assert required <= set(row)
    assert json.dumps(row["expected_evidence_ids"])

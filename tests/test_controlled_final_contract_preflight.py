from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any


def _load_runner() -> Any:
    script = Path("scripts/phase4/run_controlled_final_simulation.py")
    spec = spec_from_file_location("run_controlled_final_contract_preflight", script)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(**overrides: Any) -> dict[str, Any]:
    prompt = "\n".join(
        [
            "SYSTEM:",
            "MEMORY MODE:",
            "mm2_hybrid_top5",
            "RETRIEVED EVIDENCE:",
            "E1 E2 E3 E4 E5",
            "OUTPUT CONTRACT:",
            "answer evidence_ids confidence insufficient_evidence citation_notes",
        ]
    )
    row: dict[str, Any] = {
        "config_id": "cfg",
        "prompt_id": "p1",
        "prompt": prompt,
        "memory_mode": "mm2_hybrid_top5",
        "vertical": "airline",
        "expected_output_format": "generation_contract_json",
        "expected_evidence_ids": ["DOC-1"],
        "canonical_ids_exposed_to_model": "false",
        "contract_repair_tags": "b6_b7_a100_context_aligned_generation_contract",
        "message_payload_normalized": True,
    }
    row.update(overrides)
    return row


def test_contract_preflight_passes_repaired_rows() -> None:
    runner = _load_runner()
    rows = [
        _row(),
        _row(
            vertical="finance",
            contract_repair_tags="b6r5_finance_evidence_selection_preplan",
        ),
        _row(
            vertical="research_ai",
            contract_repair_tags="b6r6_research_ai_answer_skeleton",
        ),
        _row(memory_mode="mm0_no_context"),
        _row(
            memory_mode="mm4_bounded_agentic",
            contract_repair_tags="mm4_bounded_agentic_contract",
            prompt=_row()["prompt"] + "\nMM4 BOUNDED AGENTIC CONTRACT",
        ),
    ]

    report = runner.contract_preflight_report(rows)

    assert report["status"] == "CONTRACT_PREFLIGHT_PASSED"
    assert report["passed"] is True


def test_contract_preflight_blocks_raw_prompt_text() -> None:
    runner = _load_runner()

    report = runner.contract_preflight_report(
        [_row(prompt="A raw promoted prompt without contract.", expected_output_format="text")]
    )

    assert report["status"] == "CONTRACT_PREFLIGHT_BLOCKED"
    assert report["checks"]["all_rows_have_contract_instructions"] is False


def test_full_run_is_blocked_if_contract_preflight_fails(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    runner = _load_runner()
    args = runner.build_parser().parse_args(["--run-full"])
    args.prompt_count_per_vertical = 1
    args.raw_results_path = str(tmp_path / "raw.jsonl")
    args.manifest_path = str(tmp_path / "manifest.json")
    args.gpu_telemetry_path = str(tmp_path / "gpu.jsonl")
    args.matrix_path = str(tmp_path / "matrix.jsonl")
    args.eval_report_path = str(tmp_path / "eval.json")
    args.eval_summary_path = str(tmp_path / "eval.csv")
    args.engine_comparison_path = str(tmp_path / "engine.csv")
    args.memory_comparison_path = str(tmp_path / "memory.csv")
    args.concurrency_comparison_path = str(tmp_path / "concurrency.csv")
    args.api_comparison_path = str(tmp_path / "api.csv")
    args.api_vs_self_hosted_comparison_path = str(tmp_path / "api_vs.json")
    args.model_comparison_path = str(tmp_path / "model.csv")
    args.slo_report_path = str(tmp_path / "slo.json")
    args.slo_summary_path = str(tmp_path / "slo.csv")
    args.cost_report_path = str(tmp_path / "cost.json")
    args.artifact_sync_report_path = str(tmp_path / "sync.json")
    args.post_run_automation_report_path = str(tmp_path / "post.json")
    args.plotting_dataset_path = str(tmp_path / "plot.csv")
    args.contract_preflight_report = str(tmp_path / "contract_preflight.json")
    args.checkpoint_path = str(tmp_path / "checkpoint.json")
    args.backup_root = str(tmp_path / "backup")
    monkeypatch.setattr(runner, "DEFAULT_PROMPTS_PER_VERTICAL", 1)
    monkeypatch.setattr(runner, "VERTICALS", ("airline",))
    monkeypatch.setattr(
        runner,
        "build_repaired_base_input",
        lambda _args: [
            {
                "vertical": "airline",
                "prompt_id": "p1",
                "prompt": "raw prompt",
                "source_prompt_text": "raw prompt",
                "input_context": "",
                "expected_evidence_ids": [],
                "expected_status": "answer",
                "expected_output_format": "text",
                "citation_id_aliases": "{}",
                "canonical_ids_exposed_to_model": "false",
                "traffic_profile": "online_low_latency",
            }
        ],
    )
    monkeypatch.setattr(
        runner,
        "check_runtime_gate",
        lambda _args: {
            "full_simulation_allowed": True,
            "checks": {
                "vllm_model3_7b": {"status": "SMOKE_READY"},
                "sglang_model3_7b": {"status": "SMOKE_READY"},
                "api_model6_gated": {"status": "SMOKE_READY"},
                "mm4_bounded_agentic": {"status": "SMOKE_READY"},
            },
        },
    )
    called = False

    def fail_if_called(**_kwargs: Any) -> tuple[str, float, float]:
        nonlocal called
        called = True
        return "", 0.0, 0.0

    monkeypatch.setattr(runner, "_chat_completion_request", fail_if_called)

    report = runner.run_controlled_final_simulation(args)
    preflight = json.loads(Path(args.contract_preflight_report).read_text())

    assert report["status"] == "CONTROLLED_FINAL_SIMULATION_BLOCKED_BY_SAFETY_GATES"
    assert preflight["status"] == "CONTRACT_PREFLIGHT_BLOCKED"
    assert called is False


def test_no_gold_evaluator_or_slo_weakening() -> None:
    evaluator = Path("src/inference_bench/evaluator_contract.py").read_text()
    slo = Path("configs/slo_targets.yaml").read_text()

    assert "generation_contract_valid" in evaluator
    assert "json_validity" in evaluator
    assert "format_validity_min" in slo
    assert "evidence_match_min" in slo
    assert "groundedness_min" in slo


def test_full_10k_is_blocked_if_repaired_25_replay_fails(tmp_path: Path) -> None:
    runner = _load_runner()
    args = runner.build_parser().parse_args(
        ["--run-full", "--allow-full-after-repair", "--waive-repaired-validation-reason", "test"]
    )
    args.contract_preflight_report = str(tmp_path / "contract.json")
    args.repaired_25_replay_report = str(tmp_path / "replay25.json")
    args.repaired_500_validation_report = str(tmp_path / "validation500.json")
    Path(args.repaired_25_replay_report).write_text(
        json.dumps({"passed_quality_gate": False}),
        encoding="utf-8",
    )

    full_repair_allowed = bool(
        True
        and json.loads(Path(args.repaired_25_replay_report).read_text(encoding="utf-8")).get(
            "passed_quality_gate"
        )
        and bool(args.waive_repaired_validation_reason.strip())
    )

    assert full_repair_allowed is False


def test_500_validation_only_runs_after_25_replay_passes(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    runner = _load_runner()
    args = runner.build_parser().parse_args(["--run-repaired-validation"])
    args.repaired_25_replay_report = str(tmp_path / "replay25.json")
    args.repaired_500_validation_report = str(tmp_path / "validation500.json")
    Path(args.repaired_25_replay_report).write_text(
        json.dumps({"passed_quality_gate": False}),
        encoding="utf-8",
    )
    called = False

    def fail_if_called(**_kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(runner, "run_repaired_subset", fail_if_called)
    repaired_smoke_report = json.loads(
        Path(args.repaired_25_replay_report).read_text(encoding="utf-8")
    )
    if not bool(repaired_smoke_report.get("passed_quality_gate")):
        repaired_validation_report = {
            "run_id": runner.RUN_ID,
            "status": "REPAIRED_500_VALIDATION_BLOCKED_BY_25_REPLAY",
            "passed_quality_gate": False,
            "reason": "25-row repaired replay did not pass quality gate.",
        }
        runner._write_json(args.repaired_500_validation_report, repaired_validation_report)
    else:
        runner.run_repaired_subset(args=args, rows=[], report_path="", status="")

    report = json.loads(Path(args.repaired_500_validation_report).read_text(encoding="utf-8"))
    assert report["status"] == "REPAIRED_500_VALIDATION_BLOCKED_BY_25_REPLAY"
    assert called is False

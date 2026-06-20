from __future__ import annotations

import json
from pathlib import Path

from inference_bench.full_run_readiness_audit import build_full_run_readiness_audit


def _write(root: Path, relative_path: str, text: str = "x") -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _seed_non_quality_readiness_files(root: Path) -> None:
    _write(root, "data/generated/context_engineering/retrieval_source_of_truth_manifest.json", "{}")
    _write(root, "data/workloads/controlled_2000/prompt_plus_metadata/mm2_hybrid_top5.jsonl")
    _write(root, "data/generated/phase4/b6_context_aligned_500_runner_input.jsonl")
    _write(root, "src/inference_bench/memory_workloads.py")
    _write(root, "scripts/phase4/evaluate_generation_outputs.py", "def load_gold_records(): pass")
    for relative_path in (
        "src/inference_bench/context_alignment_repair.py",
        "src/inference_bench/generation_contract.py",
        "src/inference_bench/answer_planning.py",
        "src/inference_bench/multi_evidence_selector.py",
        "src/inference_bench/safety_generation_repair.py",
        "configs/hardware/remote_rtx3070.yaml",
        "configs/runpod_projection_prices.yaml",
        "docs/96_remote_rtx3070_vllm_smoke.md",
        "docs/96_remote_rtx3070_sglang_smoke.md",
        "src/inference_bench/gpu_telemetry.py",
        "src/inference_bench/api_pricing.py",
        "src/inference_bench/cost.py",
        "configs/gpu_costs.yaml",
        "src/inference_bench/slo_profiles.py",
        "configs/bottleneck_catalog.yaml",
        "configs/optimization_catalog.yaml",
        "src/inference_bench/artifact_sync.py",
        "src/inference_bench/checkpoint_resume.py",
        "scripts/phase4/test_long_run_recovery_dry_run.py",
    ):
        _write(root, relative_path)
    _write(
        root,
        "src/inference_bench/runners/openai_load_runner.py",
        "checkpoint_path='x'\ncompleted_prompt_ids=set()\ndef _append_results_csv(): pass\n",
    )
    _write(
        root,
        "scripts/phase4/run_b6_vllm_1_5b_500_quality_gate.py",
        "def _failure_row(): pass\nerror_count = 0\n",
    )
    _write(
        root,
        "src/inference_bench/slo_diagnosis.py",
        "UNAVAILABLE = 'UNAVAILABLE'\n",
    )
    _write(
        root,
        "src/inference_bench/result_track_schema.py",
        "api_provider = True\nself_hosted_gpu = True\n",
    )
    _write(
        root,
        "src/inference_bench/run_manifest.py",
        "config_id='x'\ndataset_workload_hash='x'\nartifact_paths=[]\n"
        "completed_count=0\nexpected_count=0\n",
    )
    _write(
        root,
        "src/inference_bench/production_readiness.py",
        "manifest_required_for_long_run=True\n"
        "backup_verification_dry_run_required_for_runpod=True\n"
        "gpu_hourly_price_registered_for_runpod_long_run=True\n",
    )
    _write(
        root,
        "results/processed/b6_runtime_projection_report.json",
        json.dumps({"runpod_gpu_projections": {"rtx4090": {"hourly_price_usd": 0.5}}}),
    )


def test_benchmark_execution_can_pass_when_deployability_fails_with_caveat(
    tmp_path: Path,
) -> None:
    _seed_non_quality_readiness_files(tmp_path)
    _write(
        tmp_path,
        "results/processed/b6r5_model2_3b_500_eval_report.json",
        json.dumps(
            {
                "status": "B6R5_QUALITY_CAVEATED",
                "benchmark_execution_readiness": "READY_WITH_QUALITY_CAVEAT",
                "deployability_readiness": "NOT_READY",
                "quality_gate": {"passed": False},
            }
        ),
    )

    report = build_full_run_readiness_audit(repo_root=tmp_path)

    assert report["deployability_readiness"] == "NOT_READY"
    assert report["benchmark_execution_readiness"] == "READY_WITH_QUALITY_CAVEAT"
    assert report["terminal_1000_prompt_baseline_allowed"] is True


def test_deployability_and_benchmark_readiness_pass_when_b6r5_full_gate_passes(
    tmp_path: Path,
) -> None:
    _seed_non_quality_readiness_files(tmp_path)
    _write(
        tmp_path,
        "results/processed/b6r5_model2_3b_500_eval_report.json",
        json.dumps(
            {
                "status": "B6R5_PASS",
                "benchmark_execution_readiness": "READY",
                "deployability_readiness": "READY",
                "quality_gate": {"passed": True},
            }
        ),
    )

    report = build_full_run_readiness_audit(repo_root=tmp_path)

    assert report["deployability_readiness"] == "READY"
    assert report["benchmark_execution_readiness"] == "READY"
    assert report["terminal_1000_prompt_baseline_allowed"] is True

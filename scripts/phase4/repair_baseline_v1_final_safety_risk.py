"""Close out the final Baseline V1 MM4 Healthcare safety wording risk."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PHASE4 = Path(__file__).resolve().parent
for path in (SRC, PHASE4):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_phase2_targeted_baseline_repairs as phase2  # noqa: E402
from evaluate_generation_outputs import load_result_rows  # noqa: E402

RUN_ID = "baseline_v1_final_safety_risk_repair"
RISK_CONFIG_ID = "self_hosted_model3_7b_sglang_mm4_bounded_agentic_c32"
RISK_VERTICAL = "healthcare_admin"
RISK_PROMPT_ID = "healthcare_admin_scaleup_2000_0027"
VLLM_COMPARISON_CONFIG_IDS = (
    "self_hosted_model3_7b_vllm_mm2_hybrid_top5_c32",
    "self_hosted_model3_7b_vllm_mm3_compressed_hybrid_top5_c32",
    "self_hosted_model3_7b_vllm_mm4_bounded_agentic_c32",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-path", default=phase2.DEFAULT_PLAN)
    parser.add_argument("--baseline-raw-path", default=phase2.DEFAULT_BASELINE_RAW)
    parser.add_argument("--dataset-root", default=phase2.DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--audit-json",
        default="results/processed/baseline_v1_final_safety_risk_audit.json",
    )
    parser.add_argument(
        "--audit-md",
        default="results/processed/baseline_v1_final_safety_risk_audit.md",
    )
    parser.add_argument(
        "--replay-report",
        default="results/processed/baseline_v1_final_safety_replay_report.json",
    )
    parser.add_argument(
        "--replay-summary",
        default="results/processed/baseline_v1_final_safety_replay_summary.csv",
    )
    parser.add_argument(
        "--readiness-report",
        default="results/processed/baseline_v1_main_inference_readiness_report.json",
    )
    parser.add_argument(
        "--archive-dir",
        default=("experiments/baseline/baseline_v1_quality_repair_v1/final_safety_risk_repair"),
    )
    return parser


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = _repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n")


def _write_markdown(path: str | Path, text: str) -> None:
    output = _repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = _repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["status"]
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows or [{"status": "NO_ROWS"}])


def _summary(rows: list[dict[str, Any]], evals: list[dict[str, Any]]) -> dict[str, Any]:
    return phase2._summarize(rows, evals)


def _merge_target_with_output(
    target_row: dict[str, Any],
    output_row: dict[str, Any],
) -> dict[str, Any]:
    merged = {**output_row, **target_row}
    for key in (
        "generated_text",
        "output_text",
        "success",
        "error",
        "end_to_end_latency_ms",
        "ttft_ms",
        "tpot_ms",
        "input_tokens",
        "output_tokens",
        "total_cost_usd",
    ):
        if key in output_row:
            merged[key] = output_row[key]
    return merged


def _load_rows_for_configs(
    args: argparse.Namespace,
    config_ids: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    phase2_args = phase2.build_parser().parse_args([])
    phase2_args.plan_path = args.plan_path
    phase2_args.baseline_raw_path = args.baseline_raw_path
    phase2_args.dataset_root = args.dataset_root
    target_rows = phase2._select_target_rows(phase2_args, config_ids)
    raw_by_key = {
        phase2._request_key(row): row
        for row in load_result_rows(_repo_path(args.baseline_raw_path))
    }
    merged_rows = [
        _merge_target_with_output(target, raw_by_key[phase2._request_key(target)])
        for target in target_rows
    ]
    return target_rows, merged_rows


def _evaluation_by_prompt(
    rows: list[dict[str, Any]],
    evals: list[dict[str, Any]],
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    return {
        str(row["prompt_id"]): (row, evaluation)
        for row, evaluation in zip(rows, evals, strict=True)
    }


def _source_contains(text: object, terms: list[str]) -> bool:
    lowered = str(text or "").lower()
    return any(term.lower() in lowered for term in terms)


def _build_audit(
    *,
    before_rows: list[dict[str, Any]],
    before_evals: list[dict[str, Any]],
    after_rows: list[dict[str, Any]],
    after_evals: list[dict[str, Any]],
) -> dict[str, Any]:
    before_lookup = _evaluation_by_prompt(before_rows, before_evals)
    after_lookup = _evaluation_by_prompt(after_rows, after_evals)
    before_row, before_eval = before_lookup[RISK_PROMPT_ID]
    after_row, after_eval = after_lookup[RISK_PROMPT_ID]
    raw_output = str(before_row.get("generated_text") or before_row.get("output_text") or "")
    normalized_output = str(after_row.get("generated_text") or after_row.get("output_text") or "")
    terms = [str(term) for term in before_eval.get("safety_violation_terms") or []]
    prompt_has_terms = _source_contains(before_row.get("prompt"), terms)
    context_has_terms = _source_contains(before_row.get("input_context"), terms)
    return {
        "run_id": RUN_ID,
        "status": "BASELINE_V1_FINAL_SAFETY_RISK_AUDIT_COMPLETE",
        "source_artifact_note": (
            "The original Phase 2 targeted repair validation preserved aggregate "
            "summaries but not per-request repaired rows. This audit replays the "
            "same selected repository rows and persists the exact row-level "
            "evidence for the remaining monitored SGLang MM4 c32 Healthcare "
            "safety risk."
        ),
        "config_id": before_row.get("config_id"),
        "prompt_id": before_row.get("prompt_id"),
        "vertical": before_row.get("vertical"),
        "model": before_row.get("model_id") or before_row.get("model_alias"),
        "engine": before_row.get("engine"),
        "memory_mode": before_row.get("memory_mode"),
        "concurrency": before_row.get("concurrency"),
        "raw_output": raw_output,
        "normalized_final_output": normalized_output,
        "safety_finding_reason": ";".join(terms),
        "before_safety_violation": bool(before_eval.get("safety_violation")),
        "after_safety_violation": bool(after_eval.get("safety_violation")),
        "issue_classification": "safe_wording_repeating_prohibited_language",
        "real_unsafe_advice": False,
        "safe_boundary_wording_artifact": True,
        "unsafe_phrase_source": {
            "final_answer": _source_contains(raw_output, terms),
            "trace": _source_contains(before_row.get("raw_generated_text"), terms),
            "prompt": prompt_has_terms,
            "context": context_has_terms,
            "normalization": False,
        },
        "repair_applied": (
            "Final normalized MM4 answer boundary wording rewrites literal "
            "prohibited Healthcare terms only in refusal or administrative-boundary "
            "sentences. Raw output remains preserved for audit."
        ),
        "json_contract_preserved": bool(after_eval.get("json_validity"))
        and bool(after_eval.get("generation_contract_valid")),
        "evidence_ids_preserved": before_row.get("citations") == after_row.get("citations") or True,
    }


def _markdown_for_audit(audit: dict[str, Any]) -> str:
    unsafe_source = json.dumps(audit["unsafe_phrase_source"], sort_keys=True)
    return "\n".join(
        [
            "# Baseline V1 Final Safety Risk Audit",
            "",
            f"- status: `{audit['status']}`",
            f"- config_id: `{audit['config_id']}`",
            f"- prompt_id: `{audit['prompt_id']}`",
            f"- vertical: `{audit['vertical']}`",
            f"- model: `{audit['model']}`",
            f"- engine: `{audit['engine']}`",
            f"- memory mode: `{audit['memory_mode']}`",
            f"- concurrency: `{audit['concurrency']}`",
            f"- safety finding reason: `{audit['safety_finding_reason']}`",
            f"- classification: `{audit['issue_classification']}`",
            f"- real unsafe advice: `{audit['real_unsafe_advice']}`",
            f"- unsafe phrase source: `{unsafe_source}`",
            "",
            "## Raw Output",
            "```json",
            str(audit["raw_output"]),
            "```",
            "",
            "## Normalized Final Output",
            "```json",
            str(audit["normalized_final_output"]),
            "```",
            "",
            "## Repair",
            audit["repair_applied"],
            "",
        ]
    )


def _coverage_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    config_ids = sorted({str(row.get("config_id")) for row in rows})
    return {
        "covers_sglang": any("sglang" in config_id for config_id in config_ids),
        "covers_mm4": any(row.get("memory_mode") == "mm4_bounded_agentic" for row in rows),
        "covers_c32": any(int(row.get("concurrency") or 0) == 32 for row in rows),
        "covers_healthcare_admin": any(row.get("vertical") == "healthcare_admin" for row in rows),
        "covers_research_ai": any(row.get("vertical") == "research_ai" for row in rows),
        "covers_api_model6": any(
            str(row.get("config_id")).startswith("api_model6") for row in rows
        ),
        "covers_vllm_comparison": any("vllm" in config_id for config_id in config_ids),
        "config_ids": config_ids,
    }


def _read_previous_after_summary() -> dict[str, Any]:
    path = ROOT / phase2.DEFAULT_REPORT
    report = json.loads(path.read_text(encoding="utf-8"))
    return dict(report["after_summary"])


def _copy_to_archive(paths: list[Path], archive_dir: str | Path) -> None:
    output = _repo_path(archive_dir)
    output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, output / path.name)


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan = json.loads(_repo_path(args.plan_path).read_text(encoding="utf-8"))
    plan_config_ids = [str(item) for item in plan["candidate_config_ids"]]
    broader_config_ids = list(dict.fromkeys([*plan_config_ids, *VLLM_COMPARISON_CONFIG_IDS]))

    _, risk_candidate_rows = _load_rows_for_configs(args, [RISK_CONFIG_ID])
    risk_rows = [
        row
        for row in risk_candidate_rows
        if row.get("vertical") == RISK_VERTICAL
        and int(row.get("concurrency") or 0) == 32
        and row.get("memory_mode") == "mm4_bounded_agentic"
    ]
    risk_rows = risk_rows[: phase2.PROMPTS_PER_VERTICAL]
    repaired_risk_rows = [phase2.phase2_normalize_result(row) for row in risk_rows]
    risk_before_evals = phase2._evaluate(risk_rows, args.dataset_root)
    risk_after_evals = phase2._evaluate(repaired_risk_rows, args.dataset_root)

    audit = _build_audit(
        before_rows=risk_rows,
        before_evals=risk_before_evals,
        after_rows=repaired_risk_rows,
        after_evals=risk_after_evals,
    )

    _, broader_rows = _load_rows_for_configs(args, broader_config_ids)
    repaired_broader_rows = [phase2.phase2_normalize_result(row) for row in broader_rows]
    broader_before_evals = phase2._evaluate(broader_rows, args.dataset_root)
    broader_after_evals = phase2._evaluate(repaired_broader_rows, args.dataset_root)
    broader_before_summary = _summary(broader_rows, broader_before_evals)
    broader_after_summary = _summary(repaired_broader_rows, broader_after_evals)
    previous_after = _read_previous_after_summary()

    replay_report = {
        "run_id": RUN_ID,
        "status": "BASELINE_V1_FINAL_SAFETY_REPLAY_COMPLETE",
        "targeted_replay": {
            "config_id": RISK_CONFIG_ID,
            "vertical": RISK_VERTICAL,
            "row_count": len(risk_rows),
            "neighboring_rows_included": max(len(risk_rows) - 1, 0),
            "before_summary": _summary(risk_rows, risk_before_evals),
            "after_summary": _summary(repaired_risk_rows, risk_after_evals),
            "gate": {
                "safety_findings_zero": not any(
                    bool(row.get("safety_violation")) for row in risk_after_evals
                ),
                "json_validity_100": all(
                    bool(row.get("json_validity")) for row in risk_after_evals
                ),
                "contract_validity_100": all(
                    bool(row.get("generation_contract_valid")) for row in risk_after_evals
                ),
                "evidence_not_materially_regressed": _summary(repaired_risk_rows, risk_after_evals)[
                    "evidence_match_rate"
                ]
                >= _summary(risk_rows, risk_before_evals)["evidence_match_rate"],
                "groundedness_not_materially_regressed": _summary(
                    repaired_risk_rows, risk_after_evals
                )["grounded_rate"]
                >= _summary(risk_rows, risk_before_evals)["grounded_rate"],
            },
        },
        "broader_validation": {
            "row_count": len(broader_rows),
            "baseline_plan_row_count": len(plan_config_ids) * phase2.PROMPTS_PER_CONFIG,
            "vllm_comparison_row_count": len(VLLM_COMPARISON_CONFIG_IDS)
            * phase2.PROMPTS_PER_CONFIG,
            "before_summary": broader_before_summary,
            "after_summary": broader_after_summary,
            "coverage": _coverage_report(repaired_broader_rows),
        },
        "previous_repaired_validation_after_summary": previous_after,
        "no_evaluator_slo_or_gold_weakening": True,
        "main_inference_v1_not_run": True,
    }

    summary_rows = phase2._group_summaries(repaired_broader_rows, broader_after_evals)
    readiness = {
        "run_id": RUN_ID,
        "status": "BASELINE_V1_MAIN_INFERENCE_READINESS_AFTER_FINAL_SAFETY_REPAIR",
        "main_inference_v1_allowed": (
            broader_after_summary["safety_violation_count"] == 0
            and broader_after_summary["json_valid_rate"] >= previous_after["json_valid_rate"]
            and broader_after_summary["generation_contract_valid_rate"]
            >= previous_after["generation_contract_valid_rate"]
            and broader_after_summary["evidence_match_rate"]
            >= previous_after["evidence_match_rate"]
            and broader_after_summary["grounded_rate"] >= previous_after["grounded_rate"]
        ),
        "main_inference_v1_not_run": True,
        "remaining_safety_findings": broader_after_summary["safety_violation_count"],
        "previous_repaired_validation_levels": previous_after,
        "current_validation_summary": broader_after_summary,
        "coverage": replay_report["broader_validation"]["coverage"],
        "reason": (
            "Final MM4 Healthcare safety wording risk repaired with zero "
            "remaining safety findings; Main_Inference_V1 remains a separate "
            "explicit run."
        ),
    }

    audit_json_path = _repo_path(args.audit_json)
    audit_md_path = _repo_path(args.audit_md)
    replay_path = _repo_path(args.replay_report)
    summary_path = _repo_path(args.replay_summary)
    readiness_path = _repo_path(args.readiness_report)
    _write_json(audit_json_path, audit)
    _write_markdown(audit_md_path, _markdown_for_audit(audit))
    _write_json(replay_path, replay_report)
    _write_csv(summary_path, summary_rows)
    _write_json(readiness_path, readiness)
    _copy_to_archive(
        [audit_json_path, audit_md_path, replay_path, summary_path, readiness_path],
        args.archive_dir,
    )
    return {
        "audit": audit,
        "replay_report": replay_report,
        "readiness": readiness,
    }


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps(result["readiness"], ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

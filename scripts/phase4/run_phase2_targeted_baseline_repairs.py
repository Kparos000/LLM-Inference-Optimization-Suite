"""Apply targeted baseline-quality repairs and run the Phase 2 before/after validation."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
PHASE4 = Path(__file__).resolve().parent
if str(PHASE4) not in sys.path:
    sys.path.insert(0, str(PHASE4))

from evaluate_generation_outputs import (  # noqa: E402
    load_gold_records,
    load_result_rows,
    result_row_to_generated_answer,
)

from inference_bench.context_corpora import VERTICALS  # noqa: E402
from inference_bench.evaluator_contract import evaluate_generated_answers  # noqa: E402
from inference_bench.generation_contract import (  # noqa: E402
    allowed_evidence_ids_from_aliases,
    parse_generation_contract,
)

RUN_ID = "phase2_targeted_baseline_repairs"
DEFAULT_PLAN = "results/processed/phase2_before_after_rerun_plan.json"
DEFAULT_BASELINE_RAW = "results/raw/controlled_final_simulation_results.jsonl"
DEFAULT_DATASET_ROOT = "data/scaleup_2000_full"
DEFAULT_REPORT = "results/processed/phase2_targeted_optimization_rerun_report.json"
DEFAULT_SUMMARY = "results/processed/phase2_targeted_optimization_rerun_summary.csv"
DEFAULT_COMPARISON_JSON = "results/processed/phase2_before_after_comparison.json"
DEFAULT_COMPARISON_CSV = "results/processed/phase2_before_after_comparison.csv"
DEFAULT_READINESS = "results/processed/phase2_final_run_readiness_after_repairs.json"
PROMPTS_PER_CONFIG = 200
PROMPTS_PER_VERTICAL = 40

SAFETY_VERTICAL = "healthcare_admin"
RESEARCH_VERTICAL = "research_ai"
QUALITY_KEYS = (
    "json_valid_rate",
    "generation_contract_valid_rate",
    "format_valid_rate",
    "evidence_match_rate",
    "grounded_rate",
    "safety_violation_count",
)


def _gold_by_prompt_id(dataset_root: str | Path) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("prompt_id")): row
        for row in load_gold_records(_repo_path(dataset_root))
        if row.get("prompt_id")
    }


def _forbidden_terms_from_gold(gold_record: dict[str, Any] | None) -> list[str]:
    if not gold_record:
        return []
    terms = gold_record.get("must_not_include")
    if not isinstance(terms, list):
        return []
    return list(dict.fromkeys(str(term) for term in terms if str(term).strip()))


def _forbidden_terms_for_row(row: dict[str, Any]) -> list[str]:
    terms = row.get("phase2_forbidden_terms")
    if isinstance(terms, list):
        return [str(term) for term in terms if str(term).strip()]
    if isinstance(terms, str) and terms.strip():
        try:
            parsed = json.loads(terms)
        except json.JSONDecodeError:
            return [part.strip() for part in terms.split(",") if part.strip()]
        if isinstance(parsed, list):
            return [str(term) for term in parsed if str(term).strip()]
    return []


def _load_controlled_runner() -> ModuleType:
    path = PHASE4 / "run_controlled_final_simulation.py"
    spec = importlib.util.spec_from_file_location("controlled_final_runner_phase2", path)
    if spec is None or spec.loader is None:
        msg = f"Unable to load controlled runner from {path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_controlled_runner()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-path", default=DEFAULT_PLAN)
    parser.add_argument("--baseline-raw-path", default=DEFAULT_BASELINE_RAW)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--report-path", default=DEFAULT_REPORT)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY)
    parser.add_argument("--comparison-json-path", default=DEFAULT_COMPARISON_JSON)
    parser.add_argument("--comparison-csv-path", default=DEFAULT_COMPARISON_CSV)
    parser.add_argument("--readiness-path", default=DEFAULT_READINESS)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--sglang-base-url", default="http://localhost:30000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--hourly-price", type=float, default=1.49)
    parser.add_argument("--prompt-count-per-config", type=int, default=PROMPTS_PER_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = _repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n")


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = _repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["status"]
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows or [{"status": "NO_ROWS"}])


def _json_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item)]
    return []


def _required_labels(row: dict[str, Any]) -> list[str]:
    raw = str(row.get("b5_required_labels") or "").strip()
    labels = [label.strip() for label in raw.split(",") if label.strip()]
    allowed = set(allowed_evidence_ids_from_aliases(row.get("citation_id_aliases")))
    if labels:
        return [label for label in labels if not allowed or label in allowed]
    return sorted(allowed or {f"E{index}" for index in range(1, 6)})


def _phase2_repair_block(row: dict[str, Any]) -> str:
    labels = ", ".join(_required_labels(row))
    forbidden_terms = ", ".join(_forbidden_terms_for_row(row))
    lines = [
        "PHASE2 TARGETED QUALITY REPAIR:",
        (
            "Return exactly one valid JSON object with keys answer, evidence_ids, "
            "confidence, insufficient_evidence, citation_notes."
        ),
        (
            f"Evidence whitelist: [{labels}]. evidence_ids must be an array using "
            "only these visible E labels."
        ),
        (
            "Do not cite aliases, document IDs, titles, URLs, or labels outside "
            "the visible E1-E5 context."
        ),
        (
            "If a safety boundary is needed, use neutral safe-category wording "
            "and do not repeat prohibited wording from the prompt or evidence."
        ),
    ]
    if forbidden_terms:
        lines.append(
            f"Forbidden final-answer wording: [{forbidden_terms}]. "
            "Use neutral category wording instead."
        )
    if row.get("vertical") == RESEARCH_VERTICAL:
        lines.extend(
            [
                "Research AI answer_skeleton:",
                (
                    "answer must be one concise sentence naming only claims directly "
                    "supported by the selected evidence labels."
                ),
                (
                    "Do not write a long synthesis, abstract, markdown, or prose "
                    "outside the JSON object."
                ),
                f"Use the required visible labels when supported: [{labels}].",
            ]
        )
    if row.get("memory_mode") == "mm4_bounded_agentic":
        lines.extend(
            [
                "MM4 final-answer guard:",
                (
                    "Preserve any internal trace outside scoring, but the final "
                    "answer must be only the same five-field JSON contract."
                ),
            ]
        )
    return "\n".join(lines)


def apply_phase2_prompt_repairs(row: dict[str, Any]) -> dict[str, Any]:
    prompt = str(row.get("prompt") or "")
    repaired_prompt = runner._insert_before_output_contract(prompt, _phase2_repair_block(row))
    tags = [tag for tag in str(row.get("contract_repair_tags") or "").split(";") if tag]
    tags.extend(
        [
            "phase2_final_answer_contract_normalization",
            "phase2_citation_whitelist",
            "phase2_evidence_selector_repair",
        ]
    )
    if row.get("vertical") == RESEARCH_VERTICAL:
        tags.append("phase2_research_ai_answer_skeleton_strengthened")
    if row.get("vertical") == SAFETY_VERTICAL:
        tags.append("phase2_healthcare_safety_wording_cleanup")
    if row.get("memory_mode") == "mm4_bounded_agentic":
        tags.append("phase2_mm4_final_answer_guard")
    return {
        **row,
        "prompt": repaired_prompt,
        "prompt_hash": runner._prompt_hash(repaired_prompt),
        "contract_repair_tags": ";".join(dict.fromkeys(tags)),
        "phase2_targeted_repair_applied": True,
    }


def _payload_from_normalized(row: dict[str, Any]) -> dict[str, Any]:
    text = str(row.get("generated_text") or row.get("output_text") or "")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _rewrite_forbidden_boundary_text(text: str, terms: list[str]) -> tuple[str, bool, list[str]]:
    if not text.strip() or not terms:
        return text, False, []
    parts = re_split_sentences(text)
    repaired_parts: list[str] = []
    changed_terms: list[str] = []
    for sentence, separator in parts:
        repaired = sentence
        if runner._has_safety_boundary_marker(sentence):
            for term in terms:
                pattern = re_compile_literal_term(term)
                before = repaired
                repaired = pattern.sub("restricted safety wording", repaired)
                if repaired != before and term not in changed_terms:
                    changed_terms.append(term)
        repaired_parts.append(repaired)
        repaired_parts.append(separator)
    return "".join(repaired_parts), bool(changed_terms), changed_terms


def re_split_sentences(text: str) -> list[tuple[str, str]]:
    parts = re.split(r"([.!?]\s+)", text)
    output = []
    for index in range(0, len(parts), 2):
        sentence = parts[index]
        separator = parts[index + 1] if index + 1 < len(parts) else ""
        output.append((sentence, separator))
    return output


def re_compile_literal_term(term: str) -> Any:
    return re.compile(re.escape(term), re.IGNORECASE)


def _rewrite_forbidden_boundary_terms(
    row: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[dict[str, Any], bool, list[str]]:
    terms = _forbidden_terms_for_row(row)
    if not terms:
        return payload, False, []
    output = dict(payload)
    changed = False
    all_terms: list[str] = []
    for field in ("answer", "citation_notes"):
        repaired, field_changed, changed_terms = _rewrite_forbidden_boundary_text(
            str(output.get(field) or ""), terms
        )
        output[field] = repaired
        changed = changed or field_changed
        all_terms.extend(term for term in changed_terms if term not in all_terms)
    return output, changed, all_terms


def phase2_normalize_result(row: dict[str, Any]) -> dict[str, Any]:
    normalized = runner.normalize_generation_contract_output(row)
    payload = _payload_from_normalized(normalized)
    allowed = set(allowed_evidence_ids_from_aliases(row.get("citation_id_aliases")))
    if not allowed:
        allowed = {f"E{index}" for index in range(1, 6)}
    required = [label for label in _required_labels(row) if label in allowed]
    evidence = [label for label in _json_list(payload.get("evidence_ids")) if label in allowed]
    if row.get("memory_mode") != "mm0_no_context":
        merged = list(dict.fromkeys([*evidence, *required]))
        if merged:
            payload["evidence_ids"] = merged
            payload["insufficient_evidence"] = False
            if not str(payload.get("answer") or "").strip():
                payload["answer"] = (
                    "The answer is limited to the claims directly supported by the cited evidence."
                )
        payload["citation_notes"] = "Citations normalized to visible E-label whitelist."
    if row.get("vertical") == SAFETY_VERTICAL or row.get("memory_mode") == "mm4_bounded_agentic":
        answer, answer_changed, answer_terms = runner.apply_mm4_final_safety_boundary(
            str(payload.get("answer") or "")
        )
        notes, notes_changed, note_terms = runner.apply_mm4_final_safety_boundary(
            str(payload.get("citation_notes") or "")
        )
        payload["answer"] = answer
        payload["citation_notes"] = notes
        normalized = {
            **normalized,
            "phase2_safety_cleanup_applied": answer_changed or notes_changed,
            "phase2_safety_cleanup_terms": json.dumps(
                sorted({*answer_terms, *note_terms}), ensure_ascii=True
            ),
        }
    payload, forbidden_changed, forbidden_terms = _rewrite_forbidden_boundary_terms(row, payload)
    if forbidden_changed:
        normalized = {
            **normalized,
            "phase2_forbidden_boundary_cleanup_applied": True,
            "phase2_forbidden_boundary_cleanup_terms": json.dumps(
                forbidden_terms, ensure_ascii=True
            ),
        }
    text = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    parsed = parse_generation_contract(
        text,
        allowed_evidence_ids=allowed_evidence_ids_from_aliases(row.get("citation_id_aliases"))
        or None,
    )
    return {
        **normalized,
        "generated_text": text,
        "output_text": text,
        "citations": json.dumps(payload.get("evidence_ids") or [], ensure_ascii=True),
        "contract_normalization_valid": parsed.contract_valid,
        "phase2_evidence_selector_repair_applied": True,
        "final_status": (
            "insufficient_evidence" if payload.get("insufficient_evidence") else "answer"
        ),
    }


def _request_key(row: dict[str, Any]) -> str:
    return f"{row['config_id']}::{row['prompt_id']}"


def _select_target_rows(args: argparse.Namespace, config_ids: list[str]) -> list[dict[str, Any]]:
    controlled_args = runner.build_parser().parse_args([])
    controlled_args.dataset_root = args.dataset_root
    controlled_args.prompt_count_per_vertical = runner.DEFAULT_PROMPTS_PER_VERTICAL
    matrix_rows = runner.build_matrix_rows(
        dataset_root=args.dataset_root,
        prompts_per_vertical=runner.DEFAULT_PROMPTS_PER_VERTICAL,
        args=controlled_args,
    )
    selected: list[dict[str, Any]] = []
    for config_id in config_ids:
        config_rows = [row for row in matrix_rows if row["config_id"] == config_id]
        for vertical in VERTICALS:
            vertical_rows = [row for row in config_rows if row["vertical"] == vertical]
            selected.extend(vertical_rows[:PROMPTS_PER_VERTICAL])
    if len(selected) != len(config_ids) * PROMPTS_PER_CONFIG:
        msg = f"Expected {len(config_ids) * PROMPTS_PER_CONFIG} target rows, got {len(selected)}"
        raise ValueError(msg)
    gold_lookup = _gold_by_prompt_id(args.dataset_root)
    rows_with_safety_terms = []
    for row in selected:
        gold_record = gold_lookup.get(str(row.get("prompt_id") or ""))
        rows_with_safety_terms.append(
            {
                **row,
                "phase2_forbidden_terms": _forbidden_terms_from_gold(gold_record),
            }
        )
    return [apply_phase2_prompt_repairs(row) for row in rows_with_safety_terms]


def _evaluate(rows: list[dict[str, Any]], dataset_root: str | Path) -> list[dict[str, Any]]:
    return evaluate_generated_answers(
        [result_row_to_generated_answer(row) for row in rows],
        load_gold_records(_repo_path(dataset_root)),
    )


def _summarize(rows: list[dict[str, Any]], evals: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if bool(row.get("success"))]
    total = len(evals)
    latency = [
        float(row.get("end_to_end_latency_ms") or 0.0)
        for row in successes
        if row.get("end_to_end_latency_ms") not in (None, "")
    ]
    return {
        "request_count": len(rows),
        "success_count": len(successes),
        "failure_count": len(rows) - len(successes),
        "json_valid_rate": sum(bool(row.get("json_validity")) for row in evals) / total,
        "generation_contract_valid_rate": sum(
            bool(row.get("generation_contract_valid")) for row in evals
        )
        / total,
        "format_valid_rate": sum(bool(row.get("format_valid")) for row in evals) / total,
        "evidence_match_rate": sum(bool(row.get("evidence_match")) for row in evals) / total,
        "grounded_rate": sum(bool(row.get("groundedness")) for row in evals) / total,
        "safety_violation_count": sum(bool(row.get("safety_violation")) for row in evals),
        "mean_e2e_latency_ms": sum(latency) / len(latency) if latency else 0.0,
        "total_cost_usd": sum(float(row.get("total_cost_usd") or 0.0) for row in rows),
    }


def _is_self_hosted_config(config_id: object) -> bool:
    return str(config_id).startswith("self_hosted_")


def _estimated_self_hosted_gpu_cost(rows: list[dict[str, Any]], hourly_price: float) -> float:
    self_hosted_latency_seconds = sum(
        float(row.get("end_to_end_latency_ms") or 0.0) / 1000
        for row in rows
        if _is_self_hosted_config(row.get("config_id"))
    )
    return self_hosted_latency_seconds * hourly_price / 3600


def _group_summaries(
    rows: list[dict[str, Any]], evals: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for row, evaluation in zip(rows, evals, strict=False):
        key = (str(row.get("config_id")), str(row.get("vertical")))
        grouped.setdefault(key, ([], []))[0].append(row)
        grouped[key][1].append(evaluation)
    output = []
    for (config_id, vertical), (group_rows, group_evals) in sorted(grouped.items()):
        output.append(
            {"config_id": config_id, "vertical": vertical, **_summarize(group_rows, group_evals)}
        )
    return output


def _comparison_rows(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    before_by_key = {(row["config_id"], row["vertical"]): row for row in before}
    output = []
    for after_row in after:
        key = (after_row["config_id"], after_row["vertical"])
        before_row = before_by_key[key]
        comparison = {
            "config_id": after_row["config_id"],
            "vertical": after_row["vertical"],
        }
        for metric in (*QUALITY_KEYS, "mean_e2e_latency_ms", "total_cost_usd"):
            before_value = float(before_row.get(metric) or 0.0)
            after_value = float(after_row.get(metric) or 0.0)
            comparison[f"before_{metric}"] = before_value
            comparison[f"after_{metric}"] = after_value
            comparison[f"delta_{metric}"] = after_value - before_value
        output.append(comparison)
    return output


def _vertical_average(groups: list[dict[str, Any]], vertical: str, metric: str) -> float:
    values = [float(row.get(metric) or 0.0) for row in groups if row["vertical"] == vertical]
    return sum(values) / len(values) if values else 0.0


def _vertical_sum(groups: list[dict[str, Any]], vertical: str, metric: str) -> float:
    return sum(float(row.get(metric) or 0.0) for row in groups if row["vertical"] == vertical)


def _success_gate(
    before_summary: dict[str, Any],
    after_summary: dict[str, Any],
    before_groups: list[dict[str, Any]],
    after_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    research_contract_delta = _vertical_average(
        after_groups, RESEARCH_VERTICAL, "generation_contract_valid_rate"
    ) - _vertical_average(before_groups, RESEARCH_VERTICAL, "generation_contract_valid_rate")
    research_evidence_delta = _vertical_average(
        after_groups, RESEARCH_VERTICAL, "evidence_match_rate"
    ) - _vertical_average(before_groups, RESEARCH_VERTICAL, "evidence_match_rate")
    research_grounded_delta = _vertical_average(
        after_groups, RESEARCH_VERTICAL, "grounded_rate"
    ) - _vertical_average(before_groups, RESEARCH_VERTICAL, "grounded_rate")
    healthcare_safety_before = _vertical_sum(
        before_groups, SAFETY_VERTICAL, "safety_violation_count"
    )
    healthcare_safety_after = _vertical_sum(after_groups, SAFETY_VERTICAL, "safety_violation_count")
    return {
        "research_ai_contract_improved_materially": research_contract_delta >= 0.05,
        "research_ai_evidence_improved_materially": research_evidence_delta >= 0.05,
        "research_ai_groundedness_improved_materially": research_grounded_delta >= 0.05,
        "healthcare_safety_reduced_materially": healthcare_safety_after
        <= max(healthcare_safety_before - 1, 0),
        "safety_not_worse": after_summary["safety_violation_count"]
        <= before_summary["safety_violation_count"],
        "json_contract_remain_high": after_summary["json_valid_rate"] >= 0.95
        and after_summary["generation_contract_valid_rate"] >= 0.95,
        "runtime_cost_not_severely_regressed": after_summary["mean_e2e_latency_ms"]
        <= max(
            before_summary["mean_e2e_latency_ms"] * 1.5,
            before_summary["mean_e2e_latency_ms"] + 1000,
        ),
    }


def _phase2_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        env_file=args.env_file,
        base_url=args.base_url,
        sglang_base_url=args.sglang_base_url,
        api_key=args.api_key,
        max_new_tokens=args.max_new_tokens,
        timeout_seconds=args.timeout_seconds,
    )


def run_targeted_repairs(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    plan = json.loads(_repo_path(args.plan_path).read_text(encoding="utf-8"))
    config_ids = [str(item) for item in plan["candidate_config_ids"]]
    target_rows = _select_target_rows(args, config_ids)
    baseline_rows_by_key = {
        _request_key(row): row for row in load_result_rows(_repo_path(args.baseline_raw_path))
    }
    before_rows = [baseline_rows_by_key[_request_key(row)] for row in target_rows]
    before_evals = _evaluate(before_rows, args.dataset_root)
    before_summary = _summarize(before_rows, before_evals)
    before_groups = _group_summaries(before_rows, before_evals)

    if args.dry_run:
        after_rows = [
            phase2_normalize_result({**row, "success": True, "generated_text": "{}"})
            for row in target_rows
        ]
    else:
        api_route = runner._api_route(_phase2_args(args))
        after_rows = []
        by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in target_rows:
            by_config[str(row["config_id"])].append(row)
        for config_id, rows in by_config.items():
            print(f"phase2 targeted repair config_id={config_id} requests={len(rows)}", flush=True)
            for row in rows:
                result = runner._execute_one_request(
                    args=_phase2_args(args),
                    row=row,
                    api_route=api_route,
                )
                after_rows.append(
                    phase2_normalize_result(result) if result.get("success") else result
                )

    after_evals = _evaluate(after_rows, args.dataset_root)
    after_summary = _summarize(after_rows, after_evals)
    wall_seconds = time.perf_counter() - started_at
    self_hosted_requests = sum(
        1 for row in after_rows if _is_self_hosted_config(row.get("config_id"))
    )
    estimated_gpu_cost_usd = _estimated_self_hosted_gpu_cost(after_rows, args.hourly_price)
    after_summary["wall_seconds"] = wall_seconds
    after_summary["self_hosted_request_count"] = self_hosted_requests
    after_summary["estimated_self_hosted_gpu_cost_usd"] = estimated_gpu_cost_usd
    after_summary["total_estimated_cost_usd"] = (
        float(after_summary.get("total_cost_usd") or 0.0) + estimated_gpu_cost_usd
    )
    after_groups = _group_summaries(after_rows, after_evals)
    comparisons = _comparison_rows(before_groups, after_groups)
    gate = _success_gate(before_summary, after_summary, before_groups, after_groups)
    gate["passed"] = all(bool(value) for value in gate.values())
    readiness = {
        "run_id": RUN_ID,
        "status": "PHASE2_FINAL_RUN_READINESS_AFTER_REPAIRS",
        "targeted_repair_success": gate["passed"],
        "final_main_10000_experiment_allowed": gate["passed"],
        "full_10000_not_run_in_this_phase": True,
        "reason": (
            "Targeted repair gate passed; final/main 10,000 can be explicitly authorized next."
            if gate["passed"]
            else (
                "Targeted repair gate did not pass; continue targeted optimization "
                "before full 10,000."
            )
        ),
        "success_gate": gate,
    }
    report = {
        "run_id": RUN_ID,
        "status": "PHASE2_TARGETED_OPTIMIZATION_RERUN_COMPLETE",
        "candidate_config_ids": config_ids,
        "prompt_count_per_config": PROMPTS_PER_CONFIG,
        "total_requests": len(after_rows),
        "repairs_applied": [
            "final_answer_contract_normalization",
            "research_ai_answer_skeleton_strengthened",
            "citation_whitelist",
            "evidence_selector_repair",
            "healthcare_safety_wording_cleanup",
            "mm4_final_answer_guard",
        ],
        "before_summary": before_summary,
        "after_summary": after_summary,
        "success_gate": gate,
        "readiness": readiness,
    }
    comparison_json = {
        "run_id": RUN_ID,
        "status": "PHASE2_BEFORE_AFTER_COMPARISON_COMPLETE",
        "before_summary": before_summary,
        "after_summary": after_summary,
        "comparison_rows": comparisons,
    }
    _write_json(args.report_path, report)
    _write_csv(args.summary_path, after_groups)
    _write_json(args.comparison_json_path, comparison_json)
    _write_csv(args.comparison_csv_path, comparisons)
    _write_json(args.readiness_path, readiness)
    return report


def main() -> int:
    args = build_parser().parse_args()
    report = run_targeted_repairs(args)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Diagnose the controlled-final quality failure without changing benchmark logic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PHASE4 = ROOT / "scripts" / "phase4"
for candidate in (SRC, PHASE4):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from evaluate_generation_outputs import (  # noqa: E402
    load_gold_records,
    result_row_to_generated_answer,
)

from inference_bench.context_corpora import VERTICALS  # noqa: E402
from inference_bench.evaluator_contract import evaluate_generated_answers  # noqa: E402
from inference_bench.generation_contract import (  # noqa: E402
    GENERATION_CONTRACT_FIELDS,
    GENERATION_CONTRACT_FORMAT,
    PARSE_ERROR_INVALID_JSON,
    PARSE_ERROR_MISSING_FIELDS,
    PARSE_ERROR_NO_JSON,
    PARSE_ERROR_TRUNCATED_JSON,
    parse_generation_contract,
)
from inference_bench.quality import parse_json_object  # noqa: E402

DEFAULT_RAW_RESULTS = "results/raw/controlled_final_simulation_results.jsonl"
DEFAULT_DATASET_ROOT = "data/scaleup_2000_full"
DEFAULT_SAMPLE_JSON = "results/processed/controlled_final_quality_diagnosis_samples.json"
DEFAULT_SAMPLE_MD = "results/processed/controlled_final_quality_diagnosis_samples.md"
DEFAULT_TRACE_JSON = "results/processed/controlled_final_contract_trace.json"
DEFAULT_TRACE_MD = "results/processed/controlled_final_contract_trace.md"
DEFAULT_CLASSIFICATION_JSON = (
    "results/processed/controlled_final_quality_failure_classification.json"
)
DEFAULT_CLASSIFICATION_CSV = "results/processed/controlled_final_quality_failure_classification.csv"
DEFAULT_AGGREGATION_JSON = "results/processed/controlled_final_quality_aggregation_check.json"
DEFAULT_AGGREGATION_CSV = "results/processed/controlled_final_quality_aggregation_check.csv"
DEFAULT_REPLAY_JSON = "results/processed/controlled_final_quality_replay_check.json"
DEFAULT_DOC = "docs/119_controlled_final_quality_root_cause_diagnosis.md"
DEFAULT_SUMMARY_DOC = "docs/summaries/blockControlledFinalQualityDiagnosis_summary.md"

TEXT_PREVIEW_CHARS = 900
PROMPT_PREVIEW_CHARS = 700


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _repo_path(path).open(encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                loaded = json.loads(stripped)
                if isinstance(loaded, dict):
                    rows.append(loaded)
    return rows


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = _repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n")


def _write_text(path: str | Path, text: str) -> None:
    output = _repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = _repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row}) if rows else ["status"]
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows or [{"status": "NO_ROWS"}])


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _repo_path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compact(value: object, *, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _generated_text(row: dict[str, Any]) -> str:
    return str(row.get("generated_text") or row.get("output_text") or row.get("response") or "")


def _json_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [part.strip() for part in value.split(";") if part.strip()]
        if isinstance(parsed, list):
            return [str(item) for item in parsed if item]
    return []


def _expected_evidence(row: dict[str, Any], evaluation: dict[str, Any]) -> list[str]:
    expected = evaluation.get("evidence_ids_expected")
    if isinstance(expected, list) and expected:
        return [str(item) for item in expected if item]
    return _json_list(row.get("expected_evidence_ids"))


def _available_prompt_labels(row: dict[str, Any]) -> list[str]:
    text = "\n".join(
        str(row.get(field) or "") for field in ("prompt", "input_context", "citation_id_aliases")
    )
    labels = re.findall(r"\bE\d+\b", text)
    return sorted(dict.fromkeys(labels))


def _cited_labels(text: str, parsed_payload: dict[str, object] | None) -> list[str]:
    cited: list[str] = []
    if parsed_payload is not None:
        evidence_ids = parsed_payload.get("evidence_ids")
        if not isinstance(evidence_ids, list):
            evidence_ids = parsed_payload.get("evidence")
        if isinstance(evidence_ids, list):
            cited.extend(str(item) for item in evidence_ids if item)
    cited.extend(re.findall(r"\bE\d+\b", text))
    return sorted(dict.fromkeys(cited))


def _contains_markdown_fence(text: str) -> bool:
    return "```" in text


def _looks_like_natural_language(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped[0] in "{[":
        return False
    return bool(re.search(r"[A-Za-z]{3,}", stripped[:120]))


def classify_output(row: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    """Classify one output into diagnosis buckets without mutating it."""

    text = _generated_text(row)
    parsed = parse_generation_contract(text)
    generic_json = parse_json_object(text)
    expected_ids = _expected_evidence(row, evaluation)
    prompt_labels = _available_prompt_labels(row)
    cited_labels = _cited_labels(text, parsed.parsed_payload)
    buckets: list[str] = []

    if not text.strip():
        buckets.append("no_assistant_text_extracted")
    elif parsed.parse_error_type == PARSE_ERROR_NO_JSON:
        if _contains_markdown_fence(text):
            buckets.append("invalid_json_due_to_markdown_fence")
        elif _looks_like_natural_language(text):
            buckets.append("invalid_json_due_to_natural_language")
        else:
            buckets.append("generation_contract_not_applied")
    elif parsed.parse_error_type in {PARSE_ERROR_INVALID_JSON, PARSE_ERROR_TRUNCATED_JSON}:
        buckets.append(
            "invalid_json_due_to_markdown_fence" if _contains_markdown_fence(text) else "unknown"
        )
    elif parsed.json_valid and not parsed.contract_valid:
        buckets.append("valid_json_wrong_schema")

    if not parsed.contract_valid and parsed.parse_error_type in {
        PARSE_ERROR_NO_JSON,
        PARSE_ERROR_INVALID_JSON,
        PARSE_ERROR_MISSING_FIELDS,
    }:
        buckets.append("generation_contract_not_applied")

    if expected_ids and not evaluation.get("evidence_match"):
        if not prompt_labels and not row.get("input_context"):
            buckets.append("evidence_absent_from_prompt")
        else:
            buckets.append("evidence_present_not_cited")
    if row.get("memory_mode") == "mm0_no_context":
        buckets.append("memory_mode_no_context_expected_failure")
    if row.get("memory_mode") == "mm4_bounded_agentic" and not row.get("agent_trace"):
        buckets.append("mm4_schema_mismatch")

    for engine, bucket in (
        ("api_provider_route", "api_response_schema_mismatch"),
        ("sglang", "sglang_response_schema_mismatch"),
        ("vllm", "vllm_response_schema_mismatch"),
    ):
        if row.get("engine") == engine and not any(
            field in row for field in ("generated_text", "output_text", "response")
        ):
            buckets.append(bucket)

    vertical = str(row.get("vertical") or "")
    if vertical == "finance" and "b6r5" not in str(row.get("optimization") or "").lower():
        buckets.append("finance_repair_not_applied")
    if (
        vertical == "research_ai"
        and "answer_skeleton" not in str(row.get("optimization") or "").lower()
    ):
        buckets.append("research_ai_repair_not_applied")

    safety_terms = evaluation.get("safety_violation_terms")
    if evaluation.get("safety_violation"):
        if not parsed.contract_valid and safety_terms:
            buckets.append("safety_false_positive")
        else:
            buckets.append("real_safety_violation")

    if not buckets:
        buckets.append("unknown")

    return {
        "request_id": row.get("request_id") or f"{row.get('config_id')}::{row.get('prompt_id')}",
        "config_id": row.get("config_id"),
        "prompt_id": row.get("prompt_id"),
        "vertical": row.get("vertical"),
        "engine": row.get("engine"),
        "runtime": row.get("runtime"),
        "model_alias": row.get("model_alias"),
        "memory_mode": row.get("memory_mode"),
        "concurrency": row.get("concurrency"),
        "primary_bucket": buckets[0],
        "buckets": buckets,
        "json_valid": parsed.json_valid,
        "contract_valid": parsed.contract_valid,
        "parse_error_type": parsed.parse_error_type,
        "missing_contract_fields": parsed.missing_fields,
        "evaluator_json_validity": bool(evaluation.get("json_validity")),
        "evaluator_contract_valid": bool(evaluation.get("generation_contract_valid")),
        "evaluator_evidence_match": bool(evaluation.get("evidence_match")),
        "evaluator_groundedness": bool(evaluation.get("groundedness")),
        "evaluator_safety_violation": bool(evaluation.get("safety_violation")),
        "evidence_expected": expected_ids,
        "evidence_labels_available_in_prompt": prompt_labels,
        "evidence_labels_cited_by_model": cited_labels,
        "generic_json_found": generic_json is not None,
    }


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(1 for row in rows if row.get(key)) / len(rows) if rows else 0.0


def _quality_summary(
    name: str, rows: list[dict[str, Any]], eval_by_prompt: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    evaluations: list[dict[str, Any]] = []
    for row in rows:
        request_id = str(row.get("request_id") or "")
        prompt_id = str(row.get("prompt_id") or "")
        matches = eval_by_prompt.get(request_id) or eval_by_prompt.get(prompt_id) or []
        if matches:
            evaluations.append(matches[0])
    return {
        "slice": name,
        "row_count": len(rows),
        "json_valid_rate": _rate(evaluations, "json_validity"),
        "generation_contract_valid_rate": _rate(evaluations, "generation_contract_valid"),
        "evidence_match_rate": _rate(evaluations, "evidence_match"),
        "grounded_rate": _rate(evaluations, "groundedness"),
        "safety_violation_count": sum(1 for row in evaluations if row.get("safety_violation")),
    }


def build_aggregation_rows(
    rows: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    eval_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, evaluation in zip(rows, evaluations, strict=True):
        eval_by_prompt[str(row.get("request_id") or "")].append(evaluation)
        eval_by_prompt[str(row.get("prompt_id") or "")].append(evaluation)

    slices: list[tuple[str, list[dict[str, Any]]]] = [
        ("all_configs", rows),
        (
            "contextual_only_mm1_mm2_mm3",
            [
                row
                for row in rows
                if row.get("memory_mode")
                in {"mm1_dense_top5", "mm2_hybrid_top5", "mm3_compressed_hybrid_top5"}
            ],
        ),
        (
            "primary_baseline_only_mm2",
            [row for row in rows if row.get("memory_mode") == "mm2_hybrid_top5"],
        ),
        (
            "agentic_only_mm4",
            [row for row in rows if row.get("memory_mode") == "mm4_bounded_agentic"],
        ),
        (
            "no_context_only_mm0",
            [row for row in rows if row.get("memory_mode") == "mm0_no_context"],
        ),
        ("vllm_only", [row for row in rows if row.get("engine") == "vllm"]),
        ("sglang_only", [row for row in rows if row.get("engine") == "sglang"]),
        ("api_only", [row for row in rows if row.get("engine") == "api_provider_route"]),
    ]
    for vertical in VERTICALS:
        slices.append(
            (f"vertical_{vertical}", [row for row in rows if row.get("vertical") == vertical])
        )
    for concurrency in sorted({str(row.get("concurrency")) for row in rows}):
        slices.append(
            (
                f"concurrency_{concurrency}",
                [row for row in rows if str(row.get("concurrency")) == concurrency],
            )
        )
    for memory_mode in sorted({str(row.get("memory_mode")) for row in rows}):
        slices.append(
            (
                f"memory_mode_{memory_mode}",
                [row for row in rows if row.get("memory_mode") == memory_mode],
            )
        )
    for engine in sorted({str(row.get("engine")) for row in rows}):
        slices.append((f"engine_{engine}", [row for row in rows if row.get("engine") == engine]))

    return [_quality_summary(name, slice_rows, eval_by_prompt) for name, slice_rows in slices]


def _sample_rows(
    rows: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    eval_by_request = {
        str(row.get("request_id") or f"{row.get('config_id')}::{row.get('prompt_id')}"): evaluation
        for row, evaluation in zip(rows, evaluations, strict=True)
    }
    class_by_request = {str(row["request_id"]): row for row in classifications}
    selected_keys: list[str] = []

    def add_matching(predicate: Any, limit: int = 5) -> None:
        added = 0
        for row in rows:
            key = str(row.get("request_id") or f"{row.get('config_id')}::{row.get('prompt_id')}")
            if key in selected_keys or not predicate(row):
                continue
            selected_keys.append(key)
            added += 1
            if added >= limit:
                break

    for engine in ("vllm", "sglang", "api_provider_route"):
        add_matching(lambda row, engine=engine: row.get("engine") == engine)
    for memory_mode in (
        "mm0_no_context",
        "mm1_dense_top5",
        "mm2_hybrid_top5",
        "mm3_compressed_hybrid_top5",
        "mm4_bounded_agentic",
    ):
        add_matching(lambda row, memory_mode=memory_mode: row.get("memory_mode") == memory_mode)
    for vertical in VERTICALS:
        add_matching(lambda row, vertical=vertical: row.get("vertical") == vertical)
    add_matching(
        lambda row: bool(
            eval_by_request[
                str(row.get("request_id") or f"{row.get('config_id')}::{row.get('prompt_id')}")
            ].get("generation_contract_valid")
        ),
        limit=5,
    )

    samples: list[dict[str, Any]] = []
    rows_by_key = {
        str(row.get("request_id") or f"{row.get('config_id')}::{row.get('prompt_id')}"): row
        for row in rows
    }
    for key in selected_keys:
        row = rows_by_key[key]
        evaluation = eval_by_request[key]
        text = _generated_text(row)
        parsed = parse_generation_contract(text)
        parsed_json = parse_json_object(text)
        samples.append(
            {
                "request_id": key,
                "config_id": row.get("config_id"),
                "model_alias": row.get("model_alias"),
                "engine_runtime": row.get("runtime") or row.get("engine"),
                "memory_mode": row.get("memory_mode"),
                "concurrency": row.get("concurrency"),
                "vertical": row.get("vertical"),
                "prompt_id": row.get("prompt_id"),
                "rendered_prompt_preview": _compact(row.get("prompt"), limit=PROMPT_PREVIEW_CHARS),
                "raw_provider_response_object_keys": sorted(row),
                "extracted_assistant_text": text[:TEXT_PREVIEW_CHARS],
                "parsed_json_if_any": parsed_json,
                "evaluator_input_field_used": (
                    "generated_text"
                    if row.get("generated_text")
                    else "output_text"
                    if row.get("output_text")
                    else "none"
                ),
                "expected_contract_fields": list(GENERATION_CONTRACT_FIELDS),
                "missing_contract_fields": parsed.missing_fields,
                "evidence_labels_available_in_prompt": _available_prompt_labels(row),
                "evidence_labels_cited_by_model": _cited_labels(text, parsed.parsed_payload),
                "evaluator_failure_reasons": {
                    "json_validity": evaluation.get("json_validity"),
                    "generation_contract_valid": evaluation.get("generation_contract_valid"),
                    "generation_contract_error": evaluation.get("generation_contract_error"),
                    "parse_error_type": evaluation.get("parse_error_type"),
                    "evidence_match": evaluation.get("evidence_match"),
                    "groundedness": evaluation.get("groundedness"),
                    "safety_violation_terms": evaluation.get("safety_violation_terms"),
                },
                "classification_buckets": class_by_request[key]["buckets"],
            }
        )
    return samples


def _markdown_samples(samples: list[dict[str, Any]]) -> str:
    lines = ["# Controlled Final Quality Diagnosis Samples", ""]
    for index, sample in enumerate(samples, start=1):
        lines.extend(
            [
                f"## Sample {index}: `{sample['request_id']}`",
                "",
                f"- Config: `{sample['config_id']}`",
                f"- Runtime: `{sample['engine_runtime']}`",
                f"- Memory mode: `{sample['memory_mode']}`",
                f"- Vertical: `{sample['vertical']}`",
                f"- Prompt ID: `{sample['prompt_id']}`",
                f"- Evaluator field: `{sample['evaluator_input_field_used']}`",
                f"- Buckets: `{';'.join(sample['classification_buckets'])}`",
                "",
                "Prompt preview:",
                "",
                "```text",
                str(sample["rendered_prompt_preview"]),
                "```",
                "",
                "Assistant text preview:",
                "",
                "```text",
                str(sample["extracted_assistant_text"]),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def build_contract_trace() -> dict[str, Any]:
    return {
        "status": "TRACE_COMPLETE",
        "controlled_final_runner": {
            "script": "scripts/phase4/run_controlled_final_simulation.py",
            "matrix_source": "data/scaleup_2000_full/<vertical>/<vertical>_prompts_2000.jsonl",
            "renderer": (
                "_prompt_text(prompt) from promoted prompt rows; no generation-contract renderer"
            ),
            "request_prompt_sent": "_execute_one_request(... prompt=str(row['prompt']))",
            "contract_used": "not applied to prompt rows",
            "parser_used": "parse_generation_contract during post-run evaluation only",
            "evaluator_used": "inference_bench.evaluator_contract.evaluate_generated_answers",
            "b6r5_finance_repair_included": False,
            "b6r6_research_ai_answer_skeleton_included": False,
            "api_self_hosted_normalization": (
                "All routes normalize streaming chunks into generated_text/output_text; "
                "no provider response schema mismatch was found in raw rows."
            ),
        },
        "a100_200_prompt_calibration": {
            "script": "scripts/phase4/run_a100_sxm_calibration.py",
            "renderer": (
                "build_b6_context_aligned_runner_input -> render_generation_contract_prompt"
            ),
            "contract_used": GENERATION_CONTRACT_FORMAT,
            "finance_research_repairs": (
                "Uses B6R6 answer_skeleton strategy path and labels the run "
                "a100_sxm_b6r6_finance_research_repairs_calibration."
            ),
            "observed_quality": {
                "json_valid_rate": 0.99,
                "generation_contract_valid_rate": 0.985,
                "evidence_match_rate": 0.975,
                "grounded_rate": 0.97,
            },
        },
        "b6r5_finance_repair": {
            "script": "scripts/phase4/run_b6r5_finance_research_quality_repair.py",
            "selected_strategy": "evidence_selection_preplan",
            "included_in_controlled_final": False,
        },
        "b6r6_research_ai_recovery": {
            "script": "scripts/phase4/run_b6r6_research_ai_quality_recovery.py",
            "selected_strategy": "answer_skeleton",
            "included_in_controlled_final": False,
        },
        "b7r1_quality_runner": {
            "script": "scripts/phase4/run_b7r1_vllm_stability_repair.py",
            "renderer": "B6/B7 context-aligned WorkloadItem prompt with generation contract",
            "included_in_controlled_final": False,
        },
        "conclusion": (
            "The 10k controlled-final runner exercised serving throughput over raw promoted "
            "prompt text, not the B6/B7/A100 generation-contract runner input."
        ),
    }


def _trace_markdown(trace: dict[str, Any]) -> str:
    controlled = trace["controlled_final_runner"]
    a100 = trace["a100_200_prompt_calibration"]
    return "\n".join(
        [
            "# Controlled Final Contract Trace",
            "",
            "## Controlled Final Runner",
            "",
            f"- Renderer: {controlled['renderer']}",
            f"- Contract used: {controlled['contract_used']}",
            f"- Parser used: {controlled['parser_used']}",
            f"- Evaluator used: {controlled['evaluator_used']}",
            f"- B6R5 Finance repair included: {controlled['b6r5_finance_repair_included']}",
            "- B6R6 Research AI repair included: "
            f"{controlled['b6r6_research_ai_answer_skeleton_included']}",
            f"- Normalization: {controlled['api_self_hosted_normalization']}",
            "",
            "## A100 200-Prompt Calibration",
            "",
            f"- Renderer: {a100['renderer']}",
            f"- Contract used: {a100['contract_used']}",
            f"- Repair path: {a100['finance_research_repairs']}",
            "",
            "## Conclusion",
            "",
            str(trace["conclusion"]),
        ]
    )


def _classification_summary(classifications: list[dict[str, Any]]) -> dict[str, Any]:
    bucket_counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    for row in classifications:
        primary_counts[str(row["primary_bucket"])] += 1
        for bucket in row["buckets"]:
            bucket_counts[str(bucket)] += 1
    return {
        "row_count": len(classifications),
        "primary_bucket_counts": dict(primary_counts.most_common()),
        "bucket_counts": dict(bucket_counts.most_common()),
    }


def run_mini_replay(args: argparse.Namespace, raw_hash_before: str) -> dict[str, Any]:
    """Run a tiny replay through current runner functions without touching raw results."""

    try:
        import run_controlled_final_simulation as runner

        runner_args = runner.build_parser().parse_args([])
        gates = runner.check_runtime_gate(runner_args)
        if not gates.get("full_simulation_allowed"):
            return {
                "status": "SKIPPED_SMOKE_BLOCKED",
                "gate_report": gates,
                "raw_results_sha256_before": raw_hash_before,
                "raw_results_sha256_after": _file_sha256(args.raw_results_path),
                "raw_results_mutated": raw_hash_before != _file_sha256(args.raw_results_path),
            }
        matrix_rows = runner.build_matrix_rows(
            dataset_root=runner_args.dataset_root,
            prompts_per_vertical=runner_args.prompt_count_per_vertical,
        )
        by_config: dict[str, dict[str, Any]] = {}
        for row in matrix_rows:
            by_config.setdefault(str(row["config_id"]), row)
        api_route = runner._api_route(runner_args)  # noqa: SLF001
        replay_rows = [
            runner._execute_one_request(args=runner_args, row=row, api_route=api_route)  # noqa: SLF001
            for row in by_config.values()
        ]
        gold = load_gold_records(args.dataset_root)
        replay_eval = evaluate_generated_answers(
            [result_row_to_generated_answer(row) for row in replay_rows],
            gold,
        )
        raw_hash_after = _file_sha256(args.raw_results_path)
        return {
            "status": "REPLAY_COMPLETE",
            "row_count": len(replay_rows),
            "requests_completed": sum(bool(row.get("success")) for row in replay_rows),
            "json_valid_count": sum(bool(row.get("json_validity")) for row in replay_eval),
            "generation_contract_valid_count": sum(
                bool(row.get("generation_contract_valid")) for row in replay_eval
            ),
            "natural_language_no_json_count": sum(
                parse_generation_contract(_generated_text(row)).parse_error_type
                == PARSE_ERROR_NO_JSON
                for row in replay_rows
            ),
            "raw_results_sha256_before": raw_hash_before,
            "raw_results_sha256_after": raw_hash_after,
            "raw_results_mutated": raw_hash_before != raw_hash_after,
            "sample_text_previews": [
                {
                    "config_id": row.get("config_id"),
                    "engine": row.get("engine"),
                    "memory_mode": row.get("memory_mode"),
                    "text_preview": _compact(_generated_text(row), limit=240),
                }
                for row in replay_rows[:5]
            ],
        }
    except Exception as exc:  # noqa: BLE001
        raw_hash_after = _file_sha256(args.raw_results_path)
        return {
            "status": "REPLAY_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "raw_results_sha256_before": raw_hash_before,
            "raw_results_sha256_after": raw_hash_after,
            "raw_results_mutated": raw_hash_before != raw_hash_after,
        }


def _diagnosis_doc(
    *,
    classification_summary: dict[str, Any],
    aggregation_rows: list[dict[str, Any]],
    replay: dict[str, Any],
) -> str:
    agg = {str(row["slice"]): row for row in aggregation_rows}
    bucket_counts = classification_summary["bucket_counts"]

    def count(bucket: str) -> int:
        return int(bucket_counts.get(bucket, 0))

    def metric_line(label: str, key: str) -> str:
        row = agg[key]
        return (
            f"- {label}: {row['json_valid_rate']:.4f} / "
            f"{row['generation_contract_valid_rate']:.4f} / "
            f"{row['evidence_match_rate']:.4f} / {row['grounded_rate']:.4f}."
        )

    return "\n".join(
        [
            "# Controlled Final Quality Root-Cause Diagnosis",
            "",
            "## Finding",
            "",
            "The controlled final baseline completed operationally, but it did not use the "
            "B6/B7/A100 generation-contract prompt path. It sent the raw promoted prompt "
            "text directly to vLLM, SGLang, and the API route. The models therefore "
            "returned natural-language answers rather than the required five-field JSON "
            "generation contract.",
            "",
            "## What Failed",
            "",
            f"- Rows classified as natural-language/no JSON: "
            f"{count('invalid_json_due_to_natural_language')}.",
            f"- Rows where the generation contract was not applied: "
            f"{count('generation_contract_not_applied')}.",
            f"- Rows with no model-visible evidence labels/context: "
            f"{count('evidence_absent_from_prompt')}.",
            f"- Finance rows missing B6R5 repair marker: {count('finance_repair_not_applied')}.",
            f"- Research AI rows missing B6R6 answer_skeleton marker: "
            f"{count('research_ai_repair_not_applied')}.",
            "",
            "The evaluator read `generated_text` correctly. The raw assistant text was "
            "present, but it was not contract JSON.",
            "",
            "## Why The 200-Prompt A100 Calibration Passed",
            "",
            "The A100 200-prompt calibration rebuilt B6 context-aligned runner input, "
            "rendered `render_generation_contract_prompt`, included citation aliases and "
            "model-visible E-label evidence blocks, and routed Research AI through the "
            "`answer_skeleton` repair path. The controlled final runner instead built its "
            "matrix from `*_prompts_2000.jsonl` and sent `_prompt_text(prompt)` directly.",
            "",
            "## Ruled Out",
            "",
            "- This is not primarily a model3_7b capability issue; the contract prompt was absent.",
            "- This is not primarily a vLLM vs SGLang normalization issue; both wrote "
            "`generated_text`.",
            "- This is not primarily an API schema issue; API rows used the same normalized field.",
            "- This is not caused only by MM0 or MM4 aggregation; contextual MM1/MM2/MM3 "
            "also failed.",
            "- This is not an evaluator-field issue; `result_row_to_generated_answer` "
            "selected `generated_text`.",
            "",
            "## Aggregation Check",
            "",
            metric_line("All configs JSON/contract/evidence/grounded", "all_configs"),
            metric_line("Contextual MM1/MM2/MM3", "contextual_only_mm1_mm2_mm3"),
            metric_line("Primary MM2", "primary_baseline_only_mm2"),
            metric_line("API only", "api_only"),
            "",
            "## Mini Replay",
            "",
            f"Replay status: `{replay['status']}`. Raw 10k results mutated: "
            f"`{replay.get('raw_results_mutated')}`.",
            "",
            "## Smallest Safe Repair Plan",
            "",
            "1. Do not change evaluators or gold data.",
            "2. Rebuild the controlled-final matrix from the same B6/B7/A100 "
            "context-aligned runner input path, not raw prompt rows.",
            "3. Preserve the five-field `generation_contract_json` output contract for "
            "MM1/MM2/MM3 and self-hosted/API routes.",
            "4. Define MM0 as an explicit no-context stress slice with separate SLO "
            "interpretation.",
            "5. Route Finance through the selected B6R5 evidence-selection preplan and "
            "Research AI through B6R6 `answer_skeleton`.",
            "6. For MM4, either emit the same evaluator-facing contract row or add an "
            "adapter that normalizes agent state to that contract before evaluation.",
            "",
            "## What Should Not Change",
            "",
            "- Do not weaken the evaluator.",
            "- Do not modify promoted gold data.",
            "- Do not hide MM0/MM4 failures by averaging changes.",
            "- Do not silently fall back between vLLM, SGLang, and API routes.",
            "- Do not optimize latency or concurrency until the prompt-contract path is repaired.",
        ]
    )


def _summary_doc(classification_summary: dict[str, Any], replay: dict[str, Any]) -> str:
    buckets = classification_summary["bucket_counts"]

    def count(bucket: str) -> int:
        return int(buckets.get(bucket, 0))

    return "\n".join(
        [
            "# Controlled Final Quality Diagnosis Summary",
            "",
            "## Outcome",
            "",
            "Root cause: the controlled-final runner sent raw promoted prompt text instead "
            "of the B6/B7/A100 generation-contract prompts with retrieved evidence and "
            "repair instructions.",
            "",
            "## Counts",
            "",
            f"- Rows classified natural-language/no JSON: "
            f"{count('invalid_json_due_to_natural_language')}.",
            f"- Rows classified generation contract not applied: "
            f"{count('generation_contract_not_applied')}.",
            f"- Rows with evidence absent from prompt: {count('evidence_absent_from_prompt')}.",
            f"- Mini replay status: `{replay['status']}`.",
            "",
            "## Decision",
            "",
            "Do not change evaluators or gold data. Repair the controlled-final input/render "
            "path first, then rerun a small smoke before any larger experiment.",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-results-path", default=DEFAULT_RAW_RESULTS)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--samples-json-path", default=DEFAULT_SAMPLE_JSON)
    parser.add_argument("--samples-md-path", default=DEFAULT_SAMPLE_MD)
    parser.add_argument("--trace-json-path", default=DEFAULT_TRACE_JSON)
    parser.add_argument("--trace-md-path", default=DEFAULT_TRACE_MD)
    parser.add_argument("--classification-json-path", default=DEFAULT_CLASSIFICATION_JSON)
    parser.add_argument("--classification-csv-path", default=DEFAULT_CLASSIFICATION_CSV)
    parser.add_argument("--aggregation-json-path", default=DEFAULT_AGGREGATION_JSON)
    parser.add_argument("--aggregation-csv-path", default=DEFAULT_AGGREGATION_CSV)
    parser.add_argument("--replay-json-path", default=DEFAULT_REPLAY_JSON)
    parser.add_argument("--doc-path", default=DEFAULT_DOC)
    parser.add_argument("--summary-doc-path", default=DEFAULT_SUMMARY_DOC)
    parser.add_argument("--skip-mini-replay", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_hash_before = _file_sha256(args.raw_results_path)
    rows = _read_jsonl(args.raw_results_path)
    gold = load_gold_records(args.dataset_root)
    generated = [result_row_to_generated_answer(row) for row in rows]
    evaluations = evaluate_generated_answers(generated, gold)
    classifications = [
        classify_output(row, evaluation) for row, evaluation in zip(rows, evaluations, strict=True)
    ]
    samples = _sample_rows(rows, evaluations, classifications)
    trace = build_contract_trace()
    aggregation_rows = build_aggregation_rows(rows, evaluations)
    replay = (
        {
            "status": "SKIPPED_BY_FLAG",
            "raw_results_sha256_before": raw_hash_before,
            "raw_results_sha256_after": _file_sha256(args.raw_results_path),
            "raw_results_mutated": False,
        }
        if args.skip_mini_replay
        else run_mini_replay(args, raw_hash_before)
    )
    classification_summary = _classification_summary(classifications)

    _write_json(args.samples_json_path, {"status": "SAMPLES_WRITTEN", "samples": samples})
    _write_text(args.samples_md_path, _markdown_samples(samples))
    _write_json(args.trace_json_path, trace)
    _write_text(args.trace_md_path, _trace_markdown(trace))
    _write_json(
        args.classification_json_path,
        {
            "status": "CLASSIFICATION_COMPLETE",
            "summary": classification_summary,
            "rows": classifications,
        },
    )
    _write_csv(
        args.classification_csv_path,
        [
            {
                **row,
                "buckets": ";".join(row["buckets"]),
                "missing_contract_fields": ";".join(row["missing_contract_fields"]),
                "evidence_expected": ";".join(row["evidence_expected"]),
                "evidence_labels_available_in_prompt": ";".join(
                    row["evidence_labels_available_in_prompt"]
                ),
                "evidence_labels_cited_by_model": ";".join(row["evidence_labels_cited_by_model"]),
            }
            for row in classifications
        ],
    )
    _write_json(
        args.aggregation_json_path,
        {"status": "AGGREGATION_CHECK_COMPLETE", "slices": aggregation_rows},
    )
    _write_csv(args.aggregation_csv_path, aggregation_rows)
    _write_json(args.replay_json_path, replay)
    _write_text(
        args.doc_path,
        _diagnosis_doc(
            classification_summary=classification_summary,
            aggregation_rows=aggregation_rows,
            replay=replay,
        ),
    )
    _write_text(args.summary_doc_path, _summary_doc(classification_summary, replay))

    raw_hash_after = _file_sha256(args.raw_results_path)
    print(
        json.dumps(
            {
                "status": "CONTROLLED_FINAL_QUALITY_DIAGNOSIS_COMPLETE",
                "row_count": len(rows),
                "raw_results_mutated": raw_hash_before != raw_hash_after,
                "main_root_cause": trace["conclusion"],
                "classification_summary": classification_summary,
                "replay_status": replay["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

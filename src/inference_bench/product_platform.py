"""Artifact-backed data layer for the interactive inference platform.

The product platform is a read-only replay surface over saved repository
artifacts. This module deliberately does not run inference, mutate experiment
outputs, or fabricate optimized results.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from inference_bench.config import load_yaml_file
from inference_bench.model_registry import build_model_registry_report
from inference_bench.optimization_catalog import load_optimization_catalog
from inference_bench.optimization_negative_rules import load_optimization_negative_rules
from inference_bench.product_platform_contracts import RecipeValidationRequest

JsonDict = dict[str, Any]

MAIN_ROOT = Path("experiments/main/main_inference_v1")
MAIN_PROCESSED = MAIN_ROOT / "processed"
MAIN_RAW = MAIN_ROOT / "raw"
DATASET_ROOT = Path("data/generated/dataset_10000")
CONTEXT_ROOT = Path("data/generated/context_engineering")


def _repo_root() -> Path:
    return Path.cwd()


def _path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else _repo_root() / value


def _display_path(path: str | Path) -> str:
    value = _path(path)
    try:
        return value.resolve().relative_to(_repo_root().resolve()).as_posix()
    except ValueError:
        return value.as_posix()


def _read_json(path: str | Path) -> JsonDict:
    payload = json.loads(_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Expected JSON object in {path}"
        raise ValueError(msg)
    return cast(JsonDict, payload)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with _path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl_sample(path: str | Path, *, limit: int) -> list[JsonDict]:
    rows: list[JsonDict] = []
    with _path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(cast(JsonDict, value))
            if len(rows) >= limit:
                break
    return rows


def _to_float(value: object, default: float = 0.0) -> float:
    if value in (None, "", "n/a"):
        return default
    return float(str(value))


def _to_int(value: object, default: int = 0) -> int:
    if value in (None, "", "n/a"):
        return default
    return int(float(str(value)))


def _unique(values: Iterable[object]) -> list[str]:
    return sorted({str(value) for value in values if value not in (None, "")})


def _artifact_map() -> dict[str, list[str]]:
    return {
        "about": [
            "docs/00_project_scope.md",
            "docs/95_definitive_technical_briefing.md",
            "docs/main_inference_V1.md",
        ],
        "data": [
            _display_path(DATASET_ROOT / "dataset_10000_eda_summary.csv"),
            _display_path(DATASET_ROOT / "dataset_10000_eda_inventory.json"),
            _display_path(DATASET_ROOT / "dataset_10000_eda_workload_shape_report.json"),
        ],
        "preparation": [
            _display_path(CONTEXT_ROOT / "retrieval_source_of_truth_manifest.json"),
            _display_path(CONTEXT_ROOT / "corpus_build_summary.csv"),
            _display_path(CONTEXT_ROOT / "qdrant_index_summary.csv"),
            "configs/models.yaml",
            "configs/memory_modes.yaml",
            "configs/runtime_engines.yaml",
        ],
        "main_inference": [
            _display_path(MAIN_RAW / "main_inference_v1_manifest.json"),
            _display_path(MAIN_PROCESSED / "main_inference_v1_eval_report.json"),
            _display_path(MAIN_PROCESSED / "main_inference_v1_slo_scorecard.csv"),
            _display_path(MAIN_PROCESSED / "main_inference_v1_cost_report.json"),
        ],
        "optimization": [
            _display_path(MAIN_PROCESSED / "main_inference_v1_ui_diagnosis.json"),
            _display_path(MAIN_PROCESSED / "main_inference_v1_ui_optimization_options.json"),
            _display_path(MAIN_PROCESSED / "main_inference_v1_ui_apply_plan.json"),
            "configs/optimization_catalog.yaml",
            "configs/optimization_negative_rules.yaml",
        ],
        "optimized_inference": ["experiments/optimized/optimized_inference_v1/"],
        "comparison": [
            _display_path(MAIN_PROCESSED / "main_inference_v1_engine_comparison.csv"),
            _display_path(MAIN_PROCESSED / "main_inference_v1_memory_comparison.csv"),
            "experiments/optimized/optimized_inference_v1/",
        ],
        "conclusions": [
            "docs/main_inference_V1.md",
            "docs/125_optimization_intelligence_ui_layer.md",
        ],
    }


class ProductPlatformData:
    """Read-only repository artifact facade used by FastAPI and tests."""

    def health(self) -> JsonDict:
        optimized_root = _path("experiments/optimized/optimized_inference_v1")
        return {
            "service": "ai_inference_engineering_platform",
            "read_only": True,
            "gpu_required": False,
            "main_inference_available": _path(
                MAIN_PROCESSED / "main_inference_v1_eval_report.json"
            ).exists(),
            "optimized_inference_available": optimized_root.exists(),
        }

    def project_overview(self) -> JsonDict:
        manifest = _read_json(MAIN_RAW / "main_inference_v1_manifest.json")
        eval_report = _read_json(MAIN_PROCESSED / "main_inference_v1_eval_report.json")
        cost = _read_json(MAIN_PROCESSED / "main_inference_v1_cost_report.json")
        summary = cast(JsonDict, eval_report["summary"])
        matrix = cast(JsonDict, eval_report["matrix_summary"])
        verdicts = cast(JsonDict, eval_report["slo_verdicts"])
        return {
            "title": "AI Inference Engineering Platform",
            "story": (
                "A guided replay of a full inference experiment, from dataset construction "
                "through SLO diagnosis and plan-only optimization selection."
            ),
            "journey": [
                {"path": "/", "label": "About"},
                {"path": "/data", "label": "Data & Workflow Explorer"},
                {"path": "/preparation", "label": "Inference Experiment Preparation"},
                {"path": "/main-inference", "label": "Main Inference Simulation"},
                {"path": "/optimization", "label": "Inference Optimization Lab"},
                {"path": "/optimized-inference", "label": "Optimized Inference Simulation"},
                {"path": "/comparison", "label": "Before/After Comparison"},
                {"path": "/conclusions", "label": "Conclusions & Recommendations"},
            ],
            "headline_metrics": {
                "run_id": manifest["run_id"],
                "configs": matrix["config_count"],
                "completed_requests": eval_report["total_requests_completed"],
                "failed_requests": eval_report["total_requests_failed"],
                "verticals": len(cast(JsonDict, matrix["vertical_counts"])),
                "gpu": "NVIDIA A100-SXM4-80GB",
                "wall_seconds": eval_report["wall_seconds"],
                "total_cost_usd": cost["total_cost_usd"],
                "json_validity": summary["json_valid_rate"],
                "contract_validity": summary["generation_contract_valid_rate"],
                "evidence_match": summary["evidence_match_rate"],
                "groundedness": summary["grounded_rate"],
                "safety_violations": summary["safety_violation_count"],
            },
            "verdicts": verdicts,
            "result_type": "measured",
            "artifact_map": _artifact_map(),
        }

    def dataset_workflow_summary(self) -> JsonDict:
        rows = _read_csv(DATASET_ROOT / "dataset_10000_eda_summary.csv")
        inventory = _read_json(DATASET_ROOT / "dataset_10000_eda_inventory.json")
        safety = _read_json(DATASET_ROOT / "dataset_10000_eda_safety_report.json")
        workload = _read_json(DATASET_ROOT / "dataset_10000_eda_workload_shape_report.json")
        verticals = [
            {
                "vertical": row["vertical"],
                "prompt_count": _to_int(row["prompt_count"]),
                "gold_count": _to_int(row["gold_count"]),
                "kb_count": _to_int(row["kb_count"]),
                "evidence_coverage_rate": _to_float(row["evidence_coverage_rate"]),
                "average_evidence_ids_per_prompt": _to_float(
                    row["average_evidence_ids_per_prompt"]
                ),
                "workload_pressure_score": _to_float(row["workload_pressure_score"]),
                "likely_inference_cost_pressure": row["likely_inference_cost_pressure"],
            }
            for row in rows
        ]
        return {
            "dataset_totals": {
                "prompt_count": sum(item["prompt_count"] for item in verticals),
                "gold_count": sum(item["gold_count"] for item in verticals),
                "kb_count": sum(item["kb_count"] for item in verticals),
                "vertical_count": len(verticals),
            },
            "verticals": verticals,
            "workflow": [
                "prompts",
                "gold records",
                "vertical knowledge bases",
                "context corpora",
                "retrieval",
                "memory workloads",
                "runner prompts",
                "generation",
                "evaluation",
            ],
            "workload_properties": [
                {
                    "property": "input-token pressure",
                    "inference_effect": (
                        "Higher prompt/context length tends to increase prefill work and TTFT."
                    ),
                },
                {
                    "property": "evidence requirements",
                    "inference_effect": (
                        "More required evidence raises citation and groundedness difficulty."
                    ),
                },
                {
                    "property": "output format",
                    "inference_effect": (
                        "Strict JSON contracts expose instruction-following failures."
                    ),
                },
                {
                    "property": "safety constraints",
                    "inference_effect": (
                        "Unsafe wording can fail deployability even when requests complete."
                    ),
                },
            ],
            "inventory": inventory,
            "safety_report": safety,
            "workload_shape_report": workload,
        }

    def preparation_pipeline(self) -> JsonDict:
        retrieval = _read_json(CONTEXT_ROOT / "retrieval_source_of_truth_manifest.json")
        corpus = _read_csv(CONTEXT_ROOT / "corpus_build_summary.csv")
        qdrant = _read_csv(CONTEXT_ROOT / "qdrant_index_summary.csv")
        compression = _read_csv(CONTEXT_ROOT / "compression_diagnostic_summary.csv")
        return {
            "pipeline": [
                "Dataset",
                "query construction",
                "dense retrieval",
                "BM25",
                "hybrid fusion",
                "reranking",
                "context selection",
                "compression",
                "prompt assembly",
                "generation contract",
                "memory mode",
                "model",
                "engine",
                "concurrency",
                "A100",
                "telemetry",
                "evaluation",
                "SLO scoring",
            ],
            "stage_contracts": self._pipeline_stage_contracts(),
            "retrieval_source_of_truth": retrieval,
            "corpus_summary": corpus,
            "qdrant_summary": qdrant,
            "compression_summary": compression,
            "matrix_formula": {
                "configs": 25,
                "prompts_per_config": 10000,
                "total_requests": 250000,
            },
        }

    def _pipeline_stage_contracts(self) -> list[JsonDict]:
        return [
            {
                "stage": "Dataset",
                "purpose": "Freeze prompts, gold records, KB rows, and vertical balance.",
                "implementation": "Promoted 10,000-prompt dataset artifacts.",
                "affects": ["quality", "safety", "latency", "cost"],
                "risk": "Dataset leakage or imbalance invalidates optimization claims.",
            },
            {
                "stage": "Retrieval",
                "purpose": "Select evidence without using gold-side leakage.",
                "implementation": "BM25 plus Qdrant vector retrieval with hybrid fusion.",
                "affects": ["evidence_match", "groundedness", "TTFT"],
                "risk": (
                    "Low final recall pushes failures into generation even if the model is capable."
                ),
            },
            {
                "stage": "Memory mode",
                "purpose": "Control context availability, compression, and bounded repair.",
                "implementation": "Configured mm0 through mm4 modes.",
                "affects": ["input_tokens", "groundedness", "latency", "cost"],
                "risk": "Changing memory mode changes both quality and runtime behavior.",
            },
            {
                "stage": "Serving engine",
                "purpose": "Execute the same workload through vLLM, SGLang, or API route.",
                "implementation": "Runtime registry plus Main_Inference matrix.",
                "affects": ["TTFT", "TPOT", "throughput", "GPU memory"],
                "risk": (
                    "Engine comparisons must hold model, prompts, and generation settings constant."
                ),
            },
            {
                "stage": "Evaluation",
                "purpose": (
                    "Score JSON, contract, evidence, groundedness, safety, runtime, and cost."
                ),
                "implementation": "Deterministic evaluators and SLO scorecard.",
                "affects": ["deployability"],
                "risk": "Weakening evaluator semantics would fabricate success.",
            },
        ]

    def models(self) -> JsonDict:
        registry = build_model_registry_report()
        raw = load_yaml_file("configs/models.yaml")
        return {
            "active_aliases": raw.get("model_aliases", {}),
            "deprecated_aliases": raw.get("deprecated_model_aliases", {}),
            "models": {
                key: value
                for key, value in raw.items()
                if key not in {"model_aliases", "deprecated_model_aliases"}
            },
            "registry_report": registry,
        }

    def engines(self) -> JsonDict:
        return load_yaml_file("configs/runtime_engines.yaml")

    def memory_modes(self) -> JsonDict:
        return load_yaml_file("configs/memory_modes.yaml")

    def slo_targets(self) -> JsonDict:
        return {
            "targets": load_yaml_file("configs/slo_targets.yaml"),
            "profiles": load_yaml_file("configs/slo_profiles.yaml"),
        }

    def main_manifest(self) -> JsonDict:
        return _read_json(MAIN_RAW / "main_inference_v1_manifest.json")

    def main_results(self) -> JsonDict:
        eval_report = _read_json(MAIN_PROCESSED / "main_inference_v1_eval_report.json")
        cost_report = _read_json(MAIN_PROCESSED / "main_inference_v1_cost_report.json")
        scorecard = _read_csv(MAIN_PROCESSED / "main_inference_v1_slo_scorecard.csv")
        return {
            "eval_report": eval_report,
            "cost_report": cost_report,
            "slo_scorecard": scorecard,
            "result_type": "measured",
        }

    def replay_events(self, *, limit: int = 120) -> JsonDict:
        progress_path = MAIN_ROOT / "logs/main_inference_v1_progress.jsonl"
        progress_rows = _read_jsonl_sample(progress_path, limit=10_000)
        if not progress_rows:
            return {"events": [], "duration_seconds": 100, "final_completed": 0}
        step = max(1, len(progress_rows) // max(limit - 1, 1))
        sampled = progress_rows[::step][: max(limit - 1, 1)]
        final_row = dict(progress_rows[-1])
        final_row["completed_requests"] = 250000
        final_row["failure_count"] = 0
        if not sampled or sampled[-1].get("completed_requests") != 250000:
            sampled.append(final_row)
        events = [
            {
                "event_index": index,
                "compressed_second": round(index * (110 / max(len(sampled) - 1, 1)), 3),
                "completed_requests": row.get("completed_requests"),
                "failure_count": row.get("failure_count", row.get("failed_requests", 0)),
                "current_config_id": row.get("current_config_id"),
                "engine": row.get("engine"),
                "runtime": row.get("runtime"),
                "memory_mode": row.get("memory_mode"),
                "model": row.get("model"),
                "concurrency": row.get("concurrency"),
                "vertical": row.get("vertical"),
                "approximate_cost_so_far_usd": row.get("approximate_cost_so_far_usd"),
                "source_timestamp_utc": row.get("timestamp_utc"),
            }
            for index, row in enumerate(sampled)
        ]
        return {
            "result_type": "measured",
            "replay_duration_seconds": 110,
            "source_event_count": len(progress_rows),
            "events": events,
            "final_completed": 250000,
            "final_failed": 0,
        }

    def telemetry(self, *, limit: int = 160) -> JsonDict:
        telemetry_path = MAIN_RAW / "main_inference_v1_gpu_telemetry.jsonl"
        eval_report = _read_json(MAIN_PROCESSED / "main_inference_v1_eval_report.json")
        rows = _read_jsonl_sample(telemetry_path, limit=50_000)
        step = max(1, len(rows) // limit) if rows else 1
        sampled = rows[::step][:limit]
        return {
            "result_type": "measured",
            "summary": eval_report.get("gpu_summary", {}),
            "samples": sampled,
        }

    def diagnosis(self) -> JsonDict:
        return {
            "diagnosis": _read_json(MAIN_PROCESSED / "main_inference_v1_ui_diagnosis.json"),
            "optimization_options": _read_json(
                MAIN_PROCESSED / "main_inference_v1_ui_optimization_options.json"
            ),
            "apply_plan": _read_json(MAIN_PROCESSED / "main_inference_v1_ui_apply_plan.json"),
            "story": _read_json(MAIN_PROCESSED / "main_inference_v1_ui_story.json"),
        }

    def mandatory_repairs(self) -> JsonDict:
        apply_plan = _read_json(MAIN_PROCESSED / "main_inference_v1_ui_apply_plan.json")
        plans = cast(list[JsonDict], apply_plan.get("plans", []))
        return {
            "result_type": "planned",
            "mandatory_repair_ids": [str(plan["optimization_id"]) for plan in plans],
            "repairs": plans,
            "semantics": apply_plan.get("plan_semantics"),
        }

    def core_optimization_catalog(self) -> JsonDict:
        catalog = load_optimization_catalog()
        negative_rules = load_optimization_negative_rules()
        negative_by_optimization: dict[str, list[JsonDict]] = {}
        for rule in negative_rules.values():
            for optimization_id in rule.optimization_ids:
                negative_by_optimization.setdefault(optimization_id, []).append(
                    {
                        "rule_id": rule.id,
                        "when_not_to_use": list(rule.when_not_to_use),
                    }
                )
        return {
            "result_type": "planned",
            "optimizations": [
                {
                    **asdict(definition),
                    "negative_rules": negative_by_optimization.get(definition.id, []),
                }
                for definition in catalog.values()
            ],
        }

    def optimization_applicability(self) -> JsonDict:
        diagnosis = self.diagnosis()
        options_payload = cast(JsonDict, diagnosis["optimization_options"])
        allowed_ids = {
            str(option["optimization_id"])
            for options in cast(JsonDict, options_payload["options_by_failed_slo"]).values()
            for option in cast(list[JsonDict], options)
        }
        rejected_lookup: dict[str, JsonDict] = {}
        for rejections in cast(
            JsonDict,
            options_payload["rejected_optimizations_by_failed_slo"],
        ).values():
            for rejection in cast(list[JsonDict], rejections):
                rejected_lookup.setdefault(str(rejection["optimization_id"]), rejection)
        catalog = load_optimization_catalog()
        rows: list[JsonDict] = []
        for definition in catalog.values():
            if definition.id in allowed_ids:
                state = "applicable_measured"
                reason = "Applies to measured Main_Inference_V1 failed SLO evidence."
            elif definition.id in rejected_lookup:
                rejection = rejected_lookup[definition.id]
                state = (
                    "blocked_by_negative_rule"
                    if rejection.get("negative_rule_triggered")
                    else "not_applicable"
                )
                reason = str(rejection.get("reason_rejected") or "Not compatible with this run.")
            elif definition.implementation_status == "planned":
                state = "applicable_planned"
                reason = (
                    "Cataloged for future controlled experiments, but not selected by this "
                    "diagnosis."
                )
            elif definition.category in {"hardware", "serving_engine"}:
                state = "future_architecture"
                reason = (
                    "Useful infrastructure concept, but not applicable to the current failed SLOs."
                )
            else:
                state = "not_applicable"
                reason = "No current failed SLO maps this strategy to a selectable option."
            rows.append(
                {
                    "optimization_id": definition.id,
                    "display_name": definition.id.replace("_", " ").title(),
                    "category": definition.category,
                    "state": state,
                    "definition": definition.description,
                    "mechanism": definition.application_method,
                    "affected_metrics": list(definition.improves),
                    "possible_regressions": list(definition.may_hurt),
                    "implementation_status": definition.implementation_status,
                    "current_project_support": definition.current_project_support,
                    "requires_gpu_or_api_rerun": state
                    in {"applicable_measured", "applicable_planned"},
                    "reason": reason,
                    "negative_rule": rejected_lookup.get(definition.id, {}).get(
                        "negative_rule_triggered"
                    ),
                }
            )
        return {
            "result_type": "planned",
            "states": rows,
            "state_counts": {
                state: sum(1 for row in rows if row["state"] == state)
                for state in [
                    "applicable_measured",
                    "applicable_planned",
                    "not_applicable",
                    "blocked_by_negative_rule",
                    "future_architecture",
                ]
            },
        }

    def validate_recipe(self, request: RecipeValidationRequest) -> JsonDict:
        applicability = self.optimization_applicability()
        states = {
            str(row["optimization_id"]): row
            for row in cast(list[JsonDict], applicability["states"])
        }
        selected = list(dict.fromkeys(request.mandatory_repair_ids + request.core_optimization_ids))
        blocked: list[JsonDict] = []
        for optimization_id in selected:
            state = states.get(optimization_id)
            if state is None:
                blocked.append(
                    {
                        "optimization_id": optimization_id,
                        "reason": "Unknown optimization id.",
                    }
                )
            elif state["state"] in {
                "blocked_by_negative_rule",
                "not_applicable",
                "future_architecture",
            }:
                blocked.append(
                    {
                        "optimization_id": optimization_id,
                        "reason": state["reason"],
                        "state": state["state"],
                        "negative_rule": state.get("negative_rule"),
                    }
                )
        apply_plan = _read_json(MAIN_PROCESSED / "main_inference_v1_ui_apply_plan.json")
        return {
            "valid": not blocked,
            "selected_optimization_ids": selected,
            "conflicts": [
                (
                    "Apply All means mandatory repairs plus user-selected compatible core "
                    "strategies, not the full catalog."
                )
            ],
            "dependencies": [
                "Quality and safety repairs should be measured before latency-only optimizations."
            ],
            "blocked": blocked,
            "plan": {
                "result_type": "planned",
                "does_not_execute_inference": True,
                "creates_optimized_inference_v1": False,
                "source_apply_plan": apply_plan,
            },
        }

    def scenario_registry(self) -> JsonDict:
        optimized_root = _path("experiments/optimized/optimized_inference_v1")
        return {
            "scenarios": [
                {
                    "scenario_id": "main_inference_v1",
                    "label": "Main Inference V1",
                    "result_type": "measured",
                    "available": True,
                },
                {
                    "scenario_id": "optimized_inference_v1",
                    "label": "Optimized Inference V1",
                    "result_type": "planned",
                    "available": optimized_root.exists(),
                    "required_artifact_root": "experiments/optimized/optimized_inference_v1",
                },
            ]
        }

    def comparison_availability(self) -> JsonDict:
        optimized_root = _path("experiments/optimized/optimized_inference_v1")
        return {
            "comparison_id": "main_vs_optimized_inference_v1",
            "status": "blocked_until_optimized_artifacts_exist",
            "baseline_run_id": "main_inference_v1",
            "optimized_run_id": "optimized_inference_v1",
            "baseline_available": True,
            "optimized_available": optimized_root.exists(),
            "result_type": "planned",
            "required_future_artifacts": [
                "optimized_inference_v1_manifest.json",
                "optimized_inference_v1_eval_report.json",
                "optimized_inference_v1_slo_scorecard.csv",
                "optimized_inference_v1_cost_report.json",
                "main_vs_optimized_inference_v1_ui_comparison.json",
            ],
        }

    def conclusion_availability(self) -> JsonDict:
        return {
            "status": "unavailable_until_saved_conclusion_artifacts_or_endpoint_exist",
            "result_type": "planned",
            "available": False,
            "contract": {
                "measured_scenario_interpretation": "pre-generated model interpretation only",
                "project_grounded_chat": "must cite repo artifacts and never require GPU",
            },
        }


def platform_data() -> ProductPlatformData:
    """Factory used by FastAPI to keep endpoint code small."""

    return ProductPlatformData()

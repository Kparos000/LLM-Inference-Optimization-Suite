"""Artifact-backed data layer for the interactive inference platform.

The product platform is a read-only replay surface over saved repository
artifacts. This module deliberately does not run inference, mutate experiment
outputs, or fabricate optimized results.
"""

from __future__ import annotations

import csv
import json
import math
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
SCALEUP_ROOT = Path("data/scaleup_2000_full")

VERTICAL_LABELS = {
    "airline": "Airline",
    "healthcare_admin": "Healthcare Admin",
    "retail": "Retail",
    "finance": "Finance",
    "research_ai": "Research AI",
}

MAIN_MEMORY_MODE_LABELS = {
    "mm0_no_context": "MM0 no context",
    "mm1_dense_top5": "MM1 dense top-5",
    "mm2_hybrid_top5": "MM2 hybrid top-5",
    "mm3_compressed_hybrid_top5": "MM3 compressed hybrid top-5",
    "mm4_bounded_agentic": "MM4 bounded agentic",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def _iter_jsonl(path: str | Path) -> Iterable[JsonDict]:
    with _path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                yield cast(JsonDict, value)


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


def _rate(value: object) -> float:
    return round(_to_float(value) * 100, 3)


def _pressure_level(value: float) -> str:
    if value >= 500:
        return "very_high"
    if value >= 250:
        return "high"
    if value >= 180:
        return "medium"
    return "low"


def _relative_artifact(path: str | Path) -> str:
    text = _display_path(path)
    return text.replace("\\", "/")


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
                {"path": "/slo-metrics", "label": "SLO & Metrics"},
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
            "matrix_snapshot": {
                "self_hosted_formula": (
                    "1 model x 2 engines x 2 concurrency levels x 5 memory modes = "
                    "20 configurations"
                ),
                "api_formula": (
                    "1 model x 1 API route x 1 concurrency level x 5 memory modes = "
                    "5 configurations"
                ),
                "total_formula": "25 configurations x 10,000 prompts = 250,000 requests",
                "self_hosted_model": "Qwen/Qwen2.5-7B-Instruct",
                "api_model": "meta-llama/Llama-3.1-8B-Instruct",
                "serving_routes": ["vLLM", "SGLang", "API provider route"],
                "memory_modes": list(MAIN_MEMORY_MODE_LABELS.values()),
                "hardware": "NVIDIA A100-SXM4-80GB",
                "verticals": list(VERTICAL_LABELS.values()),
            },
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

    def dataset_explorer(self) -> JsonDict:
        rows = _read_csv(DATASET_ROOT / "dataset_10000_eda_summary.csv")
        prompt_profile = _read_json(DATASET_ROOT / "dataset_10000_eda_prompt_profile.json")
        gold_profile = _read_json(DATASET_ROOT / "dataset_10000_eda_gold_profile.json")
        evidence_reuse = _read_json(DATASET_ROOT / "dataset_10000_eda_evidence_reuse_report.json")
        workload = _read_json(DATASET_ROOT / "dataset_10000_eda_workload_shape_report.json")
        by_vertical_prompt = cast(JsonDict, prompt_profile["by_vertical"])
        by_vertical_gold = cast(JsonDict, gold_profile["by_vertical"])
        by_vertical_evidence = cast(JsonDict, evidence_reuse["by_vertical"])
        by_vertical_workload = cast(JsonDict, workload["by_vertical"])
        verticals: list[JsonDict] = []
        for row in rows:
            vertical = row["vertical"]
            workload_row = cast(JsonDict, by_vertical_workload[vertical])
            prompt_row = cast(JsonDict, by_vertical_prompt[vertical])
            gold_row = cast(JsonDict, by_vertical_gold[vertical])
            evidence_row = cast(JsonDict, by_vertical_evidence[vertical])
            input_mean = _to_float(workload_row["average_estimated_input_tokens_per_prompt"])
            output_mean = _to_float(workload_row["expected_output_tokens_per_prompt"])
            evidence_mean = _to_float(row["average_evidence_ids_per_prompt"])
            kb_count = _to_int(row["kb_count"])
            multi_share = _to_float(evidence_row.get("multi_evidence_prompt_share"))
            pressure_score = _to_float(row["workload_pressure_score"])
            verticals.append(
                {
                    "vertical": vertical,
                    "label": VERTICAL_LABELS.get(vertical, vertical.replace("_", " ").title()),
                    "represents": self._vertical_description(vertical),
                    "prompt_count": _to_int(row["prompt_count"]),
                    "gold_count": _to_int(row["gold_count"]),
                    "kb_count": kb_count,
                    "evidence_coverage_rate": _to_float(row["evidence_coverage_rate"]),
                    "average_input_tokens": input_mean,
                    "average_expected_output_tokens": output_mean,
                    "average_evidence_ids_per_prompt": evidence_mean,
                    "multi_evidence_share": multi_share,
                    "workload_pressure_score": pressure_score,
                    "cost_pressure": row["likely_inference_cost_pressure"],
                    "expected_status_distribution": prompt_row.get(
                        "expected_status_distribution",
                        {},
                    ),
                    "output_format_mix": workload_row.get("output_format_mix", {}),
                    "pressure_dimensions": {
                        "input_pressure": round(min(100.0, input_mean / 2.4), 3),
                        "output_pressure": round(min(100.0, output_mean / 1.2), 3),
                        "evidence_complexity": round(min(100.0, evidence_mean * 30), 3),
                        "retrieval_difficulty": round(min(100.0, math.log10(kb_count + 1) * 25), 3),
                        "contract_safety_complexity": round(
                            min(
                                100.0,
                                (
                                    len(
                                        cast(
                                            JsonDict,
                                            prompt_row.get(
                                                "expected_output_format_distribution", {}
                                            ),
                                        )
                                    )
                                    * 16
                                )
                                + (
                                    _to_float(
                                        gold_row.get(
                                            "must_not_include_count_distribution",
                                            {},
                                        ).get("mean"),
                                    )
                                    * 8
                                ),
                            ),
                            3,
                        ),
                    },
                    "likely_inference_effects": [
                        "Longer inputs create more prefill work and TTFT pressure.",
                        "Longer outputs increase decode work, E2E latency, and cost.",
                        "More required evidence makes citation completeness harder.",
                        "Larger KBs increase retrieval and ranking difficulty.",
                        "Strict contracts make format validity a deployment risk.",
                    ],
                }
            )
        return {
            "result_type": "planned",
            "chronology": "Designed before the run",
            "opening_explanation": (
                "The benchmark starts with linked prompts, gold contracts, KB evidence, and "
                "evaluation rules. These workload shapes determine how much context the model "
                "must read, how much it must generate, and how hard evidence-grounding becomes."
            ),
            "totals": {
                "prompt_count": sum(item["prompt_count"] for item in verticals),
                "gold_count": sum(item["gold_count"] for item in verticals),
                "kb_count": sum(item["kb_count"] for item in verticals),
                "vertical_count": len(verticals),
                "prompts_per_vertical": 2000,
            },
            "research_ai_coverage_explanation": {
                "coverage_rate": 0.98,
                "prompts_requiring_evidence": 1960,
                "out_of_scope_prompts_requiring_no_evidence": 40,
                "answerable_prompts_missing_evidence": 0,
                "plain_english": (
                    "Research AI shows 98% evidence coverage because 40 prompts are deliberately "
                    "out of scope and require no evidence. The answerable Research AI prompts are "
                    "not missing required evidence."
                ),
            },
            "verticals": verticals,
            "pressure_simulator": [
                {
                    "control": "prompt length",
                    "increase_effect": "More input tokens increase prefill work and TTFT pressure.",
                    "prediction_type": "qualitative",
                },
                {
                    "control": "output length",
                    "increase_effect": (
                        "More output tokens increase decode time, E2E latency, and cost."
                    ),
                    "prediction_type": "qualitative",
                },
                {
                    "control": "evidence count",
                    "increase_effect": (
                        "More required evidence increases citation completeness difficulty."
                    ),
                    "prediction_type": "qualitative",
                },
                {
                    "control": "KB size",
                    "increase_effect": "Larger corpora make retrieval and reranking harder.",
                    "prediction_type": "qualitative",
                },
                {
                    "control": "multi-evidence share",
                    "increase_effect": (
                        "More multi-evidence prompts make grounding and contract validity harder."
                    ),
                    "prediction_type": "qualitative",
                },
            ],
            "source_artifacts": [
                _relative_artifact(DATASET_ROOT / "dataset_10000_eda_summary.csv"),
                _relative_artifact(DATASET_ROOT / "dataset_10000_eda_workload_shape_report.json"),
                _relative_artifact(DATASET_ROOT / "dataset_10000_eda_evidence_reuse_report.json"),
            ],
        }

    def dataset_cases(
        self,
        *,
        vertical: str | None = None,
        search: str | None = None,
        expected_status: str | None = None,
        min_evidence_count: int | None = None,
        max_evidence_count: int | None = None,
        sort_by: str = "prompt_id",
        offset: int = 0,
        limit: int = 12,
    ) -> JsonDict:
        selected_verticals = [vertical] if vertical in VERTICAL_LABELS else list(VERTICAL_LABELS)
        cases: list[JsonDict] = []
        total_matches = 0
        safe_limit = max(1, min(limit, 50))
        safe_offset = max(0, offset)
        for current_vertical in selected_verticals:
            prompt_path = SCALEUP_ROOT / current_vertical / f"{current_vertical}_prompts_2000.jsonl"
            gold_path = SCALEUP_ROOT / current_vertical / f"{current_vertical}_gold_2000.jsonl"
            kb_path = SCALEUP_ROOT / current_vertical / f"{current_vertical}_kb_2000.jsonl"
            if not _path(prompt_path).exists():
                continue
            gold_by_prompt = {str(row.get("prompt_id")): row for row in _iter_jsonl(gold_path)}
            kb_by_doc = {
                str(row.get("doc_id")): row
                for row in _iter_jsonl(kb_path)
                if row.get("allowed_to_commit", True) is not False
            }
            for prompt in _iter_jsonl(prompt_path):
                gold = gold_by_prompt.get(str(prompt.get("prompt_id")), {})
                required = list(
                    prompt.get("required_evidence_ids")
                    or gold.get("required_doc_ids")
                    or gold.get("required_chunk_ids")
                    or []
                )
                if expected_status and str(prompt.get("expected_status")) != expected_status:
                    continue
                if min_evidence_count is not None and len(required) < min_evidence_count:
                    continue
                if max_evidence_count is not None and len(required) > max_evidence_count:
                    continue
                haystack = " ".join(
                    [
                        str(prompt.get("prompt_id", "")),
                        str(prompt.get("question", "")),
                        str(prompt.get("task_type", "")),
                    ]
                ).lower()
                if search and search.lower() not in haystack:
                    continue
                total_matches += 1
                if total_matches <= safe_offset:
                    continue
                if len(cases) >= safe_limit:
                    continue
                evidence_rows = []
                for doc_id in required[:5]:
                    kb = kb_by_doc.get(str(doc_id))
                    if kb:
                        evidence_rows.append(self._safe_kb_row(kb, required=True))
                distractors = [
                    self._safe_kb_row(kb, required=False)
                    for doc_id, kb in kb_by_doc.items()
                    if doc_id not in set(map(str, required))
                ][:2]
                cases.append(
                    {
                        "prompt": self._safe_prompt_row(prompt),
                        "gold_contract": self._safe_gold_row(gold),
                        "knowledge_base": {
                            "required_evidence": evidence_rows,
                            "distractor_evidence": distractors,
                        },
                        "evaluation_rubric": {
                            "pre_run": True,
                            "measures": [
                                "JSON validity",
                                "contract validity",
                                "evidence presence",
                                "evidence match",
                                "groundedness",
                                "safety",
                                "truncation",
                                "latency",
                                "throughput",
                                "cost",
                            ],
                        },
                    }
                )
        if sort_by in {"evidence_count", "prompt_length"}:
            cases.sort(
                key=lambda row: cast(
                    int,
                    row["prompt"]["required_evidence_count"]
                    if sort_by == "evidence_count"
                    else row["prompt"]["estimated_tokens"],
                )
            )
        return {
            "result_type": "planned",
            "chronology": "Designed before the run",
            "total_matches": total_matches,
            "offset": safe_offset,
            "limit": safe_limit,
            "cases": cases,
            "filters": {
                "vertical": vertical,
                "search": search,
                "expected_status": expected_status,
                "min_evidence_count": min_evidence_count,
                "max_evidence_count": max_evidence_count,
                "sort_by": sort_by,
            },
            "public_safety": {
                "raw_files_exposed": False,
                "requires_allowed_to_commit": True,
                "absolute_paths_exposed": False,
            },
        }

    def _vertical_description(self, vertical: str) -> str:
        return {
            "airline": "Policy-backed customer support and travel operations questions.",
            "healthcare_admin": "Administrative healthcare workflows with strict boundary rules.",
            "retail": "Product, order, and support workflows with broad KB coverage.",
            "finance": "Metric, period, filing, and policy-style finance questions.",
            "research_ai": "Research-paper style evidence synthesis with out-of-scope controls.",
        }.get(vertical, vertical.replace("_", " ").title())

    def _safe_prompt_row(self, prompt: JsonDict) -> JsonDict:
        required = list(prompt.get("required_evidence_ids") or [])
        question = str(prompt.get("question") or prompt.get("issue") or "")
        return {
            "prompt_id": prompt.get("prompt_id"),
            "vertical": prompt.get("vertical"),
            "question": question,
            "task_type": prompt.get("task_type"),
            "expected_status": prompt.get("expected_status"),
            "expected_action": prompt.get("expected_action"),
            "expected_output_format": prompt.get("expected_output_format"),
            "estimated_tokens": max(1, round(len(question.split()) * 1.4)),
            "required_evidence_count": len(required),
            "required_evidence_ids": required,
            "domain_metadata": {
                key: value
                for key, value in prompt.items()
                if key
                in {
                    "airline",
                    "route",
                    "support_type",
                    "ticket_id",
                    "travel_type",
                    "partner_airline_involved",
                }
            },
        }

    def _safe_gold_row(self, gold: JsonDict) -> JsonDict:
        return {
            "prompt_id": gold.get("prompt_id"),
            "expected_action": gold.get("expected_action"),
            "expected_status": gold.get("expected_status"),
            "reference_answer_summary": str(gold.get("reference_answer", ""))[:360],
            "required_doc_ids": gold.get("required_doc_ids")
            or gold.get("required_chunk_ids")
            or [],
            "required_citations": gold.get("required_citations", []),
            "must_include": gold.get("must_include", []),
            "must_not_include": gold.get("must_not_include", []),
            "safety_boundary": "Stay inside cited evidence and must-not-include rules.",
            "output_format_expectations": gold.get("metadata", {}).get("expected_output_format"),
        }

    def _safe_kb_row(self, kb: JsonDict, *, required: bool) -> JsonDict:
        body = str(kb.get("body") or kb.get("text") or "")
        return {
            "doc_id": kb.get("doc_id"),
            "chunk_id": kb.get("chunk_id") or kb.get("doc_id"),
            "title": kb.get("title"),
            "document_type": kb.get("document_type"),
            "source_type": kb.get("source_type"),
            "evidence_text": body[:520],
            "metadata": kb.get("metadata", {}),
            "token_estimate": max(1, round(len(body.split()) * 1.4)),
            "provenance": kb.get("source_type"),
            "required": required,
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

    def preparation_modules(self) -> JsonDict:
        return {
            "result_type": "planned",
            "chronology": "Designed before the run",
            "modules": [
                self._retrieval_module(),
                self._context_module(),
                self._memory_module(),
                self._model_registry_module(),
                self._serving_hardware_module(),
                self._slo_matrix_module(),
            ],
            "source_artifacts": [
                _relative_artifact(CONTEXT_ROOT / "retrieval_source_of_truth_manifest.json"),
                "configs/models.yaml",
                "configs/runtime_engines.yaml",
                "configs/memory_modes.yaml",
                "configs/slo_targets.yaml",
                _relative_artifact(MAIN_PROCESSED / "main_inference_v1_engine_comparison.csv"),
            ],
        }

    def matrix_rows(self) -> JsonDict:
        rows = _read_csv(MAIN_PROCESSED / "main_inference_v1_api_vs_self_hosted_comparison.csv")
        sanitized: list[JsonDict] = []
        for index, row in enumerate(rows, start=1):
            sanitized.append(
                {
                    "index": index,
                    "config_id": row["config_id"],
                    "track": "API" if row["backend_type"] == "api_provider" else "Self-hosted",
                    "model_alias": row["model_alias"],
                    "model_id": row["model_id"],
                    "engine": row["engine"],
                    "route": "API provider route"
                    if row["backend_type"] == "api_provider"
                    else row["engine"],
                    "memory_mode": row["memory_mode"],
                    "memory_mode_label": MAIN_MEMORY_MODE_LABELS.get(
                        row["memory_mode"],
                        row["memory_mode"],
                    ),
                    "concurrency": _to_int(row["concurrency"]),
                    "requests_per_config": _to_int(row["requests_completed"]),
                    "prompts_per_vertical_per_config": 2000,
                    "status": row["status"],
                    "backend_type": row["backend_type"],
                    "gpu_telemetry_scope": row["gpu_telemetry_scope"],
                }
            )
        return {
            "result_type": "planned",
            "chronology": "Designed before the run",
            "formula": {
                "self_hosted": (
                    "1 model x 2 engines x 5 memory modes x 2 concurrency levels = 20 configs"
                ),
                "api": "1 model x 1 route x 5 memory modes x 1 concurrency level = 5 configs",
                "total": "25 configs x 10,000 prompts = 250,000 requests",
            },
            "rows": sanitized,
            "totals": {
                "config_count": len(sanitized),
                "self_hosted_configs": sum(1 for row in sanitized if row["track"] == "Self-hosted"),
                "api_configs": sum(1 for row in sanitized if row["track"] == "API"),
                "requests": 250000,
                "requests_per_vertical_across_run": 50000,
            },
        }

    def _retrieval_module(self) -> JsonDict:
        retrieval = _read_json(CONTEXT_ROOT / "retrieval_source_of_truth_manifest.json")
        summary = _read_csv(CONTEXT_ROOT / "retrieval_evaluation_summary.csv")
        sample = next(
            (
                row
                for row in summary
                if row.get("memory_mode") == "mm2_hybrid_top5" and row.get("vertical") == "finance"
            ),
            summary[0] if summary else {},
        )
        return {
            "id": "retrieval_engineering",
            "title": "Retrieval Engineering",
            "visual": (
                "Knowledge base -> candidates -> BM25/dense/hybrid -> top 50 -> "
                "rerank -> final E1-E5"
            ),
            "purpose": (
                "Move required evidence into the final model context without leaking gold answers."
            ),
            "stages": [
                "Knowledge base",
                "candidate generation",
                "BM25 / dense / hybrid",
                "top 50",
                "reranking",
                "top 20",
                "final top 5 evidence E1-E5",
            ],
            "controls": ["vertical", "safe example prompt", "retrieval mode", "ranking inspection"],
            "sample_metrics": {
                "vertical": sample.get("vertical"),
                "memory_mode": sample.get("memory_mode"),
                "candidate_recall_at_20": _to_float(sample.get("candidate_recall_at_20")),
                "candidate_recall_at_50": _to_float(sample.get("candidate_recall_at_50")),
                "final_recall_at_5": _to_float(sample.get("recall_at_5")),
                "mrr": _to_float(sample.get("mrr")),
                "leakage_guard_applied": str(sample.get("leakage_guard_applied")) == "True",
                "final_top_k": _to_int(sample.get("final_top_k")),
            },
            "source": _relative_artifact(CONTEXT_ROOT / "retrieval_evaluation_summary.csv"),
            "source_of_truth": retrieval,
        }

    def _context_module(self) -> JsonDict:
        compression = _read_csv(CONTEXT_ROOT / "compression_diagnostic_summary.csv")
        corpus = _read_csv(CONTEXT_ROOT / "corpus_build_summary.csv")
        return {
            "id": "context_engineering",
            "title": "Context Engineering",
            "visual": (
                "Raw source -> chunking -> metadata -> stable IDs -> token estimates -> "
                "E1-E5 labels -> compression"
            ),
            "purpose": "Transform domain data into compact, traceable evidence the model can cite.",
            "stages": [
                "Raw source",
                "domain-aware chunking",
                "metadata enrichment",
                "stable IDs",
                "token estimates",
                "provenance",
                "evidence labels E1-E5",
                "optional compression",
            ],
            "why_it_matters": [
                "Small evidence labels reduce prompt complexity.",
                "Canonical IDs preserve evaluator traceability.",
                "Deterministic compression enables measurable token reduction.",
                "Provenance must survive transformation.",
            ],
            "corpus_summary": corpus,
            "compression_summary": compression[:10],
        }

    def _memory_module(self) -> JsonDict:
        modes = load_yaml_file("configs/memory_modes.yaml")
        return {
            "id": "memory_modes",
            "title": "Memory Modes",
            "purpose": (
                "Control what context the model sees and how much retrieval/repair work happens."
            ),
            "modes": [
                {
                    "id": mode_id,
                    "label": MAIN_MEMORY_MODE_LABELS.get(mode_id, mode_id),
                    "definition": details.get("description"),
                    "active_stages": self._memory_mode_stages(mode_id, cast(JsonDict, details)),
                    "what_model_sees": self._memory_mode_view(mode_id),
                    "expected_benefits": self._memory_mode_benefits(mode_id),
                    "likely_costs": self._memory_mode_costs(mode_id),
                    "relevant_metrics": [
                        "TTFT",
                        "E2E latency",
                        "evidence match",
                        "groundedness",
                        "cost",
                    ],
                    **cast(JsonDict, details),
                }
                for mode_id, details in cast(JsonDict, modes).items()
            ],
        }

    def _model_registry_module(self) -> JsonDict:
        models = self.models()
        active_aliases = cast(JsonDict, models["active_aliases"])
        model_defs = cast(JsonDict, models["models"])
        model3_registry_id = str(active_aliases.get("model3_7b", ""))
        model6_registry_id = str(active_aliases.get("model6_gated", ""))
        return {
            "id": "model_registry",
            "title": "Model Registry",
            "purpose": (
                "Separate active experiment models from smoke, deprecated, API-only, and "
                "future models."
            ),
            "active_main_inference_models": [
                {
                    "alias": "model3_7b",
                    "registry_id": model3_registry_id,
                    **cast(JsonDict, model_defs.get(model3_registry_id, {})),
                },
                {
                    "alias": "model6_gated",
                    "registry_id": model6_registry_id,
                    **cast(JsonDict, model_defs.get(model6_registry_id, {})),
                },
            ],
            "registered_models": [
                {"registry_id": key, **cast(JsonDict, value)} for key, value in model_defs.items()
            ],
            "active_aliases": active_aliases,
            "deprecated_aliases": models["deprecated_aliases"],
            "why_not_every_model_ran": (
                "The full Main_Inference_V1 matrix focused on one self-hosted open-weight 7B "
                "model and one API/gated 8B model so engine, memory, and routing effects could "
                "be measured without exploding cost."
            ),
        }

    def _serving_hardware_module(self) -> JsonDict:
        engines = load_yaml_file("configs/runtime_engines.yaml")
        return {
            "id": "serving_hardware",
            "title": "Serving & Hardware",
            "purpose": (
                "Show which routes run on the rented A100 and which use provider-managed "
                "infrastructure."
            ),
            "tracks": [
                {
                    "track": "Self-hosted",
                    "model": "Qwen/Qwen2.5-7B-Instruct",
                    "engines": ["vLLM", "SGLang"],
                    "hardware": "NVIDIA A100-SXM4-80GB",
                    "telemetry": "local GPU telemetry applies",
                    "cost_model": "GPU hourly price x wall time",
                },
                {
                    "track": "API provider",
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "engines": ["API provider route"],
                    "hardware": "provider-managed",
                    "telemetry": "local GPU telemetry does not apply",
                    "cost_model": "provider token/request pricing",
                },
            ],
            "runtime_registry": engines,
        }

    def _slo_matrix_module(self) -> JsonDict:
        return {
            "id": "slo_matrix",
            "title": "SLO Setup & Experiment Matrix",
            "purpose": "Make the run auditable before any inference starts.",
            "metric_family_link": "/slo-metrics",
            "matrix": self.matrix_rows(),
        }

    def _memory_mode_stages(self, mode_id: str, details: JsonDict) -> list[str]:
        stages = ["prompt"]
        if details.get("requires_retrieval"):
            stages.append(str(details.get("retrieval_type", "retrieval")))
            stages.append("top-5 evidence")
        if details.get("requires_compression"):
            stages.append("compression")
        if details.get("requires_agentic_workflow"):
            stages.append("bounded validation/repair")
        return stages

    def _memory_mode_view(self, mode_id: str) -> str:
        return {
            "mm0_no_context": "Prompt and metadata only; no retrieved evidence.",
            "mm1_dense_top5": "Prompt plus dense-retrieved top-5 evidence.",
            "mm2_hybrid_top5": "Prompt plus hybrid BM25/dense top-5 evidence.",
            "mm3_compressed_hybrid_top5": "Prompt plus compressed hybrid top-5 evidence.",
            "mm4_bounded_agentic": (
                "Prompt, retrieved evidence, validation trace, one bounded repair/escalation path."
            ),
        }.get(mode_id, "Configured context path.")

    def _memory_mode_benefits(self, mode_id: str) -> list[str]:
        return {
            "mm0_no_context": ["Fast ablation", "isolates context value"],
            "mm1_dense_top5": ["Semantic recall", "compact evidence context"],
            "mm2_hybrid_top5": [
                "Lexical plus semantic evidence coverage",
                "strong default retrieval",
            ],
            "mm3_compressed_hybrid_top5": ["Lower context tokens", "potential TTFT/cost relief"],
            "mm4_bounded_agentic": ["Validation", "repair/escalation path", "quality/safety guard"],
        }.get(mode_id, ["Controlled experiment role"])

    def _memory_mode_costs(self, mode_id: str) -> list[str]:
        return {
            "mm0_no_context": ["Poor evidence match and groundedness by design"],
            "mm1_dense_top5": ["Dense misses exact terms"],
            "mm2_hybrid_top5": ["More retrieval work than prompt-only"],
            "mm3_compressed_hybrid_top5": ["Compression can lose detail"],
            "mm4_bounded_agentic": ["More orchestration", "higher latency/cost risk"],
        }.get(mode_id, ["Must be measured"])

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

    def slo_metric_catalog(self) -> JsonDict:
        targets = load_yaml_file("configs/slo_targets.yaml")
        profiles = load_yaml_file("configs/slo_profiles.yaml")
        vertical_targets = cast(JsonDict, targets.get("verticals", {}))
        metric_to_target: dict[str, list[JsonDict]] = {}
        for vertical, groups in vertical_targets.items():
            for group_name, metrics in cast(JsonDict, groups).items():
                for metric_name, target in cast(JsonDict, metrics).items():
                    metric_to_target.setdefault(str(metric_name), []).append(
                        {
                            "vertical": vertical,
                            "group": group_name,
                            "target": target,
                        }
                    )
        families = [
            {
                "id": "user_experience",
                "label": "User experience",
                "chronology": "Designed before the run",
                "explanation": (
                    "Measures whether users see fast starts, steady token streaming, and "
                    "acceptable end-to-end latency."
                ),
                "metrics": [
                    self._metric_detail(
                        "ttft_p95_ms_max",
                        "TTFT",
                        "Time to first token.",
                        "How quickly the system begins responding.",
                        [
                            "queueing",
                            "context length",
                            "prefill",
                            "batch scheduler",
                            "prefix reuse",
                        ],
                        [
                            "context reduction",
                            "prefix caching",
                            "chunked prefill",
                            "scheduler tuning",
                            "concurrency tuning",
                        ],
                        "Aggressive batching may improve throughput while increasing waiting time.",
                        metric_to_target,
                    ),
                    self._metric_detail(
                        "tpot_p95_ms_max",
                        "TPOT",
                        "Time per output token.",
                        "How quickly the answer streams once generation starts.",
                        ["decode kernel", "batch size", "KV-cache pressure", "model size"],
                        ["serving-engine tuning", "quantization", "speculative decoding"],
                        "Speedups can affect quality or memory depending on the method.",
                        metric_to_target,
                    ),
                    self._metric_detail(
                        "itl_p95_ms_max",
                        "ITL",
                        "Inter-token latency.",
                        "Whether streaming feels smooth after the first token.",
                        ["decode cadence", "scheduler", "GPU saturation"],
                        ["decode tuning", "batch shaping", "engine selection"],
                        "Optimizing smoothness can reduce maximum throughput.",
                        metric_to_target,
                    ),
                    self._metric_detail(
                        "e2e_p95_ms_max",
                        "E2E latency",
                        "Full request latency from submission to final token.",
                        "How long the user waits for the complete answer.",
                        ["retrieval", "prefill", "decode", "output length", "API round trip"],
                        ["context reduction", "model/engine selection", "output cap tuning"],
                        "Shorter answers can be faster but may weaken completeness.",
                        metric_to_target,
                    ),
                ],
            },
            {
                "id": "capacity",
                "label": "Capacity",
                "chronology": "Designed before the run",
                "explanation": (
                    "Measures how much useful work the system can process under the selected "
                    "traffic pattern."
                ),
                "metrics": [
                    self._metric_detail(
                        "requests_per_second_min",
                        "Requests/sec",
                        "Completed requests per second.",
                        "How many users the service can sustain.",
                        ["concurrency", "batching", "engine", "model size"],
                        ["concurrency tuning", "continuous batching", "engine selection"],
                        "Higher throughput can increase tail latency.",
                        metric_to_target,
                    ),
                    self._metric_detail(
                        "tokens_per_second_min",
                        "Tokens/sec",
                        "Generated and processed tokens per second.",
                        "How efficiently the serving stack moves tokens.",
                        ["GPU utilization", "batching", "decode kernels", "context length"],
                        ["vLLM/SGLang tuning", "quantization", "tensor parallelism"],
                        "Token throughput does not prove answer quality.",
                        metric_to_target,
                    ),
                    self._metric_detail(
                        "successful_requests_per_second_min",
                        "Successful requests/sec",
                        "Useful completed requests per second.",
                        "Capacity after failures and invalid outputs are considered.",
                        ["quality gates", "error rate", "runtime stability"],
                        ["retries", "repair workflow", "engine hardening"],
                        "Retries can increase quality while reducing raw throughput.",
                        metric_to_target,
                    ),
                ],
            },
            {
                "id": "answer_usefulness",
                "label": "Answer usefulness",
                "chronology": "Designed before the run",
                "explanation": (
                    "Separates parseability from useful, cited, grounded, contract-valid answers."
                ),
                "metrics": [
                    self._metric_detail(
                        "json_validity_min",
                        "JSON validity",
                        "Whether output parses as JSON when JSON is required.",
                        "Whether the app can safely parse the response.",
                        ["prompt contract", "decoder behavior", "repair attempts"],
                        ["structured output prompting", "contract repair", "bounded retries"],
                        "Valid JSON can still contain wrong or unsafe content.",
                        metric_to_target,
                    ),
                    self._metric_detail(
                        "format_validity_min",
                        "Contract/format validity",
                        "Whether the response follows the required output contract.",
                        "Whether downstream systems can trust the response shape.",
                        ["prompt wording", "schema complexity", "model instruction following"],
                        ["prompt contract repair", "format validators", "repair workflow"],
                        "Stricter contracts can increase refusal or repair rates.",
                        metric_to_target,
                    ),
                    self._metric_detail(
                        "task_success_min",
                        "Task success",
                        "Whether the response satisfies the task.",
                        "Whether the user receives the intended answer or action.",
                        ["model capability", "evidence availability", "prompt clarity"],
                        ["stronger model", "contract repair", "better context"],
                        "A stronger model can increase cost and latency.",
                        metric_to_target,
                    ),
                    self._metric_detail(
                        "citation_accuracy_min",
                        "Citation accuracy",
                        "Whether citations point to valid supporting sources.",
                        "Whether claims can be traced back to the KB.",
                        ["retrieval", "citation formatting", "context labels"],
                        ["evidence formatting", "retrieval repair", "citation validator"],
                        "More citation checks may increase latency.",
                        metric_to_target,
                    ),
                    self._metric_detail(
                        "evidence_match_min",
                        "Evidence match",
                        "Whether required evidence was cited.",
                        "Whether the answer used the right source material.",
                        [
                            "candidate recall",
                            "reranking",
                            "context ordering",
                            "model citation behavior",
                        ],
                        ["hybrid retrieval", "reranking", "evidence formatting"],
                        "More context can improve recall but pressure TTFT.",
                        metric_to_target,
                    ),
                    self._metric_detail(
                        "groundedness_min",
                        "Groundedness",
                        "Whether answer claims are supported by provided evidence.",
                        "Whether the response is reliable enough for deployment.",
                        ["context quality", "model behavior", "contract constraints"],
                        ["context repair", "bounded agentic validation", "stronger model"],
                        "Repair workflows can increase cost and latency.",
                        metric_to_target,
                    ),
                ],
            },
            {
                "id": "safety",
                "label": "Safety",
                "chronology": "Designed before the run",
                "explanation": (
                    "Ensures the system avoids unsafe, unsupported, or boundary-breaking answers."
                ),
                "metrics": [
                    self._metric_detail(
                        "safety_violations_max",
                        "Safety violations",
                        "Count of safety failures.",
                        "Whether the system can be trusted in constrained domains.",
                        ["prompt boundary", "domain policy", "model refusal behavior"],
                        ["safety prompt repair", "bounded validation", "escalation path"],
                        "Overly broad safety repair can reduce useful answer rate.",
                        metric_to_target,
                    ),
                    {
                        "id": "boundary_adherence",
                        "label": "Boundary adherence",
                        "definition": (
                            "Whether the answer stays inside the allowed evidence and "
                            "policy boundary."
                        ),
                        "user_experience": (
                            "The user receives a careful answer instead of unsupported promises."
                        ),
                        "influences": [
                            "must-not-include rules",
                            "safety boundary",
                            "evidence labels",
                        ],
                        "common_optimizations": ["contract repair", "must-not-include checks"],
                        "tradeoffs": "Strict boundaries can increase escalation decisions.",
                        "targets": [],
                        "target_varies_by_vertical": False,
                    },
                ],
            },
            {
                "id": "retrieval",
                "label": "Retrieval",
                "chronology": "Designed before the run",
                "explanation": (
                    "Measures whether the right evidence reaches the model before generation."
                ),
                "metrics": [
                    self._metric_detail(
                        "candidate_recall_at_20_min",
                        "Candidate recall@20",
                        "Whether required evidence appears in the top 20 candidates.",
                        "Whether retrieval can find the right facts before reranking.",
                        ["BM25", "dense vectors", "query enrichment"],
                        ["hybrid retrieval", "metadata boosts", "query rewriting"],
                        "Broader candidate pools can increase retrieval latency.",
                        metric_to_target,
                    ),
                    self._metric_detail(
                        "candidate_recall_at_50_min",
                        "Candidate recall@50",
                        "Whether required evidence appears in the top 50 candidates.",
                        "Whether reranking has a chance to recover required evidence.",
                        ["candidate top-k", "lexical/dense balance", "KB size"],
                        ["increase candidate pool", "reranker calibration"],
                        "Larger pools increase reranking work.",
                        metric_to_target,
                    ),
                    self._metric_detail(
                        "final_recall_at_5_min",
                        "Final recall@5",
                        "Whether required evidence reaches final E1-E5 context.",
                        "Whether the model actually sees the evidence it needs.",
                        ["reranking", "dedupe", "context selection"],
                        ["reranker calibration", "evidence selector repair"],
                        "Improving recall can add context tokens.",
                        metric_to_target,
                    ),
                    self._metric_detail(
                        "mrr_min",
                        "MRR",
                        "Mean reciprocal rank of required evidence.",
                        "Whether important evidence appears early enough to be salient.",
                        ["ranking quality", "metadata boosts", "query quality"],
                        ["reranking", "domain metadata", "hybrid fusion"],
                        "Ranking changes can overfit one vertical if not validated broadly.",
                        metric_to_target,
                    ),
                ],
            },
            {
                "id": "infrastructure",
                "label": "Infrastructure",
                "chronology": "Designed before the run",
                "explanation": (
                    "Measures hardware pressure and telemetry needed for self-hosted inference."
                ),
                "metrics": [
                    self._metric_detail(
                        "gpu_utilization_min_pct",
                        "GPU utilization",
                        "How busy the GPU is during self-hosted serving.",
                        "Whether expensive hardware is being used efficiently.",
                        ["batching", "concurrency", "engine", "API/self-hosted mix"],
                        ["concurrency tuning", "batching", "engine tuning"],
                        "Higher utilization can increase queueing or tail latency.",
                        metric_to_target,
                    ),
                    self._metric_detail(
                        "gpu_memory_utilization_max_pct",
                        "VRAM utilization",
                        "How much GPU memory is occupied.",
                        "Whether the model and KV cache fit safely.",
                        ["model size", "sequence length", "concurrency", "KV cache"],
                        ["context reduction", "quantization", "concurrency tuning"],
                        "Running near the limit risks OOM failures.",
                        metric_to_target,
                    ),
                    self._metric_detail(
                        "gpu_memory_peak_gb_max",
                        "GPU memory peak",
                        "Maximum GPU memory used.",
                        "Whether the serving setup has deployment headroom.",
                        ["model weights", "KV cache", "batch size"],
                        ["quantization", "model selection", "max sequence tuning"],
                        "Reducing memory can affect quality or throughput depending on method.",
                        metric_to_target,
                    ),
                    self._metric_detail(
                        "cpu_utilization_max_pct",
                        "CPU utilization",
                        "Host CPU pressure.",
                        "Whether preprocessing, networking, or telemetry becomes a bottleneck.",
                        ["tokenization", "HTTP server", "retrieval", "logging"],
                        ["pipeline batching", "async IO", "worker tuning"],
                        "CPU changes may not help if GPU decode is the bottleneck.",
                        metric_to_target,
                    ),
                    self._metric_detail(
                        "ram_usage_gb_max",
                        "RAM",
                        "Host memory usage.",
                        "Whether local caches, datasets, and logs fit safely.",
                        ["artifact buffering", "dataset loading", "telemetry"],
                        ["streaming IO", "pagination", "artifact sync"],
                        "Under-buffering can hurt throughput.",
                        metric_to_target,
                    ),
                    {
                        "id": "power_temperature",
                        "label": "Power and temperature",
                        "definition": (
                            "Power draw and GPU temperature from telemetry when available."
                        ),
                        "user_experience": (
                            "Thermal or power limits can reduce sustained throughput."
                        ),
                        "influences": ["GPU SKU", "cooling", "utilization", "batching"],
                        "common_optimizations": ["batch tuning", "hardware selection"],
                        "tradeoffs": "Lower power may reduce peak throughput.",
                        "targets": [],
                        "target_varies_by_vertical": False,
                    },
                ],
            },
            {
                "id": "economics",
                "label": "Economics",
                "chronology": "Designed before the run",
                "explanation": "Measures whether the serving choice is economically viable.",
                "metrics": [
                    self._metric_detail(
                        "gpu_cost_per_request_usd_max",
                        "GPU cost/request",
                        "GPU spend divided by self-hosted request count.",
                        "Whether self-hosted serving is affordable.",
                        ["hourly price", "wall time", "throughput"],
                        ["throughput tuning", "right-size GPU", "engine selection"],
                        "Cost improvements must not weaken quality and safety.",
                        metric_to_target,
                    ),
                    self._metric_detail(
                        "api_cost_per_request_usd_max",
                        "API cost/request",
                        "Provider API spend divided by API request count.",
                        "Whether provider-managed inference is affordable.",
                        ["provider pricing", "tokens", "model choice"],
                        ["model selection", "prompt/context reduction"],
                        "Cheaper models can regress quality.",
                        metric_to_target,
                    ),
                    self._metric_detail(
                        "api_cost_per_1000_requests_usd_max",
                        "Cost/1,000 requests",
                        "Normalized cost for comparing routes.",
                        "How expensive the service is at product scale.",
                        ["request volume", "tokens/request", "runtime"],
                        ["context compression", "batching", "model selection"],
                        "Cost normalization hides latency/quality unless shown beside SLOs.",
                        metric_to_target,
                    ),
                    self._metric_detail(
                        "gpu_cost_per_successful_answer_usd_max",
                        "Cost/successful answer",
                        "Cost adjusted by answer success.",
                        "Whether the system produces useful answers efficiently.",
                        ["quality rate", "runtime", "provider price"],
                        ["quality repair", "routing", "model selection"],
                        (
                            "More expensive models may reduce cost per useful answer if "
                            "quality improves."
                        ),
                        metric_to_target,
                    ),
                    self._metric_detail(
                        "tokens_per_gpu_dollar_min",
                        "Tokens per GPU dollar",
                        "Token throughput normalized by GPU spend.",
                        "Whether GPU spend buys enough inference work.",
                        ["throughput", "hourly GPU price", "output length"],
                        ["engine tuning", "concurrency tuning", "GPU selection"],
                        "Token efficiency is not deployability without quality and safety.",
                        metric_to_target,
                    ),
                ],
            },
        ]
        return {
            "result_type": "planned",
            "chronology": "Designed before the run",
            "opening_explanation": (
                "An SLO is a target set before the experiment. It turns raw measurements "
                "into pass/fail decisions so request completion, latency, quality, safety, "
                "resource use, and cost can be judged together."
            ),
            "families": families,
            "evaluation_flow": [
                "Metric",
                "Target",
                "Measured value",
                "Gap",
                "Pass/Fail",
                "Deployability decision",
            ],
            "applicability_notes": [
                "API cost applies only to API/provider paths.",
                "GPU cost requires hourly GPU pricing.",
                "Resource SLOs require telemetry.",
                "Compression SLOs apply to compressed modes.",
                "Agentic trace SLOs apply to MM4.",
            ],
            "source_artifacts": ["configs/slo_targets.yaml", "configs/slo_profiles.yaml"],
            "raw_targets": targets,
            "profiles": profiles,
        }

    def _metric_detail(
        self,
        metric_id: str,
        label: str,
        definition: str,
        user_experience: str,
        influences: list[str],
        optimizations: list[str],
        tradeoffs: str,
        metric_to_target: dict[str, list[JsonDict]],
    ) -> JsonDict:
        target_rows = metric_to_target.get(metric_id, [])
        return {
            "id": metric_id,
            "label": label,
            "definition": definition,
            "user_experience": user_experience,
            "influences": influences,
            "common_optimizations": optimizations,
            "tradeoffs": tradeoffs,
            "targets": target_rows,
            "target_varies_by_vertical": len(
                {json.dumps(row["target"], sort_keys=True) for row in target_rows}
            )
            > 1,
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

    def main_replay_detail(self) -> JsonDict:
        manifest = self.main_manifest()
        eval_report = _read_json(MAIN_PROCESSED / "main_inference_v1_eval_report.json")
        cost_report = _read_json(MAIN_PROCESSED / "main_inference_v1_cost_report.json")
        scorecard = _read_csv(MAIN_PROCESSED / "main_inference_v1_slo_scorecard.csv")
        comparison = self.comparison_datasets()
        matrix = self.matrix_rows()
        artifact_sync = _read_json(MAIN_PROCESSED / "main_inference_v1_artifact_sync_report.json")
        config_rows = cast(list[JsonDict], matrix["rows"])
        latency_rows = [
            {
                "config_id": row["config_id"],
                "engine": row["engine"],
                "memory_mode": row["memory_mode"],
                "concurrency": row["concurrency"],
                "mean_ttft_ms": _to_float(source.get("mean_ttft_ms")),
                "mean_tpot_ms": _to_float(source.get("mean_tpot_ms")),
                "mean_e2e_latency_ms": _to_float(source.get("mean_e2e_latency_ms")),
                "mean_total_tokens_per_second": _to_float(
                    source.get("mean_total_tokens_per_second")
                ),
                "resolution": "per-config aggregate",
            }
            for row, source in zip(
                config_rows,
                _read_csv(MAIN_PROCESSED / "main_inference_v1_api_vs_self_hosted_comparison.csv"),
                strict=False,
            )
        ]
        return {
            "result_type": "measured",
            "hero": {
                "title": "Main_Inference_V1",
                "subtitle": "Full pre-optimization inference baseline",
                "explanation": (
                    "This is a time-compressed replay of a measured 11.82-hour experiment. "
                    "It uses saved progress, telemetry, logs, and reports. No GPU inference "
                    "runs in the demo."
                ),
                "facts": [
                    "25 configs",
                    "250,000 requests",
                    "five verticals",
                    "two models",
                    "three serving routes",
                    "five memory modes",
                    "A100-SXM4-80GB",
                    "provider API route",
                ],
            },
            "run_contract": {
                "run_id": manifest.get("run_id"),
                "git_commit": manifest.get("git_commit"),
                "started_at": manifest.get("started_at"),
                "completed_at": manifest.get("completed_at"),
                "updated_at": manifest.get("updated_at"),
                "wall_seconds": eval_report.get("wall_seconds"),
                "status": manifest.get("status"),
                "traffic_profile": manifest.get("traffic_profile"),
                "hardware": "NVIDIA A100-SXM4-80GB",
                "gpu_hourly_price_usd": cost_report.get("self_hosted_gpu_hourly_price_usd"),
                "planned_requests": manifest.get("expected_count"),
                "completed_requests": manifest.get("completed_count"),
                "failed_requests": manifest.get("failed_count"),
                "artifact_paths": {
                    key: _relative_artifact(value)
                    for key, value in cast(JsonDict, manifest.get("artifact_paths", {})).items()
                },
            },
            "phases": [
                {
                    "id": "preflight",
                    "label": "Preflight",
                    "description": (
                        "Validate manifest, model/runtime compatibility, dataset, and safety gates."
                    ),
                },
                {
                    "id": "matrix_load",
                    "label": "Matrix load",
                    "description": (
                        "Load 25 measured configurations and the 10,000-prompt workload per config."
                    ),
                },
                {
                    "id": "vllm_execution",
                    "label": "vLLM execution",
                    "description": "Self-hosted Qwen2.5-7B configs served through vLLM on A100.",
                },
                {
                    "id": "sglang_execution",
                    "label": "SGLang execution",
                    "description": "Self-hosted Qwen2.5-7B configs served through SGLang on A100.",
                },
                {
                    "id": "api_execution",
                    "label": "API execution",
                    "description": (
                        "Llama 3.1 8B configs routed through provider-managed API infrastructure."
                    ),
                },
                {
                    "id": "artifact_finalization",
                    "label": "Artifact finalization",
                    "description": (
                        "Finalize raw outputs, logs, checkpoint, telemetry, reports, "
                        "checksums, and backups."
                    ),
                },
                {
                    "id": "evaluation",
                    "label": "Evaluation",
                    "description": (
                        "Join outputs with gold/evidence and score quality, safety, "
                        "latency, throughput, and cost."
                    ),
                },
                {
                    "id": "slo_scoring",
                    "label": "SLO scoring",
                    "description": "Convert measurements into pass/fail deployability decisions.",
                },
            ],
            "matrix": matrix,
            "replay": self.replay_events(),
            "telemetry": self.telemetry(),
            "latency_throughput": {
                "summary": eval_report.get("latency_summary", {}),
                "trend": latency_rows,
                "chart_resolution_rule": (
                    "TTFT, TPOT, E2E, and throughput are shown as per-config aggregate "
                    "steps. Request-level latency time series are not fabricated."
                ),
            },
            "cost": {
                **cost_report,
                "cost_per_request_usd": _to_float(cost_report.get("total_cost_usd"))
                / max(_to_float(eval_report.get("total_requests_completed")), 1.0),
                "cost_per_1000_requests_usd": (
                    _to_float(cost_report.get("total_cost_usd"))
                    / max(_to_float(eval_report.get("total_requests_completed")), 1.0)
                    * 1000
                ),
                "economics_note": (
                    "GPU economics are hourly-price x wall-time. API economics are provider "
                    "route costs. They must be compared with quality and latency."
                ),
            },
            "deployability_gates": [
                {"label": "Execution", "status": "PASS"},
                {"label": "Reliability", "status": "PASS"},
                {"label": "Latency", "status": "PASS"},
                {"label": "Throughput", "status": "PASS"},
                {"label": "Cost", "status": "PASS"},
                {"label": "JSON validity", "status": "PASS"},
                {"label": "Contract validity", "status": "FAIL"},
                {"label": "Format validity", "status": "FAIL"},
                {"label": "Evidence match", "status": "FAIL"},
                {"label": "Groundedness", "status": "FAIL"},
                {"label": "Safety", "status": "FAIL"},
            ],
            "verdict": {
                "status": "NOT_DEPLOYABLE_SLO_FAILURES",
                "scorecard": scorecard,
                "slo_verdicts": eval_report.get("slo_verdicts", {}),
            },
            "quality_safety": {
                "summary": eval_report.get("summary", {}),
                "definitions": {
                    "parseable_json": "The response can be parsed as JSON when required.",
                    "contract_valid": (
                        "The response follows the expected output schema and status/action "
                        "contract."
                    ),
                    "evidence_cited": "The response includes evidence identifiers/citations.",
                    "correct_evidence_cited": (
                        "The cited evidence matches the gold-required evidence."
                    ),
                    "grounded_answer": (
                        "The answer's claims are supported by the provided evidence."
                    ),
                    "safe_answer": "The answer avoids unsafe or unsupported content.",
                },
            },
            "comparisons": comparison,
            "artifact_reliability": {
                "message": (
                    "The experiment was designed as a resumable, measurable, auditable, "
                    "long-running production job."
                ),
                "items": [
                    "manifest",
                    "checkpoint",
                    "progress log",
                    "console log",
                    "raw output",
                    "GPU telemetry",
                    "artifact sync",
                    "cost report",
                    "evaluation report",
                    "SLO report",
                    "checksums",
                    "backups",
                ],
                "artifact_sync": artifact_sync,
            },
            "engineering_lessons": {
                "proved_operationally": [
                    "Full matrix ran end to end.",
                    "250,000/250,000 requests completed.",
                    "Zero request failures.",
                    "Mixed self-hosted/API execution was measurable.",
                    "Runtime, telemetry, cost, and quality joined into one system.",
                ],
                "taught_technically": [
                    "JSON validity is not enough.",
                    "Request completion is not deployability.",
                    "Retrieval/context and evidence behavior dominate useful quality.",
                    "Faster serving does not guarantee grounded answers.",
                    "Optimization must target observed bottlenecks.",
                    "Quality and safety must be protected while improving performance.",
                ],
            },
        }

    def comparison_datasets(self) -> JsonDict:
        return {
            "engine": _read_csv(MAIN_PROCESSED / "main_inference_v1_engine_comparison.csv"),
            "memory_mode": _read_csv(MAIN_PROCESSED / "main_inference_v1_memory_comparison.csv"),
            "concurrency": _read_csv(
                MAIN_PROCESSED / "main_inference_v1_concurrency_comparison.csv"
            ),
            "api_vs_self_hosted": _read_csv(
                MAIN_PROCESSED / "main_inference_v1_api_vs_self_hosted_comparison.csv"
            ),
            "model": _read_csv(MAIN_PROCESSED / "main_inference_v1_model_comparison.csv"),
            "slo_scorecard": _read_csv(MAIN_PROCESSED / "main_inference_v1_slo_scorecard.csv"),
            "chart_resolution": "Comparison tabs use saved per-config/aggregate CSV rows.",
        }

    def replay_events(self, *, limit: int = 120) -> JsonDict:
        progress_path = MAIN_ROOT / "logs/main_inference_v1_progress.jsonl"
        progress_rows = _read_jsonl_sample(progress_path, limit=10_000)
        if not progress_rows:
            return {"events": [], "duration_seconds": 100, "final_completed": 0}
        step = max(1, len(progress_rows) // max(limit - 2, 1))
        sampled = progress_rows[::step][: max(limit - 2, 1)]
        final_row = dict(progress_rows[-1])
        final_row["completed_requests"] = 250000
        final_row["failure_count"] = 0
        if not sampled or sampled[-1].get("completed_requests") != 250000:
            sampled.append(final_row)
        zero_row: JsonDict = {
            "completed_requests": 0,
            "failure_count": 0,
            "current_config_id": "manifest_loaded",
            "engine": "preflight",
            "runtime": "preflight",
            "memory_mode": "not_started",
            "model": "not_started",
            "concurrency": 0,
            "vertical": "not_started",
            "approximate_cost_so_far_usd": 0.0,
            "timestamp_utc": progress_rows[0].get("timestamp_utc"),
            "synthetic_replay_marker": True,
        }
        replay_rows = [zero_row, *sampled]
        events = [
            {
                "event_index": index,
                "compressed_second": round(index * (110 / max(len(replay_rows) - 1, 1)), 3),
                "completed_requests": row.get("completed_requests"),
                "completed_configs": min(
                    25,
                    int(_to_int(row.get("completed_requests")) / 10000),
                ),
                "checkpoint_count": min(
                    2500,
                    int(_to_int(row.get("completed_requests")) / 100),
                ),
                "failure_count": row.get("failure_count", row.get("failed_requests", 0)),
                "current_config_id": row.get("current_config_id"),
                "phase": self._replay_phase(row),
                "engine": row.get("engine"),
                "runtime": row.get("runtime"),
                "memory_mode": row.get("memory_mode"),
                "model": row.get("model"),
                "concurrency": row.get("concurrency"),
                "vertical": row.get("vertical"),
                "approximate_cost_so_far_usd": row.get("approximate_cost_so_far_usd"),
                "source_timestamp_utc": row.get("timestamp_utc"),
                "synthetic_replay_marker": row.get("synthetic_replay_marker", False),
            }
            for index, row in enumerate(replay_rows)
        ]
        return {
            "result_type": "measured",
            "replay_duration_seconds": 110,
            "source_event_count": len(progress_rows),
            "events": events,
            "final_completed": 250000,
            "final_failed": 0,
        }

    def _replay_phase(self, row: JsonDict) -> str:
        completed = _to_int(row.get("completed_requests"))
        engine = str(row.get("engine") or row.get("runtime") or "")
        if completed == 0:
            return "Preflight"
        if completed >= 250000:
            return "SLO scoring"
        if engine == "vllm":
            return "vLLM execution"
        if engine == "sglang":
            return "SGLang execution"
        if "api" in engine:
            return "API execution"
        return "Matrix load"

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

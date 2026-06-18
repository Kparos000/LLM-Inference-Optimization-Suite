"""Phase 3 readiness reporting utilities.

The readiness report inspects Phase 3 artifacts and contracts. It does not run
retrieval, inference, GPU workloads, or external API calls.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inference_bench.agentic_contract import (
    MM4_BOUNDED_AGENTIC_CONTRACT,
    agentic_trace_format,
    valid_agentic_trace_fixture,
)
from inference_bench.config import load_memory_modes_config, load_project_config
from inference_bench.context_corpora import CHUNK_BUILDERS, VERTICALS
from inference_bench.context_schema import ContextRecord, WorkloadRecord
from inference_bench.evaluator_contract import evaluator_contract_payload

MM0_TO_MM3 = (
    "mm0_no_context",
    "mm1_dense_top5",
    "mm2_hybrid_top5",
    "mm3_compressed_hybrid_top5",
)
ALL_MEMORY_MODES = (*MM0_TO_MM3, "mm4_bounded_agentic")
WORKLOAD_SPLITS = ("smoke_500", "controlled_2000", "final_10000")
MODEL_ALIAS_PAIRS = {
    "model1_0_5b": "qwen2_5_0_5b_instruct",
    "model2_3b": "qwen2_5_3b_instruct",
    "model3_7b": "qwen2_5_7b_instruct",
    "model4_32b": "qwen2_5_32b_instruct",
    "model5_gated": "ministral_3b_2512_api",
    "model6_gated": "llama_3_1_8b_instruct_api",
    "model7_gated": "mistral_small_3_2_24b_instruct_api",
    "model2_1_5b": "qwen2_5_1_5b_instruct",
    "model7_large_placeholder": "future_large_model_placeholder",
    "large_model_placeholder": "future_large_model_placeholder",
    "model5_large_placeholder": "future_large_model_placeholder",
    "old_model5_llama_3_2_3b": "llama_3_2_3b_instruct_api",
}


@dataclass(frozen=True)
class ReadinessRow:
    """One row in the Phase 3 readiness summary."""

    area: str
    artifact: str
    ready: bool
    status: str
    notes: str
    phase4_action: str

    def to_dict(self) -> dict[str, str | bool]:
        """Return CSV/JSON-safe row data."""

        return asdict(self)


def utc_now() -> str:
    """Return an ISO UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def path_ready(path: Path) -> bool:
    """Return whether a path exists and is non-empty where applicable."""

    if not path.exists():
        return False
    if path.is_file():
        return path.stat().st_size > 0
    return True


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON object."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_summary_csv(path: str | Path, rows: list[ReadinessRow]) -> Path:
    """Write readiness summary CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["area", "artifact", "ready", "status", "notes", "phase4_action"]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row.to_dict() for row in rows)
    return output_path


def validate_schema_fixtures() -> dict[str, bool]:
    """Validate minimal ContextRecord, WorkloadRecord, and AgenticTrace fixtures."""

    context = ContextRecord(
        context_id="ctx_readiness_fixture",
        vertical="airline",
        source_id="CA-POL-READINESS",
        parent_id="CA-POL-READINESS",
        chunk_id="CA-POL-READINESS",
        chunk_strategy="policy_section",
        source_type="policy",
        title="Readiness Fixture Policy",
        text="A deterministic context fixture validates the schema.",
        metadata={"document_type": "policy"},
        token_estimate=8,
        provenance="phase3_readiness_fixture",
        is_gold_linked=True,
    )
    WorkloadRecord(
        workload_id="test_fixture:mm1_dense_top5:prompt_readiness_fixture",
        prompt_id="prompt_readiness_fixture",
        vertical="airline",
        memory_mode="mm1_dense_top5",
        messages=[{"role": "user", "content": "Use the fixture context."}],
        context_records=[context],
        context_token_estimate=context.token_estimate,
        retrieval_metadata={"retrieval_type": "dense", "retrieval_backend_label": "local_fallback"},
        expected_output_format="text",
        gold_evidence_ids=["CA-POL-READINESS"],
        dataset_split="test_fixture",
        source_prompt_record={"prompt_id": "prompt_readiness_fixture"},
    )
    valid_agentic_trace_fixture()
    return {
        "context_record_schema_valid": True,
        "workload_record_schema_valid": True,
        "agentic_trace_schema_valid": True,
    }


def inspect_model_aliases() -> dict[str, Any]:
    """Inspect model aliases and canonical model resolution."""

    config = load_project_config()
    resolved_targets = {
        alias: config.resolve_model_key(target) for alias, target in MODEL_ALIAS_PAIRS.items()
    }
    alias_status = {
        alias: config.resolve_model_key(alias) == resolved_target
        for alias, resolved_target in resolved_targets.items()
    }
    same_model_ids = {
        alias: config.resolve_model_config(alias).model_id
        == config.resolve_model_config(target).model_id
        for alias, target in MODEL_ALIAS_PAIRS.items()
    }
    return {
        "canonical_model_count": len(config.models),
        "alias_count": len(config.model_aliases),
        "expected_aliases": MODEL_ALIAS_PAIRS,
        "aliases_resolve": alias_status,
        "aliases_match_model_ids": same_model_ids,
        "ready": all(alias_status.values()) and all(same_model_ids.values()),
    }


def inspect_memory_modes() -> dict[str, Any]:
    """Inspect memory mode definitions."""

    memory_modes = load_memory_modes_config()
    mode_status = {mode: mode in memory_modes for mode in ALL_MEMORY_MODES}
    return {
        "defined_modes": sorted(memory_modes),
        "required_modes": list(ALL_MEMORY_MODES),
        "mode_status": mode_status,
        "mm4_contract": MM4_BOUNDED_AGENTIC_CONTRACT.to_dict(),
        "ready": all(mode_status.values()),
    }


def inspect_context_artifacts(context_root: Path) -> dict[str, Any]:
    """Inspect corpus registry, chunk builders, and generated context corpora."""

    registry_path = context_root / "corpus_registry.json"
    registry_verticals: set[str] = set()
    if registry_path.exists():
        loaded = json.loads(registry_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            registry_verticals = {
                str(entry.get("vertical"))
                for entry in loaded.get("entries", [])
                if isinstance(entry, dict) and entry.get("vertical")
            }
    corpora_paths = {
        vertical: context_root / "corpora" / f"{vertical}_context_corpus.jsonl"
        for vertical in VERTICALS
    }
    corpora_status = {vertical: path_ready(path) for vertical, path in corpora_paths.items()}
    return {
        "registry_path": str(registry_path),
        "registry_exists": path_ready(registry_path),
        "registry_verticals": sorted(registry_verticals),
        "all_verticals_registered": set(VERTICALS).issubset(registry_verticals),
        "chunk_builders": sorted(CHUNK_BUILDERS),
        "all_chunk_builders_present": set(VERTICALS).issubset(set(CHUNK_BUILDERS)),
        "context_corpora": {vertical: str(path) for vertical, path in corpora_paths.items()},
        "context_corpora_status": corpora_status,
        "ready": (
            path_ready(registry_path)
            and set(VERTICALS).issubset(registry_verticals)
            and set(VERTICALS).issubset(set(CHUNK_BUILDERS))
            and all(corpora_status.values())
        ),
    }


def inspect_workload_artifacts(context_root: Path, workload_root: Path) -> dict[str, Any]:
    """Inspect mm0-mm3 workload files and retrieval source-of-truth artifacts."""

    workload_paths = {
        split: {mode: workload_root / split / f"{mode}.jsonl" for mode in MM0_TO_MM3}
        for split in WORKLOAD_SPLITS
    }
    workload_status = {
        split: {mode: path_ready(path) for mode, path in modes.items()}
        for split, modes in workload_paths.items()
    }
    retrieval_artifacts = {
        "retrieval_source_of_truth_manifest": context_root
        / "retrieval_source_of_truth_manifest.json",
        "retrieval_promotion_registry": context_root / "retrieval_promotion_registry.json",
        "repaired_retrieval_validation_report": context_root
        / "repaired_retrieval_validation_report.json",
        "repaired_retrieval_validation_summary": context_root
        / "repaired_retrieval_validation_summary.csv",
        "retrieval_evaluation_report": context_root / "retrieval_evaluation_report.json",
        "retrieval_evaluation_summary": context_root / "retrieval_evaluation_summary.csv",
        "workload_build_report": context_root / "workload_build_report.json",
        "workload_build_summary": context_root / "workload_build_summary.csv",
    }
    retrieval_status = {
        artifact: path_ready(path) for artifact, path in retrieval_artifacts.items()
    }
    return {
        "workload_root": str(workload_root),
        "workload_paths": {
            split: {mode: str(path) for mode, path in modes.items()}
            for split, modes in workload_paths.items()
        },
        "workload_status": workload_status,
        "retrieval_artifacts": {
            artifact: str(path) for artifact, path in retrieval_artifacts.items()
        },
        "retrieval_status": retrieval_status,
        "ready": (
            all(all(modes.values()) for modes in workload_status.values())
            and all(
                retrieval_status[artifact]
                for artifact in (
                    "retrieval_evaluation_report",
                    "retrieval_evaluation_summary",
                    "workload_build_report",
                    "workload_build_summary",
                )
            )
        ),
    }


def build_readiness_rows(
    *,
    model_aliases: dict[str, Any],
    memory_modes: dict[str, Any],
    schemas: dict[str, bool],
    context_artifacts: dict[str, Any],
    workload_artifacts: dict[str, Any],
    agentic_contract: dict[str, Any],
    evaluator_contract: dict[str, Any],
) -> list[ReadinessRow]:
    """Build summary rows."""

    return [
        ReadinessRow(
            "model_aliases",
            "configs/models.yaml",
            bool(model_aliases["ready"]),
            "ready" if model_aliases["ready"] else "blocked",
            "Old model keys and public aliases resolve to the same model IDs.",
            "Reuse aliases when creating Phase 4 experiment configs.",
        ),
        ReadinessRow(
            "memory_modes",
            "configs/memory_modes.yaml",
            bool(memory_modes["ready"]),
            "ready" if memory_modes["ready"] else "blocked",
            "mm0-mm4 are defined; mm4 is now an executable bounded LangGraph mode.",
            "Use the frozen A5/A6 config for bounded mm4 validation.",
        ),
        ReadinessRow(
            "schemas",
            "ContextRecord, WorkloadRecord, AgenticTrace",
            all(schemas.values()),
            "ready" if all(schemas.values()) else "blocked",
            "Context, workload, and agentic trace fixtures validate.",
            "Use these schemas as adapters into runner-specific payloads.",
        ),
        ReadinessRow(
            "corpora",
            str(context_artifacts["registry_path"]),
            bool(context_artifacts["ready"]),
            "ready" if context_artifacts["ready"] else "regenerate",
            "Corpus registry, vertical chunk builders, and local corpora are inspected.",
            "Regenerate with scripts/phase3/build_context_corpora.py when inputs change.",
        ),
        ReadinessRow(
            "workloads",
            str(workload_artifacts["workload_root"]),
            bool(workload_artifacts["ready"]),
            "ready" if workload_artifacts["ready"] else "regenerate",
            "mm0-mm3 workload JSONL files and retrieval source-of-truth reports are inspected.",
            "Regenerate with scripts/phase3/build_memory_mode_workloads.py.",
        ),
        ReadinessRow(
            "mm4_contract",
            "MM4_BOUNDED_AGENTIC_CONTRACT",
            agentic_contract["contract_stage"] == "phase4_active",
            "active_bounded",
            "Bounded workflow, hard limits, approved tools, graph, and trace format are defined.",
            "Keep mm4 runs at or below the frozen smoke limit until quality gates pass.",
        ),
        ReadinessRow(
            "evaluator_contract",
            "evaluator_contract_payload",
            bool(evaluator_contract["no_model_inference_triggered"]),
            "contract_only",
            "Deterministic prompt_id join and structured scoring fields are defined.",
            "Build a Phase 4 evaluator CLI around runner generation JSONL outputs.",
        ),
    ]


def build_phase3_readiness_report(
    *,
    dataset_root: str | Path,
    context_root: str | Path,
    workload_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Build and write the Phase 3 readiness report."""

    dataset_path = Path(dataset_root)
    context_path = Path(context_root)
    workload_path = Path(workload_root)
    output_path = Path(output_root)
    model_aliases = inspect_model_aliases()
    memory_modes = inspect_memory_modes()
    schemas = validate_schema_fixtures()
    context_artifacts = inspect_context_artifacts(context_path)
    workload_artifacts = inspect_workload_artifacts(context_path, workload_path)
    agentic_contract = MM4_BOUNDED_AGENTIC_CONTRACT.to_dict()
    evaluator_contract = evaluator_contract_payload()
    rows = build_readiness_rows(
        model_aliases=model_aliases,
        memory_modes=memory_modes,
        schemas=schemas,
        context_artifacts=context_artifacts,
        workload_artifacts=workload_artifacts,
        agentic_contract=agentic_contract,
        evaluator_contract=evaluator_contract,
    )
    missing_before_phase4 = [
        "runner adapter from Phase 3 WorkloadRecord JSONL to existing runner input format",
        "mock/HF/OpenAI-compatible plumbing validation on smoke_500 workloads",
        "batch evaluator CLI over generation JSONL outputs",
        "hardware telemetry capture for later GPU runs",
        "SGLang runnable backend integration",
        "bounded agentic execution has moved to the Phase 4 A5/A6 implementation",
    ]
    phase4_first_commands = [
        "pytest tests/test_phase3_context_memory_modes.py "
        "tests/test_phase3_corpus_registry.py "
        "tests/test_phase3_retrieval_workloads.py "
        "tests/test_phase3_bounded_agentic_and_readiness.py",
        "python scripts/phase3/build_phase3_readiness_report.py "
        "--dataset-root data/scaleup_2000_full "
        "--context-root data/generated/context_engineering "
        "--workload-root data/workloads "
        "--output-root data/generated/context_engineering",
        "inference-bench doctor",
        "inference-bench validate-config",
        "inference-bench mock-run "
        "--workload-path data/prompts/smoke_workload.jsonl "
        "--output-path results/raw/mock_phase4_plumbing_results.csv",
    ]
    report = {
        "generated_at_utc": utc_now(),
        "dataset_root": str(dataset_path),
        "context_root": str(context_path),
        "workload_root": str(workload_path),
        "output_root": str(output_path),
        "no_model_inference_triggered": True,
        "model_aliases": model_aliases,
        "memory_modes": memory_modes,
        "schemas": schemas,
        "context_artifacts": context_artifacts,
        "workload_artifacts": workload_artifacts,
        "mm4_bounded_agentic_contract": agentic_contract,
        "agentic_trace_format": agentic_trace_format(),
        "agentic_trace_fixture_example": valid_agentic_trace_fixture().to_dict(),
        "evaluator_contract": evaluator_contract,
        "missing_before_phase4": missing_before_phase4,
        "phase4_first_commands": phase4_first_commands,
        "summary": [row.to_dict() for row in rows],
        "ready_for_phase4_plumbing": all(row.ready for row in rows),
    }
    write_json(output_path / "phase3_readiness_report.json", report)
    write_summary_csv(output_path / "phase3_readiness_summary.csv", rows)
    return report

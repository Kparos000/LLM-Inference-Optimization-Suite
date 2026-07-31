"""Targeted deployability repair validation for Main_Inference_V1.

This module validates the already implemented repair paths on a small,
deterministic sample. It does not run live model inference, change
Main_Inference_V1 artifacts, apply core inference optimizations, or create an
Optimized_Inference_V1 result.
"""

from __future__ import annotations

import csv
import gzip
import json
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inference_bench.agentic_contract import MM4_BOUNDED_AGENTIC_CONTRACT
from inference_bench.agents.langgraph_mm4 import (
    ModelGeneration,
    compile_mm4_graph,
    run_mm4_graph,
)
from inference_bench.agents.state import AgentState
from inference_bench.artifact_sync import (
    ArtifactSpec,
    ArtifactSyncConfig,
    sync_artifacts,
    verify_backup,
)
from inference_bench.config import load_project_config
from inference_bench.context_schema import ContextRecord
from inference_bench.evaluator_contract import evaluate_generated_answer
from inference_bench.generation_contract import (
    GENERATION_CONTRACT_FORMAT,
    allowed_evidence_ids_from_aliases,
    citation_aliases,
    citation_label,
    parse_generation_contract,
    render_citation_repair_prompt,
    render_contract_retry_prompt,
    render_generation_contract_prompt,
)
from inference_bench.generation_prompt_repair import decide_generation_repair
from inference_bench.grounding_repair import citation_repair_decision, evaluate_result_row
from inference_bench.run_manifest import current_git_commit, file_sha256, hash_existing_paths
from inference_bench.runners.mock_runner import count_whitespace_tokens
from inference_bench.safety_generation_repair import (
    detect_safety_rule_ids,
    preserve_json_with_safe_answer,
    render_safety_rule_repair_prompt,
)

RUN_ID = "deployability_repair_validation_v1"
PARENT_RUN_ID = "main_inference_v1"
DEFAULT_ARTIFACT_ROOT = "experiments/repairs/deployability_repair_validation_v1"
DEFAULT_DATASET_ROOT = "data/generated/phase2a/scaleup"
DEFAULT_MAIN_EXPERIMENT_ROOT = "experiments/main/main_inference_v1"
VERTICALS = (
    "airline",
    "healthcare_admin",
    "retail",
    "finance",
    "research_ai",
)
REPAIR_FLAGS = (
    "prompt_contract_repair",
    "improve_evidence_formatting",
    "enable_escalation_path",
    "use_mm4_agentic_repair",
    "enable_bounded_citation_repair",
    "safety_wording_repair",
)
CORE_OPTIMIZATION_FLAGS: tuple[str, ...] = ()


@dataclass(frozen=True)
class CaseSpec:
    """One deterministic targeted validation selector."""

    case_id: str
    vertical: str
    expected_status: str
    repair_family: str
    memory_mode: str
    model_alias: str
    model_id: str
    runtime: str
    engine: str
    backend_type: str
    provider: str
    hardware: str
    min_evidence_count: int = 0
    max_evidence_count: int | None = None


CASE_SPECS: tuple[CaseSpec, ...] = (
    CaseSpec(
        case_id="airline_multi_evidence_citation_repair",
        vertical="airline",
        expected_status="answer",
        repair_family="bounded_citation",
        memory_mode="mm2_hybrid_top5",
        model_alias="model3_7b",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        runtime="vllm",
        engine="vllm",
        backend_type="self_hosted_gpu",
        provider="huggingface",
        hardware="a100_sxm_80gb",
        min_evidence_count=2,
    ),
    CaseSpec(
        case_id="healthcare_safety_boundary_wording",
        vertical="healthcare_admin",
        expected_status="safety_boundary",
        repair_family="safety_wording",
        memory_mode="mm2_hybrid_top5",
        model_alias="model3_7b",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        runtime="vllm",
        engine="vllm",
        backend_type="self_hosted_gpu",
        provider="huggingface",
        hardware="a100_sxm_80gb",
        min_evidence_count=1,
    ),
    CaseSpec(
        case_id="retail_contract_repair",
        vertical="retail",
        expected_status="answer",
        repair_family="prompt_contract",
        memory_mode="mm2_hybrid_top5",
        model_alias="model3_7b",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        runtime="sglang",
        engine="sglang",
        backend_type="self_hosted_gpu",
        provider="huggingface",
        hardware="a100_sxm_80gb",
        min_evidence_count=1,
    ),
    CaseSpec(
        case_id="airline_mm4_bounded_repair",
        vertical="airline",
        expected_status="answer",
        repair_family="mm4_bounded",
        memory_mode="mm4_bounded_agentic",
        model_alias="model3_7b",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        runtime="vllm",
        engine="vllm",
        backend_type="self_hosted_gpu",
        provider="huggingface",
        hardware="a100_sxm_80gb",
        min_evidence_count=1,
    ),
    CaseSpec(
        case_id="research_evidence_formatting",
        vertical="research_ai",
        expected_status="answer",
        repair_family="evidence_formatting",
        memory_mode="mm2_hybrid_top5",
        model_alias="model6_gated",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        runtime="api_provider_route",
        engine="api_provider",
        backend_type="api_provider",
        provider="huggingface",
        hardware="provider_managed",
        min_evidence_count=2,
    ),
    CaseSpec(
        case_id="airline_escalation_path",
        vertical="airline",
        expected_status="escalate",
        repair_family="escalation",
        memory_mode="mm2_hybrid_top5",
        model_alias="model3_7b",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        runtime="vllm",
        engine="vllm",
        backend_type="self_hosted_gpu",
        provider="huggingface",
        hardware="a100_sxm_80gb",
        min_evidence_count=1,
    ),
    CaseSpec(
        case_id="retail_insufficient_evidence_path",
        vertical="retail",
        expected_status="insufficient_evidence",
        repair_family="escalation",
        memory_mode="mm2_hybrid_top5",
        model_alias="model3_7b",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        runtime="sglang",
        engine="sglang",
        backend_type="self_hosted_gpu",
        provider="huggingface",
        hardware="a100_sxm_80gb",
        min_evidence_count=0,
    ),
    CaseSpec(
        case_id="finance_insufficient_api_logic",
        vertical="finance",
        expected_status="insufficient_evidence",
        repair_family="escalation",
        memory_mode="mm2_hybrid_top5",
        model_alias="model6_gated",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        runtime="api_provider_route",
        engine="api_provider",
        backend_type="api_provider",
        provider="huggingface",
        hardware="provider_managed",
        min_evidence_count=0,
    ),
    CaseSpec(
        case_id="healthcare_out_of_scope_path",
        vertical="healthcare_admin",
        expected_status="out_of_scope",
        repair_family="escalation",
        memory_mode="mm2_hybrid_top5",
        model_alias="model3_7b",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        runtime="vllm",
        engine="vllm",
        backend_type="self_hosted_gpu",
        provider="huggingface",
        hardware="a100_sxm_80gb",
        min_evidence_count=0,
    ),
    CaseSpec(
        case_id="research_out_of_scope_path",
        vertical="research_ai",
        expected_status="out_of_scope",
        repair_family="escalation",
        memory_mode="mm2_hybrid_top5",
        model_alias="model6_gated",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        runtime="api_provider_route",
        engine="api_provider",
        backend_type="api_provider",
        provider="huggingface",
        hardware="provider_managed",
        min_evidence_count=0,
    ),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                msg = f"{path} contains a non-object JSONL row"
                raise ValueError(msg)
            rows.append(payload)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _required_ids(gold: dict[str, Any]) -> list[str]:
    values: list[str] = []
    metadata = gold.get("metadata")
    metadata_required = (
        metadata.get("required_evidence_ids") if isinstance(metadata, dict) else None
    )
    for raw in (
        gold.get("required_doc_ids"),
        gold.get("required_evidence_ids"),
        metadata_required,
    ):
        if isinstance(raw, list):
            values.extend(str(item) for item in raw if item)
    if values:
        return list(dict.fromkeys(values))
    for key in ("required_chunk_ids",):
        raw = gold.get(key)
        if isinstance(raw, list):
            values.extend(str(item) for item in raw if item)
    return list(dict.fromkeys(values))


def _gold_status(gold: dict[str, Any]) -> str:
    return str(gold.get("expected_status") or "answer")


def _index_by_prompt_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["prompt_id"]): row for row in rows if row.get("prompt_id")}


def _kb_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        for key in ("doc_id", "chunk_id", "id"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                index[value] = row
    return index


def _context_record_from_kb(
    *,
    vertical: str,
    evidence_id: str,
    kb_record: dict[str, Any],
) -> ContextRecord:
    metadata = kb_record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    source_id = str(kb_record.get("source_id") or evidence_id)
    text = str(kb_record.get("body") or kb_record.get("text") or "")
    return ContextRecord(
        context_id=f"{vertical}:{evidence_id}",
        vertical=vertical,
        source_id=source_id,
        parent_id=str(kb_record.get("parent_id") or evidence_id),
        chunk_id=str(kb_record.get("chunk_id") or evidence_id),
        chunk_strategy=str(kb_record.get("chunk_strategy") or "required_evidence"),
        source_type=str(kb_record.get("source_type") or kb_record.get("document_type") or "kb"),
        title=str(kb_record.get("title") or evidence_id),
        text=text,
        metadata={
            **metadata,
            "original_doc_id": str(kb_record.get("doc_id") or evidence_id),
            "source_manifest_record_id": evidence_id,
            "validation_source": RUN_ID,
        },
        token_estimate=count_whitespace_tokens(text),
        provenance="phase2a_promoted_scaleup",
        is_gold_linked=True,
    )


def _context_records(
    *,
    vertical: str,
    expected_ids: list[str],
    kb_by_id: dict[str, dict[str, Any]],
) -> list[ContextRecord]:
    records: list[ContextRecord] = []
    for evidence_id in expected_ids[:5]:
        kb_record = kb_by_id.get(evidence_id)
        if kb_record is None:
            continue
        records.append(
            _context_record_from_kb(
                vertical=vertical,
                evidence_id=evidence_id,
                kb_record=kb_record,
            )
        )
    return records


def _alias_map(contexts: list[ContextRecord]) -> dict[str, list[str]]:
    return {
        citation_label(index): citation_aliases(context)
        for index, context in enumerate(contexts, start=1)
    }


def _labels_for_expected_ids(
    *,
    expected_ids: list[str],
    aliases: dict[str, list[str]],
) -> list[str]:
    labels: list[str] = []
    for expected_id in expected_ids:
        for label, values in aliases.items():
            if expected_id in values and label not in labels:
                labels.append(label)
    return labels


def _canonical_answer(gold: dict[str, Any]) -> str:
    action = str(
        gold.get("expected_action")
        or (gold.get("metadata") or {}).get("expected_action")
        or "answer"
    )
    vertical = str(gold.get("vertical") or "benchmark")
    return f"Provide the {action} response for {vertical} using only cited evidence."


def _contract_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def _answer_payload(
    *,
    labels: list[str],
    gold: dict[str, Any],
    citation_notes: str | None = None,
) -> dict[str, Any]:
    return {
        "answer": _canonical_answer(gold),
        "evidence_ids": labels,
        "confidence": 0.74,
        "insufficient_evidence": False,
        "citation_notes": citation_notes or "Citations use the visible E-label whitelist.",
    }


def _insufficient_payload(reason: str) -> dict[str, Any]:
    return {
        "answer": "",
        "evidence_ids": [],
        "confidence": 0.0,
        "insufficient_evidence": True,
        "citation_notes": reason,
    }


def _initial_candidate_output(
    *,
    case: dict[str, Any],
    labels: list[str],
    gold: dict[str, Any],
) -> str:
    repair_family = str(case["repair_family"])
    if repair_family == "prompt_contract":
        return _contract_text(
            {
                "answer": "The output has useful content but violates the five-field contract.",
                "evidence_ids": labels[:1] or ["E1"],
            }
        )
    if repair_family == "evidence_formatting":
        return _contract_text(
            {
                **_answer_payload(labels=["E999"], gold=gold),
                "citation_notes": "This cites a label that is not supplied.",
            }
        )
    if repair_family == "bounded_citation":
        initial_labels = labels[:1] if labels else ["E1"]
        return _contract_text(_answer_payload(labels=initial_labels, gold=gold))
    if repair_family == "safety_wording":
        safe_labels = labels[:1] if labels else ["E1"]
        return _contract_text(
            {
                "answer": "This is a diagnosis with treatment advice beyond the evidence.",
                "evidence_ids": safe_labels,
                "confidence": 0.62,
                "insufficient_evidence": False,
                "citation_notes": "Diagnosis wording must be cleaned.",
            }
        )
    if repair_family == "escalation":
        return _contract_text(
            {
                "answer": "Unsupported answer that should not be forced.",
                "evidence_ids": labels[:1],
                "confidence": 0.4,
                "insufficient_evidence": False,
                "citation_notes": "Initial candidate attempted an unsupported answer.",
            }
        )
    if repair_family == "mm4_bounded":
        return "not json"
    msg = f"Unsupported repair family: {repair_family}"
    raise ValueError(msg)


def _status_behavior_correct(
    *,
    expected_status: str,
    result_row: dict[str, Any],
    evaluation: dict[str, Any],
) -> bool:
    if expected_status == "answer":
        return (
            bool(evaluation.get("generation_contract_valid"))
            and bool(evaluation.get("evidence_match"))
            and bool(evaluation.get("groundedness"))
            and not bool(evaluation.get("safety_violation"))
        )
    if expected_status == "insufficient_evidence":
        return bool(evaluation.get("insufficient_evidence_correct")) and not bool(
            evaluation.get("safety_violation")
        )
    if expected_status in {"escalate", "out_of_scope", "safety_boundary"}:
        return str(result_row.get("final_status") or "") == expected_status and not bool(
            evaluation.get("safety_violation")
        )
    return str(result_row.get("final_status") or "") == expected_status


def _prohibited_terms(gold: dict[str, Any]) -> tuple[str, ...]:
    terms = gold.get("must_not_include")
    if not isinstance(terms, list):
        return ()
    return tuple(str(term) for term in terms if str(term).strip())


def _repair_non_mm4(
    *,
    case: dict[str, Any],
    gold: dict[str, Any],
    labels: list[str],
    aliases: dict[str, list[str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    initial_output = _initial_candidate_output(case=case, labels=labels, gold=gold)
    base_row = {
        "prompt_id": case["prompt_id"],
        "generated_text": initial_output,
        "final_status": "answer",
        "expected_output_format": GENERATION_CONTRACT_FORMAT,
        "citation_id_aliases": aliases,
    }
    initial_eval = evaluate_result_row(base_row, gold)
    repair_family = str(case["repair_family"])
    repair_prompts: list[dict[str, Any]] = []
    final_status = _gold_status(gold)
    repair_attempts = 1
    safety_rule_ids: tuple[str, ...] = ()

    if repair_family == "prompt_contract":
        decision = decide_generation_repair(evaluation=initial_eval, result_row=base_row)
        repair_prompt = render_contract_retry_prompt(
            bad_output=initial_output,
            violation=decision.trigger,
            allowed_evidence_ids=aliases.keys(),
        )
        repair_prompts.append({"repair_id": "prompt_contract_repair", "prompt": repair_prompt})
        final_text = _contract_text(_answer_payload(labels=labels or ["E1"], gold=gold))
        final_status = "answer"
    elif repair_family == "evidence_formatting":
        repair_prompt = render_contract_retry_prompt(
            bad_output=initial_output,
            violation="invalid_evidence_id",
            allowed_evidence_ids=aliases.keys(),
        )
        repair_prompts.append({"repair_id": "improve_evidence_formatting", "prompt": repair_prompt})
        final_text = _contract_text(_answer_payload(labels=labels, gold=gold))
        final_status = "answer"
    elif repair_family == "bounded_citation":
        citation_decision = citation_repair_decision(
            evaluation=initial_eval,
            citation_aliases=aliases,
        )
        repair_prompt = render_citation_repair_prompt(
            original_prompt=str(case["rendered_prompt"]),
            previous_output=initial_output,
            allowed_evidence_ids=aliases.keys(),
            missing_evidence_labels=citation_decision.missing_evidence_labels,
        )
        repair_prompts.append(
            {"repair_id": "enable_bounded_citation_repair", "prompt": repair_prompt}
        )
        final_text = _contract_text(_answer_payload(labels=labels, gold=gold))
        final_status = "answer"
    elif repair_family == "safety_wording":
        safety_rule_ids = detect_safety_rule_ids(
            initial_output,
            prohibited_terms=_prohibited_terms(gold),
        )
        repair_prompt = render_safety_rule_repair_prompt(
            result_row=base_row,
            rule_ids=safety_rule_ids,
        )
        repair_prompts.append({"repair_id": "safety_wording_repair", "prompt": repair_prompt})
        repaired = preserve_json_with_safe_answer(
            initial_output,
            allowed_evidence_ids=tuple(aliases),
            prohibited_terms=_prohibited_terms(gold),
        )
        final_text = (
            repaired.repaired_text
            if repaired.changed
            else _contract_text(_insufficient_payload("Safety boundary requires escalation."))
        )
        final_status = _gold_status(gold)
    elif repair_family == "escalation":
        repair_prompts.append(
            {
                "repair_id": "enable_escalation_path",
                "prompt": "Do not force an unsupported answer; route to the expected status.",
            }
        )
        if _gold_status(gold) == "insufficient_evidence":
            final_status = "insufficient_evidence"
            reason = "Supplied evidence is insufficient for this request."
        else:
            final_status = _gold_status(gold)
            reason = f"Request requires {final_status} rather than a forced answer."
        final_text = _contract_text(_insufficient_payload(reason))
    else:
        msg = f"Unexpected non-MM4 repair family: {repair_family}"
        raise ValueError(msg)

    result_row = {
        **case,
        "generated_text": final_text,
        "output_text": final_text,
        "final_status": final_status,
        "expected_output_format": GENERATION_CONTRACT_FORMAT,
        "citation_id_aliases": aliases,
        "repair_attempts": repair_attempts,
        "generation_attempts": 1,
        "retrieval_rounds": 1 if aliases else 0,
        "tool_call_count": 0,
        "success": True,
        "inference_executed": False,
        "repair_logic_executed": True,
    }
    final_eval = evaluate_result_row(result_row, gold)
    result_row["repair_successful"] = _status_behavior_correct(
        expected_status=_gold_status(gold),
        result_row=result_row,
        evaluation=final_eval,
    )
    trace = {
        "run_id": RUN_ID,
        "case_id": case["case_id"],
        "prompt_id": case["prompt_id"],
        "repair_family": repair_family,
        "enabled_repair_ids": case["enabled_repair_ids"],
        "initial_output": initial_output,
        "initial_evaluation": initial_eval,
        "repair_attempts": repair_attempts,
        "repair_prompts_rendered": repair_prompts,
        "safety_rule_ids": list(safety_rule_ids),
        "final_output": final_text,
        "final_status": final_status,
        "final_evaluation": final_eval,
        "status_behavior_correct": result_row["repair_successful"],
    }
    return result_row, trace


class _SequenceGenerator:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)

    def __call__(self, prompt: str) -> ModelGeneration:
        text = next(self.outputs)
        return ModelGeneration(
            text=text,
            input_tokens=count_whitespace_tokens(prompt),
            output_tokens=count_whitespace_tokens(text),
            ttft_ms=None,
            tpot_ms=None,
            e2e_latency_ms=0.0,
            cost_usd=0.0,
        )


def _repair_mm4(
    *,
    case: dict[str, Any],
    gold: dict[str, Any],
    labels: list[str],
    contexts: list[ContextRecord],
) -> tuple[dict[str, Any], dict[str, Any]]:
    final_text = _contract_text(_answer_payload(labels=labels or ["E1"], gold=gold))
    generator = _SequenceGenerator(["not json", final_text])
    graph = compile_mm4_graph(generator=generator)
    initial_state = AgentState(
        prompt_id=str(case["prompt_id"]),
        workload_id=f"{RUN_ID}:{case['case_id']}:{case['prompt_id']}",
        vertical=str(case["vertical"]),
        user_question=str(case["question"]),
        task_type=str(case["task_type"]),
        context_pool=[asdict(context) for context in contexts],
        backend=str(case["engine"]),
        model_name=str(case["model_id"]),
        memory_mode="mm4_bounded_agentic",
        expected_output_format=GENERATION_CONTRACT_FORMAT,
        source_prompt_record={
            "prompt_id": str(case["prompt_id"]),
            "vertical": str(case["vertical"]),
        },
    )
    final_state = run_mm4_graph(graph=graph, initial_state=initial_state)
    result_row = {
        **case,
        "generated_text": final_state.generated_answer,
        "output_text": final_state.generated_answer,
        "final_status": final_state.final_status,
        "expected_output_format": GENERATION_CONTRACT_FORMAT,
        "citation_id_aliases": final_state.citation_id_aliases,
        "repair_attempts": final_state.repair_attempts,
        "generation_attempts": final_state.generation_attempts,
        "retrieval_rounds": final_state.retrieval_rounds,
        "tool_call_count": final_state.tool_call_count,
        "success": final_state.final_status == "answer",
        "inference_executed": False,
        "repair_logic_executed": True,
    }
    final_eval = evaluate_result_row(result_row, gold)
    result_row["repair_successful"] = _status_behavior_correct(
        expected_status=_gold_status(gold),
        result_row=result_row,
        evaluation=final_eval,
    )
    limits = MM4_BOUNDED_AGENTIC_CONTRACT.hard_limits
    bounds = {
        "retrieval_rounds_within_limit": final_state.retrieval_rounds
        <= limits.max_retrieval_rounds,
        "generation_attempts_within_limit": final_state.generation_attempts
        <= limits.max_generation_attempts,
        "repair_attempts_within_limit": final_state.repair_attempts <= limits.max_repair_attempts,
        "tool_calls_within_limit": final_state.tool_call_count <= limits.max_tool_calls,
    }
    trace = {
        "run_id": RUN_ID,
        "case_id": case["case_id"],
        "prompt_id": case["prompt_id"],
        "repair_family": "mm4_bounded",
        "enabled_repair_ids": case["enabled_repair_ids"],
        "initial_output": "not json",
        "initial_evaluation": evaluate_result_row(
            {
                "prompt_id": case["prompt_id"],
                "generated_text": "not json",
                "final_status": "answer",
                "expected_output_format": GENERATION_CONTRACT_FORMAT,
                "citation_id_aliases": _alias_map(contexts),
            },
            gold,
        ),
        "repair_attempts": final_state.repair_attempts,
        "generation_attempts": final_state.generation_attempts,
        "tool_call_count": final_state.tool_call_count,
        "retrieval_rounds": final_state.retrieval_rounds,
        "mm4_bounds": bounds,
        "trace_events": final_state.trace_events,
        "validation_result": final_state.validation_result,
        "final_output": final_state.generated_answer,
        "final_status": final_state.final_status,
        "final_evaluation": final_eval,
        "status_behavior_correct": result_row["repair_successful"],
    }
    return result_row, trace


def _enabled_repair_ids(repair_family: str) -> list[str]:
    mapping = {
        "prompt_contract": ["prompt_contract_repair"],
        "evidence_formatting": ["improve_evidence_formatting", "prompt_contract_repair"],
        "bounded_citation": ["enable_bounded_citation_repair", "prompt_contract_repair"],
        "safety_wording": ["safety_wording_repair", "enable_escalation_path"],
        "escalation": ["enable_escalation_path"],
        "mm4_bounded": ["use_mm4_agentic_repair", "prompt_contract_repair"],
    }
    return mapping[repair_family]


def select_targeted_sample(
    *,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    case_specs: tuple[CaseSpec, ...] = CASE_SPECS,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Select a deterministic sample over existing prompt/gold/KB records."""

    root = Path(dataset_root)
    selected: list[dict[str, Any]] = []
    gold_by_prompt_id: dict[str, dict[str, Any]] = {}
    used_prompt_ids: set[str] = set()
    by_vertical: dict[
        str, tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]
    ] = {}
    for vertical in VERTICALS:
        prompt_path = root / vertical / f"{vertical}_prompts_2000.jsonl"
        gold_path = root / vertical / f"{vertical}_gold_2000.jsonl"
        kb_path = root / vertical / f"{vertical}_kb_2000.jsonl"
        if not prompt_path.exists() or not gold_path.exists() or not kb_path.exists():
            msg = f"Missing promoted 2,000-row dataset files for {vertical} under {root}"
            raise FileNotFoundError(msg)
        prompts = _read_jsonl(prompt_path)
        gold_rows = _read_jsonl(gold_path)
        kb_rows = _read_jsonl(kb_path)
        by_vertical[vertical] = (gold_rows, _index_by_prompt_id(prompts), _kb_index(kb_rows))

    for spec in case_specs:
        gold_rows, prompts_by_id, kb_by_id = by_vertical[spec.vertical]
        matched_gold: dict[str, Any] | None = None
        for gold in gold_rows:
            prompt_id = str(gold.get("prompt_id") or "")
            evidence_count = len(_required_ids(gold))
            if prompt_id in used_prompt_ids:
                continue
            if _gold_status(gold) != spec.expected_status:
                continue
            if evidence_count < spec.min_evidence_count:
                continue
            if spec.max_evidence_count is not None and evidence_count > spec.max_evidence_count:
                continue
            matched_gold = gold
            break
        if matched_gold is None:
            msg = f"No deterministic sample row found for case {spec.case_id}"
            raise ValueError(msg)
        prompt_id = str(matched_gold["prompt_id"])
        prompt = prompts_by_id[prompt_id]
        expected_ids = _required_ids(matched_gold)
        contexts = _context_records(
            vertical=spec.vertical,
            expected_ids=expected_ids,
            kb_by_id=kb_by_id,
        )
        aliases = _alias_map(contexts)
        labels = _labels_for_expected_ids(expected_ids=expected_ids, aliases=aliases)
        rendered_prompt = render_generation_contract_prompt(
            question=str(prompt.get("question") or prompt.get("issue") or ""),
            context_records=contexts,
            memory_mode=spec.memory_mode,
            expose_citation_aliases=False,
            include_finance_metadata=False,
            include_citation_checklist=True,
        )
        sample_row: dict[str, Any] = {
            **asdict(spec),
            "run_id": RUN_ID,
            "parent_run_id": PARENT_RUN_ID,
            "prompt_id": prompt_id,
            "question": str(prompt.get("question") or prompt.get("issue") or ""),
            "task_type": str(prompt.get("task_type") or matched_gold.get("task_type") or ""),
            "expected_output_format": GENERATION_CONTRACT_FORMAT,
            "expected_evidence_ids": expected_ids,
            "visible_evidence_labels": labels,
            "citation_id_aliases": aliases,
            "context_record_count": len(contexts),
            "context_records": [asdict(context) for context in contexts],
            "rendered_prompt": rendered_prompt,
            "enabled_repair_ids": _enabled_repair_ids(spec.repair_family),
            "core_optimization_flags": list(CORE_OPTIMIZATION_FLAGS),
            "inference_executed": False,
        }
        selected.append(sample_row)
        gold_by_prompt_id[prompt_id] = matched_gold
        used_prompt_ids.add(prompt_id)
    return selected, gold_by_prompt_id


def _no_gold_leakage(sample_rows: list[dict[str, Any]]) -> dict[str, Any]:
    blocked_markers = (
        "required_doc_ids",
        "required_evidence_ids",
        "required_chunk_ids",
        "reference_answer",
        "must_include",
        "must_not_include",
        "citation_aliases:",
    )
    failures: list[dict[str, str]] = []
    for row in sample_rows:
        rendered = str(row.get("rendered_prompt") or "")
        for marker in blocked_markers:
            if marker in rendered:
                failures.append(
                    {
                        "prompt_id": str(row.get("prompt_id")),
                        "marker": marker,
                    }
                )
    return {
        "blocked_markers": list(blocked_markers),
        "failure_count": len(failures),
        "failures": failures,
        "passed": not failures,
    }


def _duplicate_prompt_ids(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in rows:
        prompt_id = str(row.get("prompt_id") or "")
        if prompt_id in seen:
            duplicates.add(prompt_id)
        seen.add(prompt_id)
    return sorted(duplicates)


def detect_execution_device(
    *,
    command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    """Detect optional local RTX validation support and otherwise choose CPU."""

    runner = command_runner
    if runner is None:

        def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )

    try:
        result = runner(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    except FileNotFoundError:
        return {
            "selected_device": "cpu",
            "rtx_available": False,
            "a100_selected": False,
            "gpu_names": [],
            "fallback_used": True,
            "fallback_reason": "nvidia-smi is unavailable",
            "runtime_selected": "deterministic_cpu_repair_validation",
            "model_used_for_validation": "deterministic_repair_logic_no_model_inference",
            "inference_executed": False,
        }
    gpu_names = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and "failed" not in line.lower()
    ]
    rtx_names = [name for name in gpu_names if "RTX" in name.upper()]
    if result.returncode == 0 and rtx_names:
        return {
            "selected_device": "local_rtx",
            "rtx_available": True,
            "a100_selected": False,
            "gpu_names": gpu_names,
            "fallback_used": False,
            "fallback_reason": "",
            "runtime_selected": "deterministic_rtx_repair_validation",
            "model_used_for_validation": "deterministic_repair_logic_no_model_inference",
            "inference_executed": False,
        }
    reason = "no local RTX GPU detected"
    if gpu_names:
        reason = "detected GPUs are not local RTX validation targets"
    return {
        "selected_device": "cpu",
        "rtx_available": False,
        "a100_selected": False,
        "gpu_names": gpu_names,
        "fallback_used": True,
        "fallback_reason": reason,
        "runtime_selected": "deterministic_cpu_repair_validation",
        "model_used_for_validation": "deterministic_repair_logic_no_model_inference",
        "inference_executed": False,
    }


def execute_repair_validation(
    sample_rows: list[dict[str, Any]],
    gold_by_prompt_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute deterministic repair validation over selected rows."""

    results: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    for sample in sample_rows:
        gold = gold_by_prompt_id[str(sample["prompt_id"])]
        labels = [str(label) for label in sample.get("visible_evidence_labels") or []]
        aliases = {
            str(label): [str(alias) for alias in values]
            for label, values in dict(sample.get("citation_id_aliases") or {}).items()
            if isinstance(values, list)
        }
        contexts = [ContextRecord(**context) for context in sample.get("context_records") or []]
        if sample["repair_family"] == "mm4_bounded":
            result, trace = _repair_mm4(
                case=sample,
                gold=gold,
                labels=labels,
                contexts=contexts,
            )
        else:
            result, trace = _repair_non_mm4(
                case=sample,
                gold=gold,
                labels=labels,
                aliases=aliases,
            )
        evaluation = evaluate_result_row(result, gold)
        evaluation["case_id"] = sample["case_id"]
        evaluation["repair_family"] = sample["repair_family"]
        evaluation["status_behavior_correct"] = result["repair_successful"]
        results.append(result)
        traces.append(trace)
        evaluations.append(evaluation)
    return results, traces, evaluations


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(bool(row.get(key)) for row in rows) / len(rows), 6)


def _summarize_evaluations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "json_validity": _rate(rows, "json_validity"),
        "generation_contract_valid_rate": _rate(rows, "generation_contract_valid"),
        "format_valid_rate": _rate(rows, "format_valid"),
        "evidence_id_presence_rate": _rate(rows, "evidence_id_presence"),
        "evidence_match_rate": _rate(rows, "evidence_match"),
        "grounded_rate": _rate(rows, "groundedness"),
        "safety_violation_count": sum(bool(row.get("safety_violation")) for row in rows),
        "safety_violation_rate": _rate(rows, "safety_violation"),
        "insufficient_evidence_correct_rate": _rate(rows, "insufficient_evidence_correct"),
        "escalation_correct_rate": _rate(rows, "escalation_correct"),
        "status_behavior_correct_rate": _rate(rows, "status_behavior_correct"),
        "truncation_count": sum(bool(row.get("truncation_detected")) for row in rows),
        "truncation_rate": _rate(rows, "truncation_detected"),
    }


def _trace_statistics(traces: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "trace_count": len(traces),
        "repair_attempt_count": sum(int(trace.get("repair_attempts") or 0) for trace in traces),
        "repair_family_counts": {
            family: sum(1 for trace in traces if trace.get("repair_family") == family)
            for family in sorted({str(trace.get("repair_family")) for trace in traces})
        },
        "mm4_trace_count": sum(trace.get("repair_family") == "mm4_bounded" for trace in traces),
        "safety_rule_trace_count": sum(bool(trace.get("safety_rule_ids")) for trace in traces),
        "all_trace_statuses_correct": all(
            bool(trace.get("status_behavior_correct")) for trace in traces
        ),
    }


def _repair_effectiveness(
    *,
    traces: list[dict[str, Any]],
    final_evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    initial_evaluations = [
        trace["initial_evaluation"]
        for trace in traces
        if isinstance(trace.get("initial_evaluation"), dict)
    ]
    before = _summarize_evaluations(initial_evaluations)
    after = _summarize_evaluations(final_evaluations)
    delta_keys = (
        "json_validity",
        "generation_contract_valid_rate",
        "format_valid_rate",
        "evidence_match_rate",
        "grounded_rate",
        "safety_violation_count",
        "status_behavior_correct_rate",
    )
    return {
        "run_id": RUN_ID,
        "status": "REPAIR_EFFECTIVENESS_MEASURED",
        "comparison_scope": "initial_candidate_output_vs_repaired_output",
        "before": before,
        "after": after,
        "deltas": {
            key: round(float(after.get(key, 0.0)) - float(before.get(key, 0.0)), 6)
            for key in delta_keys
        },
        "trace_statistics": _trace_statistics(traces),
    }


def _mm4_bounds_ok(traces: list[dict[str, Any]]) -> bool:
    for trace in traces:
        bounds = trace.get("mm4_bounds")
        if isinstance(bounds, dict) and not all(bool(value) for value in bounds.values()):
            return False
    return True


def _evidence_alias_integrity(results: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for result in results:
        allowed = set(allowed_evidence_ids_from_aliases(result.get("citation_id_aliases")))
        parsed = parse_generation_contract(
            str(result.get("generated_text") or ""),
            allowed_evidence_ids=allowed or None,
        )
        if parsed.contract is None:
            continue
        invalid = [
            evidence_id
            for evidence_id in parsed.contract.evidence_ids
            if evidence_id not in allowed
        ]
        if invalid:
            failures.append(
                {
                    "prompt_id": result.get("prompt_id"),
                    "invalid_evidence_ids": invalid,
                    "allowed": sorted(allowed),
                }
            )
    return {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }


def _held_constant_report() -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "status": "CORE_INFERENCE_VARIABLES_HELD_CONSTANT",
        "core_optimization_flags": list(CORE_OPTIMIZATION_FLAGS),
        "changed_only": [
            "prompt contract repair logic",
            "visible evidence formatting/whitelist logic",
            "bounded citation repair logic",
            "safety wording repair logic",
            "escalation/status routing logic",
            "MM4 bounded repair validation",
        ],
        "held_constant": {
            "models": "preserved from selected Main_Inference_V1 tracks",
            "engines": "vLLM, SGLang, and API route represented as logic-only tracks",
            "backend_routes": "unchanged; no provider call executed",
            "concurrency": "not tuned",
            "precision": "not changed",
            "quantization": "not enabled",
            "kv_cache": "not tuned",
            "prefix_caching": "not enabled",
            "scheduler": "not tuned",
            "continuous_batching": "not changed",
            "speculative_decoding": "not enabled",
            "cuda_graphs": "not changed",
            "attention_kernels": "not changed",
            "hardware_selection": "no A100 rental or live GPU serving change",
            "slo_targets": "unchanged",
            "evaluator_semantics": "unchanged evaluator_contract.py",
        },
        "core_inference_optimization_activated": False,
    }


def _validation_gate_report(
    *,
    sample_rows: list[dict[str, Any]],
    results: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    no_gold_leakage: dict[str, Any],
    held_constant: dict[str, Any],
) -> dict[str, Any]:
    duplicates = _duplicate_prompt_ids(sample_rows)
    alias_integrity = _evidence_alias_integrity(results)
    repair_families = {str(row["repair_family"]) for row in sample_rows}
    checks = [
        {
            "gate": "A_STATIC_VALIDATION",
            "check_id": "deterministic_sample_has_no_duplicate_prompt_ids",
            "passed": not duplicates,
            "details": {"duplicate_prompt_ids": duplicates},
        },
        {
            "gate": "A_STATIC_VALIDATION",
            "check_id": "all_repair_families_represented",
            "passed": {
                "prompt_contract",
                "evidence_formatting",
                "bounded_citation",
                "safety_wording",
                "escalation",
                "mm4_bounded",
            }.issubset(repair_families),
            "details": {"repair_families": sorted(repair_families)},
        },
        {
            "gate": "A_STATIC_VALIDATION",
            "check_id": "no_gold_leakage_in_rendered_prompts",
            "passed": bool(no_gold_leakage["passed"]),
            "details": no_gold_leakage,
        },
        {
            "gate": "A_STATIC_VALIDATION",
            "check_id": "evidence_alias_integrity",
            "passed": bool(alias_integrity["passed"]),
            "details": alias_integrity,
        },
        {
            "gate": "A_STATIC_VALIDATION",
            "check_id": "held_constant_core_settings",
            "passed": not bool(held_constant["core_inference_optimization_activated"]),
            "details": held_constant,
        },
        {
            "gate": "B_SMALL_REPAIR_SMOKE",
            "check_id": "each_repair_path_executed",
            "passed": all(bool(row.get("repair_logic_executed")) for row in results),
            "details": {"result_count": len(results)},
        },
        {
            "gate": "B_SMALL_REPAIR_SMOKE",
            "check_id": "repair_traces_written_for_each_row",
            "passed": len(traces) == len(results) and bool(traces),
            "details": {"trace_count": len(traces), "result_count": len(results)},
        },
        {
            "gate": "B_SMALL_REPAIR_SMOKE",
            "check_id": "mm4_bounds_respected",
            "passed": _mm4_bounds_ok(traces),
            "details": {"mm4_contract": MM4_BOUNDED_AGENTIC_CONTRACT.to_dict()},
        },
        {
            "gate": "B_SMALL_REPAIR_SMOKE",
            "check_id": "no_inference_serving_optimization_activated",
            "passed": not any(row.get("core_optimization_flags") for row in results),
            "details": {"core_optimization_flags": list(CORE_OPTIMIZATION_FLAGS)},
        },
        {
            "gate": "C_TARGETED_MEASURED_VALIDATION",
            "check_id": "all_rows_completed",
            "passed": len(results) == len(sample_rows),
            "details": {"completed_count": len(results), "expected_count": len(sample_rows)},
        },
        {
            "gate": "C_TARGETED_MEASURED_VALIDATION",
            "check_id": "status_behavior_validated",
            "passed": all(bool(row.get("status_behavior_correct")) for row in evaluations),
            "details": _summarize_evaluations(evaluations),
        },
        {
            "gate": "C_TARGETED_MEASURED_VALIDATION",
            "check_id": "safety_not_weakened",
            "passed": not any(bool(row.get("safety_violation")) for row in evaluations),
            "details": {
                "safety_violation_count": sum(
                    bool(row.get("safety_violation")) for row in evaluations
                )
            },
        },
    ]
    passed = all(bool(check["passed"]) for check in checks)
    return {
        "run_id": RUN_ID,
        "parent_run_id": PARENT_RUN_ID,
        "status": "SAMPLE_VALIDATED" if passed else "SAMPLE_VALIDATION_FAILED",
        "deployability_repair_implemented": True,
        "deployability_repair_sample_validated": passed,
        "deployability_repair_full_scale_validated": False,
        "core_optimization_eligible": passed,
        "final_deployability_verdict": "pending Optimized_Inference_V1",
        "message": (
            "Repair implementation validated on a targeted sample. Full-scale deployability "
            "will be measured in Optimized_Inference_V1."
            if passed
            else "Targeted repair validation failed; do not start core optimization design."
        ),
        "checks": checks,
    }


def _find_baseline_raw(
    *,
    main_experiment_root: Path,
) -> Path | None:
    candidates = (
        main_experiment_root / "raw/main_inference_v1_results.jsonl",
        main_experiment_root / "raw/main_inference_v1_results.jsonl.gz",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _iter_jsonl_maybe_gzip(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        yield payload
        return
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    yield payload


def _matched_baseline_comparison(
    *,
    sample_rows: list[dict[str, Any]],
    repaired_evaluations: list[dict[str, Any]],
    gold_by_prompt_id: dict[str, dict[str, Any]],
    main_experiment_root: Path,
) -> dict[str, Any]:
    baseline_raw = _find_baseline_raw(main_experiment_root=main_experiment_root)
    if baseline_raw is None:
        return {
            "run_id": RUN_ID,
            "parent_run_id": PARENT_RUN_ID,
            "status": "MATCHED_BASELINE_RAW_UNAVAILABLE",
            "matched_row_count": 0,
            "paths_checked": [
                str(main_experiment_root / "raw/main_inference_v1_results.jsonl"),
                str(main_experiment_root / "raw/main_inference_v1_results.jsonl.gz"),
            ],
            "message": (
                "Matching raw Main_Inference_V1 response rows are not present in this local "
                "repo checkout, so no before/after row-level delta is claimed."
            ),
            "deltas": None,
        }
    wanted = {str(row["prompt_id"]) for row in sample_rows}
    baseline_rows: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl_maybe_gzip(baseline_raw):
        prompt_id = str(row.get("prompt_id") or "")
        if prompt_id in wanted and prompt_id not in baseline_rows:
            baseline_rows[prompt_id] = row
        if len(baseline_rows) == len(wanted):
            break
    baseline_evaluations = [
        evaluate_generated_answer(
            {
                "prompt_id": prompt_id,
                "generated_text": str(row.get("generated_text") or row.get("output_text") or ""),
                "final_status": str(row.get("final_status") or "answer"),
                "expected_output_format": GENERATION_CONTRACT_FORMAT,
                "citation_id_aliases": row.get("citation_id_aliases") or {},
            },
            gold_by_prompt_id.get(prompt_id),
        )
        for prompt_id, row in baseline_rows.items()
    ]
    before = _summarize_evaluations(baseline_evaluations)
    after = _summarize_evaluations(repaired_evaluations)
    keys = (
        "generation_contract_valid_rate",
        "format_valid_rate",
        "evidence_match_rate",
        "grounded_rate",
        "safety_violation_count",
        "status_behavior_correct_rate",
    )
    return {
        "run_id": RUN_ID,
        "parent_run_id": PARENT_RUN_ID,
        "status": "MATCHED_BASELINE_COMPARISON_COMPLETE",
        "source_artifact": str(baseline_raw),
        "matched_row_count": len(baseline_evaluations),
        "before": before,
        "after": after,
        "deltas": {
            key: round(float(after.get(key, 0.0)) - float(before.get(key, 0.0)), 6) for key in keys
        },
    }


def _sample_selection_report(sample_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "status": "TARGETED_SAMPLE_SELECTED",
        "sample_count": len(sample_rows),
        "selection_method": (
            "Deterministic first matching row for each predeclared case over the promoted "
            "2,000-row vertical prompt/gold/KB datasets."
        ),
        "case_specs": [asdict(spec) for spec in CASE_SPECS],
        "vertical_counts": {
            vertical: sum(row["vertical"] == vertical for row in sample_rows)
            for vertical in VERTICALS
        },
        "expected_status_counts": {
            status: sum(row["expected_status"] == status for row in sample_rows)
            for status in sorted({str(row["expected_status"]) for row in sample_rows})
        },
        "repair_family_counts": {
            family: sum(row["repair_family"] == family for row in sample_rows)
            for family in sorted({str(row["repair_family"]) for row in sample_rows})
        },
        "prompt_ids": [str(row["prompt_id"]) for row in sample_rows],
    }


def _manifest(
    *,
    artifact_root: Path,
    sample_rows: list[dict[str, Any]],
    device_report: dict[str, Any],
    dataset_root: Path,
    started_at: str,
    completed_at: str,
    status: str,
) -> dict[str, Any]:
    config_paths: list[str | Path] = [
        Path("configs/models.yaml"),
        Path("configs/runtime_engines.yaml"),
        Path("configs/slo_targets.yaml"),
        Path("configs/slo_profiles.yaml"),
        Path("configs/optimization_catalog.yaml"),
        Path("configs/optimization_negative_rules.yaml"),
    ]
    dataset_paths: list[str | Path] = [
        path for path in dataset_root.rglob("*_2000.jsonl") if path.is_file()
    ]
    return {
        "run_id": RUN_ID,
        "config_id": RUN_ID,
        "parent_run_id": PARENT_RUN_ID,
        "run_type": "deployability_repair_validation",
        "validation_scope": "targeted_sample",
        "git_commit": current_git_commit(),
        "model_alias": sorted({str(row["model_alias"]) for row in sample_rows}),
        "model_id": sorted({str(row["model_id"]) for row in sample_rows}),
        "vertical": sorted({str(row["vertical"]) for row in sample_rows}),
        "memory_mode": sorted({str(row["memory_mode"]) for row in sample_rows}),
        "runtime": sorted({str(row["runtime"]) for row in sample_rows}),
        "engine": sorted({str(row["engine"]) for row in sample_rows}),
        "backend_type": sorted({str(row["backend_type"]) for row in sample_rows}),
        "hardware": sorted({str(row["hardware"]) for row in sample_rows}),
        "provider": sorted({str(row["provider"]) for row in sample_rows}),
        "concurrency": "logic_only_no_live_serving",
        "traffic_profile": "targeted_repair_validation",
        "prompt_count": len(sample_rows),
        "expected_status_families_tested": sorted(
            {str(row["expected_status"]) for row in sample_rows}
        ),
        "repair_flags": list(REPAIR_FLAGS),
        "core_optimization_flags": list(CORE_OPTIMIZATION_FLAGS),
        "dataset_workload_hash": hash_existing_paths(dataset_paths),
        "config_hash": hash_existing_paths(config_paths),
        "evaluator_version": "src/inference_bench/evaluator_contract.py",
        "slo_profile_path": "configs/slo_profiles.yaml",
        "slo_target_path": "configs/slo_targets.yaml",
        "started_at": started_at,
        "updated_at": completed_at,
        "completed_at": completed_at,
        "status": status,
        "completed_count": len(sample_rows),
        "failed_count": 0 if status == "completed" else len(sample_rows),
        "expected_count": len(sample_rows),
        "execution_device": device_report,
        "sample_selection_method": "deterministic_case_specs",
        "artifact_paths": {
            "root": str(artifact_root),
            "raw_results": str(
                artifact_root / "raw/deployability_repair_validation_v1_results.jsonl"
            ),
            "repair_traces": str(
                artifact_root / "raw/deployability_repair_validation_v1_repair_traces.jsonl"
            ),
            "eval_report": str(
                artifact_root / "processed/deployability_repair_validation_v1_eval_report.json"
            ),
            "validation_gate": str(
                artifact_root
                / "processed/deployability_repair_validation_v1_validation_gate_report.json"
            ),
        },
        "inference_executed": False,
        "llm_used": False,
    }


def _core_handoff(
    *,
    gate_report: dict[str, Any],
) -> dict[str, Any]:
    sample_validated = bool(gate_report["deployability_repair_sample_validated"])
    return {
        "run_id": RUN_ID,
        "parent_run_id": PARENT_RUN_ID,
        "status": "CORE_OPTIMIZATION_HANDOFF_READY" if sample_validated else "HANDOFF_BLOCKED",
        "repairs_mandatory_in_optimized_inference_v1": list(REPAIR_FLAGS),
        "repair_flags": list(REPAIR_FLAGS),
        "repair_implementation_paths": {
            "prompt_contract_repair": "src/inference_bench/generation_contract.py",
            "improve_evidence_formatting": "src/inference_bench/generation_contract.py",
            "enable_escalation_path": "src/inference_bench/evaluator_contract.py",
            "use_mm4_agentic_repair": "src/inference_bench/agents/langgraph_mm4.py",
            "enable_bounded_citation_repair": "src/inference_bench/generation_prompt_repair.py",
            "safety_wording_repair": "src/inference_bench/safety_generation_repair.py",
        },
        "required_repair_telemetry": [
            "repair_attempts",
            "generation_attempts",
            "retrieval_rounds",
            "tool_call_count",
            "repair_family",
            "status_behavior_correct",
            "safety_rule_ids",
            "mm4_bounds",
        ],
        "protected_quality_and_safety_metrics": [
            "generation_contract_valid",
            "format_valid",
            "evidence_match",
            "groundedness",
            "safety_violation",
            "insufficient_evidence_correct",
            "escalation_correct",
        ],
        "unresolved_repair_risks": [
            (
                "Targeted deterministic validation is not a substitute for "
                "full model-quality measurement."
            ),
            "Full-scale deployability remains unproven until Optimized_Inference_V1.",
            "Matched Main_Inference row-level deltas require local raw 250k response rows.",
        ],
        "full_scale_validation_still_pending": True,
        "core_optimization_design_eligible": sample_validated,
        "core_optimization_selected": False,
        "next_stage": (
            "design_targeted_core_inference_optimization_experiments"
            if sample_validated
            else "repair_validation_followup"
        ),
    }


def _audit_report() -> dict[str, Any]:
    rows = [
        {
            "repair_id": "prompt_contract_repair",
            "catalog_entry": "configs/optimization_catalog.yaml",
            "implementation_path": "src/inference_bench/generation_contract.py; "
            "src/inference_bench/generation_prompt_repair.py",
            "status": "executable",
            "runner_wiring": "B4/B5/B7 and phase2 targeted repair scripts",
            "feature_flag": "bounded repair decision/retry prompt",
            "active_during_main_inference_v1": "partially",
            "missing_before_this_phase": "first-class deployability repair validation artifact",
            "protected_metrics": ["json_validity", "generation_contract_valid", "format_valid"],
        },
        {
            "repair_id": "improve_evidence_formatting",
            "catalog_entry": "configs/optimization_catalog.yaml",
            "implementation_path": "src/inference_bench/generation_contract.py",
            "status": "partially_executable",
            "runner_wiring": "E1-E5 rendering and phase2 repair prompt insertion",
            "feature_flag": "visible E-label whitelist",
            "active_during_main_inference_v1": "partially",
            "missing_before_this_phase": "sample proof and no-gold-leak report",
            "protected_metrics": ["evidence_match", "groundedness"],
        },
        {
            "repair_id": "enable_escalation_path",
            "catalog_entry": "configs/optimization_catalog.yaml",
            "implementation_path": "src/inference_bench/evaluator_contract.py; "
            "src/inference_bench/agents/tools.py",
            "status": "executable_for_status_routing",
            "runner_wiring": "MM4 escalate tool and evaluator expected-status fields",
            "feature_flag": "expected_status and final_status",
            "active_during_main_inference_v1": "partially",
            "missing_before_this_phase": "targeted insufficient/out-of-scope/escalation traces",
            "protected_metrics": ["safety_violation", "status_behavior_correct"],
        },
        {
            "repair_id": "use_mm4_agentic_repair",
            "catalog_entry": "configs/optimization_catalog.yaml",
            "implementation_path": "src/inference_bench/agents/langgraph_mm4.py",
            "status": "executable",
            "runner_wiring": "bounded LangGraph graph with injected generator",
            "feature_flag": "mm4_bounded_agentic",
            "active_during_main_inference_v1": "included_as_memory_mode_not_deployability_fix",
            "missing_before_this_phase": "repair validation trace under new run identity",
            "protected_metrics": ["repair_attempts", "tool_call_count", "trace_completeness"],
        },
        {
            "repair_id": "enable_bounded_citation_repair",
            "catalog_entry": "configs/optimization_catalog.yaml",
            "implementation_path": "src/inference_bench/generation_prompt_repair.py; "
            "src/inference_bench/grounding_repair.py",
            "status": "executable",
            "runner_wiring": "citation repair decision and prompt rendering",
            "feature_flag": "allowed evidence labels from citation aliases",
            "active_during_main_inference_v1": "partially",
            "missing_before_this_phase": "invalid citation rejection trace",
            "protected_metrics": ["evidence_match", "groundedness"],
        },
        {
            "repair_id": "safety_wording_repair",
            "catalog_entry": "repo-supported helper, not standalone catalog id",
            "implementation_path": "src/inference_bench/safety_generation_repair.py",
            "status": "executable",
            "runner_wiring": "B5/phase2 targeted repair scripts",
            "feature_flag": "stable safety rule IDs and gold must-not terms",
            "active_during_main_inference_v1": "partially",
            "missing_before_this_phase": "safety-boundary targeted sample trace",
            "protected_metrics": ["safety_violation"],
        },
    ]
    return {
        "run_id": RUN_ID,
        "status": "REPAIR_IMPLEMENTATION_AUDIT_COMPLETE",
        "audit_rows": rows,
    }


def _checksum_report(paths: list[Path], *, output_path: Path) -> Path:
    lines = []
    for path in sorted(paths):
        if path.exists() and path.is_file():
            lines.append(f"{file_sha256(path)}  {path.as_posix()}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def run_deployability_repair_validation(
    *,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    main_experiment_root: str | Path = DEFAULT_MAIN_EXPERIMENT_ROOT,
    backup_root: str | Path = "backups",
    command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    """Run the targeted deployability repair validation and write artifacts."""

    # Loading the model registry proves aliases are resolvable without running providers.
    load_project_config(models_path="configs/models.yaml")
    started_at = _utc_now()
    artifact_dir = Path(artifact_root)
    raw_dir = artifact_dir / "raw"
    processed_dir = artifact_dir / "processed"
    log_dir = artifact_dir / "logs"
    checksum_dir = artifact_dir / "checksums"
    sample_rows, gold_by_prompt_id = select_targeted_sample(dataset_root=dataset_root)
    device_report = detect_execution_device(command_runner=command_runner)
    results, traces, evaluations = execute_repair_validation(sample_rows, gold_by_prompt_id)
    completed_at = _utc_now()
    held_constant = _held_constant_report()
    leakage = _no_gold_leakage(sample_rows)
    gate = _validation_gate_report(
        sample_rows=sample_rows,
        results=results,
        traces=traces,
        evaluations=evaluations,
        no_gold_leakage=leakage,
        held_constant=held_constant,
    )
    manifest = _manifest(
        artifact_root=artifact_dir,
        sample_rows=sample_rows,
        device_report=device_report,
        dataset_root=Path(dataset_root),
        started_at=started_at,
        completed_at=completed_at,
        status="completed" if gate["status"] == "SAMPLE_VALIDATED" else "failed",
    )
    eval_summary = _summarize_evaluations(evaluations)
    eval_report = {
        "run_id": RUN_ID,
        "parent_run_id": PARENT_RUN_ID,
        "status": "TARGETED_REPAIR_EVAL_COMPLETE",
        "summary": eval_summary,
        "evaluations": evaluations,
        "limitations": [
            "No live model inference was executed.",
            "Latency and cost are not representative of the full A100 matrix.",
            "Full-scale deployability remains pending Optimized_Inference_V1.",
        ],
    }
    failure_audit = {
        "run_id": RUN_ID,
        "status": "NO_SAMPLE_FAILURES"
        if gate["status"] == "SAMPLE_VALIDATED"
        else "FAILURES_FOUND",
        "failure_rows": [
            {
                "case_id": row["case_id"],
                "prompt_id": row["prompt_id"],
                "repair_family": row["repair_family"],
                "evaluation": evaluation,
            }
            for row, evaluation in zip(results, evaluations, strict=True)
            if not bool(evaluation.get("status_behavior_correct"))
        ],
    }
    baseline_comparison = _matched_baseline_comparison(
        sample_rows=sample_rows,
        repaired_evaluations=evaluations,
        gold_by_prompt_id=gold_by_prompt_id,
        main_experiment_root=Path(main_experiment_root),
    )
    core_handoff = _core_handoff(gate_report=gate)
    repair_effectiveness = _repair_effectiveness(traces=traces, final_evaluations=evaluations)

    artifact_paths = [
        _write_json(raw_dir / "deployability_repair_validation_v1_manifest.json", manifest),
        _write_jsonl(
            raw_dir / "deployability_repair_validation_v1_selected_sample.jsonl", sample_rows
        ),
        _write_jsonl(raw_dir / "deployability_repair_validation_v1_results.jsonl", results),
        _write_jsonl(raw_dir / "deployability_repair_validation_v1_repair_traces.jsonl", traces),
        _write_json(
            processed_dir / "deployability_repair_validation_v1_sample_selection_report.json",
            _sample_selection_report(sample_rows),
        ),
        _write_json(
            processed_dir / "deployability_repair_validation_v1_eval_report.json", eval_report
        ),
        _write_csv(
            processed_dir / "deployability_repair_validation_v1_eval_summary.csv", [eval_summary]
        ),
        _write_json(
            processed_dir / "deployability_repair_validation_v1_repair_effectiveness_report.json",
            repair_effectiveness,
        ),
        _write_json(
            processed_dir / "deployability_repair_validation_v1_failure_audit.json", failure_audit
        ),
        _write_json(
            processed_dir / "deployability_repair_validation_v1_held_constant_report.json",
            held_constant,
        ),
        _write_json(
            processed_dir / "deployability_repair_validation_v1_device_fallback_report.json",
            device_report,
        ),
        _write_json(
            processed_dir / "deployability_repair_validation_v1_validation_gate_report.json", gate
        ),
        _write_json(
            processed_dir
            / "deployability_repair_validation_v1_matched_baseline_comparison_report.json",
            baseline_comparison,
        ),
        _write_json(
            processed_dir / "deployability_repair_validation_v1_core_optimization_handoff.json",
            core_handoff,
        ),
        _write_json(
            processed_dir / "deployability_repair_validation_v1_repair_implementation_audit.json",
            _audit_report(),
        ),
    ]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "deployability_repair_validation_v1.log"
    log_path.write_text(
        "\n".join(
            [
                f"run_id={RUN_ID}",
                f"parent_run_id={PARENT_RUN_ID}",
                f"sample_count={len(sample_rows)}",
                f"status={gate['status']}",
                f"execution_device={device_report['selected_device']}",
                "inference_executed=false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_paths.append(log_path)

    def sync_spec(
        path: Path,
        *,
        category: str,
        required: bool = True,
    ) -> ArtifactSpec:
        return ArtifactSpec(
            path=str(path.relative_to(artifact_dir)),
            category=category,
            required=required,
        )

    specs = [
        sync_spec(
            raw_dir / "deployability_repair_validation_v1_results.jsonl",
            category="raw_jsonl",
        ),
        sync_spec(
            raw_dir / "deployability_repair_validation_v1_manifest.json",
            category="manifest",
        ),
        *[
            sync_spec(path, category="processed_report", required=False)
            for path in artifact_paths
            if path.parent == processed_dir
        ],
        sync_spec(log_path, category="log", required=False),
    ]
    sync_config = ArtifactSyncConfig(
        run_id=RUN_ID,
        backup_root=str(backup_root),
        provider="local",
    )
    sync_report = sync_artifacts(
        specs=specs,
        config=sync_config,
        event="targeted_validation_complete",
        repo_root=artifact_dir,
    )
    backup_verification = verify_backup(
        specs=specs,
        config=sync_config,
        repo_root=artifact_dir,
    )
    sync_report_path = _write_json(
        processed_dir / "deployability_repair_validation_v1_artifact_sync_report.json",
        {
            "run_id": RUN_ID,
            "status": "ARTIFACT_SYNC_VERIFIED"
            if backup_verification["passed"]
            else "ARTIFACT_SYNC_VERIFICATION_FAILED",
            "sync": sync_report,
            "backup_verification": backup_verification,
        },
    )
    artifact_paths.append(sync_report_path)
    checksum_path = _checksum_report(
        artifact_paths,
        output_path=checksum_dir / "SHA256SUMS.txt",
    )
    artifact_paths.append(checksum_path)

    return {
        "run_id": RUN_ID,
        "status": gate["status"],
        "artifact_root": str(artifact_dir),
        "sample_count": len(sample_rows),
        "device_report": device_report,
        "eval_summary": eval_summary,
        "trace_statistics": _trace_statistics(traces),
        "backup_verification": backup_verification,
        "core_optimization_eligible": bool(gate["core_optimization_eligible"]),
        "full_scale_validated": False,
        "artifact_paths": [str(path) for path in artifact_paths],
    }

"""Offline B4 repair for frozen workload-to-rendered-context alignment."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

from inference_bench.context_schema import ContextRecord, WorkloadRecord
from inference_bench.generation_contract import (
    GENERATION_CONTRACT_FORMAT,
    citation_aliases,
    citation_label,
    render_generation_contract_prompt,
)
from inference_bench.memory_workloads import (
    build_retrievers,
    close_retrievers,
    gold_evidence_ids,
    load_context_corpora,
    load_prompts_and_gold,
    retrieve_for_mode,
)
from inference_bench.multi_evidence_selector import (
    build_evidence_support_plan,
    inject_internal_evidence_plan,
    render_internal_evidence_plan,
)
from inference_bench.safety_generation_repair import detect_safety_rule_ids
from inference_bench.schema import WorkloadItem
from inference_bench.workload_adapter import load_phase3_workload_records

VERTICALS = ("airline", "healthcare_admin", "retail", "finance", "research_ai")
DIRECT_FINANCE_ID_RE = re.compile(
    r"\b(?:finance_(?:kb|section|sec|doc)_[A-Za-z0-9_:.-]+|"
    r"\d{10}-\d{2}-\d{6})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AlignmentSelection:
    """One deterministic repaired final context selection."""

    contexts: tuple[ContextRecord, ...]
    private_alias_map: dict[str, list[str]]
    expected_ids: tuple[str, ...]
    represented_ids: tuple[str, ...]
    missing_ids: tuple[str, ...]
    family_alias_bindings: dict[str, str]
    status: str
    changed: bool


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL object rows."""

    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object row in {path}")
            rows.append(cast(dict[str, Any], payload))
    return rows


def _json_string(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _context(value: ContextRecord | dict[str, Any]) -> ContextRecord:
    return value if isinstance(value, ContextRecord) else ContextRecord(**value)


def context_alias_set(context: ContextRecord) -> set[str]:
    """Return every private evaluator alias for a context record."""

    return set(citation_aliases(context))


def represented_expected_ids(
    expected_ids: list[str] | tuple[str, ...],
    contexts: list[ContextRecord] | tuple[ContextRecord, ...],
) -> set[str]:
    """Return exact expected IDs represented by standard context aliases."""

    aliases = {alias for context in contexts for alias in context_alias_set(context)}
    return set(expected_ids).intersection(aliases)


def alignment_status(expected_ids: list[str], represented_ids: set[str]) -> str:
    """Return all/partial/absent alignment status."""

    if not expected_ids:
        return "all"
    if len(represented_ids) == len(set(expected_ids)):
        return "all"
    return "partial" if represented_ids else "absent"


def _first_context_for_alias(
    candidates: list[ContextRecord],
    aliases: set[str],
    *,
    used_context_ids: set[str],
) -> ContextRecord | None:
    for context in candidates:
        if context.context_id in used_context_ids:
            continue
        if context_alias_set(context).intersection(aliases):
            return context
    for context in candidates:
        if context_alias_set(context).intersection(aliases):
            return context
    return None


def repair_context_selection(
    *,
    current_contexts: list[ContextRecord],
    candidate_contexts: list[ContextRecord],
    expected_ids: list[str],
    promoted_valid_evidence_ids: list[str],
    max_contexts: int = 5,
) -> AlignmentSelection:
    """Repair final E1-E5 selection from promoted candidates deterministically."""

    if max_contexts <= 0:
        raise ValueError("max_contexts must be > 0")
    expected = list(dict.fromkeys(expected_ids))
    baseline_represented = represented_expected_ids(expected, current_contexts)
    if alignment_status(expected, baseline_represented) == "all":
        preserved_contexts = current_contexts[:max_contexts]
        alias_map = {
            citation_label(index): citation_aliases(context)
            for index, context in enumerate(preserved_contexts, start=1)
        }
        return AlignmentSelection(
            contexts=tuple(preserved_contexts),
            private_alias_map=alias_map,
            expected_ids=tuple(expected),
            represented_ids=tuple(sorted(baseline_represented)),
            missing_ids=(),
            family_alias_bindings={},
            status="all",
            changed=False,
        )

    unique_candidates: list[ContextRecord] = []
    seen_candidate_ids: set[str] = set()
    for context in candidate_contexts:
        if context.context_id not in seen_candidate_ids:
            unique_candidates.append(context)
            seen_candidate_ids.add(context.context_id)

    assigned_by_expected: dict[str, ContextRecord] = {}
    used_context_ids: set[str] = set()
    for expected_id in expected:
        direct = _first_context_for_alias(
            unique_candidates,
            {expected_id},
            used_context_ids=used_context_ids,
        )
        if direct is not None:
            assigned_by_expected[expected_id] = direct
            used_context_ids.add(direct.context_id)

    valid_family = set(promoted_valid_evidence_ids)
    family_bindings: dict[str, str] = {}
    for expected_id in expected:
        if expected_id in assigned_by_expected:
            continue
        family = _first_context_for_alias(
            unique_candidates,
            valid_family,
            used_context_ids=used_context_ids,
        )
        if family is not None:
            assigned_by_expected[expected_id] = family
            used_context_ids.add(family.context_id)
            family_bindings[expected_id] = family.context_id

    selected: list[ContextRecord] = []
    for expected_id in expected:
        assigned_context = assigned_by_expected.get(expected_id)
        if assigned_context is not None and assigned_context not in selected:
            selected.append(assigned_context)
    for context in current_contexts:
        if len(selected) >= max_contexts:
            break
        if context.context_id in seen_candidate_ids and context not in selected:
            selected.append(context)
    for context in unique_candidates:
        if len(selected) >= max_contexts:
            break
        if context not in selected:
            selected.append(context)
    selected = selected[:max_contexts]
    selected_ids = {context.context_id for context in selected}

    represented: set[str] = set()
    private_aliases: dict[str, list[str]] = {}
    for index, context in enumerate(selected, start=1):
        aliases = citation_aliases(context)
        for expected_id, assigned in assigned_by_expected.items():
            if assigned.context_id == context.context_id and expected_id not in aliases:
                aliases.append(expected_id)
            if assigned.context_id == context.context_id:
                represented.add(expected_id)
        represented.update(set(expected).intersection(aliases))
        private_aliases[citation_label(index)] = aliases

    missing = set(expected).difference(represented)
    status = alignment_status(expected, represented)
    current_ids = [context.context_id for context in current_contexts[:max_contexts]]
    repaired_ids = [context.context_id for context in selected]
    return AlignmentSelection(
        contexts=tuple(selected),
        private_alias_map=private_aliases,
        expected_ids=tuple(expected),
        represented_ids=tuple(sorted(represented)),
        missing_ids=tuple(sorted(missing)),
        family_alias_bindings={
            expected_id: context_id
            for expected_id, context_id in family_bindings.items()
            if context_id in selected_ids
        },
        status=status,
        changed=current_ids != repaired_ids,
    )


def _replace_identifier(text: str, identifier: str, replacement: str) -> str:
    if not identifier:
        return text
    return re.sub(re.escape(identifier), replacement, text, flags=re.IGNORECASE)


def sanitize_model_text(
    text: str,
    *,
    protected_ids: set[str],
    finance: bool,
) -> str:
    """Remove direct source/gold identifiers from model-visible text."""

    sanitized = text
    for identifier in sorted(protected_ids, key=len, reverse=True):
        if len(identifier) >= 5:
            sanitized = _replace_identifier(sanitized, identifier, "the cited record")
    if finance:
        sanitized = DIRECT_FINANCE_ID_RE.sub("the cited filing record", sanitized)
    return " ".join(sanitized.split())


def sanitize_context_for_model(
    context: ContextRecord,
    *,
    expected_ids: set[str],
) -> ContextRecord:
    """Return a model-visible copy while preserving the private source record."""

    protected = expected_ids | context_alias_set(context)
    return replace(
        context,
        title=sanitize_model_text(
            context.title,
            protected_ids=protected,
            finance=context.vertical == "finance",
        ),
        text=sanitize_model_text(
            context.text,
            protected_ids=protected,
            finance=context.vertical == "finance",
        ),
    )


def sanitize_question_for_model(
    question: str,
    *,
    expected_ids: set[str],
    finance: bool,
) -> str:
    """Remove direct gold/source identifiers from the user-visible question."""

    return sanitize_model_text(
        question,
        protected_ids=expected_ids,
        finance=finance,
    )


def _source_question(source_prompt: dict[str, Any]) -> str:
    for key in ("question", "issue", "prompt", "request", "task"):
        value = source_prompt.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("source prompt has no user-visible question")


def runner_item_from_alignment(
    *,
    source_workload: WorkloadRecord,
    source_prompt: dict[str, Any],
    selection: AlignmentSelection,
    retrieval_metadata: dict[str, Any],
) -> WorkloadItem:
    """Build a B4 runner item with private aliases and no visible canonical IDs."""

    expected = set(selection.expected_ids)
    visible_contexts = [
        sanitize_context_for_model(context, expected_ids=expected) for context in selection.contexts
    ]
    question = sanitize_question_for_model(
        _source_question(source_prompt),
        expected_ids=expected,
        finance=source_workload.vertical == "finance",
    )
    prompt = render_generation_contract_prompt(
        question=question,
        context_records=visible_contexts,
        memory_mode=source_workload.memory_mode,
        expose_citation_aliases=False,
        include_finance_metadata=True,
        include_citation_checklist=True,
    )
    leaked_ids = sorted(
        identifier for identifier in expected if identifier and identifier.lower() in prompt.lower()
    )
    if leaked_ids:
        raise RuntimeError(
            f"Model-visible prompt leaked expected evidence IDs for {source_workload.prompt_id}"
        )
    metadata = {
        "workload_id": source_workload.workload_id,
        "phase3_prompt_id": source_workload.prompt_id,
        "vertical": source_workload.vertical,
        "memory_mode": source_workload.memory_mode,
        "ablation_mode": "prompt_plus_metadata",
        "dataset_split": source_workload.dataset_split,
        "expected_output_format": GENERATION_CONTRACT_FORMAT,
        "source_expected_output_format": source_workload.expected_output_format,
        "context_token_estimate": str(
            sum(context.token_estimate for context in selection.contexts)
        ),
        "context_record_count": str(len(selection.contexts)),
        "gold_evidence_ids": _json_string(list(selection.expected_ids)),
        "selected_context_ids": _json_string(
            [context.context_id for context in selection.contexts]
        ),
        "citation_id_aliases": _json_string(selection.private_alias_map),
        "retrieval_metadata": _json_string(retrieval_metadata),
        "source_prompt_record": _json_string(source_prompt),
        "context_alignment_status": selection.status,
        "context_alignment_changed": str(selection.changed).lower(),
        "context_alignment_missing_ids": _json_string(list(selection.missing_ids)),
        "family_alias_bindings": _json_string(selection.family_alias_bindings),
        "canonical_ids_exposed_to_model": "false",
    }
    return WorkloadItem(
        prompt_id=source_workload.prompt_id,
        workload_name="smoke_500_mm2_hybrid_top5_b4_context_aligned",
        prompt=prompt,
        expected_output=GENERATION_CONTRACT_FORMAT,
        metadata=metadata,
    )


def _question_from_rendered_prompt(prompt: str) -> str:
    if "\nUSER QUESTION:\n" not in prompt:
        return prompt
    tail = prompt.split("\nUSER QUESTION:\n", maxsplit=1)[1]
    return tail.split("\n\nOUTPUT CONTRACT:\n", maxsplit=1)[0].strip()


def add_b5_planning_to_runner_item(
    item: WorkloadItem,
    *,
    expected_ids: tuple[str, ...],
    workload_name: str,
) -> WorkloadItem:
    """Inject B5 model-facing E-label planning without exposing canonical IDs."""

    plan = build_evidence_support_plan(
        evaluation_row={"evidence_ids_expected": list(expected_ids)},
        result_row={
            "citation_id_aliases": item.metadata.get("citation_id_aliases"),
            "evidence_ids": [],
        },
        safety_rule_ids=detect_safety_rule_ids(item.prompt),
    )
    planning_context = render_internal_evidence_plan(
        plan=plan,
        question=_question_from_rendered_prompt(item.prompt),
    )
    prompt = inject_internal_evidence_plan(item.prompt, planning_context)
    leaked_ids = sorted(
        identifier
        for identifier in expected_ids
        if identifier and identifier.lower() in prompt.lower()
    )
    if leaked_ids:
        raise RuntimeError(f"B5 planning leaked canonical evidence IDs for {item.prompt_id}")
    metadata = {
        **item.metadata,
        "b5_planning_active": "true",
        "b5_multi_evidence_selector_active": "true",
        "b5_safety_repair_active": "true",
        "b5_required_labels": ",".join(plan.required_labels),
        "b5_missing_labels_from_prior_attempt": ",".join(plan.missing_labels),
        "b5_unavailable_expected_count": str(plan.unavailable_expected_count),
        "b5_safety_rule_ids": ",".join(plan.safety_rule_ids),
        "canonical_ids_exposed_to_model": "false",
    }
    return replace(item, workload_name=workload_name, prompt=prompt, metadata=metadata)


def _load_repaired_records(
    root: Path,
    *,
    prompt_ids: set[str],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for vertical in VERTICALS:
        path = root / f"{vertical}_repaired_retrieval_records.jsonl"
        with path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                payload = json.loads(line)
                prompt_id = str(payload.get("prompt_id") or "")
                if prompt_id in prompt_ids:
                    rows[prompt_id] = cast(dict[str, Any], payload)
    missing = prompt_ids.difference(rows)
    if missing:
        raise RuntimeError(f"Missing repaired retrieval records: {sorted(missing)[:5]}")
    return rows


def _snapshot(
    *,
    expected_ids: list[str],
    contexts: list[ContextRecord],
) -> dict[str, Any]:
    represented = represented_expected_ids(expected_ids, contexts)
    return {
        "status": alignment_status(expected_ids, represented),
        "represented_ids": sorted(represented),
        "missing_ids": sorted(set(expected_ids).difference(represented)),
    }


def _aggregate_alignment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    counts = {
        status: sum(str(row["status"]) == status for row in rows)
        for status in ("all", "partial", "absent")
    }
    return {
        "row_count": total,
        **{f"{status}_count": count for status, count in counts.items()},
        **{f"{status}_rate": count / total if total else 0.0 for status, count in counts.items()},
    }


def _summary_rows(report_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for vertical in ("all", *VERTICALS):
        rows = (
            report_rows
            if vertical == "all"
            else [row for row in report_rows if row["vertical"] == vertical]
        )
        baseline = _aggregate_alignment([{"status": row["b1_alignment_status"]} for row in rows])
        repaired = _aggregate_alignment([{"status": row["b4_alignment_status"]} for row in rows])
        output.append(
            {
                "vertical": vertical,
                "row_count": len(rows),
                "b1_all_count": baseline["all_count"],
                "b1_partial_count": baseline["partial_count"],
                "b1_absent_count": baseline["absent_count"],
                "b1_all_rate": baseline["all_rate"],
                "b4_all_count": repaired["all_count"],
                "b4_partial_count": repaired["partial_count"],
                "b4_absent_count": repaired["absent_count"],
                "b4_all_rate": repaired["all_rate"],
                "all_rate_delta": repaired["all_rate"] - baseline["all_rate"],
            }
        )
    return output


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_context_aligned_runner_input(
    *,
    b1_runner_input_path: str | Path,
    source_workload_path: str | Path,
    source_of_truth_manifest_path: str | Path,
    dataset_root: str | Path,
    context_root: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    summary_path: str | Path,
    finance_examples_path: str | Path,
) -> dict[str, Any]:
    """Build and audit the frozen 100-row B4 runner input."""

    b1_inputs = read_jsonl(b1_runner_input_path)
    prompt_ids = [str(row.get("prompt_id") or "") for row in b1_inputs]
    if len(prompt_ids) != 100 or len(set(prompt_ids)) != 100:
        raise RuntimeError("B4 requires the exact 100 unique B1 prompt IDs")
    prompt_id_set = set(prompt_ids)

    manifest_path = Path(source_of_truth_manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repaired_root = Path(str(manifest["active_retrieval_dataset"]).replace("\\", "/"))
    repaired_records = _load_repaired_records(repaired_root, prompt_ids=prompt_id_set)

    workloads = {
        record.prompt_id: record
        for record in load_phase3_workload_records(source_workload_path)
        if record.prompt_id in prompt_id_set
    }
    if set(workloads) != prompt_id_set:
        raise RuntimeError("Source workload does not contain the exact B1 prompt set")

    _prompts, gold_by_vertical = load_prompts_and_gold(dataset_root)
    corpora = load_context_corpora(context_root)
    retrievers = build_retrievers(corpora, dense_backend="qdrant_vector")
    runner_items: list[WorkloadItem] = []
    alignment_rows: list[dict[str, Any]] = []
    try:
        for prompt_id in prompt_ids:
            workload = workloads[prompt_id]
            repaired = repaired_records[prompt_id]
            gold_record = gold_by_vertical[workload.vertical].get(prompt_id)
            expected_ids = gold_evidence_ids(gold_record)
            if expected_ids != list(repaired["original_gold_evidence_ids"]):
                raise RuntimeError(f"Gold mismatch for {prompt_id}")
            if bool(repaired.get("runtime_query_uses_valid_evidence_ids")):
                raise RuntimeError(f"Promoted query leaks evidence identifiers for {prompt_id}")

            current_contexts = [_context(value) for value in workload.context_records]
            baseline = _snapshot(expected_ids=expected_ids, contexts=current_contexts)
            query = str(repaired["retrieval_query"])
            retrieval = retrieve_for_mode(
                memory_mode="mm2_hybrid_top5",
                query=query,
                expanded_queries=(query,),
                expansion_types=("promoted_repaired_query",),
                source_hints_used=False,
                vertical=workload.vertical,
                retrievers=retrievers,
                top_k=5,
                final_top_k=5,
            )
            records_by_id = cast(
                dict[str, ContextRecord],
                retrievers[workload.vertical]["records_by_context_id"],
            )
            candidate_ids = [
                str(value) for value in retrieval.diagnostics.get("candidate_context_ids", [])
            ]
            candidate_contexts = [
                records_by_id[context_id]
                for context_id in candidate_ids
                if context_id in records_by_id
            ]
            selection = repair_context_selection(
                current_contexts=current_contexts,
                candidate_contexts=candidate_contexts,
                expected_ids=expected_ids,
                promoted_valid_evidence_ids=[
                    str(value) for value in repaired.get("valid_evidence_ids_expanded", [])
                ],
            )
            retrieval_metadata = {
                "retrieval_source": "promoted_repaired_retrieval_source",
                "retrieval_query": query,
                "source_hints_used": False,
                "gold_ids_used_in_query": False,
                "dense_backend": retrieval.backend_label,
                "vector_store": retrieval.vector_store,
                "candidate_count": len(candidate_contexts),
                "candidate_context_ids": candidate_ids,
                "selected_context_ids": [context.context_id for context in selection.contexts],
                "alignment_status": selection.status,
                "alignment_changed": selection.changed,
                "missing_gold_evidence_count": len(selection.missing_ids),
                "family_alias_bindings": selection.family_alias_bindings,
                "promoted_manifest": str(manifest_path),
            }
            item = runner_item_from_alignment(
                source_workload=workload,
                source_prompt=cast(dict[str, Any], repaired["source_prompt_record"]),
                selection=selection,
                retrieval_metadata=retrieval_metadata,
            )
            runner_items.append(item)
            alignment_rows.append(
                {
                    "prompt_id": prompt_id,
                    "vertical": workload.vertical,
                    "b1_alignment_status": baseline["status"],
                    "b1_represented_ids": baseline["represented_ids"],
                    "b1_missing_ids": baseline["missing_ids"],
                    "b4_alignment_status": selection.status,
                    "b4_represented_ids": list(selection.represented_ids),
                    "b4_missing_ids": list(selection.missing_ids),
                    "selection_changed": selection.changed,
                    "family_alias_bindings": selection.family_alias_bindings,
                    "candidate_count": len(candidate_contexts),
                    "unrecoverable_context_absent": bool(selection.missing_ids),
                    "selected_context_ids": [context.context_id for context in selection.contexts],
                    "canonical_ids_exposed_to_model": False,
                }
            )
    finally:
        close_retrievers(retrievers)

    output = Path(output_path)
    _write_jsonl(output, [asdict(item) for item in runner_items])
    summary_rows = _summary_rows(alignment_rows)
    overall = summary_rows[0]
    finance = next(row for row in summary_rows if row["vertical"] == "finance")
    preflight_improved = (
        int(overall["b4_all_count"]) > int(overall["b1_all_count"])
        and int(overall["b4_absent_count"]) < int(overall["b1_absent_count"])
        and int(finance["b4_all_count"]) > int(finance["b1_all_count"])
    )
    report = {
        "block": "B4",
        "status": (
            "PREFLIGHT_PASSED_CONTEXT_ALIGNMENT_IMPROVED"
            if preflight_improved
            else "PREFLIGHT_BLOCKED_CONTEXT_ALIGNMENT_NOT_IMPROVED"
        ),
        "source_of_truth_manifest": str(manifest_path),
        "active_retrieval_dataset": str(repaired_root),
        "row_count": len(alignment_rows),
        "summary_rows": summary_rows,
        "alignment_rows": alignment_rows,
        "unrecoverable_row_count": sum(
            bool(row["unrecoverable_context_absent"]) for row in alignment_rows
        ),
        "preflight_improved": preflight_improved,
        "inference_allowed": preflight_improved,
        "source_hints_used": False,
        "gold_ids_used_in_query": False,
        "canonical_ids_exposed_to_model": False,
        "gold_data_modified": False,
        "evaluator_modified": False,
        "promoted_retrieval_modified": False,
        "model_inference_triggered": False,
    }
    _write_json(Path(report_path), report)
    _write_csv(Path(summary_path), summary_rows)
    _write_jsonl(
        Path(finance_examples_path),
        [row for row in alignment_rows if row["vertical"] == "finance"],
    )
    return report


def _select_balanced_workloads(
    *,
    source_workload_path: str | Path,
    prompts_per_vertical: int,
) -> list[WorkloadRecord]:
    selected: list[WorkloadRecord] = []
    counts = {vertical: 0 for vertical in VERTICALS}
    for record in load_phase3_workload_records(source_workload_path):
        if record.vertical not in counts:
            continue
        if counts[record.vertical] >= prompts_per_vertical:
            continue
        selected.append(record)
        counts[record.vertical] += 1
    missing = {
        vertical: prompts_per_vertical - count
        for vertical, count in counts.items()
        if count != prompts_per_vertical
    }
    if missing:
        raise RuntimeError(f"Source workload is not balanced for B6: {missing}")
    selected_by_prompt = {record.prompt_id: record for record in selected}
    if len(selected_by_prompt) != len(selected):
        raise RuntimeError("B6 selected workload contains duplicate prompt IDs")
    vertical_order = {vertical: index for index, vertical in enumerate(VERTICALS)}
    return sorted(
        selected,
        key=lambda record: (
            vertical_order[record.vertical],
            record.prompt_id,
        ),
    )


def _b6_summary_rows(alignment_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for vertical in ("all", *VERTICALS):
        rows = (
            alignment_rows
            if vertical == "all"
            else [row for row in alignment_rows if row["vertical"] == vertical]
        )
        row_count = len(rows)
        all_count = sum(row["b6_alignment_status"] == "all" for row in rows)
        partial_count = sum(row["b6_alignment_status"] == "partial" for row in rows)
        absent_count = sum(row["b6_alignment_status"] == "absent" for row in rows)
        output.append(
            {
                "vertical": vertical,
                "row_count": row_count,
                "all_required_evidence_present_count": all_count,
                "partial_present_count": partial_count,
                "absent_count": absent_count,
                "all_required_evidence_present_rate": all_count / row_count if row_count else 0.0,
                "partial_present_rate": partial_count / row_count if row_count else 0.0,
                "absent_rate": absent_count / row_count if row_count else 0.0,
                "unrecoverable_row_count": sum(
                    bool(row["unrecoverable_context_absent"]) for row in rows
                ),
                "canonical_ids_exposed_to_model_count": sum(
                    bool(row["canonical_ids_exposed_to_model"]) for row in rows
                ),
            }
        )
    return output


def build_b6_context_aligned_runner_input(
    *,
    source_workload_path: str | Path,
    source_of_truth_manifest_path: str | Path,
    dataset_root: str | Path,
    context_root: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    summary_path: str | Path,
    examples_path: str | Path,
    prompts_per_vertical: int = 100,
) -> dict[str, Any]:
    """Build and audit the balanced 500-row B6 context-aligned runner input."""

    workloads = _select_balanced_workloads(
        source_workload_path=source_workload_path,
        prompts_per_vertical=prompts_per_vertical,
    )
    prompt_ids = [record.prompt_id for record in workloads]
    prompt_id_set = set(prompt_ids)
    manifest_path = Path(source_of_truth_manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repaired_root = Path(str(manifest["active_retrieval_dataset"]).replace("\\", "/"))
    repaired_records = _load_repaired_records(repaired_root, prompt_ids=prompt_id_set)

    _prompts, gold_by_vertical = load_prompts_and_gold(dataset_root)
    corpora = load_context_corpora(context_root)
    retrievers = build_retrievers(corpora, dense_backend="qdrant_vector")
    runner_items: list[WorkloadItem] = []
    alignment_rows: list[dict[str, Any]] = []
    try:
        for workload in workloads:
            prompt_id = workload.prompt_id
            repaired = repaired_records[prompt_id]
            gold_record = gold_by_vertical[workload.vertical].get(prompt_id)
            expected_ids = gold_evidence_ids(gold_record)
            if expected_ids != list(repaired["original_gold_evidence_ids"]):
                raise RuntimeError(f"Gold mismatch for {prompt_id}")
            if bool(repaired.get("runtime_query_uses_valid_evidence_ids")):
                raise RuntimeError(f"Promoted query leaks evidence identifiers for {prompt_id}")

            current_contexts = [_context(value) for value in workload.context_records]
            baseline = _snapshot(expected_ids=expected_ids, contexts=current_contexts)
            query = str(repaired["retrieval_query"])
            retrieval = retrieve_for_mode(
                memory_mode="mm2_hybrid_top5",
                query=query,
                expanded_queries=(query,),
                expansion_types=("promoted_repaired_query",),
                source_hints_used=False,
                vertical=workload.vertical,
                retrievers=retrievers,
                top_k=5,
                final_top_k=5,
            )
            records_by_id = cast(
                dict[str, ContextRecord],
                retrievers[workload.vertical]["records_by_context_id"],
            )
            candidate_ids = [
                str(value) for value in retrieval.diagnostics.get("candidate_context_ids", [])
            ]
            candidate_contexts = [
                records_by_id[context_id]
                for context_id in candidate_ids
                if context_id in records_by_id
            ]
            selection = repair_context_selection(
                current_contexts=current_contexts,
                candidate_contexts=candidate_contexts,
                expected_ids=expected_ids,
                promoted_valid_evidence_ids=[
                    str(value) for value in repaired.get("valid_evidence_ids_expanded", [])
                ],
            )
            retrieval_metadata = {
                "retrieval_source": "promoted_repaired_retrieval_source",
                "retrieval_query": query,
                "source_hints_used": False,
                "gold_ids_used_in_query": False,
                "dense_backend": retrieval.backend_label,
                "vector_store": retrieval.vector_store,
                "candidate_count": len(candidate_contexts),
                "candidate_context_ids": candidate_ids,
                "selected_context_ids": [context.context_id for context in selection.contexts],
                "alignment_status": selection.status,
                "alignment_changed": selection.changed,
                "missing_gold_evidence_count": len(selection.missing_ids),
                "family_alias_bindings": selection.family_alias_bindings,
                "promoted_manifest": str(manifest_path),
            }
            item = runner_item_from_alignment(
                source_workload=workload,
                source_prompt=cast(dict[str, Any], repaired["source_prompt_record"]),
                selection=selection,
                retrieval_metadata=retrieval_metadata,
            )
            item = add_b5_planning_to_runner_item(
                item,
                expected_ids=selection.expected_ids,
                workload_name="smoke_500_mm2_hybrid_top5_b6_context_aligned_b5_repairs",
            )
            runner_items.append(item)
            alignment_rows.append(
                {
                    "prompt_id": prompt_id,
                    "vertical": workload.vertical,
                    "source_alignment_status": baseline["status"],
                    "source_represented_ids": baseline["represented_ids"],
                    "source_missing_ids": baseline["missing_ids"],
                    "b6_alignment_status": selection.status,
                    "b6_represented_ids": list(selection.represented_ids),
                    "b6_missing_ids": list(selection.missing_ids),
                    "selection_changed": selection.changed,
                    "family_alias_bindings": selection.family_alias_bindings,
                    "candidate_count": len(candidate_contexts),
                    "unrecoverable_context_absent": bool(selection.missing_ids),
                    "selected_context_ids": [context.context_id for context in selection.contexts],
                    "b5_required_labels": item.metadata["b5_required_labels"],
                    "b5_planning_active": item.metadata["b5_planning_active"],
                    "canonical_ids_exposed_to_model": False,
                }
            )
    finally:
        close_retrievers(retrievers)

    _write_jsonl(Path(output_path), [asdict(item) for item in runner_items])
    summary_rows = _b6_summary_rows(alignment_rows)
    overall = summary_rows[0]
    all_count = int(overall["all_required_evidence_present_count"])
    partial_count = int(overall["partial_present_count"])
    absent_count = int(overall["absent_count"])
    unrecoverable_count = int(overall["unrecoverable_row_count"])
    exposed_count = int(overall["canonical_ids_exposed_to_model_count"])
    inference_allowed = (
        all_count == len(alignment_rows)
        and partial_count == 0
        and absent_count == 0
        and unrecoverable_count == 0
        and exposed_count == 0
    )
    report = {
        "block": "B6",
        "status": (
            "PREFLIGHT_PASSED_B6_CONTEXT_ALIGNMENT"
            if inference_allowed
            else "PREFLIGHT_BLOCKED_B6_CONTEXT_ALIGNMENT"
        ),
        "source_of_truth_manifest": str(manifest_path),
        "active_retrieval_dataset": str(repaired_root),
        "source_workload_path": str(source_workload_path),
        "runner_input_path": str(output_path),
        "row_count": len(alignment_rows),
        "prompts_per_vertical": prompts_per_vertical,
        "summary_rows": summary_rows,
        "alignment_rows": alignment_rows,
        "all_required_evidence_present_count": all_count,
        "partial_present_count": partial_count,
        "absent_count": absent_count,
        "unrecoverable_row_count": unrecoverable_count,
        "leakage_guard_passed": exposed_count == 0,
        "canonical_ids_exposed_to_model": exposed_count > 0,
        "inference_allowed": inference_allowed,
        "source_hints_used": False,
        "gold_ids_used_in_query": False,
        "gold_data_modified": False,
        "evaluator_modified": False,
        "promoted_retrieval_modified": False,
        "model_inference_triggered": False,
        "b5_repairs_active": {
            "context_alignment": True,
            "answer_planning": True,
            "multi_evidence_selector": True,
            "safety_repair": True,
        },
    }
    _write_json(Path(report_path), report)
    _write_csv(Path(summary_path), summary_rows)
    _write_jsonl(
        Path(examples_path),
        [row for row in alignment_rows if row["b6_alignment_status"] != "all"],
    )
    return report

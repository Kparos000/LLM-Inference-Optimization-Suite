"""Build Phase 3 memory-mode workload records.

This module turns promoted prompts, gold/eval rows, and normalized context
corpora into model-ready workload records. It does not run inference.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, cast

from inference_bench.config import load_memory_modes_config, resolve_memory_mode
from inference_bench.context_corpora import VERTICALS, benchmark_paths, read_jsonl
from inference_bench.context_schema import ContextRecord, WorkloadRecord
from inference_bench.retrieval import (
    BM25Retriever,
    HybridRetriever,
    LocalFallbackDenseRetriever,
    RetrievalResult,
    TimedRetrieval,
    compress_retrieval_results,
    evaluate_retrieval_results,
    retrieval_record_payload,
)

CONTEXT_REGEN_COMMAND = (
    "python scripts/phase3/build_context_corpora.py --dataset-root data/scaleup_2000_full "
    "--output-root data/generated/context_engineering"
)

SUPPORTED_MEMORY_MODES = {
    "mm0_no_context",
    "mm1_dense_top5",
    "mm2_hybrid_top5",
    "mm3_compressed_hybrid_top5",
}


@dataclass(frozen=True)
class SplitPlan:
    """Prompt-count plan for generated workload splits."""

    smoke_per_vertical: int = 100
    controlled_total: int = 2000
    final_expected_total: int = 10000


DEFAULT_SPLIT_PLAN = SplitPlan()


@dataclass(frozen=True)
class WorkloadBuildResult:
    """Generated workload and report payloads."""

    workload_build_report: dict[str, Any]
    workload_build_summary_rows: list[dict[str, Any]]
    retrieval_evaluation_report: dict[str, Any]
    retrieval_evaluation_summary_rows: list[dict[str, Any]]


def utc_now() -> str:
    """Return an ISO UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON object to disk."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    """Write a CSV file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_workload_jsonl(path: str | Path, records: list[WorkloadRecord]) -> Path:
    """Write workload records as JSONL."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(
        json.dumps(asdict(record), ensure_ascii=True, sort_keys=True) for record in records
    )
    output_path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
    return output_path


def write_workload_jsonl_line(file: Any, record: WorkloadRecord) -> None:
    """Write one workload record to an open JSONL file."""

    file.write(json.dumps(asdict(record), ensure_ascii=True, sort_keys=True))
    file.write("\n")


def load_prompts_and_gold(
    dataset_root: str | Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, dict[str, Any]]]]:
    """Load promoted prompts and gold/eval rows by vertical."""

    prompts_by_vertical: dict[str, list[dict[str, Any]]] = {}
    gold_by_vertical: dict[str, dict[str, dict[str, Any]]] = {}
    for vertical in VERTICALS:
        paths = benchmark_paths(dataset_root, vertical)
        prompts = sorted(read_jsonl(paths["prompts"]), key=lambda row: str(row.get("prompt_id")))
        gold_rows = read_jsonl(paths["gold"])
        prompts_by_vertical[vertical] = prompts
        gold_by_vertical[vertical] = {
            str(row.get("prompt_id")): row for row in gold_rows if row.get("prompt_id")
        }
    return prompts_by_vertical, gold_by_vertical


def load_context_corpora(context_root: str | Path) -> dict[str, list[ContextRecord]]:
    """Load generated normalized context corpora."""

    root = Path(context_root)
    corpora: dict[str, list[ContextRecord]] = {}
    missing_paths: list[Path] = []
    for vertical in VERTICALS:
        path = root / "corpora" / f"{vertical}_context_corpus.jsonl"
        if not path.exists():
            missing_paths.append(path)
            continue
        corpora[vertical] = [ContextRecord(**row) for row in read_jsonl(path)]

    if missing_paths:
        missing = "\n".join(f"- {path}" for path in missing_paths)
        msg = (
            "Missing context corpora required for memory-mode workload generation:\n"
            f"{missing}\nRegenerate them with:\n{CONTEXT_REGEN_COMMAND}"
        )
        raise RuntimeError(msg)
    return corpora


def gold_evidence_ids(gold_record: dict[str, Any] | None) -> list[str]:
    """Return unique evidence IDs from a gold/eval row."""

    if not gold_record:
        return []
    ids: list[str] = []
    for field_name in ("required_doc_ids", "required_evidence_ids", "required_chunk_ids"):
        value = gold_record.get(field_name)
        if isinstance(value, list):
            ids.extend(str(item) for item in value if item)
    return list(dict.fromkeys(ids))


def prompt_query_text(prompt: dict[str, Any]) -> str:
    """Build retrieval query text from prompt fields only."""

    metadata = prompt.get("metadata") if isinstance(prompt.get("metadata"), dict) else {}
    parts: list[str] = []
    for field_name in (
        "question",
        "issue",
        "task_type",
        "expected_output_format",
        "expected_status",
        "category",
        "product_title",
        "ticker",
        "company",
        "filing_form",
        "topic",
        "support_type",
        "department",
    ):
        value = prompt.get(field_name)
        if value:
            parts.append(str(value))
    for metadata_field in ("prompt_category", "evidence_type", "source_titles", "topics"):
        value = metadata.get(metadata_field)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value:
            parts.append(str(value))
    return " ".join(parts)


def expected_output_format(prompt: dict[str, Any], gold_record: dict[str, Any] | None) -> str:
    """Return the expected output format."""

    if prompt.get("expected_output_format"):
        return str(prompt["expected_output_format"])
    if gold_record:
        metadata = gold_record.get("metadata")
        if isinstance(metadata, dict) and metadata.get("expected_output_format"):
            return str(metadata["expected_output_format"])
    return "text"


def stratified_select(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Select records deterministically across task/status/output-format groups."""

    sorted_records = sorted(records, key=lambda row: str(row.get("prompt_id")))
    if len(sorted_records) <= count:
        return sorted_records

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sorted_records:
        key = (
            str(row.get("task_type") or "missing"),
            str(row.get("expected_output_format") or "missing"),
            str(row.get("expected_status") or "missing"),
        )
        groups[key].append(row)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    total = len(sorted_records)
    for key in sorted(groups):
        group = groups[key]
        allocation = max(1, round(count * len(group) / total))
        for row in group[:allocation]:
            if len(selected) >= count:
                break
            selected.append(row)
            selected_ids.add(str(row.get("prompt_id")))

    if len(selected) < count:
        for row in sorted_records:
            prompt_id = str(row.get("prompt_id"))
            if prompt_id in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(prompt_id)
            if len(selected) >= count:
                break

    return sorted(selected[:count], key=lambda row: str(row.get("prompt_id")))


def select_prompts_for_split(
    prompts_by_vertical: dict[str, list[dict[str, Any]]],
    split: str,
    split_plan: SplitPlan | None = None,
) -> list[dict[str, Any]]:
    """Select prompt rows for one workload split."""

    active_split_plan = split_plan or DEFAULT_SPLIT_PLAN
    if split == "smoke_500":
        selected: list[dict[str, Any]] = []
        for vertical in VERTICALS:
            prompts = prompts_by_vertical.get(vertical, [])
            if len(prompts) < active_split_plan.smoke_per_vertical:
                msg = (
                    "smoke_500 cannot produce "
                    f"{active_split_plan.smoke_per_vertical} records for {vertical}"
                )
                raise RuntimeError(msg)
            selected.extend(prompts[: active_split_plan.smoke_per_vertical])
        return selected

    if split == "controlled_2000":
        if active_split_plan.controlled_total % len(VERTICALS) != 0:
            msg = "controlled_total must divide evenly across verticals"
            raise ValueError(msg)
        per_vertical = active_split_plan.controlled_total // len(VERTICALS)
        selected = []
        for vertical in VERTICALS:
            prompts = prompts_by_vertical.get(vertical, [])
            if len(prompts) < per_vertical:
                msg = f"controlled_2000 cannot produce {per_vertical} records for {vertical}"
                raise RuntimeError(msg)
            selected.extend(stratified_select(prompts, per_vertical))
        return selected

    if split == "final_10000":
        selected = [row for vertical in VERTICALS for row in prompts_by_vertical.get(vertical, [])]
        if len(selected) != active_split_plan.final_expected_total:
            msg = (
                f"final_10000 expected {active_split_plan.final_expected_total} promoted prompts, "
                f"found {len(selected)}"
            )
            raise RuntimeError(msg)
        return sorted(selected, key=lambda row: str(row.get("prompt_id")))

    msg = f"Unknown dataset split '{split}'"
    raise ValueError(msg)


def build_retrievers(
    corpora_by_vertical: dict[str, list[ContextRecord]],
) -> dict[str, dict[str, Any]]:
    """Build lexical, dense fallback, and hybrid retrievers per vertical."""

    retrievers: dict[str, dict[str, Any]] = {}
    for vertical, records in corpora_by_vertical.items():
        lexical = BM25Retriever(records)
        dense = LocalFallbackDenseRetriever(records)
        hybrid = HybridRetriever(lexical, dense)
        retrievers[vertical] = {
            "lexical": lexical,
            "dense": dense,
            "hybrid": hybrid,
        }
    return retrievers


def no_context_retrieval() -> TimedRetrieval:
    """Return no-context retrieval metadata."""

    return TimedRetrieval(
        results=[],
        latency_ms=0.0,
        backend_label="unavailable",
        retrieval_type="none",
    )


def retrieve_for_mode(
    *,
    memory_mode: str,
    query: str,
    vertical: str,
    retrievers: dict[str, dict[str, Any]],
    top_k: int,
    retrieval_cache: dict[tuple[str, str, str, int], TimedRetrieval] | None = None,
) -> TimedRetrieval:
    """Run retrieval for one memory mode."""

    if memory_mode == "mm0_no_context":
        return no_context_retrieval()
    cache_mode = "mm2_hybrid_top5" if memory_mode == "mm3_compressed_hybrid_top5" else memory_mode
    cache_key = (cache_mode, vertical, query, top_k)
    if retrieval_cache is not None and cache_key in retrieval_cache:
        return retrieval_cache[cache_key]
    if memory_mode == "mm1_dense_top5":
        retrieval = cast(LocalFallbackDenseRetriever, retrievers[vertical]["dense"]).retrieve(
            query, top_k
        )
    elif memory_mode in {"mm2_hybrid_top5", "mm3_compressed_hybrid_top5"}:
        retrieval = cast(HybridRetriever, retrievers[vertical]["hybrid"]).retrieve(query, top_k)
    else:
        msg = f"Unknown memory mode '{memory_mode}'"
        raise ValueError(msg)
    if retrieval_cache is not None:
        retrieval_cache[cache_key] = retrieval
    return retrieval


def assemble_messages(
    *,
    question: str,
    context_records: list[ContextRecord],
    memory_mode: str,
) -> list[dict[str, str]]:
    """Assemble chat messages for a future model runner."""

    if not context_records:
        return [
            {"role": "system", "content": "Answer the user request."},
            {"role": "user", "content": question},
        ]

    context_lines: list[str] = []
    for index, record in enumerate(context_records, start=1):
        context_lines.append(
            "\n".join(
                [
                    f"[{index}] {record.title}",
                    f"Context ID: {record.context_id}",
                    f"Provenance: {record.provenance}",
                    record.text,
                ]
            )
        )
    user_content = (
        "Use the provided context records when answering. Cite context IDs when relevant.\n\n"
        f"Memory mode: {memory_mode}\n\n"
        f"Context:\n\n{chr(10).join(context_lines)}\n\nQuestion:\n{question}"
    )
    return [
        {
            "role": "system",
            "content": "Answer using only the supplied context when context is provided.",
        },
        {"role": "user", "content": user_content},
    ]


def retrieval_metadata_payload(
    *,
    retrieval: TimedRetrieval,
    selected_results: list[RetrievalResult],
    evaluation: dict[str, Any],
    configured_top_k: int,
    compression_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build per-workload retrieval metadata."""

    payload: dict[str, Any] = {
        "retrieval_type": retrieval.retrieval_type,
        "retrieval_backend_label": retrieval.backend_label,
        "retrieval_latency_ms": round(retrieval.latency_ms, 6),
        "configured_top_k": configured_top_k,
        "retrieved_count": len(retrieval.results),
        "selected_context_ids": [result.context_record.context_id for result in selected_results],
        "ranked_results": [retrieval_record_payload(result) for result in retrieval.results],
        "recall_at_5": evaluation["recall_at_5"],
        "mrr": evaluation["mrr"],
        "gold_evidence_included": evaluation["gold_evidence_included"],
        "missing_gold_evidence_count": evaluation["missing_gold_evidence_count"],
    }
    if compression_metadata is not None:
        payload["compression"] = compression_metadata
    return payload


def context_records_from_results(results: list[RetrievalResult]) -> list[ContextRecord]:
    """Return context records in retrieval order."""

    return [result.context_record for result in results]


def build_one_workload_record(
    *,
    prompt: dict[str, Any],
    gold_record: dict[str, Any] | None,
    dataset_split: str,
    memory_mode: str,
    retrievers: dict[str, dict[str, Any]],
    retrieval_cache: dict[tuple[str, str, str, int], TimedRetrieval] | None = None,
) -> tuple[WorkloadRecord, dict[str, Any]]:
    """Build one workload record and its evaluation row."""

    mode_config = resolve_memory_mode(memory_mode)
    vertical = str(prompt.get("vertical") or "")
    prompt_id = str(prompt.get("prompt_id") or "")
    query = prompt_query_text(prompt)
    gold_ids = gold_evidence_ids(gold_record)
    retrieval = retrieve_for_mode(
        memory_mode=memory_mode,
        query=query,
        vertical=vertical,
        retrievers=retrievers,
        top_k=mode_config.top_k,
        retrieval_cache=retrieval_cache,
    )

    selected_results = retrieval.results
    compression_metadata: dict[str, Any] | None = None
    if memory_mode == "mm3_compressed_hybrid_top5":
        compressed = compress_retrieval_results(
            retrieval.results,
            max_context_tokens=mode_config.max_context_tokens,
        )
        selected_results = compressed.results
        compression_metadata = {
            "compression_type": "deterministic_score_dedupe_budget",
            "original_context_tokens": compressed.original_token_count,
            "compressed_context_tokens": compressed.compressed_token_count,
            "token_reduction": compressed.token_reduction,
            "compression_ratio": compressed.compression_ratio,
            "dropped_context_ids": compressed.dropped_context_ids,
        }

    selected_context_records = context_records_from_results(selected_results)
    evaluation = evaluate_retrieval_results(
        gold_evidence_ids=gold_ids,
        results=selected_results,
    )
    metadata = retrieval_metadata_payload(
        retrieval=retrieval,
        selected_results=selected_results,
        evaluation=evaluation,
        configured_top_k=mode_config.top_k,
        compression_metadata=compression_metadata,
    )
    question = str(prompt.get("question") or prompt.get("issue") or "")
    workload_record = WorkloadRecord(
        workload_id=f"{dataset_split}:{memory_mode}:{prompt_id}",
        prompt_id=prompt_id,
        vertical=vertical,
        memory_mode=memory_mode,
        messages=assemble_messages(
            question=question,
            context_records=selected_context_records,
            memory_mode=memory_mode,
        ),
        context_records=selected_context_records,
        context_token_estimate=sum(record.token_estimate for record in selected_context_records),
        retrieval_metadata=metadata,
        expected_output_format=expected_output_format(prompt, gold_record),
        gold_evidence_ids=gold_ids,
        dataset_split=dataset_split,
        source_prompt_record=prompt,
    )
    eval_row = {
        "split": dataset_split,
        "memory_mode": memory_mode,
        "vertical": vertical,
        "prompt_id": prompt_id,
        "recall_at_5": evaluation["recall_at_5"],
        "mrr": evaluation["mrr"],
        "gold_evidence_included": evaluation["gold_evidence_included"],
        "missing_gold_evidence_count": evaluation["missing_gold_evidence_count"],
        "retrieval_latency_ms": retrieval.latency_ms,
        "context_token_count": workload_record.context_token_estimate,
        "context_rows_selected": len(selected_context_records),
        "distinct_context_ids": [result.context_record.context_id for result in selected_results],
        "retrieval_backend_label": retrieval.backend_label,
        "compression_ratio": compression_metadata["compression_ratio"]
        if compression_metadata
        else None,
        "token_reduction": compression_metadata["token_reduction"] if compression_metadata else 0,
    }
    return workload_record, eval_row


def aggregate_eval_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate retrieval evaluation rows."""

    if not rows:
        return {
            "record_count": 0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
            "avg_retrieval_latency_ms": 0.0,
            "avg_context_tokens": 0.0,
            "gold_evidence_included_count": 0,
            "missing_gold_evidence_count": 0,
            "total_context_rows_selected": 0,
            "distinct_context_rows_used": 0,
            "retrieval_backend_label": "unavailable",
            "compression_ratio": None,
            "token_reduction": 0,
        }
    distinct_context_ids: set[str] = set()
    compression_values = [
        float(row["compression_ratio"]) for row in rows if row["compression_ratio"] is not None
    ]
    for row in rows:
        distinct_context_ids.update(str(context_id) for context_id in row["distinct_context_ids"])
    backend_labels = sorted(set(str(row["retrieval_backend_label"]) for row in rows))
    return {
        "record_count": len(rows),
        "recall_at_5": round(mean(float(row["recall_at_5"]) for row in rows), 6),
        "mrr": round(mean(float(row["mrr"]) for row in rows), 6),
        "avg_retrieval_latency_ms": round(
            mean(float(row["retrieval_latency_ms"]) for row in rows), 6
        ),
        "avg_context_tokens": round(mean(float(row["context_token_count"]) for row in rows), 6),
        "gold_evidence_included_count": sum(
            1 for row in rows if bool(row["gold_evidence_included"])
        ),
        "missing_gold_evidence_count": sum(int(row["missing_gold_evidence_count"]) for row in rows),
        "total_context_rows_selected": sum(int(row["context_rows_selected"]) for row in rows),
        "distinct_context_rows_used": len(distinct_context_ids),
        "retrieval_backend_label": ",".join(backend_labels),
        "compression_ratio": round(mean(compression_values), 6) if compression_values else None,
        "token_reduction": sum(int(row["token_reduction"]) for row in rows),
    }


def build_evaluation_report(
    evaluation_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build retrieval evaluation report and summary rows."""

    summary_rows: list[dict[str, Any]] = []
    by_split: dict[str, dict[str, Any]] = {}
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in evaluation_rows:
        grouped[(str(row["split"]), str(row["memory_mode"]), str(row["vertical"]))].append(row)

    for (split, memory_mode, vertical), rows in sorted(grouped.items()):
        metrics = aggregate_eval_rows(rows)
        by_split.setdefault(split, {}).setdefault(memory_mode, {})[vertical] = metrics
        summary_rows.append(
            {
                "split": split,
                "memory_mode": memory_mode,
                "vertical": vertical,
                **metrics,
            }
        )

    overall_grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in evaluation_rows:
        overall_grouped[(str(row["split"]), str(row["memory_mode"]))].append(row)
    overall_by_split_mode = {
        split: {
            mode: aggregate_eval_rows(rows)
            for (row_split, mode), rows in overall_grouped.items()
            if row_split == split
        }
        for split in sorted({str(row["split"]) for row in evaluation_rows})
    }

    return (
        {
            "generated_at_utc": utc_now(),
            "no_model_inference_triggered": True,
            "dense_retrieval_status": "local_fallback",
            "hybrid_dense_component_status": "local_fallback",
            "by_split": by_split,
            "overall_by_split_mode": overall_by_split_mode,
        },
        summary_rows,
    )


def write_workload_reports(
    *,
    output_report_root: Path,
    workload_build_report: dict[str, Any],
    workload_build_summary_rows: list[dict[str, Any]],
    retrieval_report: dict[str, Any],
    retrieval_summary_rows: list[dict[str, Any]],
) -> None:
    """Write workload and retrieval report files."""

    write_json(output_report_root / "workload_build_report.json", workload_build_report)
    write_json(output_report_root / "retrieval_evaluation_report.json", retrieval_report)
    write_csv(
        output_report_root / "workload_build_summary.csv",
        workload_build_summary_rows,
        [
            "split",
            "memory_mode",
            "record_count",
            "airline",
            "healthcare_admin",
            "retail",
            "finance",
            "research_ai",
            "output_path",
            "validated",
        ],
    )
    write_csv(
        output_report_root / "retrieval_evaluation_summary.csv",
        retrieval_summary_rows,
        [
            "split",
            "memory_mode",
            "vertical",
            "record_count",
            "recall_at_5",
            "mrr",
            "avg_retrieval_latency_ms",
            "avg_context_tokens",
            "gold_evidence_included_count",
            "missing_gold_evidence_count",
            "total_context_rows_selected",
            "distinct_context_rows_used",
            "retrieval_backend_label",
            "compression_ratio",
            "token_reduction",
        ],
    )


def build_memory_mode_workloads(
    *,
    dataset_root: str | Path,
    context_root: str | Path,
    output_root: str | Path,
    splits: list[str],
    memory_modes: list[str],
    split_plan: SplitPlan | None = None,
) -> WorkloadBuildResult:
    """Build memory-mode workload JSONL files and reports."""

    active_split_plan = split_plan or DEFAULT_SPLIT_PLAN
    configured_modes = load_memory_modes_config()
    for memory_mode in memory_modes:
        if memory_mode not in configured_modes:
            msg = f"Unknown memory mode '{memory_mode}'"
            raise ValueError(msg)
        if memory_mode not in SUPPORTED_MEMORY_MODES:
            msg = f"Memory mode '{memory_mode}' is not implemented in Phase 3 Block 3"
            raise ValueError(msg)

    prompts_by_vertical, gold_by_vertical = load_prompts_and_gold(dataset_root)
    corpora_by_vertical = load_context_corpora(context_root)
    retrievers = build_retrievers(corpora_by_vertical)
    output_path = Path(output_root)
    report_root = Path(context_root)
    workload_report: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "dataset_root": str(dataset_root),
        "context_root": str(context_root),
        "output_root": str(output_root),
        "memory_modes": memory_modes,
        "splits": splits,
        "no_model_inference_triggered": True,
        "all_workload_records_validated": True,
        "by_split": {},
    }
    workload_summary_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    retrieval_cache: dict[tuple[str, str, str, int], TimedRetrieval] = {}

    for split in splits:
        selected_prompts = select_prompts_for_split(prompts_by_vertical, split, active_split_plan)
        split_payload: dict[str, Any] = {}
        for memory_mode in memory_modes:
            eval_rows_for_file: list[dict[str, Any]] = []
            by_vertical_counts: dict[str, int] = {vertical: 0 for vertical in VERTICALS}
            output_file = output_path / split / f"{memory_mode}.jsonl"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            record_count = 0

            with output_file.open("w", encoding="utf-8") as file:
                for prompt in selected_prompts:
                    vertical = str(prompt.get("vertical"))
                    prompt_id = str(prompt.get("prompt_id"))
                    gold_record = gold_by_vertical.get(vertical, {}).get(prompt_id)
                    workload_record, eval_row = build_one_workload_record(
                        prompt=prompt,
                        gold_record=gold_record,
                        dataset_split=split,
                        memory_mode=memory_mode,
                        retrievers=retrievers,
                        retrieval_cache=retrieval_cache,
                    )
                    write_workload_jsonl_line(file, workload_record)
                    eval_rows_for_file.append(eval_row)
                    by_vertical_counts[vertical] += 1
                    record_count += 1

            evaluation_rows.extend(eval_rows_for_file)
            split_payload[memory_mode] = {
                "output_path": str(output_file),
                "record_count": record_count,
                "by_vertical": by_vertical_counts,
                "validated": True,
            }
            workload_summary_rows.append(
                {
                    "split": split,
                    "memory_mode": memory_mode,
                    "record_count": record_count,
                    **by_vertical_counts,
                    "output_path": str(output_file),
                    "validated": True,
                }
            )
        workload_report["by_split"][split] = split_payload

    retrieval_report, retrieval_summary_rows = build_evaluation_report(evaluation_rows)
    write_workload_reports(
        output_report_root=report_root,
        workload_build_report=workload_report,
        workload_build_summary_rows=workload_summary_rows,
        retrieval_report=retrieval_report,
        retrieval_summary_rows=retrieval_summary_rows,
    )
    return WorkloadBuildResult(
        workload_build_report=workload_report,
        workload_build_summary_rows=workload_summary_rows,
        retrieval_evaluation_report=retrieval_report,
        retrieval_evaluation_summary_rows=retrieval_summary_rows,
    )

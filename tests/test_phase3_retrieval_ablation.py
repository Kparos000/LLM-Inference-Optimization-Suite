import json
from pathlib import Path
from typing import Any

from inference_bench.context_corpora import VERTICALS, build_context_corpora, read_jsonl
from inference_bench.memory_workloads import (
    SplitPlan,
    build_memory_mode_workloads,
    prompt_query_text,
)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def kb_row(vertical: str) -> dict[str, Any]:
    if vertical == "finance":
        return {
            "allowed_to_commit": True,
            "body": "AAPL 10-K revenue evidence for fiscal year 2024.",
            "doc_id": "finance_kb_xbrl_AAPL_revenue",
            "document_type": "xbrl_fact_evidence",
            "metadata": {
                "company_name": "Apple Inc.",
                "concept": "Revenue",
                "form": "10-K",
                "fiscal_year": "2024",
                "ticker": "AAPL",
            },
            "source_type": "synthetic_public_inspired",
            "tags": ["finance", "xbrl"],
            "title": "AAPL revenue fact",
            "vertical": vertical,
        }
    return {
        "allowed_to_commit": True,
        "body": f"{vertical} fixture evidence for grounded retrieval.",
        "doc_id": f"{vertical}_kb_fixture",
        "document_type": "fixture",
        "metadata": {"category": "fixture"},
        "source_type": "synthetic_public_inspired",
        "tags": [vertical],
        "title": f"{vertical} fixture",
        "vertical": vertical,
    }


def make_fixture_dataset(root: Path, per_vertical: int = 2) -> Path:
    for vertical in VERTICALS:
        row = kb_row(vertical)
        doc_id = str(row["doc_id"])
        prompts: list[dict[str, Any]] = []
        gold_rows: list[dict[str, Any]] = []
        for index in range(per_vertical):
            prompt_id = f"{vertical}_fixture_{index}"
            prompts.append(
                {
                    "prompt_id": prompt_id,
                    "question": "What does the evidence say about revenue?"
                    if vertical == "finance"
                    else f"What does the {vertical} evidence say?",
                    "issue": "Use the cited source if available.",
                    "expected_output_format": "text",
                    "expected_status": "answer",
                    "task_type": "answer_grounded",
                    "vertical": vertical,
                    "ticker": "AAPL" if vertical == "finance" else "",
                    "company": "Apple Inc." if vertical == "finance" else "",
                    "required_doc_ids": [doc_id],
                    "required_evidence_ids": [doc_id],
                    "required_chunk_ids": [doc_id],
                }
            )
            gold_rows.append(
                {
                    "prompt_id": prompt_id,
                    "reference_answer": f"Ground answer in {doc_id}.",
                    "required_doc_ids": [doc_id],
                    "required_evidence_ids": [doc_id],
                    "required_chunk_ids": [doc_id],
                    "must_include": [doc_id],
                    "must_not_include": ["unsupported"],
                    "task_type": "answer_grounded",
                    "vertical": vertical,
                }
            )
        write_jsonl(root / vertical / f"{vertical}_kb_2000.jsonl", [row])
        write_jsonl(root / vertical / f"{vertical}_gold_2000.jsonl", gold_rows)
        write_jsonl(root / vertical / f"{vertical}_prompts_2000.jsonl", prompts)
    return root


def ablation_prompt() -> dict[str, Any]:
    return {
        "prompt_id": "finance_fixture",
        "question": "What does Apple report for revenue?",
        "issue": "Answer from the finance evidence.",
        "task_type": "answer_grounded",
        "expected_output_format": "text",
        "expected_status": "answer",
        "vertical": "finance",
        "ticker": "AAPL",
        "company": "Apple Inc.",
        "required_doc_ids": ["finance_kb_xbrl_AAPL_revenue"],
        "required_evidence_ids": ["finance_kb_xbrl_AAPL_revenue"],
        "source_parent_asins": ["SHOULD_NOT_APPEAR"],
        "metadata": {"source_titles": ["AAPL revenue source"]},
    }


def test_prompt_text_only_ablation_does_not_use_gold_or_source_ids() -> None:
    query = prompt_query_text(ablation_prompt(), "prompt_text_only")

    assert query.ablation_mode == "prompt_text_only"
    assert query.uses_source_hints is False
    assert query.uses_gold_ids is False
    assert "finance_kb_xbrl_AAPL_revenue" not in query.query_text
    assert "AAPL revenue source" not in query.query_text


def test_prompt_plus_metadata_does_not_use_gold_evidence_ids() -> None:
    query = prompt_query_text(ablation_prompt(), "prompt_plus_metadata")

    assert query.uses_metadata is True
    assert query.uses_source_hints is False
    assert query.uses_gold_ids is False
    assert "AAPL" in query.query_text
    assert "finance_kb_xbrl_AAPL_revenue" not in query.query_text
    assert "AAPL revenue source" not in query.query_text


def test_prompt_plus_source_hints_is_clearly_labeled() -> None:
    query = prompt_query_text(ablation_prompt(), "prompt_plus_source_hints")

    assert query.ablation_mode == "prompt_plus_source_hints"
    assert query.uses_source_hints is True
    assert query.uses_gold_ids is False
    assert "finance_kb_xbrl_AAPL_revenue" in query.query_text


def test_retrieval_evaluation_report_includes_ablation_and_dense_backend(
    tmp_path: Path,
) -> None:
    dataset_root = make_fixture_dataset(tmp_path / "dataset")
    context_root = tmp_path / "context"
    output_root = tmp_path / "workloads"
    build_context_corpora(dataset_root=dataset_root, output_root=context_root)

    result = build_memory_mode_workloads(
        dataset_root=dataset_root,
        context_root=context_root,
        output_root=output_root,
        splits=["smoke_500"],
        memory_modes=["mm1_dense_top5", "mm2_hybrid_top5"],
        ablation_modes=[
            "prompt_text_only",
            "prompt_plus_metadata",
            "prompt_plus_source_hints",
        ],
        split_plan=SplitPlan(smoke_per_vertical=1, controlled_total=5, final_expected_total=10),
    )

    assert result.retrieval_evaluation_report["ablation_modes"] == [
        "prompt_plus_metadata",
        "prompt_plus_source_hints",
        "prompt_text_only",
    ]
    assert result.retrieval_evaluation_report["dense_retrieval_status"] == "local_fallback"
    assert all("ablation_mode" in row for row in result.retrieval_evaluation_summary_rows)
    assert all("dense_backend" in row for row in result.retrieval_evaluation_summary_rows)
    assert (output_root / "smoke_500" / "prompt_text_only" / "mm1_dense_top5.jsonl").exists()
    row = read_jsonl(
        output_root / "smoke_500" / "prompt_plus_source_hints" / "mm2_hybrid_top5.jsonl"
    )[0]
    assert row["retrieval_metadata"]["ablation_mode"] == "prompt_plus_source_hints"
    assert row["retrieval_metadata"]["source_hints_used"] is True


def test_no_model_inference_or_gpu_api_calls_are_triggered(tmp_path: Path) -> None:
    dataset_root = make_fixture_dataset(tmp_path / "dataset")
    context_root = tmp_path / "context"
    output_root = tmp_path / "workloads"
    build_context_corpora(dataset_root=dataset_root, output_root=context_root)

    result = build_memory_mode_workloads(
        dataset_root=dataset_root,
        context_root=context_root,
        output_root=output_root,
        splits=["smoke_500"],
        memory_modes=["mm0_no_context"],
        ablation_modes=["prompt_text_only"],
        split_plan=SplitPlan(smoke_per_vertical=1, controlled_total=5, final_expected_total=10),
    )

    assert result.workload_build_report["no_model_inference_triggered"] is True
    assert result.retrieval_evaluation_report["no_model_inference_triggered"] is True

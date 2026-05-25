import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from inference_bench.context_corpora import VERTICALS, build_context_corpora, read_jsonl
from inference_bench.context_schema import WorkloadRecord
from inference_bench.memory_workloads import (
    CONTEXT_REGEN_COMMAND,
    SplitPlan,
    build_memory_mode_workloads,
    load_prompts_and_gold,
    select_prompts_for_split,
)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def fixture_kb_row(vertical: str) -> dict[str, Any]:
    common = {
        "allowed_to_commit": True,
        "source_type": "synthetic_public_inspired",
        "vertical": vertical,
    }
    if vertical == "airline":
        return {
            **common,
            "body": "Canada Air refund policy allows eligible cancellation within 24 hours.",
            "doc_id": "CA-POL-TEST",
            "document_type": "policy",
            "metadata": {"base_doc_id": "CA-POL-TEST", "fictional_airline": "Canada Air"},
            "tags": ["airline", "refund"],
            "title": "Canada Air Refund Policy",
        }
    if vertical == "healthcare_admin":
        return {
            **common,
            "body": "MapleCare scheduling staff verify identity and avoid clinical diagnosis.",
            "doc_id": "MCH-POL-TEST",
            "document_type": "procedure",
            "metadata": {
                "base_doc_id": "MCH-POL-TEST",
                "fictional_provider": "MapleCare Health",
            },
            "tags": ["healthcare-admin", "scheduling"],
            "title": "MapleCare Scheduling Procedure",
        }
    if vertical == "retail":
        return {
            **common,
            "body": "Review evidence for a towel notes soft texture and good kitchen fit.",
            "doc_id": "retail_review_test",
            "document_type": "review_evidence",
            "metadata": {
                "category": "Home_and_Kitchen",
                "parent_asin": "B000TEST",
                "product_title": "Test Towel",
                "rating": 5.0,
            },
            "tags": ["retail", "review"],
            "title": "Test Towel Review",
        }
    if vertical == "finance":
        return {
            **common,
            "body": "Curated XBRL fact evidence for Test Corp revenue equals 100 USD.",
            "doc_id": "finance_kb_xbrl_TEST_revenue",
            "document_type": "xbrl_fact_evidence",
            "metadata": {
                "company_name": "Test Corp",
                "concept": "Revenue",
                "ticker": "TEST",
            },
            "tags": ["finance", "xbrl"],
            "title": "TEST XBRL Revenue",
        }
    if vertical == "research_ai":
        return {
            **common,
            "body": "ABSTRACT This paper studies efficient long-context inference evaluation.",
            "doc_id": "research_ai_kb_test_abstract",
            "document_type": "paper_section",
            "metadata": {
                "paper_id": "paper_test_001",
                "section_title": "Abstract",
                "section_type": "abstract",
                "title": "Efficient Long-Context Inference",
            },
            "tags": ["research_ai", "abstract"],
            "title": "Efficient Long-Context Inference - Abstract",
        }
    raise AssertionError(f"Unexpected vertical: {vertical}")


def make_fixture_dataset(root: Path, per_vertical: int = 4) -> Path:
    for vertical in VERTICALS:
        kb_row = fixture_kb_row(vertical)
        doc_id = str(kb_row["doc_id"])
        prompts: list[dict[str, Any]] = []
        gold_rows: list[dict[str, Any]] = []
        for index in range(1, per_vertical + 1):
            prompt_id = f"{vertical}_fixture_{index:03d}"
            prompts.append(
                {
                    "prompt_id": prompt_id,
                    "question": f"Answer {vertical} fixture request {index} using cited evidence.",
                    "issue": f"{vertical} request needs grounded evidence from {doc_id}.",
                    "expected_output_format": "text",
                    "expected_status": "answer",
                    "task_type": "answer_grounded",
                    "vertical": vertical,
                }
            )
            gold_rows.append(
                {
                    "prompt_id": prompt_id,
                    "reference_answer": f"Use cited evidence {doc_id}.",
                    "required_doc_ids": [doc_id],
                    "required_evidence_ids": [doc_id],
                    "required_chunk_ids": [doc_id],
                    "must_include": [doc_id],
                    "must_not_include": ["unsupported claim"],
                    "task_type": "answer_grounded",
                    "vertical": vertical,
                }
            )
        write_jsonl(root / vertical / f"{vertical}_kb_2000.jsonl", [kb_row])
        write_jsonl(root / vertical / f"{vertical}_gold_2000.jsonl", gold_rows)
        write_jsonl(root / vertical / f"{vertical}_prompts_2000.jsonl", prompts)
    return root


@pytest.fixture()
def generated_fixture(
    tmp_path: Path,
) -> tuple[Any, Path, Path]:
    dataset_root = make_fixture_dataset(tmp_path / "dataset")
    context_root = tmp_path / "context"
    output_root = tmp_path / "workloads"
    build_context_corpora(dataset_root=dataset_root, output_root=context_root)
    result = build_memory_mode_workloads(
        dataset_root=dataset_root,
        context_root=context_root,
        output_root=output_root,
        splits=["smoke_500", "controlled_2000", "final_10000"],
        memory_modes=[
            "mm0_no_context",
            "mm1_dense_top5",
            "mm2_hybrid_top5",
            "mm3_compressed_hybrid_top5",
        ],
        split_plan=SplitPlan(smoke_per_vertical=2, controlled_total=10, final_expected_total=20),
    )
    return result, context_root, output_root


def first_workload_row(output_root: Path, split: str, memory_mode: str) -> dict[str, Any]:
    rows = read_jsonl(output_root / split / f"{memory_mode}.jsonl")
    assert rows
    return rows[0]


def test_mm0_workload_has_no_context_records(generated_fixture: tuple[Any, Path, Path]) -> None:
    _, _, output_root = generated_fixture
    row = first_workload_row(output_root, "smoke_500", "mm0_no_context")

    assert row["context_records"] == []
    assert row["context_token_estimate"] == 0
    assert row["retrieval_metadata"]["retrieval_type"] == "none"
    assert row["retrieval_metadata"]["retrieval_backend_label"] == "unavailable"


def test_mm1_workload_has_retrieval_metadata(generated_fixture: tuple[Any, Path, Path]) -> None:
    _, _, output_root = generated_fixture
    row = first_workload_row(output_root, "smoke_500", "mm1_dense_top5")

    assert row["context_records"]
    assert row["retrieval_metadata"]["retrieval_type"] == "dense"
    assert row["retrieval_metadata"]["retrieval_backend_label"] == "local_fallback"
    assert row["retrieval_metadata"]["retrieved_count"] >= 1


def test_mm2_workload_has_hybrid_retrieval_metadata(
    generated_fixture: tuple[Any, Path, Path],
) -> None:
    _, _, output_root = generated_fixture
    row = first_workload_row(output_root, "smoke_500", "mm2_hybrid_top5")

    assert row["context_records"]
    assert row["retrieval_metadata"]["retrieval_type"] == "hybrid"
    assert row["retrieval_metadata"]["retrieval_backend_label"] == "local_fallback"
    assert {"bm25", "dense"}.issubset(
        row["retrieval_metadata"]["ranked_results"][0]["component_scores"]
    )


def test_mm3_workload_has_compression_metadata(
    generated_fixture: tuple[Any, Path, Path],
) -> None:
    _, _, output_root = generated_fixture
    row = first_workload_row(output_root, "smoke_500", "mm3_compressed_hybrid_top5")

    compression = row["retrieval_metadata"]["compression"]
    assert compression["compression_type"] == "deterministic_score_dedupe_budget"
    assert compression["compression_ratio"] is not None
    assert compression["compressed_context_tokens"] <= compression["original_context_tokens"]


def test_workload_records_validate_against_schema(
    generated_fixture: tuple[Any, Path, Path],
) -> None:
    _, _, output_root = generated_fixture
    rows = read_jsonl(output_root / "smoke_500" / "mm2_hybrid_top5.jsonl")

    for row in rows:
        WorkloadRecord(**row)


def test_smoke_500_has_100_prompts_per_vertical() -> None:
    prompts_by_vertical, _ = load_prompts_and_gold("data/scaleup_2000_full")
    selected = select_prompts_for_split(prompts_by_vertical, "smoke_500")
    counts = Counter(str(row["vertical"]) for row in selected)

    assert len(selected) == 500
    assert counts == {vertical: 100 for vertical in VERTICALS}


def test_controlled_and_final_counts_validate_with_fixture_equivalent(
    generated_fixture: tuple[Any, Path, Path],
) -> None:
    _, _, output_root = generated_fixture

    assert len(read_jsonl(output_root / "controlled_2000" / "mm0_no_context.jsonl")) == 10
    assert len(read_jsonl(output_root / "final_10000" / "mm0_no_context.jsonl")) == 20


def test_retrieval_evaluation_report_is_created(
    generated_fixture: tuple[Any, Path, Path],
) -> None:
    result, context_root, _ = generated_fixture

    assert (context_root / "retrieval_evaluation_report.json").exists()
    assert (context_root / "retrieval_evaluation_summary.csv").exists()
    assert result.retrieval_evaluation_report["no_model_inference_triggered"] is True
    assert result.retrieval_evaluation_report["dense_retrieval_status"] == "local_fallback"


def test_compression_ratio_is_calculated(generated_fixture: tuple[Any, Path, Path]) -> None:
    result, _, _ = generated_fixture
    mm3_rows = [
        row
        for row in result.retrieval_evaluation_summary_rows
        if row["memory_mode"] == "mm3_compressed_hybrid_top5"
    ]

    assert mm3_rows
    assert all(row["compression_ratio"] is not None for row in mm3_rows)


def test_unknown_memory_mode_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown memory mode"):
        build_memory_mode_workloads(
            dataset_root=tmp_path / "dataset",
            context_root=tmp_path / "context",
            output_root=tmp_path / "workloads",
            splits=["smoke_500"],
            memory_modes=["missing_memory_mode"],
        )


def test_missing_corpora_fail_with_regeneration_instructions(tmp_path: Path) -> None:
    dataset_root = make_fixture_dataset(tmp_path / "dataset")

    with pytest.raises(RuntimeError, match="Regenerate them with") as exc_info:
        build_memory_mode_workloads(
            dataset_root=dataset_root,
            context_root=tmp_path / "missing_context",
            output_root=tmp_path / "workloads",
            splits=["smoke_500"],
            memory_modes=["mm0_no_context"],
            split_plan=SplitPlan(smoke_per_vertical=1, controlled_total=5, final_expected_total=20),
        )

    assert CONTEXT_REGEN_COMMAND in str(exc_info.value)


def test_no_model_inference_is_triggered(generated_fixture: tuple[Any, Path, Path]) -> None:
    result, _, _ = generated_fixture

    assert result.workload_build_report["no_model_inference_triggered"] is True
    assert result.retrieval_evaluation_report["no_model_inference_triggered"] is True

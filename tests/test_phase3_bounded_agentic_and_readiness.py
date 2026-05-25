import json
from pathlib import Path

import pytest

from inference_bench.agentic_contract import (
    APPROVED_AGENTIC_TOOLS,
    MM4_BOUNDED_AGENTIC_CONTRACT,
    AgenticTrace,
    valid_agentic_trace_fixture,
)
from inference_bench.config import load_memory_modes_config
from inference_bench.context_corpora import VERTICALS
from inference_bench.evaluator_contract import (
    SUPPORTED_EVALUATOR_FIELDS,
    evaluate_generated_answers,
)
from inference_bench.phase3_readiness import (
    MM0_TO_MM3,
    WORKLOAD_SPLITS,
    build_phase3_readiness_report,
)


def write_text(path: Path, content: str = "ok\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def make_readiness_artifacts(root: Path) -> tuple[Path, Path, Path, Path]:
    dataset_root = root / "dataset"
    context_root = root / "context"
    workload_root = root / "workloads"
    output_root = root / "context"
    dataset_root.mkdir(parents=True, exist_ok=True)
    registry = {
        "entries": [
            {
                "vertical": vertical,
                "corpus_id": f"{vertical}_benchmark_context_corpus",
                "corpus_role": "benchmark_kb",
                "input_path": f"dataset/{vertical}/{vertical}_kb_2000.jsonl",
                "output_path": f"context/corpora/{vertical}_context_corpus.jsonl",
                "chunk_builder": f"build_{vertical}_context_records",
                "notes": "test fixture",
            }
            for vertical in VERTICALS
        ]
    }
    write_text(context_root / "corpus_registry.json", json.dumps(registry))
    for vertical in VERTICALS:
        write_text(context_root / "corpora" / f"{vertical}_context_corpus.jsonl")
    for artifact_name in (
        "retrieval_evaluation_report.json",
        "retrieval_evaluation_summary.csv",
        "workload_build_report.json",
        "workload_build_summary.csv",
    ):
        write_text(context_root / artifact_name)
    for split in WORKLOAD_SPLITS:
        for mode in MM0_TO_MM3:
            write_text(workload_root / split / f"{mode}.jsonl")
    return dataset_root, context_root, workload_root, output_root


def test_mm4_config_exists() -> None:
    memory_modes = load_memory_modes_config()

    assert "mm4_bounded_agentic" in memory_modes
    assert memory_modes["mm4_bounded_agentic"].requires_agentic_workflow is True


def test_mm4_hard_limits_are_present() -> None:
    limits = MM4_BOUNDED_AGENTIC_CONTRACT.hard_limits

    assert limits.max_tool_calls == 3
    assert limits.max_retrieval_rounds == 2
    assert limits.max_generation_attempts == 2
    assert limits.max_repair_attempts == 1
    assert limits.no_internet is True
    assert limits.no_arbitrary_tools is True
    assert limits.corpus_scope == "project_corpus_only"
    assert set(APPROVED_AGENTIC_TOOLS) == {
        "retrieve_context",
        "assemble_context",
        "validate_citations",
        "validate_format",
        "validate_safety",
        "repair_once",
        "escalate",
    }


def test_agentic_trace_schema_validates_a_valid_trace() -> None:
    trace = valid_agentic_trace_fixture()

    assert trace.memory_mode == "mm4_bounded_agentic"
    assert trace.retrieval_rounds == 1
    assert trace.final_status == "answer"


def test_agentic_trace_rejects_too_many_retrieval_rounds() -> None:
    payload = valid_agentic_trace_fixture().to_dict()
    payload["retrieval_rounds"] = 3

    with pytest.raises(ValueError, match="retrieval_rounds exceeds"):
        AgenticTrace(**payload)


def test_agentic_trace_rejects_unapproved_tools() -> None:
    payload = valid_agentic_trace_fixture().to_dict()
    payload["steps"][0]["tool_name"] = "web_search"

    with pytest.raises(ValueError, match="unapproved tool"):
        AgenticTrace(**payload)


def test_evaluator_contract_can_join_mock_output_to_gold_by_prompt_id() -> None:
    generated = [
        {
            "prompt_id": "prompt_eval_001",
            "generated_text": "Grounded answer cites CA-POL-001.",
            "final_status": "answer",
            "citations": ["CA-POL-001"],
        }
    ]
    gold = [
        {
            "prompt_id": "prompt_eval_001",
            "expected_status": "answer",
            "metadata": {"expected_output_format": "text"},
            "must_include": ["CA-POL-001"],
            "must_not_include": ["fabricated citation"],
            "required_doc_ids": ["CA-POL-001"],
        }
    ]

    results = evaluate_generated_answers(generated, gold)

    assert results[0]["joined"] is True
    assert results[0]["prompt_id"] == "prompt_eval_001"
    assert results[0]["evidence_match"] is True


def test_evaluator_contract_returns_expected_structured_fields() -> None:
    results = evaluate_generated_answers(
        [
            {
                "prompt_id": "prompt_eval_002",
                "generated_text": '{"answer": "Use DOC-1", "citation": "DOC-1"}',
                "final_status": "answer",
                "citations": ["DOC-1"],
            }
        ],
        [
            {
                "prompt_id": "prompt_eval_002",
                "expected_status": "answer",
                "expected_output_format": "json",
                "must_include": ["DOC-1"],
                "must_not_include": ["price target"],
                "required_evidence_ids": ["DOC-1"],
            }
        ],
    )
    result = results[0]

    assert set(SUPPORTED_EVALUATOR_FIELDS).issubset(result)
    assert result["format_valid"] is True
    assert result["json_validity"] is True
    assert result["groundedness"] is True
    assert result["safety_violation"] is False


def test_phase3_readiness_report_is_generated(tmp_path: Path) -> None:
    dataset_root, context_root, workload_root, output_root = make_readiness_artifacts(tmp_path)

    report = build_phase3_readiness_report(
        dataset_root=dataset_root,
        context_root=context_root,
        workload_root=workload_root,
        output_root=output_root,
    )

    assert (output_root / "phase3_readiness_report.json").exists()
    assert (output_root / "phase3_readiness_summary.csv").exists()
    assert report["ready_for_phase4_plumbing"] is True


def test_readiness_report_marks_required_phase3_artifacts_clearly(tmp_path: Path) -> None:
    dataset_root, context_root, workload_root, output_root = make_readiness_artifacts(tmp_path)
    report = build_phase3_readiness_report(
        dataset_root=dataset_root,
        context_root=context_root,
        workload_root=workload_root,
        output_root=output_root,
    )
    areas = {row["area"] for row in report["summary"]}

    assert {
        "model_aliases",
        "memory_modes",
        "schemas",
        "corpora",
        "workloads",
        "mm4_contract",
        "evaluator_contract",
    }.issubset(areas)
    assert report["mm4_bounded_agentic_contract"]["memory_mode"] == "mm4_bounded_agentic"
    assert report["evaluator_contract"]["join_key"] == "prompt_id"


def test_no_model_inference_is_triggered(tmp_path: Path) -> None:
    dataset_root, context_root, workload_root, output_root = make_readiness_artifacts(tmp_path)
    report = build_phase3_readiness_report(
        dataset_root=dataset_root,
        context_root=context_root,
        workload_root=workload_root,
        output_root=output_root,
    )

    assert report["no_model_inference_triggered"] is True
    assert report["mm4_bounded_agentic_contract"]["no_model_inference_triggered"] is True
    assert report["evaluator_contract"]["no_model_inference_triggered"] is True

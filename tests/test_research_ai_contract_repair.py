from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

from inference_bench.research_ai_contract_repair import (
    RESEARCH_AI_BASE_MAX_NEW_TOKENS,
    RESEARCH_AI_CONCISE_STRATEGY,
    apply_research_ai_strategy,
    build_failure_audit_report,
    build_research_ai_replay_rows,
    classify_research_ai_failure,
)
from inference_bench.schema import WorkloadItem


def _runner_row(prompt_id: str = "research_ai_scaleup_2000_0013") -> dict[str, object]:
    return {
        "prompt_id": prompt_id,
        "workload_name": "smoke_500_mm2",
        "prompt": (
            "SYSTEM:\nAnswer only from supplied evidence.\n\n"
            "RETRIEVED EVIDENCE:\n[EVIDENCE 1]\nevidence_id: E1\n"
            "text: Research paper abstract.\n\nOUTPUT CONTRACT:\nReturn JSON."
        ),
        "expected_output": "generation_contract_json",
        "metadata": {
            "vertical": "research_ai",
            "memory_mode": "mm2_hybrid_top5",
            "gold_evidence_ids": '["gold-1"]',
            "citation_id_aliases": '{"E1":["gold-1"],"E2":["other-2"]}',
        },
    }


def _result_row(prompt_id: str = "research_ai_scaleup_2000_0013") -> dict[str, object]:
    return {
        "prompt_id": prompt_id,
        "vertical": "research_ai",
        "generated_text": '{"answer":"unfinished"',
        "answer": "This answer is too long " * 20,
        "output_tokens": 160,
        "truncation_detected": True,
        "parse_error_type": "truncated_json",
        "input_tokens": 100,
        "total_tokens": 260,
        "ttft_ms": 10.0,
        "tpot_ms": 5.0,
        "end_to_end_latency_ms": 100.0,
        "throughput_tokens_per_second": 2600.0,
    }


def _evaluation_row(prompt_id: str = "research_ai_scaleup_2000_0013") -> dict[str, object]:
    return {
        "prompt_id": prompt_id,
        "json_validity": False,
        "generation_contract_valid": False,
        "evidence_match": False,
        "groundedness": False,
        "generation_contract_missing_fields": ["citation_notes"],
    }


def test_replay_selection_preserves_frozen_b6_inputs() -> None:
    runner = _runner_row()
    result = _result_row()
    evaluation = _evaluation_row()
    originals = deepcopy((runner, result, evaluation))

    rows = build_research_ai_replay_rows(
        runner_rows=[runner],
        result_rows=[result],
        evaluation_rows=[evaluation],
    )

    assert (runner, result, evaluation) == originals
    assert len(rows) == 1
    assert rows[0]["prompt_id"] == "research_ai_scaleup_2000_0013"
    assert rows[0]["runner_input"] == runner
    assert rows[0]["citation_id_aliases"] == {"E1": ["gold-1"], "E2": ["other-2"]}
    assert rows[0]["b6_failure_flags"] == {
        "truncated": True,
        "invalid_json": True,
        "invalid_contract": True,
        "evidence_match_failed": True,
        "groundedness_failed": True,
    }


def test_non_research_ai_rows_are_not_replayed() -> None:
    result = {**_result_row("finance_scaleup_2000_0001"), "vertical": "finance"}
    rows = build_research_ai_replay_rows(
        runner_rows=[_runner_row("finance_scaleup_2000_0001")],
        result_rows=[result],
        evaluation_rows=[_evaluation_row("finance_scaleup_2000_0001")],
    )

    assert rows == []


def test_failure_classification_identifies_truncation_contract_and_budget_causes() -> None:
    row = build_research_ai_replay_rows(
        runner_rows=[_runner_row()],
        result_rows=[_result_row()],
        evaluation_rows=[_evaluation_row()],
    )[0]

    causes = classify_research_ai_failure(row)

    assert "output_budget_too_small" in causes
    assert "answer_too_verbose" in causes
    assert "json_closing_missing" in causes
    assert "contract_field_missing" in causes
    assert "evidence_ids_missing_due_to_truncation" in causes
    assert "model_instruction_following_failure" in causes


def test_failure_audit_reports_no_evaluator_or_gold_changes() -> None:
    row = build_research_ai_replay_rows(
        runner_rows=[_runner_row()],
        result_rows=[_result_row()],
        evaluation_rows=[_evaluation_row()],
    )[0]

    report = build_failure_audit_report([row])

    assert report["row_count"] == 1
    assert report["evaluator_modified"] is False
    assert report["gold_data_modified"] is False
    assert report["promoted_retrieval_modified"] is False
    assert report["root_cause_counts"]["output_budget_too_small"] == 1


def test_concise_strategy_preserves_aliases_and_adds_compact_rules() -> None:
    item = WorkloadItem(**cast(dict[str, Any], _runner_row()))

    repaired, max_tokens = apply_research_ai_strategy(
        item,
        strategy=RESEARCH_AI_CONCISE_STRATEGY,
    )

    assert max_tokens == RESEARCH_AI_BASE_MAX_NEW_TOKENS
    assert "RESEARCH AI COMPACT ANSWER RULES:" in repaired.prompt
    assert repaired.metadata["citation_id_aliases"] == item.metadata["citation_id_aliases"]
    assert repaired.metadata["gold_evidence_ids"] == item.metadata["gold_evidence_ids"]
    assert repaired.metadata["b6r1_strategy"] == RESEARCH_AI_CONCISE_STRATEGY

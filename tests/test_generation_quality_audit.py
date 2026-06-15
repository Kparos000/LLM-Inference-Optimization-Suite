from __future__ import annotations

from copy import deepcopy

from inference_bench.generation_quality_audit import (
    build_generation_quality_audit,
    classify_quality_failure,
)


def _evaluation(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "prompt_id": "finance-1",
        "vertical": "finance",
        "json_validity": True,
        "generation_contract_valid": True,
        "evidence_ids_expected": ["gold-1"],
        "evidence_ids_found": [],
        "evidence_match": False,
        "groundedness": False,
        "safety_violation": False,
        "truncation_detected": False,
        "expected_status": "answer",
        "observed_status": "answer",
        "must_include_missing": [],
    }
    row.update(overrides)
    return row


def _result(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "prompt_id": "finance-1",
        "vertical": "finance",
        "citation_id_aliases": {"E1": ["gold-1"], "E2": ["other-2"]},
        "evidence_ids": ["E2"],
        "generated_text": '{"answer":"x"}',
        "prompt": (
            "RETRIEVED EVIDENCE:\n[EVIDENCE 1]\n"
            "text: Apple Inc. AAPL 10-K 2024 revenue.\nUSER QUESTION:\nQuestion"
        ),
    }
    row.update(overrides)
    return row


def _runner(**source_overrides: object) -> dict[str, object]:
    source: dict[str, object] = {
        "ticker": "AAPL",
        "company": "Apple Inc. (AAPL) 10-K",
        "filing_form": "10-K",
        "period": "2024",
        "metric": "revenue",
    }
    source.update(source_overrides)
    return {
        "prompt_id": "finance-1",
        "metadata": {
            "vertical": "finance",
            "citation_id_aliases": {"E1": ["gold-1"], "E2": ["other-2"]},
            "source_prompt_record": source,
            "retrieval_metadata": {
                "gold_evidence_included": True,
                "missing_gold_evidence_count": 0,
            },
        },
    }


def test_evidence_present_but_not_cited_classification() -> None:
    row = classify_quality_failure(_evaluation(), _result(), _runner())

    assert "evidence_present_but_not_cited" in row["failure_classes"]
    assert "model_instruction_following_failure" in row["failure_classes"]
    assert row["gold_evidence_present_in_context"] is True
    assert row["wrong_evidence_labels"] == ["E2"]


def test_gold_absent_from_context_classification() -> None:
    row = classify_quality_failure(
        _evaluation(),
        _result(citation_id_aliases={"E1": ["other-1"]}),
        _runner(),
    )

    assert "retrieved_gold_absent_from_context" in row["failure_classes"]
    assert row["gold_evidence_absent_ids"] == ["gold-1"]


def test_partial_multi_evidence_citation_is_distinct_from_context_absence() -> None:
    row = classify_quality_failure(
        _evaluation(
            evidence_ids_expected=["gold-1", "gold-2"],
            evidence_ids_found=["gold-1"],
        ),
        _result(
            citation_id_aliases={"E1": ["gold-1"], "E2": ["other-2"]},
            evidence_ids=["E1"],
        ),
        _runner(),
    )

    assert "partial_multi_evidence_citation" in row["failure_classes"]
    assert "retrieved_gold_absent_from_context" in row["failure_classes"]


def test_invalid_json_classification() -> None:
    row = classify_quality_failure(
        _evaluation(json_validity=False),
        _result(),
        _runner(),
    )

    assert "invalid_json" in row["failure_classes"]
    assert "model_instruction_following_failure" in row["failure_classes"]


def test_invalid_contract_classification() -> None:
    row = classify_quality_failure(
        _evaluation(generation_contract_valid=False),
        _result(),
        _runner(),
    )

    assert "invalid_contract" in row["failure_classes"]


def test_safety_violation_classification() -> None:
    row = classify_quality_failure(
        _evaluation(
            safety_violation=True,
            safety_violation_terms=["investment recommendation"],
        ),
        _result(generated_text="This is an investment recommendation."),
        _runner(),
    )

    assert "safety_violation" in row["failure_classes"]
    assert row["finance_safety_term_matches"] == ["investment recommendation"]


def test_finance_metric_period_issue_classification() -> None:
    row = classify_quality_failure(
        _evaluation(evidence_ids_expected=["finance_kb_xbrl_AAPL_fact"]),
        _result(
            citation_id_aliases={"E1": ["unrelated"]},
            prompt=(
                "RETRIEVED EVIDENCE:\n[EVIDENCE 1]\n"
                "text: Apple Inc. AAPL 10-K.\nUSER QUESTION:\nQuestion"
            ),
        ),
        _runner(period="2024", metric="revenue"),
    )

    assert "finance_metric_period_missing" in row["failure_classes"]
    assert row["finance_metadata_presence"]["period"]["status"] == "missing"
    assert row["finance_metadata_presence"]["metric"]["status"] == "missing"


def test_audit_does_not_weaken_evaluator_or_modify_inputs() -> None:
    evaluation = _evaluation()
    result = _result()
    runner = _runner()
    originals = deepcopy((evaluation, result, runner))

    report = build_generation_quality_audit(
        evaluation_rows=[evaluation],
        result_rows=[result],
        runner_inputs=[runner],
    )

    assert (evaluation, result, runner) == originals
    assert report["evaluator_modified"] is False
    assert report["gold_data_modified"] is False
    assert report["promoted_retrieval_modified"] is False
    assert report["model_inference_triggered"] is False
    assert report["recommended_repair_block"]["id"] == (
        "B3R1_FROZEN_WORKLOAD_CONTEXT_ALIGNMENT_REPAIR"
    )

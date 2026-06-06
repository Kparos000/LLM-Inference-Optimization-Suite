from inference_bench.grounding_diagnostics import (
    build_grounding_failure_report,
    classify_grounding_failure,
)


def test_partial_multi_evidence_failure_classes_are_produced() -> None:
    evaluation = {
        "generation_contract_valid": True,
        "expected_status": "answer",
        "observed_status": "answer",
        "evidence_ids_expected": ["doc-1", "doc-2"],
        "evidence_ids_found": ["doc-1"],
        "evidence_match": False,
        "groundedness": False,
        "must_include_missing": ["required detail"],
    }
    result = {"evidence_ids": ["E1"]}

    classes = classify_grounding_failure(evaluation, result)

    assert "missing_required_evidence_id" in classes
    assert "cited_partial_evidence_only" in classes
    assert "multi_evidence_under_citation" in classes
    assert "semantic_under_answer" in classes


def test_wrong_citation_and_malformed_contract_are_classified() -> None:
    classes = classify_grounding_failure(
        {
            "generation_contract_valid": False,
            "expected_status": "answer",
            "observed_status": "answer",
            "evidence_ids_expected": ["doc-1"],
            "evidence_ids_found": [],
            "evidence_match": False,
            "groundedness": False,
            "must_include_missing": [],
        },
        {"evidence_ids": ["E9"]},
    )

    assert "malformed_contract" in classes
    assert "cited_wrong_evidence_id" in classes


def test_grounding_report_only_contains_failed_outputs() -> None:
    report = build_grounding_failure_report(
        evaluation_rows=[
            {
                "prompt_id": "good",
                "groundedness": True,
            },
            {
                "prompt_id": "bad",
                "generation_contract_valid": True,
                "expected_status": "answer",
                "observed_status": "answer",
                "evidence_ids_expected": ["doc-1", "doc-2"],
                "evidence_ids_found": ["doc-1"],
                "evidence_match": False,
                "groundedness": False,
                "must_include_missing": [],
            },
        ],
        result_rows=[
            {"prompt_id": "good", "vertical": "retail", "evidence_ids": ["E1"]},
            {"prompt_id": "bad", "vertical": "airline", "evidence_ids": ["E1"]},
        ],
    )

    assert report["row_count"] == 2
    assert report["grounded_count"] == 1
    assert report["failure_count"] == 1
    assert report["failure_rows"][0]["prompt_id"] == "bad"

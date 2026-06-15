from __future__ import annotations

from copy import deepcopy

from inference_bench.generation_prompt_repair import (
    build_generation_repair_prompt,
    decide_generation_repair,
)


def _result() -> dict[str, object]:
    return {
        "prompt": "SYSTEM:\nUse E1 and E2 only.",
        "generated_text": '{"answer":"x","evidence_ids":["E1"]',
        "citation_id_aliases": {"E1": ["gold-1"], "E2": ["gold-2"]},
    }


def test_invalid_json_repair_does_not_invent_evidence() -> None:
    result = _result()
    original = deepcopy(result)
    decision = decide_generation_repair(
        evaluation={
            "json_validity": False,
            "generation_contract_valid": False,
            "safety_violation": False,
        },
        result_row=result,
    )

    prompt = build_generation_repair_prompt(decision=decision, result_row=result)

    assert decision.trigger == "invalid_json"
    assert "Allowed evidence_id labels remain exactly: E1, E2" in prompt
    assert "Do not add facts or evidence labels" in prompt
    assert "gold-1" not in prompt
    assert result == original


def test_missing_evidence_repair_uses_only_private_short_label() -> None:
    result = _result()
    decision = decide_generation_repair(
        evaluation={
            "json_validity": True,
            "generation_contract_valid": True,
            "safety_violation": False,
            "evidence_ids_expected": ["gold-1", "gold-2"],
            "evidence_ids_found": ["gold-1"],
        },
        result_row=result,
    )

    prompt = build_generation_repair_prompt(decision=decision, result_row=result)

    assert decision.trigger == "missing_evidence_label"
    assert decision.missing_labels == ("E2",)
    assert "gold-2" not in prompt
    assert "E2" in prompt


def test_safety_repair_records_trigger_and_does_not_hide_initial_failure() -> None:
    result = _result()
    evaluation = {
        "json_validity": True,
        "generation_contract_valid": True,
        "safety_violation": True,
        "safety_violation_terms": ["prohibited literal"],
    }
    original_evaluation = deepcopy(evaluation)

    decision = decide_generation_repair(evaluation=evaluation, result_row=result)
    prompt = build_generation_repair_prompt(decision=decision, result_row=result)

    assert decision.trigger == "safety_violation"
    assert "failed a deterministic safety check" in prompt
    assert "prohibited literal" not in prompt
    assert evaluation == original_evaluation


def test_no_repair_when_missing_evidence_is_absent_from_context() -> None:
    result = _result()
    decision = decide_generation_repair(
        evaluation={
            "json_validity": True,
            "generation_contract_valid": True,
            "safety_violation": False,
            "evidence_ids_expected": ["gold-3"],
            "evidence_ids_found": [],
        },
        result_row=result,
    )

    assert decision.should_retry is False
    assert decision.trigger == "no_approved_repair"

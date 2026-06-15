from __future__ import annotations

import json

from inference_bench.safety_generation_repair import (
    apply_lexical_guard_to_text,
    decide_targeted_retry,
    detect_safety_rule_ids,
    preserve_json_with_safe_answer,
    render_safety_rule_repair_prompt,
)


def test_rule_detection_returns_ids_not_unsafe_text() -> None:
    rule_ids = detect_safety_rule_ids("This mentions verification bypass.")

    assert rule_ids == ("AIR-FRAUD-1",)
    assert "verification bypass" not in rule_ids


def test_lexical_guard_replaces_prohibited_wording() -> None:
    repaired = apply_lexical_guard_to_text("Do not request verification bypass.")

    assert repaired.changed is True
    assert repaired.rule_ids == ("AIR-FRAUD-1",)
    assert "verification bypass" not in repaired.repaired_text.lower()
    assert "identity-process circumvention" in repaired.repaired_text


def test_preserve_json_with_safe_answer_keeps_citations_and_confidence() -> None:
    generated = json.dumps(
        {
            "answer": "Route verification bypass requests to review.",
            "evidence_ids": ["E1", "E2"],
            "confidence": 0.8,
            "insufficient_evidence": False,
            "citation_notes": "E1 and E2 mention verification bypass handling.",
        }
    )

    repaired = preserve_json_with_safe_answer(
        generated,
        allowed_evidence_ids=("E1", "E2"),
    )
    payload = json.loads(repaired.repaired_text)

    assert repaired.changed is True
    assert payload["evidence_ids"] == ["E1", "E2"]
    assert payload["confidence"] == 0.8
    assert payload["insufficient_evidence"] is False
    assert "verification bypass" not in payload["answer"].lower()
    assert "verification bypass" not in payload["citation_notes"].lower()


def test_safety_repair_prompt_does_not_echo_previous_output_or_unsafe_text() -> None:
    prompt = render_safety_rule_repair_prompt(
        result_row={
            "generated_text": "Route verification bypass requests.",
            "evidence_ids": ["E1"],
            "citation_id_aliases": {"E1": ["gold-1"]},
        },
        rule_ids=("AIR-FRAUD-1",),
    )

    assert "AIR-FRAUD-1" in prompt
    assert "verification bypass" not in prompt.lower()
    assert "Route verification" not in prompt
    assert "gold-1" not in prompt
    assert "Preserve evidence_ids exactly when possible: E1." in prompt


def test_targeted_retry_logic_uses_only_approved_triggers() -> None:
    assert (
        decide_targeted_retry(
            evaluation={"safety_violation": True, "json_validity": True},
            missing_labels=("E2",),
        ).trigger
        == "safety_violation"
    )
    assert (
        decide_targeted_retry(
            evaluation={"safety_violation": False, "json_validity": False},
        ).trigger
        == "invalid_json"
    )
    assert (
        decide_targeted_retry(
            evaluation={
                "safety_violation": False,
                "json_validity": True,
                "generation_contract_valid": False,
            },
        ).trigger
        == "invalid_contract"
    )
    assert (
        decide_targeted_retry(
            evaluation={
                "safety_violation": False,
                "json_validity": True,
                "generation_contract_valid": True,
            },
            missing_labels=("E2",),
        ).trigger
        == "missing_evidence_label"
    )
    assert (
        decide_targeted_retry(
            evaluation={
                "safety_violation": False,
                "json_validity": True,
                "generation_contract_valid": True,
            },
            missing_labels=("E2",),
            attempt_count=2,
        ).should_retry
        is False
    )

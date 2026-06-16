from __future__ import annotations

from inference_bench.research_ai_contract_repair import classify_b6r1_full_gate


def _summary(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "json_valid_rate": 0.98,
        "generation_contract_valid_rate": 0.98,
        "evidence_match_rate": 0.92,
        "grounded_rate": 0.91,
        "truncation_rate": 0.01,
        "safety_violation_count": 0,
    }
    row.update(overrides)
    return row


def _vertical(
    vertical: str,
    *,
    evidence: float = 0.9,
    grounded: float = 0.9,
    json_rate: float = 0.98,
    contract_rate: float = 0.98,
    truncation: float = 0.01,
) -> dict[str, object]:
    return {
        "vertical": vertical,
        "evidence_match_rate": evidence,
        "grounded_rate": grounded,
        "json_valid_rate": json_rate,
        "generation_contract_valid_rate": contract_rate,
        "truncation_rate": truncation,
    }


def _per_vertical() -> list[dict[str, object]]:
    return [
        _vertical("airline"),
        _vertical("healthcare_admin"),
        _vertical("retail"),
        _vertical("finance"),
        _vertical("research_ai"),
    ]


def test_full_gate_ready_when_all_thresholds_pass() -> None:
    gate = classify_b6r1_full_gate(
        summary=_summary(),
        per_vertical_quality=_per_vertical(),
    )

    assert gate["status"] == "B6R1_READY"
    assert gate["passed"] is True
    assert gate["failed_metrics"] == []


def test_full_gate_blocks_on_research_ai_truncation() -> None:
    rows = _per_vertical()
    rows[-1] = _vertical("research_ai", truncation=0.03)

    gate = classify_b6r1_full_gate(summary=_summary(), per_vertical_quality=rows)

    assert gate["status"] == "B6R1_BLOCKED"
    assert gate["passed"] is False
    assert "research_ai_truncation_rate" in gate["failed_metrics"]


def test_full_gate_blocks_on_minimum_vertical_evidence() -> None:
    rows = _per_vertical()
    rows[2] = _vertical("retail", evidence=0.84)

    gate = classify_b6r1_full_gate(summary=_summary(), per_vertical_quality=rows)

    assert gate["status"] == "B6R1_BLOCKED"
    assert "vertical_evidence_match_rate_min" in gate["failed_metrics"]


def test_full_gate_blocks_on_overall_json_contract_and_safety() -> None:
    gate = classify_b6r1_full_gate(
        summary=_summary(
            json_valid_rate=0.96,
            generation_contract_valid_rate=0.96,
            safety_violation_count=1,
        ),
        per_vertical_quality=_per_vertical(),
    )

    assert "json_valid_rate" in gate["failed_metrics"]
    assert "generation_contract_valid_rate" in gate["failed_metrics"]
    assert "safety_violation_count" in gate["failed_metrics"]

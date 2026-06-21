from __future__ import annotations

from inference_bench.b6r6_research_ai_recovery import (
    STRATEGY_A_ORIGINAL,
    STRATEGY_C_EVIDENCE_WHITELIST,
    STRATEGY_D_ANSWER_SKELETON,
    apply_research_ai_strategy_prompt,
    build_failure_audit,
    build_research_ai_replay_rows,
    classify_b6r6_full_gate,
    finance_repair_candidate_passes,
    full_rerun_allowed,
    map_answer_skeleton_to_common_text,
    research_ai_failure_row_selected,
    select_b6r6_strategy,
)


def _raw_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "prompt_id": "research_ai-1",
        "vertical": "research_ai",
        "evidence_ids": ["E2"],
        "answer": "A short generic answer.",
        "generated_text": '{"limitation":"short","evidence":["E2"],"confidence":"medium"}',
    }
    row.update(overrides)
    return row


def _eval_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "prompt_id": "research_ai-1",
        "evidence_match": False,
        "groundedness": False,
        "generation_contract_valid": True,
        "json_validity": True,
        "truncation_detected": False,
        "evidence_ids_expected": ["gold-a", "gold-b"],
    }
    row.update(overrides)
    return row


def _runner_item() -> dict[str, object]:
    return {
        "prompt_id": "research_ai-1",
        "workload_name": "w",
        "expected_output": "generation_contract_json",
        "prompt": (
            "[EVIDENCE 1]\nevidence_id: E1\ntitle: Abstract\nsource_type: paper\n"
            "text: abstract evidence\n\n[EVIDENCE 2]\nevidence_id: E2\ntitle: Results\n"
            "source_type: paper\ntext: result evidence\n\nOUTPUT CONTRACT:\n{}"
        ),
        "metadata": {
            "vertical": "research_ai",
            "citation_id_aliases": '{"E1":["gold-a"],"E2":["gold-b"]}',
            "gold_evidence_ids": '["gold-a","gold-b"]',
            "b5_required_labels": "E1,E2",
            "memory_mode": "mm2_hybrid_top5",
        },
    }


def test_research_ai_failure_set_uses_only_failed_research_rows() -> None:
    assert research_ai_failure_row_selected(raw_row=_raw_row(), evaluation_row=_eval_row())
    assert not research_ai_failure_row_selected(
        raw_row=_raw_row(vertical="finance"),
        evaluation_row=_eval_row(),
    )
    assert not research_ai_failure_row_selected(
        raw_row=_raw_row(),
        evaluation_row=_eval_row(evidence_match=True, groundedness=True),
    )


def test_replay_rows_and_audit_preserve_required_labels() -> None:
    rows = build_research_ai_replay_rows(
        raw_rows=[_raw_row(evidence_ids=["E3"])],
        evaluation_rows=[_eval_row()],
        runner_items_by_prompt={"research_ai-1": _runner_item()},
    )
    audit = build_failure_audit(rows)

    assert rows[0]["required_evidence_labels"] == ["E1", "E2"]
    assert audit["row_count"] == 1
    assert "wrong_evidence_selected" in audit["examples"][0]["root_causes"]
    assert audit["gold_data_modified"] is False


def test_strategy_prompts_do_not_expose_canonical_ids_or_model_routing() -> None:
    prompt = str(_runner_item()["prompt"])

    whitelist = apply_research_ai_strategy_prompt(
        prompt=prompt,
        strategy_id=STRATEGY_C_EVIDENCE_WHITELIST,
        required_labels=["E1", "E2"],
    )
    skeleton = apply_research_ai_strategy_prompt(
        prompt=prompt,
        strategy_id=STRATEGY_D_ANSWER_SKELETON,
        required_labels=["E1"],
    )
    original = apply_research_ai_strategy_prompt(
        prompt=prompt,
        strategy_id=STRATEGY_A_ORIGINAL,
        required_labels=["E1"],
    )

    assert "Eligible evidence labels: E1, E2" in whitelist
    assert '"summary"' in skeleton
    assert "gold-a" not in whitelist
    assert "route to" not in whitelist.lower()
    assert original == prompt


def test_answer_skeleton_maps_to_common_generation_contract() -> None:
    mapped = map_answer_skeleton_to_common_text(
        '{"summary":"Short answer","evidence":["E1"],'
        '"confidence":"high","insufficient_evidence":false}'
    )

    assert '"answer":"Short answer"' in mapped
    assert '"evidence_ids":["E1"]' in mapped
    assert '"confidence":0.9' in mapped


def test_selection_prefers_85_percent_but_allows_80_percent_caveat() -> None:
    lock = {
        "effective_targeted_evidence_floor": 0.8,
        "effective_targeted_grounded_floor": 0.8,
    }
    selection = select_b6r6_strategy(
        strategy_summaries=[
            {
                "strategy_id": "low",
                "json_valid_rate": 1.0,
                "generation_contract_valid_rate": 1.0,
                "evidence_match_rate": 0.8,
                "grounded_rate": 0.8,
                "safety_violation_count": 0,
                "truncation_rate": 0.0,
                "output_tokens": 100,
            },
            {
                "strategy_id": "preferred",
                "json_valid_rate": 1.0,
                "generation_contract_valid_rate": 1.0,
                "evidence_match_rate": 0.85,
                "grounded_rate": 0.85,
                "safety_violation_count": 0,
                "truncation_rate": 0.0,
                "output_tokens": 200,
            },
        ],
        baseline_lock=lock,
    )

    assert selection["selected_strategy"] == "preferred"
    assert selection["selection_status"] == "B6R6_TARGETED_READY"


def test_full_rerun_requires_research_and_finance_floors() -> None:
    selection = {"targeted_passed": True}
    b6r5 = {
        "selection": {
            "selected_strategy": "finance",
            "strategy_summaries": [
                {
                    "strategy_id": "finance",
                    "finance_evidence_match_rate": 0.9,
                    "finance_grounded_rate": 0.9,
                }
            ],
        }
    }

    assert finance_repair_candidate_passes(b6r5)
    assert full_rerun_allowed(selection=selection, b6r5_report=b6r5)


def test_full_gate_marks_research_80_to_85_as_benchmark_caveat() -> None:
    gate = classify_b6r6_full_gate(
        summary={
            "json_valid_rate": 0.98,
            "generation_contract_valid_rate": 0.98,
            "evidence_match_rate": 0.91,
            "grounded_rate": 0.91,
            "safety_violation_count": 0,
            "truncation_rate": 0.01,
        },
        per_vertical_quality=[
            {"vertical": "finance", "evidence_match_rate": 0.9, "grounded_rate": 0.9},
            {
                "vertical": "research_ai",
                "evidence_match_rate": 0.8,
                "grounded_rate": 0.8,
            },
        ],
    )

    assert gate["status"] == "BENCHMARK_EXECUTION_READY_WITH_QUALITY_CAVEAT"
    assert gate["benchmark_execution_readiness"] == "READY_WITH_QUALITY_CAVEAT"
    assert gate["deployability_readiness"] == "NOT_READY"

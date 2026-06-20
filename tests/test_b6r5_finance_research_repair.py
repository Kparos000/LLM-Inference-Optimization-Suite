from __future__ import annotations

from inference_bench.b6r5_finance_research_repair import (
    ROOT_CAUSE_CATEGORIES,
    STRATEGY_CITATION_REMINDER,
    STRATEGY_EVIDENCE_PREPLAN,
    STRATEGY_OUTPUT_BUDGET_320,
    apply_strategy_to_prompt,
    build_failure_replay_rows,
    build_root_cause_audit,
    classify_b6r5_targeted_gate,
    failure_row_selected,
    full_rerun_allowed,
    no_policy_mutation_flags,
    parse_alias_map,
    required_labels_from_aliases,
    select_b6r5_strategy,
)


def _eval_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "prompt_id": "finance-1",
        "evidence_match": False,
        "groundedness": False,
        "generation_contract_valid": True,
        "json_validity": True,
        "truncation_detected": False,
        "evidence_ids_expected": ["gold-a", "gold-b"],
    }
    row.update(overrides)
    return row


def _raw_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "prompt_id": "finance-1",
        "vertical": "finance",
        "evidence_ids": ["E1"],
        "answer": "Short answer.",
        "generated_text": "{}",
        "input_tokens": 10,
        "output_tokens": 5,
    }
    row.update(overrides)
    return row


def _runner_item(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "prompt_id": "finance-1",
        "workload_name": "w",
        "expected_output": "generation_contract_json",
        "prompt": (
            "[EVIDENCE 1]\nevidence_id: E1\ntitle: Fact\nsource_type: sec\n"
            "metric: Revenue\nperiod: 2025\ntext: fact text\n\n[EVIDENCE 2]\n"
            "evidence_id: E2\ntitle: Filing\nsource_type: sec\n"
            "filing_form: 10-Q\ntext: filing table text\n\nOUTPUT CONTRACT:\n{}"
        ),
        "metadata": {
            "vertical": "finance",
            "citation_id_aliases": '{"E1":["gold-a"],"E2":["gold-b"]}',
            "gold_evidence_ids": '["gold-a","gold-b"]',
            "b5_required_labels": "E1,E2",
            "memory_mode": "mm2_hybrid_top5",
        },
    }
    row.update(overrides)
    return row


def test_failure_set_selection_is_limited_to_failed_finance_and_research_rows() -> None:
    assert failure_row_selected(raw_row=_raw_row(), evaluation_row=_eval_row()) is True
    assert (
        failure_row_selected(
            raw_row=_raw_row(vertical="airline"),
            evaluation_row=_eval_row(),
        )
        is False
    )
    assert (
        failure_row_selected(
            raw_row=_raw_row(),
            evaluation_row=_eval_row(evidence_match=True, groundedness=True),
        )
        is False
    )


def test_failure_replay_preserves_context_aliases_and_original_flags() -> None:
    rows = build_failure_replay_rows(
        raw_rows=[_raw_row()],
        evaluation_rows=[_eval_row()],
        runner_items_by_prompt={"finance-1": _runner_item()},
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["prompt_id"] == "finance-1"
    assert row["required_evidence_labels"] == ["E1", "E2"]
    assert row["private_alias_mapping"] == {"E1": ["gold-a"], "E2": ["gold-b"]}
    assert len(row["original_rendered_context"]) == 2
    assert row["original_b6r4_evaluation"]["evidence_match"] is False


def test_alias_required_label_mapping() -> None:
    aliases = parse_alias_map('{"E1":["gold-a"],"E2":["other","gold-b"]}')

    assert required_labels_from_aliases(
        gold_evidence_ids=["gold-a", "gold-b"],
        alias_map=aliases,
    ) == ["E1", "E2"]


def test_root_cause_categories_are_stable_and_policy_flags_are_false() -> None:
    rows = build_failure_replay_rows(
        raw_rows=[_raw_row(evidence_ids=["E1", "E3"])],
        evaluation_rows=[_eval_row()],
        runner_items_by_prompt={"finance-1": _runner_item()},
    )
    audit = build_root_cause_audit(rows)
    causes = set(audit["examples"][0]["root_causes"])

    assert set(ROOT_CAUSE_CATEGORIES).issuperset(causes)
    assert "partial_multi_evidence_citation" in causes
    assert "wrong_evidence_selected" in causes
    assert "finance_metric_ambiguity" in causes
    assert audit["gold_data_modified"] is False
    assert no_policy_mutation_flags()["promoted_retrieval_modified"] is False


def test_strategy_prompts_do_not_introduce_model_routing() -> None:
    prompt = str(_runner_item()["prompt"])

    preplan = apply_strategy_to_prompt(
        prompt=prompt,
        strategy_id=STRATEGY_EVIDENCE_PREPLAN,
        required_labels=["E1", "E2"],
        vertical="finance",
    )
    reminder = apply_strategy_to_prompt(
        prompt=prompt,
        strategy_id=STRATEGY_CITATION_REMINDER,
        required_labels=[],
        vertical="research_ai",
    )
    unchanged = apply_strategy_to_prompt(
        prompt=prompt,
        strategy_id=STRATEGY_OUTPUT_BUDGET_320,
        required_labels=["E1"],
        vertical="finance",
    )

    assert "Required evidence labels: E1, E2" in preplan
    assert "B6R5 VERTICAL CITATION REMINDER" in reminder
    assert unchanged == prompt
    assert "route to" not in preplan.lower()
    assert "best model" not in reminder.lower()


def test_strategy_selection_picks_lowest_token_passing_strategy() -> None:
    summaries = [
        {
            "strategy_id": STRATEGY_EVIDENCE_PREPLAN,
            "finance_evidence_match_rate": 0.9,
            "finance_grounded_rate": 0.9,
            "research_ai_evidence_match_rate": 0.9,
            "research_ai_grounded_rate": 0.9,
            "safety_violation_count": 0,
            "truncation_rate": 0.0,
            "output_tokens": 300,
        },
        {
            "strategy_id": STRATEGY_CITATION_REMINDER,
            "finance_evidence_match_rate": 0.9,
            "finance_grounded_rate": 0.9,
            "research_ai_evidence_match_rate": 0.9,
            "research_ai_grounded_rate": 0.9,
            "safety_violation_count": 0,
            "truncation_rate": 0.0,
            "output_tokens": 200,
        },
    ]

    selected = select_b6r5_strategy(summaries)

    assert selected["selected_strategy"] == STRATEGY_CITATION_REMINDER
    assert full_rerun_allowed(selected) is True


def test_full_rerun_blocks_when_targeted_strategy_fails() -> None:
    gate = classify_b6r5_targeted_gate(
        {
            "finance_evidence_match_rate": 0.8,
            "finance_grounded_rate": 0.8,
            "research_ai_evidence_match_rate": 0.9,
            "research_ai_grounded_rate": 0.9,
            "safety_violation_count": 0,
            "truncation_rate": 0.0,
        }
    )
    selected = select_b6r5_strategy(
        [
            {
                "strategy_id": STRATEGY_OUTPUT_BUDGET_320,
                "finance_evidence_match_rate": 0.8,
                "finance_grounded_rate": 0.8,
                "research_ai_evidence_match_rate": 0.9,
                "research_ai_grounded_rate": 0.9,
                "safety_violation_count": 0,
                "truncation_rate": 0.0,
                "output_tokens": 100,
            }
        ]
    )

    assert gate["passed"] is False
    assert selected["selection_status"] == "B6R5_QUALITY_CAVEATED"
    assert full_rerun_allowed(selected) is False

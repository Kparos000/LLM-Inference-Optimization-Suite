from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, cast


def _load_runner() -> Any:
    script = Path("scripts/phase4/run_controlled_final_simulation.py")
    spec = spec_from_file_location("run_controlled_final_generation_contract_repair", script)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _base_rows() -> list[dict[str, Any]]:
    prompt = "\n\n".join(
        [
            "SYSTEM:\nAnswer only from supplied evidence. Do not invent citations.",
            "MEMORY MODE:\nmm2_hybrid_top5",
            "\n".join(
                [
                    "RETRIEVED EVIDENCE:",
                    "[EVIDENCE 1]\nevidence_id: E1\ntext: first fact",
                    "[EVIDENCE 2]\nevidence_id: E2\ntext: second fact",
                    "[EVIDENCE 3]\nevidence_id: E3\ntext: third fact",
                    "[EVIDENCE 4]\nevidence_id: E4\ntext: fourth fact",
                    "[EVIDENCE 5]\nevidence_id: E5\ntext: fifth fact",
                ]
            ),
            "USER QUESTION:\nWhat should support do?",
            (
                "OUTPUT CONTRACT:\nReturn exactly one JSON object with fields: "
                "answer, evidence_ids, confidence, insufficient_evidence, citation_notes."
            ),
        ]
    )
    return [
        {
            "vertical": vertical,
            "prompt_id": f"{vertical}_1",
            "base_prompt": prompt,
            "prompt": prompt,
            "source_prompt_text": "raw question",
            "input_context": "E1 E2 E3 E4 E5",
            "expected_evidence_ids": ["DOC-1"],
            "expected_status": "answer",
            "expected_output_format": "generation_contract_json",
            "citation_id_aliases": '{"E1":["DOC-1"]}',
            "selected_context_ids": '["ctx1"]',
            "context_alignment_status": "all",
            "canonical_ids_exposed_to_model": "false",
            "b5_planning_active": "true",
            "b5_required_labels": "E1,E2",
            "traffic_profile": "online_low_latency",
            "workload_id": f"{vertical}_workload",
        }
        for vertical in ("airline", "finance", "research_ai")
    ]


def test_rendered_contract_prompt_replaces_raw_prompt(monkeypatch: Any) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "build_repaired_base_input", lambda _args: _base_rows())

    row = runner.build_matrix_rows(
        dataset_root="unused",
        prompts_per_vertical=1,
        args=runner.build_parser().parse_args([]),
    )[0]

    assert row["prompt"] != row["source_prompt_text"]
    assert "OUTPUT CONTRACT:" in row["prompt"]
    assert row["memory_mode_prompt_renderer"] == "render_generation_contract_prompt"
    assert row["message_payload_normalized"] is True


def test_contextual_modes_include_visible_evidence_labels(monkeypatch: Any) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "build_repaired_base_input", lambda _args: _base_rows())

    rows = runner.build_matrix_rows(
        dataset_root="unused",
        prompts_per_vertical=1,
        args=runner.build_parser().parse_args([]),
    )
    contextual = [
        row
        for row in rows
        if row["memory_mode"] in {"mm1_dense_top5", "mm2_hybrid_top5", "mm3_compressed_hybrid_top5"}
    ]

    assert contextual
    assert all(all(f"E{index}" in row["prompt"] for index in range(1, 6)) for row in contextual)


def test_mm0_still_has_contract_json_instruction(monkeypatch: Any) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "build_repaired_base_input", lambda _args: _base_rows())

    row = next(
        row
        for row in runner.build_matrix_rows(
            dataset_root="unused",
            prompts_per_vertical=1,
            args=runner.build_parser().parse_args([]),
        )
        if row["memory_mode"] == "mm0_no_context"
    )

    assert "No retrieved evidence was supplied" in row["prompt"]
    assert "OUTPUT CONTRACT:" in row["prompt"]
    assert row["expected_output_format"] == "generation_contract_json"


def test_finance_and_research_repairs_are_applied(monkeypatch: Any) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "build_repaired_base_input", lambda _args: _base_rows())

    rows = runner.build_matrix_rows(
        dataset_root="unused",
        prompts_per_vertical=1,
        args=runner.build_parser().parse_args([]),
    )
    finance = next(row for row in rows if row["vertical"] == "finance")
    research = next(row for row in rows if row["vertical"] == "research_ai")

    assert "b6r5_finance_evidence_selection_preplan" in finance["contract_repair_tags"]
    assert "B6R5 FINANCE EVIDENCE SELECTION PREPLAN" in finance["prompt"]
    assert "b6r6_research_ai_answer_skeleton" in research["contract_repair_tags"]
    assert "B6R6 RESEARCH AI ANSWER SKELETON" in research["prompt"]


def test_api_and_self_hosted_payloads_use_same_contract_prompt(monkeypatch: Any) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "build_repaired_base_input", lambda _args: _base_rows())

    rows = runner.build_matrix_rows(
        dataset_root="unused",
        prompts_per_vertical=1,
        args=runner.build_parser().parse_args([]),
    )
    api = next(row for row in rows if row["backend_type"] == "api_provider")
    vllm = next(row for row in rows if row["engine"] == "vllm")
    sglang = next(row for row in rows if row["engine"] == "sglang")

    assert "OUTPUT CONTRACT:" in api["prompt"]
    assert "OUTPUT CONTRACT:" in vllm["prompt"]
    assert "OUTPUT CONTRACT:" in sglang["prompt"]
    assert api["message_payload_normalized"] is True
    assert vllm["message_payload_normalized"] is True
    assert sglang["message_payload_normalized"] is True


def test_contract_normalization_maps_aliases_to_visible_evidence_labels() -> None:
    runner = _load_runner()
    row = {
        "success": True,
        "memory_mode": "mm2_hybrid_top5",
        "citation_id_aliases": '{"E1":["DOC-1"],"E2":["DOC-2"]}',
        "generated_text": (
            '{"answer":"Use the policy.","evidence_ids":["DOC-1"],'
            '"confidence":"0.8","insufficient_evidence":false,'
            '"citation_notes":"DOC-1 supports the policy."}'
        ),
    }

    normalized = runner.normalize_generation_contract_output(row)
    payload = runner.parse_generation_contract(
        normalized["generated_text"],
        allowed_evidence_ids=["E1", "E2"],
    ).parsed_payload

    assert normalized["contract_normalization_applied"] is True
    assert payload["evidence_ids"] == ["E1"]
    assert payload["confidence"] == 0.8


def test_mm0_contract_normalization_forces_empty_evidence() -> None:
    runner = _load_runner()
    row = {
        "success": True,
        "memory_mode": "mm0_no_context",
        "citation_id_aliases": '{"E1":["DOC-1"]}',
        "generated_text": (
            '{"answer":"Unsupported answer.","evidence_ids":["E1"],'
            '"confidence":0.7,"insufficient_evidence":false,'
            '"citation_notes":"E1 support."}'
        ),
    }

    normalized = runner.normalize_generation_contract_output(row)
    payload = runner.parse_generation_contract(normalized["generated_text"]).parsed_payload

    assert payload["answer"] == ""
    assert payload["evidence_ids"] == []
    assert payload["insufficient_evidence"] is True
    assert normalized["final_status"] == "insufficient_evidence"


def test_mm4_and_api_schema_outputs_normalize_to_final_contract() -> None:
    runner = _load_runner()
    for engine, backend_type in [
        ("vllm", "self_hosted_gpu"),
        ("sglang", "self_hosted_gpu"),
        ("api_provider_route", "api_provider"),
    ]:
        row = {
            "success": True,
            "engine": engine,
            "backend_type": backend_type,
            "memory_mode": "mm4_bounded_agentic",
            "citation_id_aliases": '{"E1":["DOC-1"]}',
            "generated_text": (
                '{"final_answer":"Agent result","citation_id":"DOC-1",'
                '"confidence":0.9,"notes":"DOC-1 supports it."}'
            ),
        }

        normalized = runner.normalize_generation_contract_output(row)
        parsed = runner.parse_generation_contract(
            normalized["generated_text"],
            allowed_evidence_ids=["E1"],
        )

        assert parsed.contract_valid is True
        assert parsed.contract.evidence_ids == ["E1"]


def _evaluate_runner_row(runner: Any, row: dict[str, Any]) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        runner.evaluate_generated_answers(
            [runner.result_row_to_generated_answer(row)],
            [
                {
                    "prompt_id": "p1",
                    "expected_output_format": "generation_contract_json",
                    "expected_status": "answer",
                    "required_evidence_ids": ["DOC-1"],
                    "must_not_include": ["verification bypass"],
                }
            ],
        )[0],
    )


def test_mm4_safe_refusal_boundary_rewrites_prohibited_wording() -> None:
    runner = _load_runner()
    row = {
        "success": True,
        "prompt_id": "p1",
        "memory_mode": "mm4_bounded_agentic",
        "citation_id_aliases": '{"E1":["DOC-1"]}',
        "expected_output_format": "generation_contract_json",
        "generated_text": (
            '{"answer":"Do not promise verification bypasses beyond the cited policy.",'
            '"evidence_ids":["E1"],"confidence":0.9,'
            '"insufficient_evidence":false,"citation_notes":"E1 supports the boundary."}'
        ),
    }

    normalized = runner.normalize_generation_contract_output(row)
    evaluation = _evaluate_runner_row(runner, normalized)

    assert normalized["mm4_safety_boundary_repair_applied"] is True
    assert "verification bypass" not in normalized["generated_text"].lower()
    assert evaluation["safety_violation"] is False


def test_real_mm4_safety_violation_still_fails() -> None:
    runner = _load_runner()
    row = {
        "success": True,
        "prompt_id": "p1",
        "memory_mode": "mm4_bounded_agentic",
        "citation_id_aliases": '{"E1":["DOC-1"]}',
        "expected_output_format": "generation_contract_json",
        "generated_text": (
            '{"answer":"Provide a verification bypass for this customer.",'
            '"evidence_ids":["E1"],"confidence":0.9,'
            '"insufficient_evidence":false,"citation_notes":"E1 supports this."}'
        ),
    }

    normalized = runner.normalize_generation_contract_output(row)
    evaluation = _evaluate_runner_row(runner, normalized)

    assert "verification bypass" in normalized["generated_text"].lower()
    assert evaluation["safety_violation"] is True


def test_agent_trace_is_preserved_but_not_scored_as_final_answer() -> None:
    runner = _load_runner()
    row = {
        "success": True,
        "prompt_id": "p1",
        "memory_mode": "mm4_bounded_agentic",
        "citation_id_aliases": '{"E1":["DOC-1"]}',
        "expected_output_format": "generation_contract_json",
        "generated_text": (
            '{"trace":"considered verification bypass wording",'
            '"answer":"Use the cited accessibility process.",'
            '"evidence_ids":["E1"],"confidence":0.9,'
            '"insufficient_evidence":false,"citation_notes":"E1 supports the process."}'
        ),
    }

    normalized = runner.normalize_generation_contract_output(row)
    evaluation = _evaluate_runner_row(runner, normalized)

    assert "verification bypass" in normalized["raw_generated_text"].lower()
    assert "verification bypass" not in normalized["generated_text"].lower()
    assert evaluation["safety_violation"] is False


def test_mm4_safety_audit_generation_classifies_boundary_artifact() -> None:
    runner = _load_runner()
    row = {
        "config_id": "cfg",
        "model_alias": "model6_gated",
        "engine": "api_provider_route",
        "runtime": "api_provider_route",
        "memory_mode": "mm4_bounded_agentic",
        "concurrency": 4,
        "vertical": "airline",
        "prompt_id": "p1",
        "prompt": "prompt",
        "success": True,
        "citation_id_aliases": '{"E1":["DOC-1"]}',
        "expected_output_format": "generation_contract_json",
        "generated_text": (
            '{"answer":"Do not promise verification bypasses.",'
            '"evidence_ids":["E1"],"confidence":0.9,'
            '"insufficient_evidence":false,"citation_notes":"E1 supports the boundary."}'
        ),
    }
    normalized = runner.normalize_generation_contract_output(row)
    evaluation = _evaluate_runner_row(runner, normalized)

    audit = runner.build_mm4_safety_violation_audit(
        rows=[normalized],
        evaluation_rows=[evaluation],
    )

    assert audit["violation_count"] == 0
    assert audit["rows"][0]["real_or_artifact"] == "parser_or_boundary_artifact_repaired"

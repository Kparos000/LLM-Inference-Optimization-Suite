import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from inference_bench.evaluator_contract import evaluate_generated_answer
from inference_bench.generation_contract import (
    GENERATION_CONTRACT_FORMAT,
    PARSE_ERROR_INVALID_EVIDENCE_ID,
    PARSE_ERROR_TRUNCATED_JSON,
    detect_json_truncation,
    parse_generation_contract,
    render_contract_retry_prompt,
)
from inference_bench.schema import WorkloadItem

SCRIPT_PATH = Path("scripts/phase4/run_local_hf_smoke.py")
spec = importlib.util.spec_from_file_location("run_local_hf_smoke_hardening", SCRIPT_PATH)
assert spec is not None
run_local_hf_smoke = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(run_local_hf_smoke)


def valid_contract_text(evidence_id: str = "E1") -> str:
    return json.dumps(
        {
            "answer": "The policy applies.",
            "evidence_ids": [evidence_id],
            "confidence": 0.9,
            "insufficient_evidence": False,
            "citation_notes": "Direct support.",
        }
    )


def runner_item() -> WorkloadItem:
    return WorkloadItem(
        prompt_id="airline_fixture_001",
        workload_name="smoke_500_mm2_hybrid_top5",
        prompt="Return grounded JSON using E1.",
        expected_output="generation_contract_json",
        metadata={
            "workload_id": "smoke:airline_fixture_001",
            "vertical": "airline",
            "memory_mode": "mm2_hybrid_top5",
            "ablation_mode": "prompt_plus_metadata",
            "dataset_split": "smoke_500",
            "citation_id_aliases": json.dumps({"E1": ["doc-1"]}),
        },
    )


def test_json_extraction_from_extra_text_records_repair() -> None:
    parsed = parse_generation_contract(
        f"Result follows:\n{valid_contract_text()}\nDone.",
        allowed_evidence_ids=["E1"],
    )

    assert parsed.contract_valid is True
    assert parsed.parse_repair_applied is True
    assert parsed.parse_error_type is None


def test_simple_trailing_comma_json_repair() -> None:
    text = (
        '{"answer":"Supported","evidence_ids":["E1"],"confidence":0.8,'
        '"insufficient_evidence":false,"citation_notes":"Direct support.",}'
    )

    parsed = parse_generation_contract(text, allowed_evidence_ids=["E1"])

    assert parsed.contract_valid is True
    assert parsed.parse_repair_applied is True
    assert parsed.contract is not None
    assert parsed.contract.evidence_ids == ["E1"]


def test_invalid_evidence_id_is_rejected() -> None:
    parsed = parse_generation_contract(
        valid_contract_text("E9"),
        allowed_evidence_ids=["E1", "E2"],
    )

    assert parsed.json_valid is True
    assert parsed.contract_valid is False
    assert parsed.parse_error_type == PARSE_ERROR_INVALID_EVIDENCE_ID
    assert parsed.contract is None


def test_evaluator_remains_strict_for_unknown_evidence_label() -> None:
    result = evaluate_generated_answer(
        {
            "prompt_id": "airline_fixture_001",
            "generated_text": valid_contract_text("E9"),
            "expected_output_format": GENERATION_CONTRACT_FORMAT,
            "citation_id_aliases": {"E1": ["doc-1"]},
        },
        {
            "prompt_id": "airline_fixture_001",
            "expected_status": "answer",
            "required_doc_ids": ["doc-1"],
            "must_include": [],
            "must_not_include": [],
        },
    )

    assert result["generation_contract_valid"] is False
    assert result["parse_error_type"] == PARSE_ERROR_INVALID_EVIDENCE_ID
    assert result["evidence_match"] is False
    assert result["groundedness"] is False


def test_retry_prompt_preserves_allowed_labels_and_bad_output() -> None:
    prompt = render_contract_retry_prompt(
        bad_output='{"answer":"incomplete"',
        violation="Generated JSON appears truncated.",
        allowed_evidence_ids=["E1", "E2"],
    )

    assert "Generated JSON appears truncated." in prompt
    assert "Allowed evidence_id labels remain exactly: E1, E2" in prompt
    assert '{"answer":"incomplete"' in prompt
    assert "Do not add facts or evidence labels" in prompt
    assert "Do not answer the original question again" in prompt
    assert prompt.rstrip().endswith("Return only the corrected compact JSON object now.")


@pytest.mark.parametrize(
    "text",
    [
        '{"answer":"unfinished',
        '{"answer":"unfinished","evidence_ids":["E1"]',
        '{"answer":"unfinished","evidence_ids":["E1"],',
    ],
)
def test_truncation_detection(text: str) -> None:
    parsed = parse_generation_contract(text, allowed_evidence_ids=["E1"])

    assert detect_json_truncation(text) is True
    assert parsed.truncation_detected is True
    assert parsed.parse_error_type == PARSE_ERROR_TRUNCATED_JSON


def test_parser_repair_does_not_invent_evidence_ids() -> None:
    text = (
        '{"answer":"Supported","evidence_ids":["E2"],"confidence":0.8,'
        '"insufficient_evidence":false,"citation_notes":"Direct support.",}'
    )

    parsed = parse_generation_contract(text, allowed_evidence_ids=["E1", "E2"])

    assert parsed.contract is not None
    assert parsed.contract.evidence_ids == ["E2"]


def test_contract_retry_metadata_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def to(self, _device: str) -> "FakeModel":
            return self

        def eval(self) -> None:
            return None

    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda _model_id, **_kwargs: object()),
        AutoModelForCausalLM=SimpleNamespace(
            from_pretrained=lambda _model_id, **_kwargs: FakeModel()
        ),
    )
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    outputs = [
        '{"answer":"truncated"',
        valid_contract_text(),
    ]

    monkeypatch.setattr(run_local_hf_smoke, "require_hf_dependencies", lambda: None)
    monkeypatch.setattr(
        run_local_hf_smoke,
        "import_module",
        lambda name: fake_torch if name == "torch" else fake_transformers,
    )

    def fake_generate_once(**kwargs: Any) -> dict[str, Any]:
        generated_text = outputs.pop(0)
        return {
            "generated_text": generated_text,
            "input_tokens": 10,
            "output_tokens": 8,
            "latency_ms": 1.0,
            "chat_template_applied": True,
            "max_new_tokens": kwargs["max_new_tokens"],
        }

    monkeypatch.setattr(run_local_hf_smoke, "_generate_once", fake_generate_once)

    rows = run_local_hf_smoke.run_real_local_hf(
        items=[runner_item()],
        run_id="fixture-run",
        model_alias="model1_0_5b",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens=64,
        max_contract_retries=1,
    )

    row = rows[0]
    assert row["generation_contract_valid"] is True
    assert row["contract_retry_count"] == 1
    assert row["generation_attempt_count"] == 2
    assert row["retry_applied"] is True
    assert row["retry_success"] is True
    assert row["truncation_detected"] is True
    assert row["input_tokens"] == 20
    assert row["output_tokens"] == 16


def test_retry_can_be_disabled_for_latency_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_local_hf_smoke.validate_max_contract_retries(0)
    with pytest.raises(ValueError, match="must be <= 1"):
        run_local_hf_smoke.validate_max_contract_retries(2)


def test_hardening_path_does_not_trigger_api_or_gpu() -> None:
    row = run_local_hf_smoke.base_result_row(
        item=runner_item(),
        run_id="fixture-run",
        model_alias="model1_0_5b",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        dry_run=True,
    )

    assert row["paid_api_call_triggered"] is False
    assert row["no_gpu_experiment_triggered"] is True

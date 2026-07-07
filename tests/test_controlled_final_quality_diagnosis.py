from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "phase4"
    / "diagnose_controlled_final_quality.py"
)
SPEC = importlib.util.spec_from_file_location("diagnose_controlled_final_quality", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
diagnosis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnosis)


def _evaluation(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "json_validity": False,
        "generation_contract_valid": False,
        "evidence_match": False,
        "groundedness": False,
        "safety_violation": False,
        "safety_violation_terms": [],
        "evidence_ids_expected": ["DOC-1"],
    }
    row.update(overrides)
    return row


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_classify_malformed_markdown_json() -> None:
    row = {
        "generated_text": '```json\n{"answer": "x"\n```',
        "engine": "vllm",
        "memory_mode": "mm2_hybrid_top5",
    }

    classified = diagnosis.classify_output(row, _evaluation(evidence_ids_expected=[]))

    assert "invalid_json_due_to_markdown_fence" in classified["buckets"]


def test_classify_wrong_schema() -> None:
    row = {
        "generated_text": '{"summary":"x","evidence":["E1"]}',
        "engine": "vllm",
        "memory_mode": "mm2_hybrid_top5",
    }

    classified = diagnosis.classify_output(row, _evaluation(evidence_ids_expected=[]))

    assert "valid_json_wrong_schema" in classified["buckets"]


def test_classify_missing_assistant_text() -> None:
    classified = diagnosis.classify_output(
        {"engine": "vllm", "memory_mode": "mm2_hybrid_top5"},
        _evaluation(evidence_ids_expected=[]),
    )

    assert classified["primary_bucket"] == "no_assistant_text_extracted"


def test_classify_memory_mode_no_context_expected_failure() -> None:
    classified = diagnosis.classify_output(
        {
            "generated_text": "Natural-language answer without evidence JSON.",
            "engine": "vllm",
            "memory_mode": "mm0_no_context",
        },
        _evaluation(),
    )

    assert "memory_mode_no_context_expected_failure" in classified["buckets"]


def test_classify_response_normalization_mismatch() -> None:
    classified = diagnosis.classify_output(
        {"engine": "sglang", "memory_mode": "mm2_hybrid_top5"},
        _evaluation(evidence_ids_expected=[]),
    )

    assert "no_assistant_text_extracted" in classified["buckets"]
    assert "sglang_response_schema_mismatch" in classified["buckets"]


def test_diagnosis_main_does_not_mutate_raw_gold_or_evaluator(tmp_path: Path) -> None:
    raw_path = tmp_path / "results" / "raw" / "controlled_final_simulation_results.jsonl"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text(
        json.dumps(
            {
                "prompt_id": "airline_1",
                "config_id": "cfg",
                "generated_text": "Natural-language answer.",
                "engine": "vllm",
                "memory_mode": "mm2_hybrid_top5",
                "vertical": "airline",
                "success": True,
                "final_status": "answer",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_root = tmp_path / "data"
    gold_paths: list[Path] = []
    for vertical in diagnosis.VERTICALS:
        vertical_root = dataset_root / vertical
        vertical_root.mkdir(parents=True)
        gold_path = vertical_root / f"{vertical}_gold_2000.jsonl"
        prompt_id = "airline_1" if vertical == "airline" else f"{vertical}_unused"
        gold_path.write_text(
            json.dumps(
                {
                    "prompt_id": prompt_id,
                    "expected_status": "answer",
                    "expected_output_format": diagnosis.GENERATION_CONTRACT_FORMAT,
                    "required_evidence_ids": ["DOC-1"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        gold_paths.append(gold_path)
    evaluator_path = Path("src/inference_bench/evaluator_contract.py")
    before = {
        "raw": _hash(raw_path),
        "gold": [_hash(path) for path in gold_paths],
        "evaluator": _hash(evaluator_path),
    }

    exit_code = diagnosis.main(
        [
            "--raw-results-path",
            str(raw_path),
            "--dataset-root",
            str(dataset_root),
            "--samples-json-path",
            str(tmp_path / "processed" / "samples.json"),
            "--samples-md-path",
            str(tmp_path / "processed" / "samples.md"),
            "--trace-json-path",
            str(tmp_path / "processed" / "trace.json"),
            "--trace-md-path",
            str(tmp_path / "processed" / "trace.md"),
            "--classification-json-path",
            str(tmp_path / "processed" / "classification.json"),
            "--classification-csv-path",
            str(tmp_path / "processed" / "classification.csv"),
            "--aggregation-json-path",
            str(tmp_path / "processed" / "aggregation.json"),
            "--aggregation-csv-path",
            str(tmp_path / "processed" / "aggregation.csv"),
            "--replay-json-path",
            str(tmp_path / "processed" / "replay.json"),
            "--doc-path",
            str(tmp_path / "docs" / "diagnosis.md"),
            "--summary-doc-path",
            str(tmp_path / "docs" / "summary.md"),
            "--skip-mini-replay",
        ]
    )

    after = {
        "raw": _hash(raw_path),
        "gold": [_hash(path) for path in gold_paths],
        "evaluator": _hash(evaluator_path),
    }
    assert exit_code == 0
    assert after == before

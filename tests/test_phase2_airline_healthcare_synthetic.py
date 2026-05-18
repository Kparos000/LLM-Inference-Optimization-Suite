import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

AIRLINE_PROMPT_PATH = ROOT / "data/real_world_samples/airline_sample.jsonl"
AIRLINE_KB_PATH = ROOT / "data/kb/airline/kb_sample.jsonl"
AIRLINE_GOLD_PATH = ROOT / "data/eval/gold/airline_gold_sample.jsonl"

HEALTHCARE_PROMPT_PATH = ROOT / "data/real_world_samples/healthcare_admin_sample.jsonl"
HEALTHCARE_KB_PATH = ROOT / "data/kb/healthcare_admin/kb_sample.jsonl"
HEALTHCARE_GOLD_PATH = ROOT / "data/eval/gold/healthcare_admin_gold_sample.jsonl"

KB_REQUIRED_FIELDS = {
    "doc_id",
    "vertical",
    "title",
    "document_type",
    "source_type",
    "body",
    "version",
    "tags",
}
GOLD_REQUIRED_FIELDS = {
    "prompt_id",
    "vertical",
    "task_type",
    "expected_status",
    "must_include",
    "must_not_include",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parsed = json.loads(line)
            assert isinstance(parsed, dict)
            rows.append(parsed)
    return rows


def test_airline_sample_files_exist() -> None:
    assert AIRLINE_PROMPT_PATH.exists()
    assert AIRLINE_KB_PATH.exists()
    assert AIRLINE_GOLD_PATH.exists()


def test_healthcare_sample_files_exist() -> None:
    assert HEALTHCARE_PROMPT_PATH.exists()
    assert HEALTHCARE_KB_PATH.exists()
    assert HEALTHCARE_GOLD_PATH.exists()


def test_airline_prompt_records() -> None:
    records = _read_jsonl(AIRLINE_PROMPT_PATH)
    required_fields = {
        "prompt_id",
        "ticket_id",
        "vertical",
        "airline",
        "support_type",
        "issue",
        "expected_status",
        "required_policy_ids",
    }

    assert len(records) == 40
    assert len({record["prompt_id"] for record in records}) == 40
    assert Counter(record["expected_status"] for record in records) == {
        "answer": 36,
        "escalate": 3,
        "spam_or_fraud": 1,
    }
    for record in records:
        assert required_fields.issubset(record)
        assert record["vertical"] == "airline"
        assert record["airline"] == "Canada Air"
        assert isinstance(record["required_policy_ids"], list)
        assert record["required_policy_ids"]


def test_airline_kb_records() -> None:
    records = _read_jsonl(AIRLINE_KB_PATH)
    combined_text = "\n".join(json.dumps(record).lower() for record in records)

    assert len(records) >= 25
    assert len({record["doc_id"] for record in records}) == len(records)
    for record in records:
        assert KB_REQUIRED_FIELDS.issubset(record)
        assert record["vertical"] == "airline"
        assert record["body"]

    assert "refund" in combined_text or "cancellation" in combined_text
    assert "baggage" in combined_text
    assert "partner" in combined_text or "codeshare" in combined_text
    assert "fraud" in combined_text or "chargeback" in combined_text


def test_airline_gold_alignment() -> None:
    prompts = _read_jsonl(AIRLINE_PROMPT_PATH)
    gold_records = _read_jsonl(AIRLINE_GOLD_PATH)
    prompt_by_id = {record["prompt_id"]: record for record in prompts}

    assert len(gold_records) == 40
    assert set(record["prompt_id"] for record in gold_records) == set(prompt_by_id)
    for record in gold_records:
        assert GOLD_REQUIRED_FIELDS.issubset(record)
        metadata = record["metadata"]
        assert metadata["required_policy_ids"]
        assert (
            metadata["required_policy_ids"]
            == prompt_by_id[record["prompt_id"]]["required_policy_ids"]
        )


def test_healthcare_prompt_records() -> None:
    records = _read_jsonl(HEALTHCARE_PROMPT_PATH)
    required_fields = {
        "prompt_id",
        "ticket_id",
        "vertical",
        "support_type",
        "issue",
        "expected_status",
        "expected_queue",
        "required_policy_ids",
        "safety_boundary",
    }

    assert len(records) == 40
    assert len({record["prompt_id"] for record in records}) == 40
    assert Counter(record["expected_status"] for record in records) == {
        "answer": 35,
        "escalate": 3,
        "boundary_response": 1,
        "spam_or_fraud": 1,
    }
    for record in records:
        assert required_fields.issubset(record)
        assert record["vertical"] == "healthcare_admin"
        assert isinstance(record["required_policy_ids"], list)
        assert record["required_policy_ids"]


def test_healthcare_kb_records() -> None:
    records = _read_jsonl(HEALTHCARE_KB_PATH)
    combined_text = "\n".join(json.dumps(record).lower() for record in records)

    assert len(records) >= 25
    assert len({record["doc_id"] for record in records}) == len(records)
    for record in records:
        assert KB_REQUIRED_FIELDS.issubset(record)
        assert record["vertical"] == "healthcare_admin"
        assert record["body"]

    assert "privacy" in combined_text
    assert "medical records" in combined_text
    assert "urgent" in combined_text or "emergency" in combined_text
    assert "billing" in combined_text or "insurance" in combined_text


def test_healthcare_gold_alignment() -> None:
    prompts = _read_jsonl(HEALTHCARE_PROMPT_PATH)
    gold_records = _read_jsonl(HEALTHCARE_GOLD_PATH)
    prompt_by_id = {record["prompt_id"]: record for record in prompts}

    assert len(gold_records) == 40
    assert set(record["prompt_id"] for record in gold_records) == set(prompt_by_id)
    for record in gold_records:
        assert GOLD_REQUIRED_FIELDS.issubset(record)
        metadata = record["metadata"]
        assert metadata["required_policy_ids"]
        assert (
            metadata["required_policy_ids"]
            == prompt_by_id[record["prompt_id"]]["required_policy_ids"]
        )
        if record["expected_status"] == "boundary_response":
            forbidden = " ".join(record["must_not_include"]).lower()
            assert "diagnosis" in forbidden
            assert "treatment" in forbidden
            assert "medical advice" in forbidden


def test_no_private_or_real_personal_data() -> None:
    paths = (
        AIRLINE_PROMPT_PATH,
        AIRLINE_KB_PATH,
        AIRLINE_GOLD_PATH,
        HEALTHCARE_PROMPT_PATH,
        HEALTHCARE_KB_PATH,
        HEALTHCARE_GOLD_PATH,
    )
    combined_text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    combined_lower = combined_text.lower()

    assert re.search(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", combined_text) is None
    assert (
        re.search(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b", combined_text) is None
    )
    assert "c:\\users" not in combined_lower
    assert "/home/" not in combined_lower
    assert "api key" not in combined_lower
    assert "token" not in combined_lower
    assert "medical record number" not in combined_lower
    assert "passport number" not in combined_lower
    assert "credit card number" not in combined_lower


def test_generator_scripts_run() -> None:
    commands = (
        (
            ROOT / "scripts/phase2/generate_airline_synthetic.py",
            "airline",
            {"answer": 36, "escalate": 3, "spam_or_fraud": 1},
        ),
        (
            ROOT / "scripts/phase2/generate_healthcare_admin_synthetic.py",
            "healthcare_admin",
            {"answer": 35, "escalate": 3, "boundary_response": 1, "spam_or_fraud": 1},
        ),
    )

    for script_path, vertical, expected_status_counts in commands:
        result = subprocess.run(
            [sys.executable, str(script_path), "--build-samples"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        summary = json.loads(result.stdout)
        assert summary["vertical"] == vertical
        assert summary["prompt_record_count"] == 40
        assert summary["gold_record_count"] == 40
        assert summary["kb_record_count"] >= 25
        assert summary["counts_by_expected_status"] == expected_status_counts

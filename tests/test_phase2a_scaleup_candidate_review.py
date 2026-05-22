import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GENERATOR_SCRIPT = ROOT / "scripts/phase2/generate_phase2a_scaleup.py"
REVIEW_SCRIPT = ROOT / "scripts/phase2/review_phase2a_scaleup_candidate.py"
DOC_PATH = ROOT / "docs/41_phase2a_airline_250_candidate_review.md"
REPORT_PATH = (
    ROOT / "data/generated/phase2a/scaleup_reports/airline_250_candidate_review_report.json"
)
CANDIDATE_PROMPTS = ROOT / "data/generated/phase2a/scaleup/airline/airline_prompts_250.jsonl"
CANDIDATE_GOLD = ROOT / "data/generated/phase2a/scaleup/airline/airline_gold_250.jsonl"
CANDIDATE_KB = ROOT / "data/generated/phase2a/scaleup/airline/airline_kb_250.jsonl"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("review_phase2a_scaleup_candidate", REVIEW_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_airline_candidate() -> None:
    if CANDIDATE_PROMPTS.exists() and CANDIDATE_GOLD.exists() and CANDIDATE_KB.exists():
        return
    result = subprocess.run(
        [
            sys.executable,
            str(GENERATOR_SCRIPT),
            "--generate-vertical",
            "--vertical",
            "airline",
            "--target-per-vertical",
            "250",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
    path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")


def test_review_script_exists() -> None:
    assert REVIEW_SCRIPT.exists()


def test_review_candidate_cli() -> None:
    _ensure_airline_candidate()
    result = subprocess.run(
        [sys.executable, str(REVIEW_SCRIPT), "--review-candidate", "--vertical", "airline"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["phase"] == "2A-9A"
    assert summary["promotion_ready"] is True
    assert REPORT_PATH.exists()


def test_review_detects_prompt_gold_mismatch() -> None:
    _ensure_airline_candidate()
    module = _load_module()
    prompts = _read_jsonl(CANDIDATE_PROMPTS)
    gold = _read_jsonl(CANDIDATE_GOLD)
    kb = _read_jsonl(CANDIDATE_KB)
    gold[0]["prompt_id"] = "orphan_airline_scaleup_prompt"

    report = module.build_review_report(
        vertical="airline",
        target_count=250,
        prompts=prompts,
        gold=gold,
        kb_rows=kb,
    )

    assert report["critical_issue_count"] > 0
    assert any(issue["check_name"] == "prompt_gold_alignment" for issue in report["issue_log"])
    assert any(issue["check_name"] == "orphan_gold_record" for issue in report["issue_log"])


def test_review_detects_hygiene_issue() -> None:
    _ensure_airline_candidate()
    module = _load_module()
    prompts = _read_jsonl(CANDIDATE_PROMPTS)
    gold = _read_jsonl(CANDIDATE_GOLD)
    kb = _read_jsonl(CANDIDATE_KB)
    prompts[0]["question"] = "Use C:\\Users\\example\\private.txt for this answer."

    report = module.build_review_report(
        vertical="airline",
        target_count=250,
        prompts=prompts,
        gold=gold,
        kb_rows=kb,
    )

    assert report["critical_issue_count"] > 0
    assert any(issue["check_name"] == "hygiene_scan" for issue in report["issue_log"])


def test_review_distribution_checks() -> None:
    _ensure_airline_candidate()
    module = _load_module()
    prompts = _read_jsonl(CANDIDATE_PROMPTS)
    gold = _read_jsonl(CANDIDATE_GOLD)
    kb = _read_jsonl(CANDIDATE_KB)

    report = module.build_review_report(
        vertical="airline",
        target_count=250,
        prompts=prompts,
        gold=gold,
        kb_rows=kb,
    )

    assert report["status_counts"] == {"answer": 225, "escalate": 20, "spam_or_fraud": 5}
    assert report["output_format_counts"] == {"json": 35, "markdown_table": 25, "text": 190}
    assert report["critical_issue_count"] == 0
    assert report["warning_count"] == 0


def test_promote_if_clean_copies_files(tmp_path: Path) -> None:
    _ensure_airline_candidate()
    candidate_dir = tmp_path / "candidate"
    promoted_dir = tmp_path / "promoted"
    report_path = tmp_path / "review_report.json"
    prompts_path = candidate_dir / "airline_prompts_250.jsonl"
    gold_path = candidate_dir / "airline_gold_250.jsonl"
    kb_path = candidate_dir / "airline_kb_250.jsonl"
    _write_jsonl(prompts_path, _read_jsonl(CANDIDATE_PROMPTS))
    _write_jsonl(gold_path, _read_jsonl(CANDIDATE_GOLD))
    _write_jsonl(kb_path, _read_jsonl(CANDIDATE_KB))

    result = subprocess.run(
        [
            sys.executable,
            str(REVIEW_SCRIPT),
            "--promote-if-clean",
            "--vertical",
            "airline",
            "--candidate-prompts",
            str(prompts_path),
            "--candidate-gold",
            str(gold_path),
            "--candidate-kb",
            str(kb_path),
            "--review-report",
            str(report_path),
            "--promoted-output-dir",
            str(promoted_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["promoted"] is True
    assert (promoted_dir / "airline_prompts_250.jsonl").exists()
    assert (promoted_dir / "airline_gold_250.jsonl").exists()
    assert (promoted_dir / "airline_kb_250.jsonl").exists()
    assert report_path.exists()


def test_docs_include_review_commands() -> None:
    docs = DOC_PATH.read_text(encoding="utf-8")

    assert "--generate-vertical --vertical airline --target-per-vertical 250" in docs
    assert "--review-candidate --vertical airline" in docs
    assert "--promote-if-clean --vertical airline" in docs
    assert "no RAG" in docs

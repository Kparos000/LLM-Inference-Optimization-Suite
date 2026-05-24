import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/phase2/generate_phase2a_scaleup.py"
DOC_PATH = ROOT / "docs/40_phase2a_250_scaleup_generator.md"
GITIGNORE_PATH = ROOT / ".gitignore"
REPORT_DIR = ROOT / "data/generated/phase2a/scaleup_reports"
OUTPUT_DIR = ROOT / "data/generated/phase2a/scaleup"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("generate_phase2a_scaleup", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parsed = json.loads(line)
            assert isinstance(parsed, dict)
            rows.append(parsed)
    return rows


def _generate_vertical(vertical: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--generate-vertical",
            "--vertical",
            vertical,
            "--target-per-vertical",
            "250",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert isinstance(summary, dict)
    return summary


def test_scaleup_generator_script_exists() -> None:
    assert SCRIPT_PATH.exists()


def test_supported_targets() -> None:
    module = _load_module()

    assert module.SUPPORTED_TARGETS == [250, 1000, 2000, 4000, 5000]


def test_get_checkpoint_for_target() -> None:
    module = _load_module()

    assert module.get_checkpoint_for_target(250) == "checkpoint_250"
    assert module.get_checkpoint_for_target(1000) == "checkpoint_1000"
    assert module.get_checkpoint_for_target(2000) == "checkpoint_2000"
    assert module.get_checkpoint_for_target(4000) == "checkpoint_4000"
    assert module.get_checkpoint_for_target(5000) == "checkpoint_5000"


def test_unsupported_target_fails() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--dry-run", "--target-per-vertical", "333"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Unsupported target_per_vertical: 333" in result.stderr
    assert "250, 1000, 2000, 4000, 5000" in result.stderr


def test_distribution_counts_sum_for_all_targets() -> None:
    module = _load_module()

    for vertical in ["finance", "airline", "healthcare_admin", "research_ai", "retail"]:
        for target in module.SUPPORTED_TARGETS:
            distributions = module.calculate_distribution_counts(vertical, target)
            assert sum(distributions["expected_status"].values()) == target
            assert sum(distributions["task_type"].values()) == target
            assert sum(distributions["expected_output_format"].values()) == target
            assert sum(distributions["difficulty"].values()) == target


def test_healthcare_distribution_counts_sum_to_250() -> None:
    module = _load_module()

    distributions = module.calculate_distribution_counts("healthcare_admin", 250)
    assert distributions["expected_status"] == {
        "answer": 220,
        "escalate": 20,
        "safety_boundary": 5,
        "spam_or_fraud": 3,
        "out_of_scope": 2,
    }
    assert distributions["expected_output_format"] == {
        "text": 195,
        "json": 35,
        "markdown_table": 20,
    }
    assert distributions["task_type"] == {
        "answer_grounded": 120,
        "policy_reasoning": 55,
        "extract_structured": 30,
        "escalation_response": 25,
        "safety_boundary": 10,
        "quality_boundary": 10,
    }
    assert distributions["difficulty"] == {"easy": 80, "medium": 130, "hard": 40}


def test_retail_distribution_counts_sum_to_250() -> None:
    module = _load_module()

    distributions = module.calculate_distribution_counts("retail", 250)
    assert distributions["expected_status"] == {
        "answer": 222,
        "insufficient_evidence": 9,
        "escalate": 9,
        "spam_or_low_quality": 7,
        "out_of_scope": 3,
    }
    assert distributions["expected_output_format"] == {
        "text": 185,
        "json": 40,
        "markdown_table": 25,
    }
    assert distributions["task_type"] == {
        "answer_grounded": 95,
        "issue_identification": 45,
        "compare_products": 25,
        "extract_structured": 35,
        "policy_reasoning": 30,
        "quality_boundary": 10,
        "escalation_response": 10,
    }
    assert distributions["difficulty"] == {"easy": 80, "medium": 130, "hard": 40}


def test_research_ai_distribution_counts_sum_to_250() -> None:
    module = _load_module()

    distributions = module.calculate_distribution_counts("research_ai", 250)
    assert distributions["expected_status"] == {
        "answer": 225,
        "insufficient_evidence": 10,
        "escalate": 10,
        "out_of_scope": 5,
    }
    assert distributions["expected_output_format"] == {
        "text": 180,
        "json": 35,
        "markdown_table": 35,
    }
    assert distributions["task_type"] == {
        "answer_grounded": 90,
        "paper_method": 45,
        "results_evaluation": 35,
        "extract_structured": 30,
        "compare_papers": 25,
        "literature_table": 15,
        "escalation_response": 10,
    }
    assert distributions["difficulty"] == {"easy": 80, "medium": 130, "hard": 40}


def test_finance_distribution_counts_sum_to_250() -> None:
    module = _load_module()

    distributions = module.calculate_distribution_counts("finance", 250)
    assert distributions["expected_status"] == {
        "answer": 230,
        "insufficient_evidence": 10,
        "escalate": 10,
    }
    assert distributions["expected_output_format"] == {
        "text": 155,
        "json": 50,
        "markdown_table": 45,
    }
    assert distributions["task_type"] == {
        "answer_grounded": 95,
        "extract_structured": 45,
        "compare_filings": 35,
        "calculation": 35,
        "escalation_response": 20,
        "evidence_citation_lookup": 20,
    }
    assert distributions["difficulty"] == {"easy": 80, "medium": 130, "hard": 40}


def test_question_template_diversity_helper() -> None:
    module = _load_module()
    repeated_prompts = [
        {
            "question": f"Traveler asks Canada Air for baggage delay help in scenario {index}.",
            "support_type": "baggage_delay",
            "airline": "Canada Air",
        }
        for index in range(10)
    ]
    varied_prompts = [
        {"question": "A traveler needs help with baggage delay.", "support_type": "baggage_delay"},
        {
            "question": "What should support do for a baggage issue?",
            "support_type": "baggage_delay",
        },
        {
            "question": "Using policy records, explain the baggage delay workflow.",
            "support_type": "baggage_delay",
        },
        {"question": "A passenger asks about delayed bags.", "support_type": "baggage_delay"},
        {
            "question": "Determine the support action for delayed baggage.",
            "support_type": "baggage_delay",
        },
    ]

    repeated = module.calculate_question_template_diversity(repeated_prompts)
    varied = module.calculate_question_template_diversity(varied_prompts)
    assert repeated["unique_question_template_count"] == 1
    assert repeated["linguistic_variation_rate"] == 0.0
    assert varied["unique_question_template_count"] == 5
    assert varied["linguistic_variation_rate"] >= 0.60


def test_dry_run_cli() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["phase"] == "2A-9"
    assert summary["mode"] == "dry_run"
    assert set(summary["verticals"]) == {
        "finance",
        "airline",
        "healthcare_admin",
        "research_ai",
        "retail",
    }


def test_dry_run_2000() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--dry-run",
            "--target-per-vertical",
            "2000",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["target_per_vertical"] == 2000
    assert summary["planned_total_prompts"] == 10000
    assert summary["checkpoint"] == "checkpoint_2000"


def test_dry_run_4000_reports_planning_only() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--dry-run",
            "--target-per-vertical",
            "4000",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    for vertical in summary["verticals"].values():
        assert vertical["generation_scope"] == "planning_only"
        assert vertical["ready_for_actual_generation"] is False
        assert "generation_not_implemented_for_vertical_target" in vertical["blockers"]
        assert "large_target_generation_requires_checkpoint_review" in vertical["blockers"]


def test_generate_plan_cli() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--generate-plan",
            "--target-per-vertical",
            "250",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["phase"] == "2A-9"
    assert summary["mode"] == "generate_plan"
    assert summary["manifest_count"] == 5
    assert (REPORT_DIR / "airline_scaleup_250_manifest.json").exists()
    assert (REPORT_DIR / "phase2a_scaleup_generation_plan_250.json").exists()


def test_airline_250_has_no_generation_blocker() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--generate-plan",
            "--target-per-vertical",
            "250",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    airline = summary["verticals"]["airline"]
    assert airline["generation_implemented"] is True
    assert airline["ready_for_actual_generation"] is True
    assert airline["blocker_count"] == 0
    assert airline["blockers"] == []


def test_dry_run_marks_healthcare_ready_for_250() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--dry-run",
            "--target-per-vertical",
            "250",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    healthcare = summary["verticals"]["healthcare_admin"]
    assert healthcare["generation_implemented"] is True
    assert healthcare["generation_scope"] == "local_candidate_generation"
    assert healthcare["ready_for_actual_generation"] is True
    assert healthcare["blockers"] == []


def test_dry_run_marks_retail_ready_for_250() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--dry-run",
            "--target-per-vertical",
            "250",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    retail = summary["verticals"]["retail"]
    assert retail["generation_implemented"] is True
    assert retail["generation_scope"] == "local_candidate_generation"
    assert retail["ready_for_actual_generation"] is True
    assert retail["blockers"] == []


def test_dry_run_marks_research_ai_ready_for_250() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--dry-run",
            "--target-per-vertical",
            "250",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    research_ai = summary["verticals"]["research_ai"]
    assert research_ai["generation_implemented"] is True
    assert research_ai["generation_scope"] == "local_candidate_generation"
    assert research_ai["ready_for_actual_generation"] is True
    assert research_ai["blockers"] == []


def test_generate_plan_healthcare_blocker_count_zero_for_250() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--generate-plan",
            "--target-per-vertical",
            "250",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    healthcare = summary["verticals"]["healthcare_admin"]
    assert healthcare["generation_implemented"] is True
    assert healthcare["ready_for_actual_generation"] is True
    assert healthcare["blocker_count"] == 0
    assert healthcare["blockers"] == []


def test_generate_plan_retail_blocker_count_zero_for_250() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--generate-plan",
            "--target-per-vertical",
            "250",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    retail = summary["verticals"]["retail"]
    assert retail["generation_implemented"] is True
    assert retail["ready_for_actual_generation"] is True
    assert retail["blocker_count"] == 0
    assert retail["blockers"] == []


def test_generate_plan_research_ai_blocker_count_zero_for_250() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--generate-plan",
            "--target-per-vertical",
            "250",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    research_ai = summary["verticals"]["research_ai"]
    assert research_ai["generation_implemented"] is True
    assert research_ai["ready_for_actual_generation"] is True
    assert research_ai["blocker_count"] == 0
    assert research_ai["blockers"] == []


def test_dry_run_marks_finance_ready_for_250() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--dry-run",
            "--target-per-vertical",
            "250",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    finance = summary["verticals"]["finance"]
    assert finance["generation_implemented"] is True
    assert finance["generation_scope"] == "local_candidate_generation"
    assert finance["ready_for_actual_generation"] is True
    assert finance["blockers"] == []


def test_generate_plan_finance_blocker_count_zero_for_250() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--generate-plan",
            "--target-per-vertical",
            "250",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    finance = summary["verticals"]["finance"]
    assert finance["generation_implemented"] is True
    assert finance["ready_for_actual_generation"] is True
    assert finance["blocker_count"] == 0
    assert finance["blockers"] == []


def test_generate_plan_5000() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--generate-plan",
            "--target-per-vertical",
            "5000",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["target_per_vertical"] == 5000
    assert summary["total_target_prompts"] == 25000
    assert summary["checkpoint"] == "checkpoint_5000"
    assert (REPORT_DIR / "phase2a_scaleup_generation_plan_5000.json").exists()


def test_generate_plan_5000_reports_generation_blockers() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--generate-plan",
            "--target-per-vertical",
            "5000",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    for vertical in summary["verticals"].values():
        assert vertical["generation_implemented"] is False
        assert vertical["ready_for_actual_generation"] is False
        assert vertical["blocker_count"] > 0
        assert "generation_not_implemented_for_vertical_target" in vertical["blockers"]
        assert "large_target_generation_requires_checkpoint_review" in vertical["blockers"]


def test_large_generation_blocked_without_explicit_support() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--generate-vertical",
            "--vertical",
            "airline",
            "--target-per-vertical",
            "4000",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert (
        "Generation for airline at 4000 requires explicit implementation and prior "
        "checkpoint review"
    ) in result.stderr


def test_finance_generation_cli() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--generate-vertical",
            "--vertical",
            "finance",
            "--target-per-vertical",
            "250",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["vertical"] == "finance"
    assert summary["prompt_count"] == 250
    assert summary["gold_count"] == 250
    assert summary["critical_issue_count"] == 0
    assert (ROOT / summary["prompts_path"]).exists()
    assert (ROOT / summary["gold_path"]).exists()
    assert (ROOT / summary["kb_path"]).exists()
    report = json.loads((ROOT / summary["report_path"]).read_text(encoding="utf-8"))
    assert report["vertical"] == "finance"
    assert report["prompt_count"] == 250
    assert report["gold_count"] == 250
    assert report["critical_issue_count"] == 0


def test_generation_report_ignores_missing_qa_warning(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--generate-vertical",
            "--vertical",
            "finance",
            "--target-per-vertical",
            "250",
            "--qa-report",
            str(tmp_path / "missing_phase2a_qa_report.json"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    report = json.loads((ROOT / summary["report_path"]).read_text(encoding="utf-8"))
    assert summary["warning_count"] == 0
    assert report["warning_count"] == 0
    assert report["warnings"] == []


def test_finance_generated_counts_and_alignment() -> None:
    summary = _generate_vertical("finance")
    prompts = _read_jsonl(ROOT / summary["prompts_path"])
    gold = _read_jsonl(ROOT / summary["gold_path"])
    kb = _read_jsonl(ROOT / summary["kb_path"])
    assert len(prompts) == 250
    assert len(gold) == 250
    assert prompts[0]["prompt_id"] == "finance_scaleup_250_0001"
    assert prompts[-1]["prompt_id"] == "finance_scaleup_250_0250"

    module = _load_module()
    assert module.validate_prompt_gold_alignment(prompts, gold) == []
    assert module.validate_evidence_coverage(gold, kb) == []
    assert module.validate_no_private_hygiene_terms(prompts + gold + kb) == []


def test_finance_negative_statuses_present() -> None:
    summary = _generate_vertical("finance")
    prompts = _read_jsonl(ROOT / summary["prompts_path"])
    gold = _read_jsonl(ROOT / summary["gold_path"])
    status_counts: dict[str, int] = {}
    for prompt in prompts:
        status = str(prompt["expected_status"])
        status_counts[status] = status_counts.get(status, 0) + 1

    assert status_counts == {
        "answer": 230,
        "insufficient_evidence": 10,
        "escalate": 10,
    }
    assert {"insufficient_evidence", "escalate"}.issubset(status_counts)
    assert all(
        gold_row["must_not_include"] for gold_row in gold if gold_row["expected_status"] != "answer"
    )


def test_finance_answerable_gold_has_evidence() -> None:
    summary = _generate_vertical("finance")
    gold = _read_jsonl(ROOT / summary["gold_path"])
    for gold_row in gold:
        if gold_row["expected_status"] != "answer":
            continue
        assert gold_row["required_doc_ids"]
        assert gold_row["required_chunk_ids"]
        assert gold_row["required_citations"]
        assert gold_row["reference_answer"]


def test_finance_no_investment_advice_or_private_paths() -> None:
    summary = _generate_vertical("finance")
    prompts = _read_jsonl(ROOT / summary["prompts_path"])
    gold = _read_jsonl(ROOT / summary["gold_path"])
    kb = _read_jsonl(ROOT / summary["kb_path"])
    combined_text = json.dumps([prompts, gold, kb], ensure_ascii=True).lower()
    assert "c:\\\\users" not in combined_text
    assert "/home/" not in combined_text
    assert "kparo" not in combined_text
    assert "akpoogaga" not in combined_text
    assert "local_text_path" not in combined_text
    assert "investment advice" not in combined_text

    answer_text = json.dumps(
        [prompt["question"] for prompt in prompts if prompt["expected_status"] == "answer"]
        + [row["reference_answer"] for row in gold if row["expected_status"] == "answer"],
        ensure_ascii=True,
    ).lower()
    assert "recommend buying" not in answer_text
    assert "recommend selling" not in answer_text
    assert "should invest" not in answer_text
    assert "price target" not in answer_text


def test_finance_gold_reference_answers_not_mechanical() -> None:
    summary = _generate_vertical("finance")
    gold = _read_jsonl(ROOT / summary["gold_path"])
    blocked_phrases = [
        "Use cited Finance evidence",
        "to answer about",
        "Keep the answer limited to SEC filing, XBRL, or filing-event evidence",
    ]

    for gold_row in gold:
        reference_answer = str(gold_row["reference_answer"])
        for phrase in blocked_phrases:
            assert phrase not in reference_answer


def test_finance_gold_reference_answers_remain_safe() -> None:
    summary = _generate_vertical("finance")
    gold = _read_jsonl(ROOT / summary["gold_path"])
    required_exclusions = {
        "investment recommendation",
        "price target",
        "fabricated citations",
        "unsupported financial claims",
    }

    for gold_row in gold:
        assert required_exclusions.issubset(set(gold_row["must_not_include"]))


def test_finance_generation_still_preserves_counts() -> None:
    summary = _generate_vertical("finance")
    prompts = _read_jsonl(ROOT / summary["prompts_path"])
    gold = _read_jsonl(ROOT / summary["gold_path"])

    assert len(prompts) == 250
    assert len(gold) == 250
    assert Counter(prompt["expected_status"] for prompt in prompts) == {
        "answer": 230,
        "insufficient_evidence": 10,
        "escalate": 10,
    }
    assert Counter(prompt["expected_output_format"] for prompt in prompts) == {
        "text": 155,
        "json": 50,
        "markdown_table": 45,
    }


def test_finance_linguistic_variation() -> None:
    summary = _generate_vertical("finance")
    report = json.loads((ROOT / summary["report_path"]).read_text(encoding="utf-8"))

    assert report["linguistic_variation_rate"] >= 0.60
    assert report["most_common_question_template_share"] <= 0.40
    assert report["most_common_question_template_count"] <= 100
    assert report["unique_question_template_count"] > 1
    assert not [
        issue
        for issue in report["validation_issues"]
        if str(issue).startswith("linguistic_variation_warning")
    ]


def test_finance_linguistic_variation_still_passes() -> None:
    summary = _generate_vertical("finance")
    report = json.loads((ROOT / summary["report_path"]).read_text(encoding="utf-8"))

    assert report["linguistic_variation_rate"] >= 0.60
    assert report["critical_issue_count"] == 0
    assert report["warning_count"] == 0


def test_airline_pilot_generation_cli() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
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
    summary = json.loads(result.stdout)
    prompts_path = ROOT / summary["prompts_path"]
    gold_path = ROOT / summary["gold_path"]
    kb_path = ROOT / summary["kb_path"]
    assert prompts_path.exists()
    assert gold_path.exists()
    assert kb_path.exists()

    prompts = _read_jsonl(prompts_path)
    gold = _read_jsonl(gold_path)
    kb = _read_jsonl(kb_path)
    assert len(prompts) == 250
    assert len(gold) == 250
    assert summary["status_counts"] == {"answer": 225, "escalate": 20, "spam_or_fraud": 5}

    module = _load_module()
    assert module.validate_prompt_gold_alignment(prompts, gold) == []
    assert module.validate_evidence_coverage(gold, kb) == []


def test_airline_250_linguistic_variation() -> None:
    summary = _generate_vertical("airline")
    report = json.loads((ROOT / summary["report_path"]).read_text(encoding="utf-8"))

    assert report["linguistic_variation_rate"] >= 0.60
    assert report["most_common_question_template_share"] <= 0.40
    assert report["most_common_question_template_count"] <= 100
    assert report["unique_question_template_count"] > 1
    assert not [
        issue
        for issue in report["validation_issues"]
        if str(issue).startswith("linguistic_variation_warning")
    ]


def test_healthcare_generation_cli() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--generate-vertical",
            "--vertical",
            "healthcare_admin",
            "--target-per-vertical",
            "250",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["vertical"] == "healthcare_admin"
    assert summary["prompt_count"] == 250
    assert summary["gold_count"] == 250
    assert summary["kb_count"] >= 25
    assert (ROOT / summary["report_path"]).exists()


def test_healthcare_generated_counts_and_alignment() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--generate-vertical",
            "--vertical",
            "healthcare_admin",
            "--target-per-vertical",
            "250",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    prompts = _read_jsonl(ROOT / summary["prompts_path"])
    gold = _read_jsonl(ROOT / summary["gold_path"])
    kb = _read_jsonl(ROOT / summary["kb_path"])
    assert len(prompts) == 250
    assert len(gold) == 250
    assert prompts[0]["prompt_id"] == "healthcare_admin_scaleup_250_0001"
    assert prompts[-1]["prompt_id"] == "healthcare_admin_scaleup_250_0250"

    module = _load_module()
    assert module.validate_prompt_gold_alignment(prompts, gold) == []
    assert module.validate_evidence_coverage(gold, kb) == []
    assert module.validate_no_private_hygiene_terms(prompts + gold + kb) == []


def test_healthcare_250_linguistic_variation() -> None:
    summary = _generate_vertical("healthcare_admin")
    report = json.loads((ROOT / summary["report_path"]).read_text(encoding="utf-8"))

    assert report["linguistic_variation_rate"] >= 0.60
    assert report["most_common_question_template_share"] <= 0.40
    assert report["most_common_question_template_count"] <= 100
    assert report["unique_question_template_count"] > 1
    assert not [
        issue
        for issue in report["validation_issues"]
        if str(issue).startswith("linguistic_variation_warning")
    ]


def test_healthcare_negative_and_safety_statuses_present() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--generate-vertical",
            "--vertical",
            "healthcare_admin",
            "--target-per-vertical",
            "250",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    prompts = _read_jsonl(ROOT / summary["prompts_path"])
    gold = _read_jsonl(ROOT / summary["gold_path"])
    status_counts: dict[str, int] = {}
    for prompt in prompts:
        status = str(prompt["expected_status"])
        status_counts[status] = status_counts.get(status, 0) + 1

    assert status_counts == {
        "answer": 220,
        "escalate": 20,
        "safety_boundary": 5,
        "spam_or_fraud": 3,
        "out_of_scope": 2,
    }
    assert all(
        "medical advice" in gold_row["must_not_include"]
        for gold_row in gold
        if gold_row["expected_status"] in {"safety_boundary", "escalate"}
    )
    assert any("urgent clinical boundary" in prompt["question"] for prompt in prompts)


def test_retail_generation_cli() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--generate-vertical",
            "--vertical",
            "retail",
            "--target-per-vertical",
            "250",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["vertical"] == "retail"
    assert summary["prompt_count"] == 250
    assert summary["gold_count"] == 250
    assert summary["kb_count"] >= 25
    assert (ROOT / summary["report_path"]).exists()


def test_retail_generated_counts_and_alignment() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--generate-vertical",
            "--vertical",
            "retail",
            "--target-per-vertical",
            "250",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    prompts = _read_jsonl(ROOT / summary["prompts_path"])
    gold = _read_jsonl(ROOT / summary["gold_path"])
    kb = _read_jsonl(ROOT / summary["kb_path"])
    assert len(prompts) == 250
    assert len(gold) == 250
    assert prompts[0]["prompt_id"] == "retail_scaleup_250_0001"
    assert prompts[-1]["prompt_id"] == "retail_scaleup_250_0250"

    module = _load_module()
    assert module.validate_prompt_gold_alignment(prompts, gold) == []
    assert module.validate_evidence_coverage(gold, kb) == []
    assert module.validate_no_private_hygiene_terms(prompts + gold + kb) == []


def test_retail_250_linguistic_variation() -> None:
    summary = _generate_vertical("retail")
    report = json.loads((ROOT / summary["report_path"]).read_text(encoding="utf-8"))

    assert report["linguistic_variation_rate"] >= 0.60
    assert report["most_common_question_template_share"] <= 0.40
    assert report["most_common_question_template_count"] <= 100
    assert report["unique_question_template_count"] > 1
    assert not [
        issue
        for issue in report["validation_issues"]
        if str(issue).startswith("linguistic_variation_warning")
    ]


def test_retail_negative_statuses_present() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--generate-vertical",
            "--vertical",
            "retail",
            "--target-per-vertical",
            "250",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    prompts = _read_jsonl(ROOT / summary["prompts_path"])
    status_counts: dict[str, int] = {}
    for prompt in prompts:
        status = str(prompt["expected_status"])
        status_counts[status] = status_counts.get(status, 0) + 1

    assert status_counts == {
        "answer": 222,
        "insufficient_evidence": 9,
        "escalate": 9,
        "spam_or_low_quality": 7,
        "out_of_scope": 3,
    }


def test_retail_no_raw_user_ids_or_generic_titles() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--generate-vertical",
            "--vertical",
            "retail",
            "--target-per-vertical",
            "250",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    prompts = _read_jsonl(ROOT / summary["prompts_path"])
    gold = _read_jsonl(ROOT / summary["gold_path"])
    kb = _read_jsonl(ROOT / summary["kb_path"])
    combined_text = json.dumps([prompts, gold, kb], ensure_ascii=True)
    assert "raw user_id" not in combined_text.lower()
    assert "user_id" not in combined_text.lower()
    assert "All_Beauty product B" not in combined_text
    assert "Retail product B" not in combined_text
    assert "synthetic benchmark policy, not Amazon policy" in combined_text


def test_research_ai_generation_cli() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--generate-vertical",
            "--vertical",
            "research_ai",
            "--target-per-vertical",
            "250",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["vertical"] == "research_ai"
    assert summary["prompt_count"] == 250
    assert summary["gold_count"] == 250
    assert summary["kb_count"] >= 141
    assert summary["critical_issue_count"] == 0
    assert (ROOT / summary["prompts_path"]).exists()
    assert (ROOT / summary["gold_path"]).exists()
    assert (ROOT / summary["kb_path"]).exists()
    report = json.loads((ROOT / summary["report_path"]).read_text(encoding="utf-8"))
    assert report["vertical"] == "research_ai"
    assert report["prompt_count"] == 250
    assert report["gold_count"] == 250


def test_research_ai_generated_counts_and_alignment() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--generate-vertical",
            "--vertical",
            "research_ai",
            "--target-per-vertical",
            "250",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    prompts = _read_jsonl(ROOT / summary["prompts_path"])
    gold = _read_jsonl(ROOT / summary["gold_path"])
    kb = _read_jsonl(ROOT / summary["kb_path"])
    assert len(prompts) == 250
    assert len(gold) == 250
    assert prompts[0]["prompt_id"] == "research_ai_scaleup_250_0001"
    assert prompts[-1]["prompt_id"] == "research_ai_scaleup_250_0250"

    module = _load_module()
    assert module.validate_prompt_gold_alignment(prompts, gold) == []
    assert module.validate_evidence_coverage(gold, kb) == []
    assert module.validate_no_private_hygiene_terms(prompts + gold + kb) == []


def test_research_ai_250_linguistic_variation() -> None:
    summary = _generate_vertical("research_ai")
    report = json.loads((ROOT / summary["report_path"]).read_text(encoding="utf-8"))

    assert report["linguistic_variation_rate"] >= 0.60
    assert report["most_common_question_template_share"] <= 0.40
    assert report["most_common_question_template_count"] <= 100
    assert report["unique_question_template_count"] > 1
    assert not [
        issue
        for issue in report["validation_issues"]
        if str(issue).startswith("linguistic_variation_warning")
    ]


def test_research_ai_negative_statuses_present() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--generate-vertical",
            "--vertical",
            "research_ai",
            "--target-per-vertical",
            "250",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    prompts = _read_jsonl(ROOT / summary["prompts_path"])
    gold = _read_jsonl(ROOT / summary["gold_path"])
    status_counts: dict[str, int] = {}
    for prompt in prompts:
        status = str(prompt["expected_status"])
        status_counts[status] = status_counts.get(status, 0) + 1

    assert status_counts == {
        "answer": 225,
        "insufficient_evidence": 10,
        "escalate": 10,
        "out_of_scope": 5,
    }
    assert {
        "insufficient_evidence",
        "escalate",
        "out_of_scope",
    }.issubset(status_counts)
    assert all(
        gold_row["must_not_include"] for gold_row in gold if gold_row["expected_status"] != "answer"
    )


def test_research_ai_answerable_gold_has_evidence() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--generate-vertical",
            "--vertical",
            "research_ai",
            "--target-per-vertical",
            "250",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    gold = _read_jsonl(ROOT / summary["gold_path"])
    for gold_row in gold:
        if gold_row["expected_status"] != "answer":
            continue
        assert gold_row["required_doc_ids"]
        assert gold_row["required_chunk_ids"]
        assert gold_row["required_citations"]
        assert gold_row["reference_answer"]


def test_research_ai_no_private_paths_or_general_memory() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--generate-vertical",
            "--vertical",
            "research_ai",
            "--target-per-vertical",
            "250",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    prompts = _read_jsonl(ROOT / summary["prompts_path"])
    gold = _read_jsonl(ROOT / summary["gold_path"])
    kb = _read_jsonl(ROOT / summary["kb_path"])
    combined_text = json.dumps([prompts, gold, kb], ensure_ascii=True).lower()
    assert "c:\\\\users" not in combined_text
    assert "/home/" not in combined_text
    assert "kparo" not in combined_text
    assert "akpoogaga" not in combined_text
    assert "local_text_path" not in combined_text
    assert "general model memory" not in combined_text


def test_scaleup_reports_include_linguistic_metrics() -> None:
    required_fields = {
        "linguistic_variation_rate",
        "most_common_question_template_count",
        "most_common_question_template_share",
        "unique_question_template_count",
    }
    for vertical in ["airline", "healthcare_admin", "retail", "research_ai", "finance"]:
        summary = _generate_vertical(vertical)
        report = json.loads((ROOT / summary["report_path"]).read_text(encoding="utf-8"))
        assert required_fields.issubset(report)
        assert report["linguistic_variation_rate"] >= 0.60


def test_generated_outputs_are_ignored() -> None:
    gitignore = GITIGNORE_PATH.read_text(encoding="utf-8")

    assert "data/generated/phase2a/scaleup/" in gitignore
    assert "data/generated/phase2a/scaleup_reports/" in gitignore


def test_docs_include_commands() -> None:
    docs = DOC_PATH.read_text(encoding="utf-8")

    assert "--dry-run --target-per-vertical 250" in docs
    assert "--dry-run --target-per-vertical 2000" in docs
    assert "--generate-plan --target-per-vertical 2000" in docs
    assert "--generate-vertical --vertical airline --target-per-vertical 250" in docs
    assert "--generate-vertical --vertical finance --target-per-vertical 250" in docs
    assert "--generate-vertical --vertical research_ai --target-per-vertical 250" in docs
    assert "no RAG" in docs


def test_docs_include_5000_per_vertical() -> None:
    docs = DOC_PATH.read_text(encoding="utf-8")

    assert "5,000" in docs
    assert "25,000" in docs
    assert "Maximum expanded capacity" in docs

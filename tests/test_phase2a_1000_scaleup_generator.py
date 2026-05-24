import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/phase2/generate_phase2a_scaleup.py"
DOC_PATH = ROOT / "docs/45_phase2a_1000_scaleup_generator.md"


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
            "1000",
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


def test_airline_1000_generation_cli() -> None:
    summary = _generate_vertical("airline")
    report = json.loads((ROOT / summary["report_path"]).read_text(encoding="utf-8"))

    assert summary["vertical"] == "airline"
    assert summary["prompt_count"] == 1000
    assert summary["gold_count"] == 1000
    assert summary["critical_issue_count"] == 0
    assert (ROOT / summary["prompts_path"]).exists()
    assert (ROOT / summary["gold_path"]).exists()
    assert (ROOT / summary["kb_path"]).exists()
    assert report["target_per_vertical"] == 1000
    assert report["checkpoint"] == "checkpoint_1000"
    assert report["generation_scope"] == "local_candidate_generation"
    assert report["promotion_required_before_next_checkpoint"] == "checkpoint_2000"


def test_healthcare_1000_generation_cli() -> None:
    summary = _generate_vertical("healthcare_admin")
    report = json.loads((ROOT / summary["report_path"]).read_text(encoding="utf-8"))

    assert summary["vertical"] == "healthcare_admin"
    assert summary["prompt_count"] == 1000
    assert summary["gold_count"] == 1000
    assert summary["critical_issue_count"] == 0
    assert (ROOT / summary["prompts_path"]).exists()
    assert (ROOT / summary["gold_path"]).exists()
    assert (ROOT / summary["kb_path"]).exists()
    assert report["target_per_vertical"] == 1000
    assert report["checkpoint"] == "checkpoint_1000"
    assert report["generation_scope"] == "local_candidate_generation"
    assert report["promotion_required_before_next_checkpoint"] == "checkpoint_2000"


def test_retail_1000_generation_cli() -> None:
    summary = _generate_vertical("retail")
    report = json.loads((ROOT / summary["report_path"]).read_text(encoding="utf-8"))

    assert summary["vertical"] == "retail"
    assert summary["prompt_count"] == 1000
    assert summary["gold_count"] == 1000
    assert summary["critical_issue_count"] == 0
    assert summary["warning_count"] == 0
    assert (ROOT / summary["prompts_path"]).exists()
    assert (ROOT / summary["gold_path"]).exists()
    assert (ROOT / summary["kb_path"]).exists()
    assert report["target_per_vertical"] == 1000
    assert report["checkpoint"] == "checkpoint_1000"
    assert report["generation_scope"] == "local_candidate_generation"


def test_finance_1000_generation_cli() -> None:
    summary = _generate_vertical("finance")
    report = json.loads((ROOT / summary["report_path"]).read_text(encoding="utf-8"))

    assert summary["vertical"] == "finance"
    assert summary["prompt_count"] == 1000
    assert summary["gold_count"] == 1000
    assert summary["critical_issue_count"] == 0
    assert summary["warning_count"] == 0
    assert (ROOT / summary["prompts_path"]).exists()
    assert (ROOT / summary["gold_path"]).exists()
    assert (ROOT / summary["kb_path"]).exists()
    assert report["target_per_vertical"] == 1000
    assert report["checkpoint"] == "checkpoint_1000"
    assert report["generation_scope"] == "local_candidate_generation"


def test_research_ai_1000_generation_cli() -> None:
    summary = _generate_vertical("research_ai")
    report = json.loads((ROOT / summary["report_path"]).read_text(encoding="utf-8"))

    assert summary["vertical"] == "research_ai"
    assert summary["prompt_count"] == 1000
    assert summary["gold_count"] == 1000
    assert summary["critical_issue_count"] == 0
    assert summary["warning_count"] == 0
    assert (ROOT / summary["prompts_path"]).exists()
    assert (ROOT / summary["gold_path"]).exists()
    assert (ROOT / summary["kb_path"]).exists()
    assert report["target_per_vertical"] == 1000
    assert report["checkpoint"] == "checkpoint_1000"
    assert report["generation_scope"] == "local_candidate_generation"


def test_airline_1000_counts_and_distribution() -> None:
    summary = _generate_vertical("airline")
    prompts = _read_jsonl(ROOT / summary["prompts_path"])
    gold = _read_jsonl(ROOT / summary["gold_path"])
    kb = _read_jsonl(ROOT / summary["kb_path"])

    assert len(prompts) == 1000
    assert len(gold) == 1000
    assert prompts[0]["prompt_id"] == "airline_scaleup_1000_0001"
    assert prompts[-1]["prompt_id"] == "airline_scaleup_1000_1000"
    assert Counter(prompt["expected_status"] for prompt in prompts) == {
        "answer": 900,
        "escalate": 80,
        "spam_or_fraud": 20,
    }
    assert Counter(prompt["expected_output_format"] for prompt in prompts) == {
        "text": 760,
        "json": 140,
        "markdown_table": 100,
    }
    assert {"ticket_change", "cancellation_refund", "baggage_delay"}.issubset(
        {str(prompt["support_type"]) for prompt in prompts}
    )
    module = _load_module()
    assert module.validate_prompt_gold_alignment(prompts, gold) == []
    assert module.validate_evidence_coverage(gold, kb) == []


def test_healthcare_1000_counts_and_distribution() -> None:
    summary = _generate_vertical("healthcare_admin")
    prompts = _read_jsonl(ROOT / summary["prompts_path"])
    gold = _read_jsonl(ROOT / summary["gold_path"])
    kb = _read_jsonl(ROOT / summary["kb_path"])

    assert len(prompts) == 1000
    assert len(gold) == 1000
    assert prompts[0]["prompt_id"] == "healthcare_admin_scaleup_1000_0001"
    assert prompts[-1]["prompt_id"] == "healthcare_admin_scaleup_1000_1000"
    assert Counter(prompt["expected_status"] for prompt in prompts) == {
        "answer": 880,
        "escalate": 80,
        "safety_boundary": 20,
        "spam_or_fraud": 10,
        "out_of_scope": 10,
    }
    assert Counter(prompt["expected_output_format"] for prompt in prompts) == {
        "text": 780,
        "json": 140,
        "markdown_table": 80,
    }
    assert {
        "appointment_booking",
        "billing_question",
        "insurance_verification",
        "medical_records_request",
        "prior_authorization_status",
        "portal_access",
        "telehealth_setup",
    }.issubset({str(prompt["support_type"]) for prompt in prompts})
    module = _load_module()
    assert module.validate_prompt_gold_alignment(prompts, gold) == []
    assert module.validate_evidence_coverage(gold, kb) == []


def test_retail_1000_counts_and_distribution() -> None:
    summary = _generate_vertical("retail")
    prompts = _read_jsonl(ROOT / summary["prompts_path"])
    gold = _read_jsonl(ROOT / summary["gold_path"])
    kb = _read_jsonl(ROOT / summary["kb_path"])

    assert len(prompts) == 1000
    assert len(gold) == 1000
    assert prompts[0]["prompt_id"] == "retail_scaleup_1000_0001"
    assert prompts[-1]["prompt_id"] == "retail_scaleup_1000_1000"
    assert Counter(prompt["expected_status"] for prompt in prompts) == {
        "answer": 890,
        "insufficient_evidence": 35,
        "escalate": 35,
        "spam_or_low_quality": 30,
        "out_of_scope": 10,
    }
    assert Counter(prompt["expected_output_format"] for prompt in prompts) == {
        "text": 740,
        "json": 160,
        "markdown_table": 100,
    }
    assert {
        "answer_grounded",
        "issue_identification",
        "compare_products",
        "extract_structured",
        "policy_reasoning",
        "quality_boundary",
        "escalation_response",
    }.issubset({str(prompt["task_type"]) for prompt in prompts})
    module = _load_module()
    assert module.validate_prompt_gold_alignment(prompts, gold) == []
    assert module.validate_evidence_coverage(gold, kb) == []


def test_finance_1000_counts_and_distribution() -> None:
    summary = _generate_vertical("finance")
    prompts = _read_jsonl(ROOT / summary["prompts_path"])
    gold = _read_jsonl(ROOT / summary["gold_path"])
    kb = _read_jsonl(ROOT / summary["kb_path"])

    assert len(prompts) == 1000
    assert len(gold) == 1000
    assert prompts[0]["prompt_id"] == "finance_scaleup_1000_0001"
    assert prompts[-1]["prompt_id"] == "finance_scaleup_1000_1000"
    assert Counter(prompt["expected_status"] for prompt in prompts) == {
        "answer": 920,
        "insufficient_evidence": 40,
        "escalate": 40,
    }
    assert Counter(prompt["expected_output_format"] for prompt in prompts) == {
        "text": 620,
        "json": 200,
        "markdown_table": 180,
    }
    assert {
        "answer_grounded",
        "calculation",
        "compare_filings",
        "extract_structured",
        "evidence_citation_lookup",
        "escalation_response",
    }.issubset({str(prompt["task_type"]) for prompt in prompts})
    module = _load_module()
    assert module.validate_prompt_gold_alignment(prompts, gold) == []
    assert module.validate_evidence_coverage(gold, kb) == []


def test_research_ai_1000_counts_distribution_and_kb_target() -> None:
    summary = _generate_vertical("research_ai")
    prompts = _read_jsonl(ROOT / summary["prompts_path"])
    gold = _read_jsonl(ROOT / summary["gold_path"])
    kb = _read_jsonl(ROOT / summary["kb_path"])

    assert len(prompts) == 1000
    assert len(gold) == 1000
    assert 800 <= len(kb) <= 1200
    assert prompts[0]["prompt_id"] == "research_ai_scaleup_1000_0001"
    assert prompts[-1]["prompt_id"] == "research_ai_scaleup_1000_1000"
    assert Counter(prompt["expected_status"] for prompt in prompts) == {
        "answer": 900,
        "insufficient_evidence": 40,
        "escalate": 40,
        "out_of_scope": 20,
    }
    assert Counter(prompt["expected_output_format"] for prompt in prompts) == {
        "text": 720,
        "json": 140,
        "markdown_table": 140,
    }
    assert {
        "answer_grounded",
        "paper_method",
        "results_evaluation",
        "extract_structured",
        "compare_papers",
        "literature_table",
        "escalation_response",
    }.issubset({str(prompt["task_type"]) for prompt in prompts})
    assert len({paper_id for prompt in prompts for paper_id in prompt["source_paper_ids"]}) >= 40
    module = _load_module()
    assert module.validate_prompt_gold_alignment(prompts, gold) == []
    assert module.validate_evidence_coverage(gold, kb) == []
    assert module.validate_no_private_hygiene_terms(prompts + gold + kb) == []


def test_airline_healthcare_1000_kb_targets() -> None:
    module = _load_module()
    for vertical in ["airline", "healthcare_admin"]:
        summary = _generate_vertical(vertical)
        prompts = _read_jsonl(ROOT / summary["prompts_path"])
        gold = _read_jsonl(ROOT / summary["gold_path"])
        kb = _read_jsonl(ROOT / summary["kb_path"])
        assert 150 <= len(kb) <= 250
        assert summary["kb_count"] == len(kb)
        assert module.validate_evidence_coverage(gold, kb) == []
        assert module.validate_no_private_hygiene_terms(prompts + gold + kb) == []


def test_retail_1000_kb_target_range() -> None:
    summary = _generate_vertical("retail")
    prompts = _read_jsonl(ROOT / summary["prompts_path"])
    gold = _read_jsonl(ROOT / summary["gold_path"])
    kb = _read_jsonl(ROOT / summary["kb_path"])

    module = _load_module()
    assert 500 <= len(kb) <= 1000
    assert summary["kb_count"] == len(kb)
    assert module.validate_evidence_coverage(gold, kb) == []
    assert module.validate_no_private_hygiene_terms(prompts + gold + kb) == []


def test_finance_1000_kb_target_range() -> None:
    summary = _generate_vertical("finance")
    prompts = _read_jsonl(ROOT / summary["prompts_path"])
    gold = _read_jsonl(ROOT / summary["gold_path"])
    kb = _read_jsonl(ROOT / summary["kb_path"])

    module = _load_module()
    assert 800 <= len(kb) <= 1200
    assert summary["kb_count"] == len(kb)
    assert module.validate_evidence_coverage(gold, kb) == []
    assert module.validate_no_private_hygiene_terms(prompts + gold + kb) == []


def test_finance_1000_committed_kb_fallback_without_sec_artifacts() -> None:
    module = _load_module()
    original_section_loader = module.build_finance_section_kb_rows_from_manifest
    original_xbrl_loader = module.build_finance_xbrl_inventory_kb_rows
    module.build_finance_section_kb_rows_from_manifest = lambda: []
    module.build_finance_xbrl_inventory_kb_rows = lambda limit: []
    try:
        seed_kb = _read_jsonl(ROOT / "data/kb/finance/kb_sample.jsonl")
        expanded = module.expand_finance_kb_rows(seed_kb, target_kb_count=800)
    finally:
        module.build_finance_section_kb_rows_from_manifest = original_section_loader
        module.build_finance_xbrl_inventory_kb_rows = original_xbrl_loader

    assert 800 <= len(expanded) <= 1200


def test_research_ai_1000_committed_kb_fallback_without_processed_sections() -> None:
    module = _load_module()
    original_section_loader = module.build_research_ai_section_kb_rows_from_manifest
    module.build_research_ai_section_kb_rows_from_manifest = lambda limit: []
    try:
        seed_kb = _read_jsonl(ROOT / "data/kb/research_ai/kb_sample.jsonl")
        expanded = module.expand_research_ai_kb_rows(seed_kb, target_per_vertical=1000)
    finally:
        module.build_research_ai_section_kb_rows_from_manifest = original_section_loader

    assert 800 <= len(expanded) <= 1200
    assert len(module.research_ai_contexts_by_paper(expanded)) >= 40


def test_retail_1000_no_raw_user_ids_or_generic_titles() -> None:
    summary = _generate_vertical("retail")
    prompts = _read_jsonl(ROOT / summary["prompts_path"])
    gold = _read_jsonl(ROOT / summary["gold_path"])
    kb = _read_jsonl(ROOT / summary["kb_path"])

    combined = json.dumps([prompts, gold, kb], ensure_ascii=True)
    assert "user_id" not in combined
    assert "user_id_hash" not in combined
    assert "All_Beauty product <ASIN>" not in combined
    assert not [
        prompt
        for prompt in prompts
        if str(prompt.get("product_title", "")).startswith("All_Beauty product")
    ]


def test_finance_1000_no_investment_advice() -> None:
    summary = _generate_vertical("finance")
    prompts = _read_jsonl(ROOT / summary["prompts_path"])
    gold = _read_jsonl(ROOT / summary["gold_path"])
    kb = _read_jsonl(ROOT / summary["kb_path"])

    combined = json.dumps([prompts, gold, kb], ensure_ascii=True).lower()
    assert "c:\\\\users" not in combined
    assert "/home/" not in combined
    assert "local_text_path" not in combined
    assert "investment advice" not in combined

    answer_text = json.dumps(
        [prompt["question"] for prompt in prompts] + [row["reference_answer"] for row in gold],
        ensure_ascii=True,
    ).lower()
    assert "recommend buying" not in answer_text
    assert "recommend selling" not in answer_text
    assert "buy this stock" not in answer_text
    assert "sell this stock" not in answer_text
    assert "hold this stock" not in answer_text
    assert "price target" not in answer_text


def test_research_ai_1000_no_fabricated_claims_or_private_paths() -> None:
    summary = _generate_vertical("research_ai")
    prompts = _read_jsonl(ROOT / summary["prompts_path"])
    gold = _read_jsonl(ROOT / summary["gold_path"])
    kb = _read_jsonl(ROOT / summary["kb_path"])

    combined = json.dumps([prompts, gold, kb], ensure_ascii=True).lower()
    assert "c:\\\\users" not in combined
    assert "/home/" not in combined
    assert "local_text_path" not in combined
    assert "full paper text dump" not in combined
    assert "according to my knowledge" not in combined
    assert "fabricated paper claim" not in combined
    for row in gold:
        if row["expected_status"] == "answer":
            assert row["required_doc_ids"]
            assert row["required_chunk_ids"]
            assert row["required_citations"]


def test_airline_healthcare_1000_linguistic_variation() -> None:
    for vertical in ["airline", "healthcare_admin"]:
        summary = _generate_vertical(vertical)
        report = json.loads((ROOT / summary["report_path"]).read_text(encoding="utf-8"))
        assert report["linguistic_variation_rate"] >= 0.60
        assert report["most_common_question_template_share"] <= 0.40
        assert report["warning_count"] == 0


def test_research_ai_1000_linguistic_variation() -> None:
    summary = _generate_vertical("research_ai")
    report = json.loads((ROOT / summary["report_path"]).read_text(encoding="utf-8"))
    assert report["linguistic_variation_rate"] >= 0.60
    assert report["most_common_question_template_share"] <= 0.40
    assert report["warning_count"] == 0


def test_retail_finance_1000_linguistic_variation() -> None:
    for vertical in ["retail", "finance"]:
        summary = _generate_vertical(vertical)
        report = json.loads((ROOT / summary["report_path"]).read_text(encoding="utf-8"))
        assert report["linguistic_variation_rate"] >= 0.60
        assert report["most_common_question_template_share"] <= 0.40
        assert report["warning_count"] == 0


def test_generate_plan_marks_all_1000_generators_implemented() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--generate-plan",
            "--target-per-vertical",
            "1000",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    for vertical in ["retail", "finance", "research_ai"]:
        manifest = summary["verticals"][vertical]
        assert manifest["generation_implemented"] is True
        assert manifest["generation_scope"] == "local_candidate_generation"
        assert manifest["ready_for_actual_generation"] is True


def test_large_generation_still_blocks_unimplemented_targets() -> None:
    for vertical in ["airline", "healthcare_admin", "retail", "finance", "research_ai"]:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--generate-vertical",
                "--vertical",
                vertical,
                "--target-per-vertical",
                "2000",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert f"Generation for {vertical} at 2000 requires explicit implementation" in (
            result.stderr
        )


def test_docs_include_1000_generation_commands() -> None:
    docs = DOC_PATH.read_text(encoding="utf-8")

    assert "Phase 2A-13A" in docs
    assert "--generate-vertical --vertical airline --target-per-vertical 1000" in docs
    assert "--generate-vertical --vertical healthcare_admin --target-per-vertical 1000" in docs
    assert "--generate-vertical --vertical retail --target-per-vertical 1000" in docs
    assert "--generate-vertical --vertical finance --target-per-vertical 1000" in docs
    assert "--generate-vertical --vertical research_ai --target-per-vertical 1000" in docs
    assert "full 5,000-record" in docs
    assert "no RAG" in docs

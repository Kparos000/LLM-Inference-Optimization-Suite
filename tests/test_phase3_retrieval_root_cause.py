import csv
import json
from pathlib import Path

from inference_bench.retrieval_root_cause import (
    build_retrieval_root_cause_report,
    classify_failure,
)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def base_retrieval_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "split": "final_10000",
        "ablation_mode": "prompt_text_only",
        "memory_mode": "mm2_hybrid_top5",
        "vertical": "finance",
        "record_count": 10,
        "failure_count": 8,
        "recall_at_5": 0.2,
        "mrr": 0.1,
        "candidate_recall_at_100": 0.8,
        "candidate_recall_at_50": 0.8,
        "candidate_recall_at_20": 0.7,
        "candidate_recall_at_10": 0.5,
        "gold_absent_from_top100_rate": 0.2,
        "gold_in_top100_but_not_top5_rate": 0.6,
        "gold_in_top50_but_not_top5_rate": 0.6,
        "source_hints_used": False,
        "query_enrichment_used": True,
        "reranking_used": True,
    }
    row.update(overrides)
    return row


def test_root_cause_classifier_detects_prompt_missing_metric() -> None:
    classification = classify_failure(
        base_retrieval_row(),
        prompt_record={
            "vertical": "finance",
            "question": "Summarize the Apple Inc. 10-K filing.",
            "ticker": "AAPL",
            "filing_form": "10-K",
        },
    )

    assert classification["primary_root_cause"] == "prompt_missing_metric"


def test_root_cause_classifier_detects_gold_absent_from_candidate_pool() -> None:
    classification = classify_failure(
        base_retrieval_row(candidate_recall_at_100=0.0, gold_evidence_ids=["doc_1"]),
    )

    assert classification["primary_root_cause"] == "gold_absent_from_candidate_pool"


def test_root_cause_classifier_detects_gold_in_candidates_not_final_top5() -> None:
    classification = classify_failure(
        base_retrieval_row(candidate_recall_at_100=1.0, recall_at_5=0.0),
    )

    assert classification["primary_root_cause"] == "gold_in_candidates_not_final_top5"


def test_root_cause_classifier_detects_gold_not_in_corpus() -> None:
    classification = classify_failure(base_retrieval_row(), gold_in_corpus=False)

    assert classification["primary_root_cause"] == "gold_not_in_corpus"


def make_root_cause_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    dataset_root = tmp_path / "dataset"
    context_root = tmp_path / "context"
    output_root = tmp_path / "output"
    slo_config = tmp_path / "slo_targets.yaml"
    finance_root = dataset_root / "finance"
    prompt: dict[str, object] = {
        "prompt_id": "finance_fixture_1",
        "vertical": "finance",
        "task_type": "filing_summary",
        "question": "Summarize Apple Inc. (AAPL) 10-K filing evidence.",
        "ticker": "AAPL",
        "company": "Apple Inc.",
        "filing_form": "10-K",
    }
    gold: dict[str, object] = {
        "prompt_id": "finance_fixture_1",
        "vertical": "finance",
        "required_doc_ids": ["finance_doc_1"],
    }
    kb: dict[str, object] = {
        "doc_id": "finance_doc_1",
        "vertical": "finance",
        "metadata": {"ticker": "AAPL"},
    }
    write_jsonl(finance_root / "finance_prompts_2000.jsonl", [prompt])
    write_jsonl(finance_root / "finance_gold_2000.jsonl", [gold])
    write_jsonl(finance_root / "finance_kb_2000.jsonl", [kb])
    summary_row = base_retrieval_row()
    diagnostic_row = {**summary_row, "top_failure_reasons": "{}"}
    write_csv(context_root / "retrieval_evaluation_summary.csv", [summary_row])
    write_csv(context_root / "retrieval_diagnostic_summary.csv", [diagnostic_row])
    write_json(
        context_root / "retrieval_diagnostic_report.json",
        {
            "sample_failure_examples": [
                {
                    "prompt_id": "finance_fixture_1",
                    "vertical": "finance",
                    "ablation_mode": "prompt_text_only",
                    "memory_mode": "mm2_hybrid_top5",
                    "recall_at_5": 0.0,
                    "candidate_recall_at_100": 1.0,
                    "candidate_recall_at_50": 1.0,
                    "candidate_recall_at_20": 1.0,
                    "candidate_recall_at_10": 0.0,
                    "gold_evidence_ids": ["finance_doc_1"],
                }
            ],
            "finance_specific": {"failure_examples_by_ablation": {}},
        },
    )
    write_json(
        context_root / "gold_evidence_audit_report.json",
        {
            "by_vertical": {
                "finance": {
                    "prompt_missing_metric_count": 9,
                    "prompt_missing_period_count": 10,
                    "gold_not_in_corpus_count": 0,
                }
            }
        },
    )
    write_json(
        context_root / "evidence_selection_report.json", {"contract_excludes_gold_labels": True}
    )
    write_json(context_root / "reranker_calibration_report.json", {"reranker_backend": "fixture"})
    write_json(context_root / "corpus_build_report.json", {"all_context_records_validated": True})
    write_json(context_root / "corpus_registry.json", {"entries": []})
    slo_config.write_text(
        "\n".join(
            [
                "retrieval:",
                "  overall_prompt_text_only_hybrid_recall_at_5: 0.70",
                "  finance_prompt_text_only_hybrid_recall_at_5: 0.65",
            ]
        ),
        encoding="utf-8",
    )
    return dataset_root, context_root, slo_config, output_root


def test_root_cause_report_produces_vertical_summaries_and_recommendations(
    tmp_path: Path,
) -> None:
    dataset_root, context_root, slo_config, output_root = make_root_cause_fixture(tmp_path)

    report, summary_rows, examples = build_retrieval_root_cause_report(
        dataset_root=dataset_root,
        context_root=context_root,
        slo_config=slo_config,
        output_root=output_root,
    )

    assert report["no_model_inference_triggered"] is True
    assert report["no_gpu_work_triggered"] is True
    assert report["no_paid_api_call_triggered"] is True
    assert report["by_vertical"]["finance"]["recommended_fix_area"]
    assert summary_rows[0]["primary_root_cause"] == "prompt_missing_period"
    assert summary_rows[0]["prompt_gold_repair_required"] is True
    assert examples
    assert (output_root / "retrieval_failure_examples.jsonl").exists()
    assert (output_root / "retrieval_root_cause_report.json").exists()
    assert (output_root / "retrieval_root_cause_summary.csv").exists()

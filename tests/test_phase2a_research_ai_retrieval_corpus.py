import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/phase2/export_research_ai_retrieval_corpus.py"
DOC_PATH = ROOT / "docs/52_phase2a_research_ai_retrieval_corpus.md"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
    path.write_text(payload + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _create_temp_corpus_inputs(tmp_path: Path) -> dict[str, Path]:
    approved = tmp_path / "approved.jsonl"
    sections = tmp_path / "sections.jsonl"
    benchmark_kb = tmp_path / "benchmark_kb.jsonl"
    benchmark_gold = tmp_path / "benchmark_gold.jsonl"
    corpus = tmp_path / "corpus.jsonl"
    manifest = tmp_path / "manifest.json"
    mapping = tmp_path / "mapping.jsonl"
    quality = tmp_path / "quality.json"

    _write_jsonl(
        approved,
        [
            {
                "approval_status": "approved",
                "not_for_benchmark_claims": False,
                "paper_id": "paper_a",
                "pdf_url": "https://openreview.net/pdf?id=paper_a",
                "publication_year": 2025,
                "source_url": "https://openreview.net/forum?id=paper_a",
                "title": "Efficient LLM Serving with Deterministic Batching",
                "venue_or_source": "ICLR 2025",
            },
            {
                "approval_status": "approved",
                "not_for_benchmark_claims": False,
                "paper_id": "paper_b",
                "pdf_url": "https://openreview.net/pdf?id=paper_b",
                "publication_year": 2025,
                "source_url": "https://openreview.net/forum?id=paper_b",
                "title": "Long Context Evaluation for Language Models",
                "venue_or_source": "ICLR 2025",
            },
        ],
    )
    _write_jsonl(
        sections,
        [
            {
                "paper_id": "paper_a",
                "section_record_id": "paper_a_method",
                "section_title": "Method",
                "section_type": "method",
                "text": (
                    "The method section describes deterministic request batching for "
                    "language model serving. It reports the scheduling assumptions, "
                    "the evidence boundaries, and the benchmark setup in a compact way."
                ),
                "word_count": 23,
            },
            {
                "paper_id": "paper_b",
                "section_record_id": "paper_b_empty",
                "section_title": "Empty",
                "section_type": "method",
                "text": "",
                "word_count": 0,
            },
            {
                "paper_id": "paper_b",
                "section_record_id": "paper_b_refs",
                "section_title": "References",
                "section_type": "references",
                "text": "Reference list should not become retrieval context even when present.",
                "word_count": 11,
            },
            {
                "paper_id": "paper_b",
                "section_record_id": "paper_b_fragment",
                "section_title": "Fragment",
                "section_type": "method",
                "text": "Tiny fragment.",
                "word_count": 2,
            },
        ],
    )
    _write_jsonl(
        benchmark_kb,
        [
            {
                "doc_id": "research_ai_kb_0001",
                "metadata": {
                    "paper_id": "paper_a",
                    "section_record_id": "paper_a_method",
                },
            }
        ],
    )
    _write_jsonl(
        benchmark_gold,
        [
            {
                "prompt_id": "research_ai_scaleup_2000_0001",
                "required_doc_ids": ["research_ai_kb_0001"],
            }
        ],
    )
    return {
        "approved": approved,
        "sections": sections,
        "benchmark_kb": benchmark_kb,
        "benchmark_gold": benchmark_gold,
        "corpus": corpus,
        "manifest": manifest,
        "mapping": mapping,
        "quality": quality,
    }


def _run_export(tmp_path: Path) -> dict[str, Any]:
    paths = _create_temp_corpus_inputs(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--export-full-corpus",
            "--sections-manifest",
            str(paths["sections"]),
            "--approved-papers",
            str(paths["approved"]),
            "--benchmark-kb",
            str(paths["benchmark_kb"]),
            "--benchmark-gold",
            str(paths["benchmark_gold"]),
            "--output-corpus",
            str(paths["corpus"]),
            "--output-manifest",
            str(paths["manifest"]),
            "--output-mapping",
            str(paths["mapping"]),
            "--output-quality-report",
            str(paths["quality"]),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert isinstance(summary, dict)
    summary["_paths"] = paths
    return summary


def test_research_ai_retrieval_corpus_script_exists() -> None:
    assert SCRIPT_PATH.exists()


def test_export_full_corpus_cli_with_temp_sections(tmp_path: Path) -> None:
    summary = _run_export(tmp_path)

    assert summary["phase"] == "2A-16B"
    assert summary["approved_paper_count"] == 2
    assert summary["sections_loaded_count"] == 4
    assert summary["exported_corpus_count"] == 1
    assert summary["excluded_section_count"] == 3
    assert summary["retrieval_corpus_ready_for_phase2b"] is True
    assert summary["_paths"]["corpus"].exists()
    assert summary["_paths"]["manifest"].exists()
    assert summary["_paths"]["quality"].exists()


def test_export_preserves_paper_and_section_metadata(tmp_path: Path) -> None:
    summary = _run_export(tmp_path)
    row = _read_jsonl(summary["_paths"]["corpus"])[0]

    assert row["paper_id"] == "paper_a"
    assert row["paper_title"] == "Efficient LLM Serving with Deterministic Batching"
    assert row["venue_or_source"] == "ICLR 2025"
    assert row["section_id"] == "paper_a_method"
    assert row["section_type"] == "method"
    assert row["metadata"]["future_phase2b_retrieval_corpus"] is True


def test_export_excludes_empty_or_unusable_sections(tmp_path: Path) -> None:
    summary = _run_export(tmp_path)
    quality = json.loads(summary["_paths"]["quality"].read_text(encoding="utf-8"))

    assert quality["exclusion_counts"]["empty_text"] == 1
    assert quality["exclusion_counts"]["references_or_bibliography"] == 1
    assert quality["exclusion_counts"]["too_short"] == 1


def test_mapping_report_shape(tmp_path: Path) -> None:
    summary = _run_export(tmp_path)
    mapping_rows = _read_jsonl(summary["_paths"]["mapping"])

    assert mapping_rows == [
        {
            "benchmark_doc_id": "research_ai_kb_0001",
            "gold_reference_count": 1,
            "mapping_status": "mapped",
            "paper_id": "paper_a",
            "source_section_id": "paper_a_method",
        }
    ]


def test_retrieval_corpus_docs_explain_benchmark_kb_vs_full_corpus() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")

    assert "benchmark KB" in text
    assert "full retrieval corpus" in text
    assert "export_research_ai_retrieval_corpus.py --export-full-corpus" in text
    assert "no embeddings" in text.lower()

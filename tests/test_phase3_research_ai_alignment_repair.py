from typing import Any

from inference_bench.context_schema import ContextRecord
from inference_bench.research_ai_alignment_repair import (
    expanded_research_ai_valid_ids,
    research_ai_repaired_record_from_prompt,
    target_section_families,
)
from inference_bench.retrieval_dataset_alignment import DIRECT_RUNTIME_ID_RE, build_promotion_plan
from inference_bench.vertical_final_selectors import (
    RankedCandidate,
    select_research_ai_section_candidates,
)


def context_record(
    context_id: str,
    text: str,
    *,
    paper_title: str = "FastKV: Efficient Inference for Long Context Models",
    section_type: str = "method",
    topic: str = "llm_serving_inference_optimization",
) -> ContextRecord:
    return ContextRecord(
        context_id=context_id,
        vertical="research_ai",
        source_id="research_fixture_source",
        parent_id="paper_fastkv",
        chunk_id=context_id,
        chunk_strategy="research_ai_fixture_chunking",
        source_type="paper_section",
        title=paper_title,
        text=text,
        metadata={
            "original_doc_id": context_id.replace("ctx", "doc"),
            "paper_id": "paper_fastkv",
            "paper_title": paper_title,
            "section_type": section_type,
            "evidence_type": section_type,
            "section_title": section_type.title(),
            "topic": topic,
        },
        token_estimate=len(text.split()),
        provenance="research_fixture",
        is_gold_linked=True,
    )


def research_prompt(section_types: list[str] | None = None) -> dict[str, Any]:
    return {
        "prompt_id": "research_ai_001",
        "vertical": "research_ai",
        "question": (
            "Which cited section supports a claim about FastKV: Efficient Inference "
            "for Long Context Models?"
        ),
        "topic": "llm_serving_inference_optimization",
        "required_evidence_ids": ["research_ai_kb_hidden_source_id"],
        "metadata": {
            "source_titles": ["FastKV: Efficient Inference for Long Context Models"],
            "evidence_type": section_types or ["abstract", "introduction"],
            "topics": ["llm_serving_inference_optimization"],
        },
    }


def test_research_ai_repair_record_has_required_fields_and_no_runtime_ids() -> None:
    records = [
        context_record(
            "research_ai_ctx_abstract",
            "Abstract contribution boundary.",
            section_type="abstract",
        ),
        context_record(
            "research_ai_ctx_intro",
            "Introduction describes the same contribution.",
            section_type="introduction",
        ),
    ]
    prompt = research_prompt()
    gold = {
        "prompt_id": "research_ai_001",
        "required_evidence_ids": ["research_ai_doc_abstract"],
    }

    repaired = research_ai_repaired_record_from_prompt(
        prompt=prompt,
        gold_record=gold,
        records=records,
        by_match_id={"research_ai_doc_abstract": [records[0]]},
    )

    assert repaired["retrieval_query"]
    assert repaired["paper_title_terms"]
    assert repaired["topic_terms"]
    assert repaired["section_type"] == "abstract"
    assert "valid_evidence_ids_expanded" in repaired
    assert repaired["runtime_query_uses_valid_evidence_ids"] is False
    assert DIRECT_RUNTIME_ID_RE.search(str(repaired["retrieval_query"])) is None


def test_expanded_ids_are_offline_only_not_query_input() -> None:
    records = [
        context_record(
            "research_ai_ctx_abstract",
            "Abstract contribution boundary.",
            section_type="abstract",
        ),
        context_record(
            "research_ai_ctx_intro",
            "Introduction contribution boundary.",
            section_type="introduction",
        ),
    ]
    prompt = research_prompt()
    repaired = research_ai_repaired_record_from_prompt(
        prompt=prompt,
        gold_record={"required_evidence_ids": ["research_ai_doc_abstract"]},
        records=records,
        by_match_id={"research_ai_doc_abstract": [records[0]]},
    )

    assert "research_ai_doc_intro" in repaired["valid_evidence_ids_expanded"]
    assert "research_ai_doc_intro" not in repaired["retrieval_query"]
    assert "research_ai_doc_abstract" not in repaired["retrieval_query"]


def test_section_type_selector_prioritizes_same_paper_target_sections() -> None:
    records = {
        "wrong_results": context_record(
            "wrong_results",
            "Different paper result text.",
            paper_title="OtherKV: Unrelated Paper",
            section_type="results",
        ),
        "fast_method": context_record(
            "fast_method",
            "Method section explains KV cache pruning.",
            section_type="method",
        ),
        "fast_results": context_record(
            "fast_results",
            "Results evaluate latency and throughput.",
            section_type="results",
        ),
    }
    ranked: list[RankedCandidate] = [
        ("wrong_results", 10.0, {}),
        ("fast_results", 8.0, {}),
        ("fast_method", 7.0, {}),
    ]

    selected = select_research_ai_section_candidates(
        rescored=ranked,
        records_by_id=records,
        query_tokens={"fastkv", "method", "efficient", "inference"},
        final_top_k=2,
    )

    assert selected[0][0] == "fast_method"
    assert selected[1][0] == "fast_results"


def test_method_result_and_limitation_signals_are_derived() -> None:
    method_metadata = research_ai_repaired_record_from_prompt(
        prompt=research_prompt(["method", "results", "limitations"]),
        gold_record={"required_evidence_ids": ["research_ai_doc_method"]},
        records=[
            context_record("research_ai_ctx_method", "Method text.", section_type="method"),
            context_record("research_ai_ctx_results", "Results text.", section_type="results"),
            context_record(
                "research_ai_ctx_limit",
                "Limitations text.",
                section_type="limitations",
            ),
        ],
        by_match_id={},
    )["canonical_retrieval_metadata"]

    families = target_section_families(method_metadata)
    assert "method" in families
    assert "results" in families
    assert "limitations" in families


def test_near_duplicate_section_handling_keeps_one_duplicate_text() -> None:
    records = {
        "dup_a": context_record("dup_a", "Duplicate method text.", section_type="method"),
        "dup_b": context_record("dup_b", "Duplicate method text.", section_type="method"),
        "result": context_record("result", "Results are different.", section_type="results"),
    }
    ranked: list[RankedCandidate] = [
        ("dup_a", 9.0, {}),
        ("dup_b", 8.5, {}),
        ("result", 8.0, {}),
    ]

    selected = select_research_ai_section_candidates(
        rescored=ranked,
        records_by_id=records,
        query_tokens={"fastkv", "method", "results"},
        final_top_k=2,
    )

    top_ids = [item[0] for item in selected[:2]]
    assert not {"dup_a", "dup_b"} <= set(top_ids)
    assert "result" in top_ids


def test_promotion_plan_updates_only_if_research_ai_slo_passes() -> None:
    passing_rows = []
    for vertical in ["airline", "retail", "healthcare_admin", "finance", "research_ai"]:
        passing_rows.append(
            {
                "dataset_variant": "repaired_generated",
                "vertical": vertical,
                "stage_size": 2000,
                "final_recall_at_5": 0.95,
                "mrr": 0.95,
                "slo_status": "PASSED",
                "primary_blocker": "none",
            }
        )
    repaired_records = [
        {"vertical": vertical}
        for vertical in ["airline", "retail", "healthcare_admin", "finance", "research_ai"]
    ]

    passing_plan = build_promotion_plan(
        summary_rows=passing_rows,
        repaired_records=repaired_records,
    )
    failing_rows = [dict(row) for row in passing_rows]
    failing_rows[-1]["slo_status"] = "FAILED"
    failing_rows[-1]["primary_blocker"] = "research_ai_alignment"
    failing_plan = build_promotion_plan(
        summary_rows=failing_rows,
        repaired_records=repaired_records,
    )

    assert passing_plan["promotion_recommended"] is True
    assert failing_plan["promotion_recommended"] is False


def test_no_model_api_or_gpu_flags_are_part_of_repair_contract() -> None:
    records = [
        context_record("research_ai_ctx_abstract", "Abstract text.", section_type="abstract"),
        context_record("research_ai_ctx_intro", "Introduction text.", section_type="introduction"),
    ]
    expanded = expanded_research_ai_valid_ids(
        prompt=research_prompt(),
        gold_record={"required_evidence_ids": ["research_ai_doc_abstract"]},
        records=records,
        metadata={
            "paper_title": "FastKV: Efficient Inference for Long Context Models",
            "section_types": ["abstract", "introduction"],
            "topic_terms": ["llm_serving_inference_optimization"],
        },
    )

    assert "research_ai_doc_intro" in expanded
    assert len(expanded) >= 2

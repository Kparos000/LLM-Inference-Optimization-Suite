from __future__ import annotations

from inference_bench.optimization_negative_rules import (
    build_negative_rule_report,
    load_optimization_negative_rules,
    negative_rules_for_optimization,
)


def test_negative_rules_cover_required_optimization_families() -> None:
    rules = load_optimization_negative_rules()

    assert {
        "quantization",
        "prefix_caching",
        "speculative_decoding",
        "tensor_parallelism",
        "disaggregated_prefill",
        "context_compression",
        "concurrency_increase",
        "stronger_model_escalation",
    }.issubset(rules)
    for rule in rules.values():
        assert rule.optimization_ids
        assert rule.when_not_to_use


def test_lookup_finds_when_not_to_use_rules_for_catalog_optimization() -> None:
    matches = negative_rules_for_optimization("enable_prefix_cache")

    assert len(matches) == 1
    assert matches[0].id == "prefix_caching"
    assert any("prefix reuse potential is low" in item for item in matches[0].when_not_to_use)


def test_negative_rule_report_preserves_post_slo_principle() -> None:
    report = build_negative_rule_report()

    assert report["post_slo_diagnosis_only"] is True
    assert report["baseline_matrix_config"] is False
    assert "not baseline matrix dimensions" in report["principle"]

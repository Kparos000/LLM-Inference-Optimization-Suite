from pathlib import Path

import pytest

from inference_bench.reranker_calibration import (
    DEFAULT_CALIBRATED_WEIGHTS,
    allowed_features_for_ablation,
    assert_no_forbidden_features,
    build_reranker_calibration_report,
    choose_reranker_backend,
    split_for_prompt_id,
)


def test_train_dev_test_split_is_deterministic() -> None:
    assert split_for_prompt_id("finance_prompt_001") == split_for_prompt_id("finance_prompt_001")
    assert split_for_prompt_id("finance_prompt_001") in {"train", "dev", "test"}


def test_calibrated_reranker_excludes_forbidden_features() -> None:
    features = allowed_features_for_ablation("prompt_plus_metadata")

    assert "source_hint_match" not in features
    assert "gold_evidence_ids" not in features
    assert_no_forbidden_features(features, ablation_mode="prompt_plus_metadata")
    with pytest.raises(ValueError, match="Forbidden reranker features"):
        assert_no_forbidden_features(
            [*features, "source_id"],
            ablation_mode="prompt_plus_metadata",
        )


def test_source_hint_features_only_enabled_for_source_hint_ablation() -> None:
    strict = allowed_features_for_ablation("prompt_text_only")
    assisted = allowed_features_for_ablation("prompt_plus_source_hints")

    assert "source_hint_match" not in strict
    assert "source_hint_match" in assisted
    assert_no_forbidden_features(assisted, ablation_mode="prompt_plus_source_hints")


def test_calibrated_weights_save_and_load(tmp_path: Path) -> None:
    path = DEFAULT_CALIBRATED_WEIGHTS.save(tmp_path / "weights.json")

    loaded = DEFAULT_CALIBRATED_WEIGHTS.load(path)

    assert loaded == DEFAULT_CALIBRATED_WEIGHTS


def test_reranker_calibration_report_is_generated() -> None:
    report, rows = build_reranker_calibration_report(
        [
            {
                "prompt_id": "finance_prompt_001",
                "ablation_mode": "prompt_plus_metadata",
                "memory_mode": "mm2_hybrid_top5",
                "vertical": "finance",
                "recall_at_5": 0.5,
                "candidate_recall_at_100": 1.0,
            }
        ]
    )

    assert report["reranker_backend"] == "calibrated_linear"
    assert report["runtime_gold_features_used"] is False
    assert rows[0]["forbidden_feature_count"] == 0
    assert rows[0]["candidate_recall_at_100"] == 1.0


def test_cross_encoder_backend_is_disabled_by_default() -> None:
    assert choose_reranker_backend("cross_encoder_optional") == "cross_encoder_optional_disabled"

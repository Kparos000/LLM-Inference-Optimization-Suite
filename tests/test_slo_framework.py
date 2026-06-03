from inference_bench.retrieval_quality_gate import build_retrieval_quality_gate_report
from inference_bench.retrieval_root_cause import load_slo_targets


def test_slo_targets_config_loads() -> None:
    targets = load_slo_targets("configs/slo_targets.yaml")

    assert targets["prompt_text_only"] == 0.70
    assert targets["prompt_plus_metadata"] == 0.80
    assert targets["finance_prompt_text_only"] == 0.65


def test_slo_targets_yaml_is_public_gate_source() -> None:
    targets = load_slo_targets("configs/slo_targets.yaml")

    assert targets["prompt_plus_source_hints"] == 0.95


def test_existing_quality_gate_still_blocks_below_slo_fixture() -> None:
    retrieval_rows = [
        {
            "split": "final_10000",
            "ablation_mode": "prompt_text_only",
            "memory_mode": "mm2_hybrid_top5",
            "vertical": "finance",
            "record_count": 10,
            "recall_at_5": 0.2,
        }
    ]
    compression_rows = [
        {
            "split": "final_10000",
            "record_count": 10,
            "token_reduction_pct": 0.1,
            "recall_loss": 0.0,
        }
    ]

    report, _summary_rows = build_retrieval_quality_gate_report(retrieval_rows, compression_rows)

    assert report["quality_gate_status"] == "BLOCKED"
    assert report["no_model_inference_triggered"] is True

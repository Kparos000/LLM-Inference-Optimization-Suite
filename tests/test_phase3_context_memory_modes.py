import pytest

from inference_bench.config import (
    ExperimentConfig,
    ProjectConfig,
    WorkloadConfig,
    load_memory_modes_config,
    load_models_config,
    load_project_config,
    resolve_memory_mode,
)
from inference_bench.context_schema import ContextRecord, WorkloadRecord

MODEL_ALIAS_PAIRS = {
    "model1_0_5b": "qwen2_5_0_5b_instruct",
    "model2_1_5b": "qwen2_5_1_5b_instruct",
    "model3_7b": "qwen2_5_7b_instruct",
    "model4_32b": "qwen2_5_32b_instruct",
    "model5_gated": "llama_3_2_3b_instruct_api",
    "model6_gated": "llama_3_1_8b_instruct_api",
    "model7_large_placeholder": "future_large_model_placeholder",
}
OLD_MODEL_KEYS = {
    "qwen2_5_0_5b_instruct",
    "qwen2_5_1_5b_instruct",
    "qwen2_5_7b_instruct",
    "qwen2_5_32b_instruct",
}
DEPRECATED_MODEL_ALIASES = {
    "large_model_placeholder": "future_large_model_placeholder",
    "model5_large_placeholder": "future_large_model_placeholder",
}


def valid_context_record() -> ContextRecord:
    return ContextRecord(
        context_id="ctx_airline_001",
        vertical="airline",
        source_id="CA-POL-001",
        parent_id="CA-POL-001",
        chunk_id="CA-POL-001",
        chunk_strategy="policy_section",
        source_type="policy",
        title="24-Hour Cancellation Policy",
        text="Canada Air permits cancellation within 24 hours for eligible bookings.",
        metadata={"document_type": "policy"},
        token_estimate=12,
        provenance="synthetic_public_inspired",
        is_gold_linked=True,
    )


def valid_workload_record() -> WorkloadRecord:
    context_record = valid_context_record()
    return WorkloadRecord(
        workload_id="dataset_10000_airline_mm1_001",
        prompt_id="airline_scaleup_2000_0001",
        vertical="airline",
        memory_mode="mm1_dense_top5",
        messages=[
            {
                "role": "user",
                "content": "Answer using the provided airline policy evidence.",
            }
        ],
        context_records=[context_record],
        context_token_estimate=context_record.token_estimate,
        retrieval_metadata={"retrieval_type": "dense", "top_k": 5},
        expected_output_format="text",
        gold_evidence_ids=["CA-POL-001"],
        dataset_split="test_fixture",
        source_prompt_record={"prompt_id": "airline_scaleup_2000_0001"},
    )


def test_old_model_keys_still_resolve() -> None:
    config = load_project_config()

    for old_key in OLD_MODEL_KEYS:
        assert old_key in config.models
        assert config.resolve_model_key(old_key) == old_key
        assert config.resolve_model_config(old_key).model_id


def test_new_model_aliases_resolve() -> None:
    config = load_project_config()

    for alias, old_key in MODEL_ALIAS_PAIRS.items():
        assert config.model_aliases[alias] == old_key
        assert config.resolve_model_key(alias) == old_key
        assert config.resolve_model_config(alias).model_id
    for alias, target in DEPRECATED_MODEL_ALIASES.items():
        assert config.model_aliases[alias] == target
        assert config.resolve_model_key(alias) == target
        assert config.resolve_model_config(alias).model_id


def test_aliases_point_to_same_model_id_as_old_keys() -> None:
    config = load_project_config()

    for alias, old_key in MODEL_ALIAS_PAIRS.items():
        assert config.resolve_model_config(alias).model_id == config.models[old_key].model_id
    for alias, old_key in DEPRECATED_MODEL_ALIASES.items():
        assert config.resolve_model_config(alias).model_id == config.models[old_key].model_id


def test_model_aliases_do_not_duplicate_canonical_model_records() -> None:
    models = load_models_config("configs/models.yaml")

    assert set(MODEL_ALIAS_PAIRS).isdisjoint(models)
    assert set(DEPRECATED_MODEL_ALIASES).isdisjoint(models)
    assert len(models) == 7


def test_memory_modes_yaml_loads() -> None:
    memory_modes = load_memory_modes_config()

    assert memory_modes["mm0_no_context"].requires_retrieval is False
    assert memory_modes["mm4_bounded_agentic"].requires_agentic_workflow is True


def test_all_five_memory_modes_exist() -> None:
    memory_modes = load_memory_modes_config()

    assert set(memory_modes) == {
        "mm0_no_context",
        "mm1_dense_top5",
        "mm2_hybrid_top5",
        "mm3_compressed_hybrid_top5",
        "mm4_bounded_agentic",
    }


def test_invalid_memory_mode_fails_clearly() -> None:
    with pytest.raises(ValueError, match="Unknown memory mode"):
        resolve_memory_mode("missing_memory_mode")


def test_valid_context_record_passes_validation() -> None:
    context_record = valid_context_record()

    assert context_record.context_id == "ctx_airline_001"
    assert context_record.is_gold_linked is True


def test_invalid_context_record_fails_validation() -> None:
    with pytest.raises(ValueError, match="text"):
        ContextRecord(
            context_id="ctx_airline_001",
            vertical="airline",
            source_id="CA-POL-001",
            parent_id="CA-POL-001",
            chunk_id="CA-POL-001",
            chunk_strategy="policy_section",
            source_type="policy",
            title="24-Hour Cancellation Policy",
            text="",
            metadata={"document_type": "policy"},
            token_estimate=12,
            provenance="synthetic_public_inspired",
            is_gold_linked=True,
        )


def test_valid_workload_record_passes_validation() -> None:
    workload_record = valid_workload_record()

    assert workload_record.memory_mode == "mm1_dense_top5"
    assert workload_record.context_token_estimate == 12


def test_invalid_workload_record_fails_validation() -> None:
    with pytest.raises(ValueError, match="Unknown memory mode"):
        WorkloadRecord(
            workload_id="dataset_10000_airline_bad_001",
            prompt_id="airline_scaleup_2000_0001",
            vertical="airline",
            memory_mode="missing_memory_mode",
            messages=[{"role": "user", "content": "hello"}],
            context_records=[],
            context_token_estimate=0,
            retrieval_metadata={},
            expected_output_format="text",
            gold_evidence_ids=[],
            dataset_split="test_fixture",
            source_prompt_record={"prompt_id": "airline_scaleup_2000_0001"},
        )


def test_existing_experiment_configs_still_validate() -> None:
    default_config = load_project_config()
    vllm_config = load_project_config(experiments_path="configs/vllm_baseline_experiments.yaml")

    assert default_config.experiments["mock_smoke"].model == "qwen2_5_0_5b_instruct"
    assert vllm_config.experiments["vllm_smoke"].model == "qwen2_5_0_5b_instruct"


def test_experiment_config_can_reference_model_alias() -> None:
    base_config = load_project_config()
    workload = WorkloadConfig(name="smoke", path="data/prompts/smoke_workload.jsonl")
    experiment = ExperimentConfig(
        name="alias_experiment",
        backend="mock",
        model="model1_0_5b",
        optimization="none",
        workload="smoke",
        output_path="results/raw/alias_results.csv",
    )

    config = ProjectConfig(
        models=base_config.models,
        workloads={"smoke": workload},
        experiments={"alias_experiment": experiment},
        model_aliases=base_config.model_aliases,
    )

    assert config.resolve_model_config(experiment.model).model_id == "Qwen/Qwen2.5-0.5B-Instruct"

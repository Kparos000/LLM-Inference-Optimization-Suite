from pathlib import Path

from inference_bench.config import load_project_config
from inference_bench.model_registry import (
    build_model_alias_rows,
    write_model_registry_artifacts,
)


def test_active_model_aliases_resolve_to_expected_models() -> None:
    config = load_project_config()

    assert config.resolve_model_key("model2_3b") == "qwen2_5_3b_instruct"
    assert config.resolve_model_config("model2_3b").model_id == "Qwen/Qwen2.5-3B-Instruct"
    assert config.resolve_model_key("model5_gated") == "ministral_3b_2512_api"
    assert config.resolve_model_config("model5_gated").model_id == ("mistralai/ministral-3b-2512")
    assert config.resolve_model_key("model6_gated") == "llama_3_1_8b_instruct_api"
    assert config.resolve_model_config("model6_gated").model_id == (
        "meta-llama/Llama-3.1-8B-Instruct"
    )
    assert config.resolve_model_key("model7_gated") == "mistral_small_3_2_24b_instruct_api"
    assert config.resolve_model_config("model7_gated").model_id == (
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    )


def test_deprecated_qwen_1_5b_entry_remains_available_but_inactive() -> None:
    config = load_project_config()

    assert "qwen2_5_1_5b_instruct" in config.models
    assert config.resolve_model_key("model2_1_5b") == "qwen2_5_1_5b_instruct"
    active_rows = [row for row in build_model_alias_rows() if row["active_public_alias"]]
    assert "model2_1_5b" not in [row["model_alias"] for row in active_rows]


def test_deprecated_llama_3_2_entry_remains_available_but_inactive() -> None:
    config = load_project_config()

    assert "llama_3_2_3b_instruct_api" in config.models
    assert config.resolve_model_key("old_model5_llama_3_2_3b") == ("llama_3_2_3b_instruct_api")
    assert config.model_aliases["model5_gated"] != "llama_3_2_3b_instruct_api"


def test_alias_report_is_human_readable_and_generated(tmp_path: Path) -> None:
    rows = build_model_alias_rows()
    outputs = write_model_registry_artifacts(output_root=tmp_path)

    public_aliases = [row["model_alias"] for row in rows if row["active_public_alias"]]
    assert public_aliases == [
        "model1_0_5b",
        "model2_3b",
        "model3_7b",
        "model4_32b",
        "model5_gated",
        "model6_gated",
        "model7_gated",
    ]
    assert all(path.is_file() for path in outputs.values())

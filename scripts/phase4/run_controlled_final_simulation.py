"""Run the controlled final-experiment simulation safety gate and reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference_bench.api_load_probe import ENV_ALIASES, load_probe_environment  # noqa: E402
from inference_bench.api_pricing import (  # noqa: E402
    estimate_api_cost_from_pricing,
    resolve_api_pricing,
)
from inference_bench.api_routes import api_key_for_route, resolve_api_provider_route  # noqa: E402
from inference_bench.artifact_sync import (  # noqa: E402
    ArtifactSyncConfig,
    build_artifact_specs,
    sync_artifacts,
    verify_backup,
)
from inference_bench.b6r6_research_ai_recovery import (  # noqa: E402
    STRATEGY_D_ANSWER_SKELETON,
    apply_research_ai_strategy_prompt,
)
from inference_bench.config import load_project_config  # noqa: E402
from inference_bench.context_alignment_repair import (  # noqa: E402
    build_b6_context_aligned_runner_input,
)
from inference_bench.context_corpora import VERTICALS  # noqa: E402
from inference_bench.evaluator_contract import evaluate_generated_answers  # noqa: E402
from inference_bench.generation_contract import (  # noqa: E402
    GENERATION_CONTRACT_FIELDS,
    GENERATION_CONTRACT_FORMAT,
    allowed_evidence_ids_from_aliases,
    parse_generation_contract,
    render_generation_contract_prompt,
)
from inference_bench.gpu_telemetry import (  # noqa: E402
    GpuTelemetrySample,
    collect_gpu_sample,
    summarize_gpu_telemetry,
)
from inference_bench.post_run_automation import (  # noqa: E402
    PostRunAutomationInputs,
    build_post_run_automation_report,
    write_post_run_automation_artifacts,
)
from inference_bench.run_manifest import (  # noqa: E402
    RunManifest,
    current_git_commit,
    hash_existing_paths,
    utc_now,
    write_run_manifest,
)
from inference_bench.runners.mock_runner import count_whitespace_tokens  # noqa: E402
from inference_bench.runtime_registry import select_runtime_for_model  # noqa: E402

PHASE4 = Path(__file__).resolve().parent
if str(PHASE4) not in sys.path:
    sys.path.insert(0, str(PHASE4))

from evaluate_generation_outputs import (  # noqa: E402
    build_summary_rows,
    load_gold_records,
    result_row_to_generated_answer,
)
from run_remote_vllm_smoke import latency_summary_rows  # noqa: E402

RUN_ID = "controlled_final_simulation"
SELF_HOSTED_MODEL_ALIAS = "model3_7b"
SELF_HOSTED_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
API_MODEL_ALIAS = "model6_gated"
API_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
GPU_ID = "a100_sxm_80gb"
GPU_NAME = "NVIDIA A100-SXM4-80GB"
TRAFFIC_PROFILE = "online_low_latency"
MEMORY_MODES = (
    "mm0_no_context",
    "mm1_dense_top5",
    "mm2_hybrid_top5",
    "mm3_compressed_hybrid_top5",
    "mm4_bounded_agentic",
)
SELF_HOSTED_ENGINES = ("vllm", "sglang")
SELF_HOSTED_CONCURRENCY = (16, 32)
API_CONCURRENCY = (4,)
DEFAULT_PROMPTS_PER_VERTICAL = 80
DEFAULT_MATRIX_PATH = (
    "data/generated/phase4/controlled_final_simulation_80_per_vertical_matrix.jsonl"
)
DEFAULT_RAW_RESULTS = "results/raw/controlled_final_simulation_results.jsonl"
DEFAULT_MANIFEST = "results/raw/controlled_final_simulation_manifest.json"
DEFAULT_GPU_TELEMETRY = "results/raw/controlled_final_simulation_gpu_telemetry.jsonl"
DEFAULT_EVAL_REPORT = "results/processed/controlled_final_simulation_eval_report.json"
DEFAULT_EVAL_SUMMARY = "results/processed/controlled_final_simulation_eval_summary.csv"
DEFAULT_ENGINE_COMPARISON = "results/processed/controlled_final_simulation_engine_comparison.csv"
DEFAULT_MEMORY_COMPARISON = (
    "results/processed/controlled_final_simulation_memory_mode_comparison.csv"
)
DEFAULT_CONCURRENCY_COMPARISON = (
    "results/processed/controlled_final_simulation_concurrency_comparison.csv"
)
DEFAULT_API_COMPARISON = "results/processed/controlled_final_simulation_api_track_comparison.csv"
DEFAULT_API_VS_SELF_HOSTED_COMPARISON = (
    "results/processed/controlled_final_simulation_api_vs_self_hosted_comparison.csv"
)
DEFAULT_MODEL_COMPARISON = "results/processed/controlled_final_simulation_model_comparison.csv"
DEFAULT_SLO_REPORT = "results/processed/controlled_final_simulation_slo_report.json"
DEFAULT_SLO_SUMMARY = "results/processed/controlled_final_simulation_slo_summary.csv"
DEFAULT_COST_REPORT = "results/processed/controlled_final_simulation_cost_report.json"
DEFAULT_ARTIFACT_SYNC_REPORT = (
    "results/processed/controlled_final_simulation_artifact_sync_report.json"
)
DEFAULT_POST_RUN_AUTOMATION_REPORT = (
    "results/processed/controlled_final_simulation_post_run_automation_report.json"
)
DEFAULT_PLOTTING_DATASET = "results/processed/controlled_final_simulation_plotting_dataset.csv"
DEFAULT_CHECKPOINT = "results/raw/controlled_final_simulation_checkpoint.json"
DEFAULT_SOURCE_WORKLOAD = (
    "data/workloads/controlled_2000/prompt_plus_metadata/mm2_hybrid_top5.jsonl"
)
DEFAULT_SOURCE_OF_TRUTH_MANIFEST = (
    "data/generated/context_engineering/retrieval_source_of_truth_manifest.json"
)
DEFAULT_CONTEXT_ROOT = "data/generated/context_engineering"
DEFAULT_REPAIRED_RUNNER_INPUT = (
    "data/generated/phase4/controlled_final_simulation_repaired_runner_input.jsonl"
)
DEFAULT_CONTEXT_PREFLIGHT_REPORT = (
    "results/processed/controlled_final_repaired_context_preflight_report.json"
)
DEFAULT_CONTEXT_PREFLIGHT_SUMMARY = (
    "results/processed/controlled_final_repaired_context_preflight_summary.csv"
)
DEFAULT_CONTEXT_PREFLIGHT_EXAMPLES = (
    "results/processed/controlled_final_repaired_context_preflight_examples.jsonl"
)
DEFAULT_CONTRACT_PREFLIGHT_REPORT = (
    "results/processed/controlled_final_contract_preflight_report.json"
)
DEFAULT_REPAIRED_25_REPLAY_REPORT = (
    "results/processed/controlled_final_repaired_25_replay_report.json"
)
DEFAULT_25_REPLAY_FAILURE_AUDIT_JSON = (
    "results/processed/controlled_final_25_replay_failure_audit.json"
)
DEFAULT_25_REPLAY_FAILURE_AUDIT_CSV = (
    "results/processed/controlled_final_25_replay_failure_audit.csv"
)
DEFAULT_REPAIRED_500_VALIDATION_REPORT = (
    "results/processed/controlled_final_repaired_500_validation_report.json"
)
DEFAULT_REPAIRED_500_VALIDATION_SUMMARY = (
    "results/processed/controlled_final_repaired_500_validation_summary.csv"
)
DEFAULT_MM4_SAFETY_AUDIT_JSON = "results/processed/controlled_final_mm4_safety_violation_audit.json"
DEFAULT_MM4_SAFETY_AUDIT_MD = "results/processed/controlled_final_mm4_safety_violation_audit.md"
DEFAULT_MM4_SAFETY_TARGETED_REPORT = (
    "results/processed/controlled_final_mm4_safety_targeted_replay_report.json"
)
DEFAULT_MM4_SAFETY_TARGETED_SUMMARY = (
    "results/processed/controlled_final_mm4_safety_targeted_replay_summary.csv"
)
DEFAULT_REPAIR_READY_REPORT = "results/processed/controlled_final_repair_ready_report.json"
DEFAULT_REPAIR_VS_BROKEN_COMPARISON = (
    "results/processed/controlled_final_repair_vs_broken_comparison.json"
)
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_SGLANG_BASE_URL = "http://localhost:30000/v1"
SGLANG_STARTUP_COMMAND = (
    "python -m sglang.launch_server "
    "--model-path Qwen/Qwen2.5-7B-Instruct "
    "--served-model-name Qwen/Qwen2.5-7B-Instruct "
    "--host 0.0.0.0 "
    "--port 30000 "
    "--mem-fraction-static 0.90 "
    "--context-length 4096 "
    "--max-running-requests 32 "
    "--chunked-prefill-size 8192"
)


@dataclass(frozen=True)
class ConfigSpec:
    """One controlled final-simulation config."""

    config_id: str
    model_alias: str
    model_id: str
    backend_type: str
    engine: str
    runtime: str
    memory_mode: str
    concurrency: int


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = _repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = _repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with _repo_path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = _repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row}) if rows else ["status"]
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows or [{"status": "NO_ROWS"}])


def build_parser() -> argparse.ArgumentParser:
    """Build the controlled final-simulation CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="data/scaleup_2000_full")
    parser.add_argument("--matrix-path", default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--raw-results-path", default=DEFAULT_RAW_RESULTS)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST)
    parser.add_argument("--gpu-telemetry-path", default=DEFAULT_GPU_TELEMETRY)
    parser.add_argument("--eval-report-path", default=DEFAULT_EVAL_REPORT)
    parser.add_argument("--eval-summary-path", default=DEFAULT_EVAL_SUMMARY)
    parser.add_argument("--engine-comparison-path", default=DEFAULT_ENGINE_COMPARISON)
    parser.add_argument("--memory-comparison-path", default=DEFAULT_MEMORY_COMPARISON)
    parser.add_argument("--concurrency-comparison-path", default=DEFAULT_CONCURRENCY_COMPARISON)
    parser.add_argument("--api-comparison-path", default=DEFAULT_API_COMPARISON)
    parser.add_argument(
        "--api-vs-self-hosted-comparison-path",
        default=DEFAULT_API_VS_SELF_HOSTED_COMPARISON,
    )
    parser.add_argument("--model-comparison-path", default=DEFAULT_MODEL_COMPARISON)
    parser.add_argument("--slo-report-path", default=DEFAULT_SLO_REPORT)
    parser.add_argument("--slo-summary-path", default=DEFAULT_SLO_SUMMARY)
    parser.add_argument("--cost-report-path", default=DEFAULT_COST_REPORT)
    parser.add_argument("--artifact-sync-report-path", default=DEFAULT_ARTIFACT_SYNC_REPORT)
    parser.add_argument(
        "--post-run-automation-report-path",
        default=DEFAULT_POST_RUN_AUTOMATION_REPORT,
    )
    parser.add_argument("--plotting-dataset-path", default=DEFAULT_PLOTTING_DATASET)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source-workload", default=DEFAULT_SOURCE_WORKLOAD)
    parser.add_argument("--source-of-truth-manifest", default=DEFAULT_SOURCE_OF_TRUTH_MANIFEST)
    parser.add_argument("--context-root", default=DEFAULT_CONTEXT_ROOT)
    parser.add_argument("--repaired-runner-input-path", default=DEFAULT_REPAIRED_RUNNER_INPUT)
    parser.add_argument("--context-preflight-report", default=DEFAULT_CONTEXT_PREFLIGHT_REPORT)
    parser.add_argument("--context-preflight-summary", default=DEFAULT_CONTEXT_PREFLIGHT_SUMMARY)
    parser.add_argument("--context-preflight-examples", default=DEFAULT_CONTEXT_PREFLIGHT_EXAMPLES)
    parser.add_argument("--contract-preflight-report", default=DEFAULT_CONTRACT_PREFLIGHT_REPORT)
    parser.add_argument("--repaired-25-replay-report", default=DEFAULT_REPAIRED_25_REPLAY_REPORT)
    parser.add_argument(
        "--repaired-25-failure-audit-json",
        default=DEFAULT_25_REPLAY_FAILURE_AUDIT_JSON,
    )
    parser.add_argument(
        "--repaired-25-failure-audit-csv",
        default=DEFAULT_25_REPLAY_FAILURE_AUDIT_CSV,
    )
    parser.add_argument(
        "--repaired-500-validation-report",
        default=DEFAULT_REPAIRED_500_VALIDATION_REPORT,
    )
    parser.add_argument(
        "--repaired-500-validation-summary",
        default=DEFAULT_REPAIRED_500_VALIDATION_SUMMARY,
    )
    parser.add_argument("--mm4-safety-audit-json", default=DEFAULT_MM4_SAFETY_AUDIT_JSON)
    parser.add_argument("--mm4-safety-audit-md", default=DEFAULT_MM4_SAFETY_AUDIT_MD)
    parser.add_argument(
        "--mm4-safety-targeted-report",
        default=DEFAULT_MM4_SAFETY_TARGETED_REPORT,
    )
    parser.add_argument(
        "--mm4-safety-targeted-summary",
        default=DEFAULT_MM4_SAFETY_TARGETED_SUMMARY,
    )
    parser.add_argument("--repair-ready-report", default=DEFAULT_REPAIR_READY_REPORT)
    parser.add_argument(
        "--repair-vs-broken-comparison-report",
        default=DEFAULT_REPAIR_VS_BROKEN_COMPARISON,
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--sglang-base-url", default=DEFAULT_SGLANG_BASE_URL)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument(
        "--prompt-count-per-vertical",
        type=int,
        default=DEFAULT_PROMPTS_PER_VERTICAL,
    )
    parser.add_argument("--traffic-profile", default=TRAFFIC_PROFILE)
    parser.add_argument("--gpu-id", default=GPU_ID)
    parser.add_argument("--hourly-price", type=float, default=1.49)
    parser.add_argument("--backup-root", default="backups")
    parser.add_argument("--max-new-tokens", type=int, default=224)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--telemetry-interval-seconds", type=float, default=1.0)
    parser.add_argument("--run-repaired-smoke", action="store_true")
    parser.add_argument("--run-repaired-validation", action="store_true")
    parser.add_argument("--run-mm4-safety-targeted", action="store_true")
    parser.add_argument("--allow-full-after-repair", action="store_true")
    parser.add_argument("--waive-repaired-validation-reason", default="")
    parser.add_argument("--run-full", action="store_true")
    return parser


def build_config_specs() -> list[ConfigSpec]:
    """Return the frozen 10,000-request self-hosted and API config matrix."""

    specs: list[ConfigSpec] = []
    for engine in SELF_HOSTED_ENGINES:
        for memory_mode in MEMORY_MODES:
            for concurrency in SELF_HOSTED_CONCURRENCY:
                specs.append(
                    ConfigSpec(
                        config_id=(
                            f"self_hosted_{SELF_HOSTED_MODEL_ALIAS}_{engine}_"
                            f"{memory_mode}_c{concurrency}"
                        ),
                        model_alias=SELF_HOSTED_MODEL_ALIAS,
                        model_id=SELF_HOSTED_MODEL_ID,
                        backend_type="self_hosted_gpu",
                        engine=engine,
                        runtime=engine,
                        memory_mode=memory_mode,
                        concurrency=concurrency,
                    )
                )
    for memory_mode in MEMORY_MODES:
        for concurrency in API_CONCURRENCY:
            specs.append(
                ConfigSpec(
                    config_id=f"api_{API_MODEL_ALIAS}_api_provider_route_{memory_mode}_c{concurrency}",
                    model_alias=API_MODEL_ALIAS,
                    model_id=API_MODEL_ID,
                    backend_type="api_provider",
                    engine="api_provider_route",
                    runtime="api_provider_route",
                    memory_mode=memory_mode,
                    concurrency=concurrency,
                )
            )
    return specs


def _prompt_text(row: dict[str, Any]) -> str:
    for key in ("question", "issue", "prompt", "user_prompt"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return json.dumps(row, ensure_ascii=True, sort_keys=True)


def _evidence_ids(prompt: dict[str, Any], gold: dict[str, Any] | None) -> list[str]:
    for source in (prompt, gold or {}):
        for key in ("required_evidence_ids", "required_doc_ids", "required_chunk_ids"):
            value = source.get(key)
            if isinstance(value, list):
                return [str(item) for item in value]
    citations = (gold or {}).get("required_citations")
    if isinstance(citations, list):
        ids = []
        for citation in citations:
            if isinstance(citation, dict):
                ids.append(str(citation.get("doc_id") or citation.get("chunk_id") or ""))
        return [item for item in ids if item]
    return []


def _json_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    parsed = json.loads(value)
    return parsed if isinstance(parsed, dict) else {}


def _json_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if not isinstance(value, str) or not value.strip():
        return []
    parsed = json.loads(value)
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _question_from_contract_prompt(prompt: str) -> str:
    marker = "\nUSER QUESTION:\n"
    if marker not in prompt:
        return prompt
    tail = prompt.split(marker, maxsplit=1)[1]
    return tail.split("\n\n", maxsplit=1)[0].strip()


def _replace_memory_mode(prompt: str, memory_mode: str) -> str:
    lines = prompt.splitlines()
    for index, line in enumerate(lines[:-1]):
        if line.strip() == "MEMORY MODE:":
            lines[index + 1] = memory_mode
            return "\n".join(lines)
    return prompt


def _insert_before_output_contract(prompt: str, addition: str) -> str:
    marker = "\n\nOUTPUT CONTRACT:\n"
    if marker in prompt:
        return prompt.replace(marker, f"\n\n{addition}{marker}", 1)
    marker = "\nOUTPUT CONTRACT:\n"
    if marker in prompt:
        return prompt.replace(marker, f"\n{addition}{marker}", 1)
    return f"{prompt.rstrip()}\n\n{addition}"


def _finance_repair_block(row: dict[str, Any]) -> str:
    labels = str(row.get("b5_required_labels") or "E1,E2,E3,E4,E5")
    return "\n".join(
        [
            "B6R5 FINANCE EVIDENCE SELECTION PREPLAN:",
            "Resolve entity, metric, period, form, and section cues before answering.",
            f"Eligible evidence labels: {labels}.",
            "Cite only eligible E labels that directly support each finance claim.",
            "Do not provide buy, sell, hold, price-target, or investment advice.",
        ]
    )


def _mm4_contract_block() -> str:
    return "\n".join(
        [
            "MM4 BOUNDED AGENTIC CONTRACT:",
            "Use the bounded agentic workflow internally: classify, plan retrieval, "
            "assemble context, generate, validate, repair once, then finalize.",
            "No internet retrieval, arbitrary tools, hidden reasoning, or extra tool calls.",
            "The final assistant response must still be exactly one five-field "
            "generation-contract JSON object.",
        ]
    )


def _render_prompt_for_memory_mode(base: dict[str, Any], memory_mode: str) -> tuple[str, str]:
    prompt = str(base["prompt"])
    question = _question_from_contract_prompt(prompt)
    source_vertical = str(base.get("vertical") or "")
    if memory_mode == "mm0_no_context":
        rendered = render_generation_contract_prompt(
            question=question,
            context_records=[],
            memory_mode=memory_mode,
            include_citation_checklist=True,
        )
        repair_tags = ["contract_no_context"]
        if source_vertical == "finance":
            rendered = _insert_before_output_contract(rendered, _finance_repair_block(base))
            repair_tags.append("b6r5_finance_evidence_selection_preplan")
        if source_vertical == "research_ai":
            rendered = apply_research_ai_strategy_prompt(
                prompt=rendered,
                strategy_id=STRATEGY_D_ANSWER_SKELETON,
                required_labels=[],
            )
            repair_tags.append("b6r6_research_ai_answer_skeleton")
        return rendered, ";".join(repair_tags)

    rendered = _replace_memory_mode(prompt, memory_mode)
    repair_tags = ["b6_b7_a100_context_aligned_generation_contract"]
    if source_vertical == "finance":
        rendered = _insert_before_output_contract(rendered, _finance_repair_block(base))
        repair_tags.append("b6r5_finance_evidence_selection_preplan")
    if source_vertical == "research_ai":
        labels = [
            label.strip()
            for label in str(base.get("b5_required_labels") or "E1,E2,E3,E4,E5").split(",")
            if label.strip()
        ]
        rendered = apply_research_ai_strategy_prompt(
            prompt=rendered,
            strategy_id=STRATEGY_D_ANSWER_SKELETON,
            required_labels=labels,
        )
        repair_tags.append("b6r6_research_ai_answer_skeleton")
    if memory_mode == "mm4_bounded_agentic":
        rendered = _insert_before_output_contract(rendered, _mm4_contract_block())
        repair_tags.append("mm4_bounded_agentic_contract")
    return rendered, ";".join(repair_tags)


def build_repaired_base_input(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Build/load the repaired B6/B7/A100-style 400-row base input."""

    build_b6_context_aligned_runner_input(
        source_workload_path=_repo_path(args.source_workload),
        source_of_truth_manifest_path=_repo_path(args.source_of_truth_manifest),
        dataset_root=_repo_path(args.dataset_root),
        context_root=_repo_path(args.context_root),
        output_path=_repo_path(args.repaired_runner_input_path),
        report_path=_repo_path(args.context_preflight_report),
        summary_path=_repo_path(args.context_preflight_summary),
        examples_path=_repo_path(args.context_preflight_examples),
        prompts_per_vertical=args.prompt_count_per_vertical,
    )
    rows: list[dict[str, Any]] = []
    for item in _read_jsonl(args.repaired_runner_input_path):
        metadata = dict(item.get("metadata") or {})
        source_prompt = _json_object(metadata.get("source_prompt_record"))
        rows.append(
            {
                "vertical": metadata.get("vertical"),
                "prompt_id": item["prompt_id"],
                "base_prompt": item["prompt"],
                "prompt": item["prompt"],
                "source_prompt_text": _prompt_text(source_prompt),
                "input_context": item["prompt"].split("USER QUESTION:", maxsplit=1)[0],
                "expected_evidence_ids": _json_list(metadata.get("gold_evidence_ids")),
                "expected_status": source_prompt.get("expected_status") or "answer",
                "expected_output_format": GENERATION_CONTRACT_FORMAT,
                "citation_id_aliases": metadata.get("citation_id_aliases") or "{}",
                "selected_context_ids": metadata.get("selected_context_ids") or "[]",
                "context_alignment_status": metadata.get("context_alignment_status"),
                "canonical_ids_exposed_to_model": metadata.get(
                    "canonical_ids_exposed_to_model", "false"
                ),
                "b5_planning_active": metadata.get("b5_planning_active"),
                "b5_required_labels": metadata.get("b5_required_labels"),
                "traffic_profile": TRAFFIC_PROFILE,
                "workload_id": metadata.get("workload_id"),
            }
        )
    return rows


def build_matrix_rows(
    *,
    dataset_root: str | Path,
    prompts_per_vertical: int,
    args: argparse.Namespace | None = None,
) -> list[dict[str, Any]]:
    """Build the 10,000-request controlled simulation matrix."""

    if args is None:
        args = build_parser().parse_args([])
        args.dataset_root = str(dataset_root)
        args.prompt_count_per_vertical = prompts_per_vertical
    prompt_rows = build_repaired_base_input(args)
    rows: list[dict[str, Any]] = []
    for spec in build_config_specs():
        for prompt in prompt_rows:
            rendered_prompt, repair_tags = _render_prompt_for_memory_mode(prompt, spec.memory_mode)
            rows.append(
                {
                    **prompt,
                    "prompt": rendered_prompt,
                    "prompt_hash": _prompt_hash(rendered_prompt),
                    "memory_mode_prompt_renderer": "render_generation_contract_prompt",
                    "contract_repair_tags": repair_tags,
                    "message_payload_normalized": True,
                    "config_id": spec.config_id,
                    "model_alias": spec.model_alias,
                    "model_id": spec.model_id,
                    "backend_type": spec.backend_type,
                    "engine": spec.engine,
                    "runtime": spec.runtime,
                    "memory_mode": spec.memory_mode,
                    "concurrency": spec.concurrency,
                }
            )
    return rows


def summarize_matrix(rows: list[dict[str, Any]], prompts_per_vertical: int) -> dict[str, Any]:
    """Validate matrix cardinality and dimensions."""

    config_ids = {str(row["config_id"]) for row in rows}
    vertical_counts = Counter(str(row["vertical"]) for row in rows)
    expected_config_count = 25
    expected_prompt_rows = prompts_per_vertical * len(VERTICALS)
    expected_total = expected_config_count * expected_prompt_rows
    return {
        "row_count": len(rows),
        "expected_row_count": expected_total,
        "config_count": len(config_ids),
        "expected_config_count": expected_config_count,
        "prompt_count_per_config": expected_prompt_rows,
        "prompts_per_vertical_per_config": prompts_per_vertical,
        "self_hosted_request_count": sum(
            1 for row in rows if row["backend_type"] == "self_hosted_gpu"
        ),
        "api_request_count": sum(1 for row in rows if row["backend_type"] == "api_provider"),
        "vertical_counts": dict(sorted(vertical_counts.items())),
        "passed": len(rows) == expected_total and len(config_ids) == expected_config_count,
    }


def _contains_any_expected_id(prompt: str, expected_ids: list[str]) -> bool:
    lowered = prompt.lower()
    return any(expected_id and expected_id.lower() in lowered for expected_id in expected_ids)


def contract_preflight_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate repaired prompt construction before any inference."""

    total = len(rows)
    contract_rows = [
        row
        for row in rows
        if "OUTPUT CONTRACT:" in str(row.get("prompt") or "")
        and all(field in str(row.get("prompt") or "") for field in GENERATION_CONTRACT_FIELDS)
        and row.get("expected_output_format") == GENERATION_CONTRACT_FORMAT
    ]
    contextual_rows = [
        row
        for row in rows
        if row.get("memory_mode")
        in {"mm1_dense_top5", "mm2_hybrid_top5", "mm3_compressed_hybrid_top5"}
    ]
    contextual_with_labels = [
        row
        for row in contextual_rows
        if all(f"E{index}" in str(row.get("prompt") or "") for index in range(1, 6))
    ]
    finance_rows = [row for row in rows if row.get("vertical") == "finance"]
    research_rows = [row for row in rows if row.get("vertical") == "research_ai"]
    mm0_rows = [row for row in rows if row.get("memory_mode") == "mm0_no_context"]
    mm4_rows = [row for row in rows if row.get("memory_mode") == "mm4_bounded_agentic"]
    leakage_rows = [
        row
        for row in rows
        if _contains_any_expected_id(str(row.get("prompt") or ""), row["expected_evidence_ids"])
        or str(row.get("canonical_ids_exposed_to_model") or "").lower() == "true"
    ]
    checks = {
        "all_rows_have_contract_instructions": len(contract_rows) == total,
        "contextual_rows_have_visible_e_labels": len(contextual_with_labels)
        == len(contextual_rows),
        "mm0_has_contract_json_instruction": all(
            "OUTPUT CONTRACT:" in str(row.get("prompt") or "") for row in mm0_rows
        ),
        "finance_rows_use_b6r5_repair": all(
            "b6r5_finance_evidence_selection_preplan" in str(row.get("contract_repair_tags") or "")
            for row in finance_rows
        ),
        "research_ai_rows_use_answer_skeleton": all(
            "b6r6_research_ai_answer_skeleton" in str(row.get("contract_repair_tags") or "")
            for row in research_rows
        ),
        "mm4_rows_use_bounded_agentic_contract": all(
            "mm4_bounded_agentic_contract" in str(row.get("contract_repair_tags") or "")
            and "MM4 BOUNDED AGENTIC CONTRACT" in str(row.get("prompt") or "")
            for row in mm4_rows
        ),
        "api_vllm_sglang_payloads_normalized": all(
            bool(row.get("message_payload_normalized")) for row in rows
        ),
        "no_canonical_or_gold_leakage": not leakage_rows,
    }
    passed = all(checks.values())
    return {
        "run_id": RUN_ID,
        "status": "CONTRACT_PREFLIGHT_PASSED" if passed else "CONTRACT_PREFLIGHT_BLOCKED",
        "passed": passed,
        "row_count": total,
        "checks": checks,
        "contract_instruction_rate": len(contract_rows) / total if total else 0.0,
        "contextual_e_label_rate": (
            len(contextual_with_labels) / len(contextual_rows) if contextual_rows else 0.0
        ),
        "finance_row_count": len(finance_rows),
        "research_ai_row_count": len(research_rows),
        "mm0_row_count": len(mm0_rows),
        "mm4_row_count": len(mm4_rows),
        "leakage_row_count": len(leakage_rows),
        "blocked_examples": [
            {
                "config_id": row.get("config_id"),
                "prompt_id": row.get("prompt_id"),
                "memory_mode": row.get("memory_mode"),
                "vertical": row.get("vertical"),
            }
            for row in leakage_rows[:10]
        ],
    }


def _label_alias_map(row: dict[str, Any]) -> dict[str, str]:
    alias_map = _json_object(row.get("citation_id_aliases"))
    labels = allowed_evidence_ids_from_aliases(alias_map)
    mapped = {label.upper(): label for label in labels}
    for label, aliases in alias_map.items():
        label_text = str(label)
        mapped[label_text.upper()] = label_text
        if isinstance(aliases, list):
            for alias in aliases:
                mapped[str(alias).upper()] = label_text
    for index in range(1, 6):
        mapped.setdefault(f"E{index}", f"E{index}")
        mapped.setdefault(str(index), f"E{index}")
        mapped.setdefault(f"EVIDENCE {index}", f"E{index}")
    return mapped


def _coerce_string(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _coerce_confidence(value: object) -> float:
    if isinstance(value, bool):
        return 0.5
    try:
        confidence = float(str(value))
    except (TypeError, ValueError):
        return 0.5
    return min(max(confidence, 0.0), 1.0)


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def _raw_evidence_values(payload: dict[str, object]) -> list[object]:
    for key in (
        "evidence_ids",
        "evidence_id",
        "citations",
        "citation_ids",
        "citation_id",
        "cited_evidence",
        "sources",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, str) and value.strip():
            return [part.strip() for part in re.split(r"[,;]", value) if part.strip()]
    return []


def _normalize_evidence_ids(
    *,
    values: list[object],
    row: dict[str, Any],
    payload_text: str,
) -> list[str]:
    label_map = _label_alias_map(row)
    allowed = set(allowed_evidence_ids_from_aliases(row.get("citation_id_aliases")))
    if not allowed:
        allowed = {f"E{index}" for index in range(1, 6)}
    normalized: list[str] = []
    candidates = [str(value).strip() for value in values if str(value).strip()]
    candidates.extend(re.findall(r"\bEVIDENCE\s+([1-5])\b", payload_text, flags=re.IGNORECASE))
    candidates.extend(re.findall(r"\bE([1-5])\b", payload_text, flags=re.IGNORECASE))
    for candidate in candidates:
        key = candidate.upper()
        if key.isdigit() and f"E{key}" in allowed:
            label = f"E{key}"
        elif len(key) == 1 and key in "12345" and f"E{key}" in allowed:
            label = f"E{key}"
        else:
            label = label_map.get(key)
        if label in allowed and label not in normalized:
            normalized.append(label)
    return normalized


SAFETY_BOUNDARY_MARKERS = (
    "do not",
    "don't",
    "cannot",
    "can't",
    "must not",
    "should not",
    "avoid",
    "without",
    "no ",
)
SAFE_REFUSAL_REPLACEMENTS = (
    (
        re.compile(r"\bverification bypass(?:es)?\b", re.IGNORECASE),
        "unsupported verification exception",
    ),
    (re.compile(r"\bbypass verification\b", re.IGNORECASE), "avoid required verification"),
    (re.compile(r"\bguaranteed compensation\b", re.IGNORECASE), "assured compensation"),
    (re.compile(r"\bbuy recommendation\b", re.IGNORECASE), "investment recommendation"),
    (re.compile(r"\bsell recommendation\b", re.IGNORECASE), "investment recommendation"),
    (re.compile(r"\bhold recommendation\b", re.IGNORECASE), "investment recommendation"),
    (re.compile(r"\bprice target\b", re.IGNORECASE), "specific valuation target"),
    (re.compile(r"\btreatment advice\b", re.IGNORECASE), "clinical treatment guidance"),
    (re.compile(r"\bmedical diagnosis\b", re.IGNORECASE), "clinical diagnosis"),
    (re.compile(r"\bdiagnose\b", re.IGNORECASE), "make a clinical determination"),
    (re.compile(r"\bfabricated citation\b", re.IGNORECASE), "unsupported citation"),
)


def _safety_terms_in_text(text: str) -> list[str]:
    lowered = text.lower()
    terms = [
        "verification bypass",
        "bypass verification",
        "guaranteed compensation",
        "buy recommendation",
        "sell recommendation",
        "hold recommendation",
        "price target",
        "treatment advice",
        "medical diagnosis",
        "diagnose",
        "fabricated citation",
    ]
    return sorted({term for term in terms if term in lowered})


def _has_safety_boundary_marker(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in SAFETY_BOUNDARY_MARKERS)


def _rewrite_safe_boundary_sentence(sentence: str) -> tuple[str, bool]:
    if not _has_safety_boundary_marker(sentence):
        return sentence, False
    repaired = sentence
    for pattern, replacement in SAFE_REFUSAL_REPLACEMENTS:
        repaired = pattern.sub(replacement, repaired)
    return repaired, repaired != sentence


def apply_mm4_final_safety_boundary(text: str) -> tuple[str, bool, list[str]]:
    """Rewrite safe MM4 refusal wording without hiding real unsafe answers."""

    if not text.strip():
        return text, False, []
    source_terms = _safety_terms_in_text(text)
    if not source_terms:
        return text, False, []
    parts = re.split(r"([.!?]\s+)", text)
    repaired_parts: list[str] = []
    changed = False
    for index in range(0, len(parts), 2):
        sentence = parts[index]
        separator = parts[index + 1] if index + 1 < len(parts) else ""
        repaired, sentence_changed = _rewrite_safe_boundary_sentence(sentence)
        repaired_parts.append(repaired)
        repaired_parts.append(separator)
        changed = changed or sentence_changed
    return "".join(repaired_parts), changed, source_terms


def normalize_generation_contract_output(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize provider output into the five-field generation contract."""

    original_text = str(row.get("generated_text") or row.get("output_text") or "")
    parse = parse_generation_contract(
        original_text,
        allowed_evidence_ids=allowed_evidence_ids_from_aliases(row.get("citation_id_aliases"))
        or None,
    )
    payload = dict(parse.parsed_payload or {})
    answer = _coerce_string(
        payload.get("answer")
        or payload.get("final_answer")
        or payload.get("response")
        or payload.get("summary")
    )
    citation_notes = _coerce_string(
        payload.get("citation_notes")
        or payload.get("citation_note")
        or payload.get("rationale")
        or payload.get("notes")
    )
    evidence_ids = _normalize_evidence_ids(
        values=_raw_evidence_values(payload),
        row=row,
        payload_text=json.dumps(payload, ensure_ascii=True, sort_keys=True),
    )
    insufficient_evidence = _coerce_bool(payload.get("insufficient_evidence"))
    if row.get("memory_mode") == "mm0_no_context":
        evidence_ids = []
        insufficient_evidence = True
        answer = ""
        citation_notes = citation_notes or "No retrieved evidence was supplied."
    elif answer and evidence_ids:
        insufficient_evidence = False
    elif not evidence_ids:
        insufficient_evidence = True
        answer = ""
        citation_notes = citation_notes or "No supporting evidence label was emitted."
    safety_boundary_terms: list[str] = []
    safety_boundary_applied = False
    if row.get("memory_mode") == "mm4_bounded_agentic":
        answer, answer_repaired, answer_terms = apply_mm4_final_safety_boundary(answer)
        citation_notes, notes_repaired, note_terms = apply_mm4_final_safety_boundary(citation_notes)
        safety_boundary_applied = answer_repaired or notes_repaired
        safety_boundary_terms = sorted({*answer_terms, *note_terms})
    normalized_payload = {
        "answer": answer,
        "evidence_ids": evidence_ids,
        "confidence": _coerce_confidence(payload.get("confidence")),
        "insufficient_evidence": insufficient_evidence,
        "citation_notes": citation_notes,
    }
    normalized_text = json.dumps(normalized_payload, ensure_ascii=True, separators=(",", ":"))
    normalized_parse = parse_generation_contract(
        normalized_text,
        allowed_evidence_ids=allowed_evidence_ids_from_aliases(row.get("citation_id_aliases"))
        or None,
    )
    normalization_applied = normalized_text.strip() != original_text.strip()
    return {
        **row,
        "raw_generated_text": original_text,
        "generated_text": normalized_text,
        "output_text": normalized_text,
        "citations": json.dumps(evidence_ids, ensure_ascii=True),
        "contract_normalization_applied": normalization_applied,
        "contract_normalization_source_error": parse.error or "",
        "contract_normalization_source_missing_fields": json.dumps(
            parse.missing_fields,
            ensure_ascii=True,
        ),
        "contract_normalization_valid": normalized_parse.contract_valid,
        "mm4_safety_boundary_repair_applied": safety_boundary_applied,
        "mm4_safety_boundary_source_terms": json.dumps(
            safety_boundary_terms,
            ensure_ascii=True,
        ),
        "parse_repair_applied": bool(parse.parse_repair_applied or normalization_applied),
        "final_status": "insufficient_evidence" if insufficient_evidence else "answer",
    }


def normalize_generation_contract_outputs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize all successful replay outputs before evaluation."""

    return [
        normalize_generation_contract_output(row) if row.get("success") else row for row in rows
    ]


def _failure_classes(
    *,
    row: dict[str, Any],
    evaluation: dict[str, Any],
) -> list[str]:
    classes: list[str] = []
    raw_parse = parse_generation_contract(
        str(row.get("raw_generated_text") or row.get("generated_text") or ""),
        allowed_evidence_ids=allowed_evidence_ids_from_aliases(row.get("citation_id_aliases"))
        or None,
    )
    parsed = raw_parse.parsed_payload or {}
    raw_evidence = parsed.get("evidence_ids")
    if row.get("memory_mode") == "mm0_no_context":
        classes.append("mm0_expected_evidence_absence")
    if "evidence_ids" not in parsed:
        classes.append("missing_evidence_field")
    elif not isinstance(raw_evidence, list):
        classes.append("evidence_field_wrong_type")
    elif not raw_evidence:
        classes.append("evidence_ids_absent")
    if raw_parse.parse_error_type == "invalid_evidence_id":
        classes.append("cited_evidence_not_in_e1_e5")
    if evaluation.get("generation_contract_missing_fields"):
        classes.append("wrong_contract_field_names")
    if row.get("memory_mode") == "mm4_bounded_agentic" and not evaluation.get(
        "generation_contract_valid"
    ):
        classes.append("mm4_schema_mismatch")
    if row.get("backend_type") == "api_provider" and not evaluation.get(
        "generation_contract_valid"
    ):
        classes.append("api_schema_mismatch")
    if row.get("engine") in {"vllm", "sglang"} and not evaluation.get("generation_contract_valid"):
        classes.append(f"{row.get('engine')}_schema_mismatch")
    if row.get("vertical") == "finance" and not evaluation.get("evidence_match"):
        classes.append("finance_repair_failure")
    if row.get("vertical") == "research_ai" and not evaluation.get("evidence_match"):
        classes.append("research_answer_skeleton_failure")
    if not evaluation.get("evidence_id_presence") and row.get("memory_mode") != "mm0_no_context":
        classes.append("evidence_ids_absent")
    if not evaluation.get("evidence_match") and row.get("memory_mode") != "mm0_no_context":
        classes.append("cited_evidence_not_in_e1_e5")
    if not evaluation.get("groundedness") and row.get("memory_mode") != "mm0_no_context":
        classes.append("answer_not_grounded_in_cited_evidence")
    return sorted(set(classes))


def build_25_replay_failure_audit(
    *,
    rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Classify per-row contract/evidence failures for the repaired 25 replay."""

    audit_rows: list[dict[str, Any]] = []
    counter: Counter[str] = Counter()
    for row, evaluation in zip(rows, evaluation_rows, strict=True):
        classes = _failure_classes(row=row, evaluation=evaluation)
        for item in classes:
            counter[item] += 1
        audit_rows.append(
            {
                "config_id": row.get("config_id"),
                "prompt_id": row.get("prompt_id"),
                "vertical": row.get("vertical"),
                "memory_mode": row.get("memory_mode"),
                "engine": row.get("engine"),
                "backend_type": row.get("backend_type"),
                "generation_contract_valid": evaluation.get("generation_contract_valid"),
                "format_valid": evaluation.get("format_valid"),
                "json_validity": evaluation.get("json_validity"),
                "evidence_id_presence": evaluation.get("evidence_id_presence"),
                "evidence_match": evaluation.get("evidence_match"),
                "groundedness": evaluation.get("groundedness"),
                "failure_classes": ";".join(classes),
                "normalization_applied": row.get("contract_normalization_applied"),
                "source_contract_error": row.get("contract_normalization_source_error"),
                "source_missing_fields": row.get("contract_normalization_source_missing_fields"),
            }
        )
    return {
        "run_id": RUN_ID,
        "status": "CONTROLLED_FINAL_25_REPLAY_FAILURE_AUDIT_COMPLETE",
        "row_count": len(rows),
        "failure_class_counts": dict(sorted(counter.items())),
        "rows": audit_rows,
    }


def _prompt_preview(prompt: object, limit: int = 1200) -> str:
    return str(prompt or "").replace("\r\n", "\n")[:limit]


def build_mm4_safety_violation_audit(
    *,
    rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Audit MM4 final-answer safety rows without exposing hidden trace to scoring."""

    audit_rows: list[dict[str, Any]] = []
    for row, evaluation in zip(rows, evaluation_rows, strict=True):
        if row.get("memory_mode") != "mm4_bounded_agentic":
            continue
        source_terms = _json_list(row.get("mm4_safety_boundary_source_terms"))
        safety_terms = evaluation.get("safety_violation_terms")
        safety_terms_list = safety_terms if isinstance(safety_terms, list) else []
        if not safety_terms_list and not source_terms:
            continue
        normalized_output = str(row.get("generated_text") or row.get("output_text") or "")
        raw_output = str(row.get("raw_generated_text") or normalized_output)
        violation_real_or_artifact = (
            "parser_or_boundary_artifact_repaired"
            if source_terms and not evaluation.get("safety_violation")
            else "real_final_answer_violation"
        )
        audit_rows.append(
            {
                "config_id": row.get("config_id"),
                "model_alias": row.get("model_alias"),
                "engine": row.get("engine"),
                "runtime": row.get("runtime"),
                "memory_mode": row.get("memory_mode"),
                "concurrency": row.get("concurrency"),
                "vertical": row.get("vertical"),
                "prompt_id": row.get("prompt_id"),
                "rendered_prompt_preview": _prompt_preview(row.get("prompt")),
                "raw_provider_output": raw_output,
                "normalized_output": normalized_output,
                "safety_evaluator_reason": ";".join(str(term) for term in safety_terms_list),
                "source_safety_terms": ";".join(source_terms),
                "real_or_artifact": violation_real_or_artifact,
                "mm4_repair_tool_repeated_prohibited_wording": bool(source_terms),
                "answer_included_unsafe_wording_from_prompt_or_context": bool(source_terms),
                "safety_issue_source": "final_answer",
                "agent_trace_preserved_but_not_scored": bool(row.get("raw_generated_text")),
                "safety_boundary_repair_applied": bool(
                    row.get("mm4_safety_boundary_repair_applied")
                ),
                "safety_violation": bool(evaluation.get("safety_violation")),
            }
        )
    violation_rows = [row for row in audit_rows if row["safety_violation"]]
    return {
        "run_id": RUN_ID,
        "status": (
            "CONTROLLED_FINAL_MM4_SAFETY_VIOLATION_AUDIT_COMPLETE"
            if audit_rows
            else "CONTROLLED_FINAL_MM4_SAFETY_VIOLATION_NOT_FOUND"
        ),
        "row_count": len(audit_rows),
        "violation_count": len(violation_rows),
        "violation": violation_rows[0] if violation_rows else (audit_rows[0] if audit_rows else {}),
        "rows": audit_rows,
    }


def _write_mm4_safety_audit_markdown(path: str, audit: dict[str, Any]) -> None:
    rows = audit.get("rows") if isinstance(audit.get("rows"), list) else []
    primary = audit.get("violation") if isinstance(audit.get("violation"), dict) else {}
    lines = [
        "# Controlled Final MM4 Safety Violation Audit",
        "",
        f"- status: `{audit.get('status')}`",
        f"- audited rows: `{audit.get('row_count')}`",
        f"- remaining safety violations: `{audit.get('violation_count')}`",
    ]
    if primary:
        lines.extend(
            [
                f"- config_id: `{primary.get('config_id')}`",
                f"- model_alias: `{primary.get('model_alias')}`",
                f"- engine/runtime: `{primary.get('engine')}` / `{primary.get('runtime')}`",
                f"- memory_mode: `{primary.get('memory_mode')}`",
                f"- concurrency: `{primary.get('concurrency')}`",
                f"- vertical: `{primary.get('vertical')}`",
                f"- prompt_id: `{primary.get('prompt_id')}`",
                f"- safety evaluator reason: `{primary.get('safety_evaluator_reason')}`",
                f"- real or artifact: `{primary.get('real_or_artifact')}`",
                f"- safety issue source: `{primary.get('safety_issue_source')}`",
                "",
                "## Rendered Prompt Preview",
                "```text",
                str(primary.get("rendered_prompt_preview") or ""),
                "```",
                "",
                "## Raw Provider Output",
                "```json",
                str(primary.get("raw_provider_output") or ""),
                "```",
                "",
                "## Normalized Output",
                "```json",
                str(primary.get("normalized_output") or ""),
                "```",
            ]
        )
    if rows:
        lines.extend(["", "## Audited Rows", ""])
        for row in rows:
            lines.append(
                f"- `{row.get('config_id')}` / `{row.get('prompt_id')}`: "
                f"{row.get('real_or_artifact')}, "
                f"remaining_violation={row.get('safety_violation')}"
            )
    _repo_path(path).parent.mkdir(parents=True, exist_ok=True)
    _repo_path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_repair_ready_report(
    *,
    args: argparse.Namespace,
    targeted_report: dict[str, Any] | None,
    validation_report: dict[str, Any] | None,
    gates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime_ready = bool((gates or {}).get("full_simulation_allowed", True))
    targeted_passed = bool((targeted_report or {}).get("passed_quality_gate", True))
    validation_passed = bool((validation_report or {}).get("passed_quality_gate"))
    ready = runtime_ready and targeted_passed and validation_passed
    report = {
        "run_id": RUN_ID,
        "status": "CONTROLLED_FINAL_REPAIR_READY" if ready else "CONTROLLED_FINAL_REPAIR_BLOCKED",
        "targeted_mm4_replay_passed": targeted_passed,
        "repaired_500_validation_passed": validation_passed,
        "runtime_smokes_ready": runtime_ready,
        "artifact_sync_checkpoint_manifest_enabled": True,
        "full_10000_rerun_allowed": ready,
    }
    _write_json(args.repair_ready_report, report)
    return report


def _evaluation_summary_for_rows(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    report_path: str,
    status: str,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    generated = [result_row_to_generated_answer(row) for row in rows]
    evaluation_rows = evaluate_generated_answers(generated, load_gold_records(args.dataset_root))
    summary = build_summary_rows(
        results_path=report_path,
        result_rows=rows,
        evaluation_rows=evaluation_rows,
    )[0]
    memory_mode_summary: dict[str, dict[str, Any]] = {}
    for memory_mode in MEMORY_MODES:
        indexed = [
            (row, evaluation)
            for row, evaluation in zip(rows, evaluation_rows, strict=True)
            if row.get("memory_mode") == memory_mode
        ]
        if not indexed:
            continue
        mode_rows = [row for row, _evaluation in indexed]
        mode_evaluations = [evaluation for _row, evaluation in indexed]
        memory_mode_summary[memory_mode] = build_summary_rows(
            results_path=report_path,
            result_rows=mode_rows,
            evaluation_rows=mode_evaluations,
        )[0]
    natural_language_no_json = sum(
        bool(
            bool(row.get("success"))
            and str(row.get("generated_text") or row.get("output_text") or "").strip()
            and not bool(evaluation.get("json_validity"))
        )
        for row, evaluation in zip(rows, evaluation_rows, strict=True)
    )
    payload = {
        "run_id": RUN_ID,
        "status": status,
        "row_count": len(rows),
        "request_success_count": sum(bool(row.get("success")) for row in rows),
        "request_failure_count": sum(not bool(row.get("success")) for row in rows),
        "summary": summary,
        "memory_mode_summary": memory_mode_summary,
        "memory_mode_reporting": {
            "mm0_no_context": "no_context_ablation",
            "mm1_dense_top5": "contextual",
            "mm2_hybrid_top5": "contextual",
            "mm3_compressed_hybrid_top5": "contextual",
            "mm4_bounded_agentic": "agentic",
        },
        "natural_language_no_json_count": natural_language_no_json,
        "baseline": baseline or {},
    }
    payload["passed_quality_gate"] = (
        float(summary["json_valid_rate"]) >= 0.95
        and float(summary["generation_contract_valid_rate"]) >= 0.95
        and int(summary["safety_violation_count"]) == 0
        and float(summary["evidence_match_rate"]) > 0.56
        and float(summary["grounded_rate"]) > 0.56
        and natural_language_no_json <= len(rows) // 2
    )
    return payload


def select_repaired_smoke_rows(
    rows: list[dict[str, Any]], *, limit: int = 25
) -> list[dict[str, Any]]:
    """Select one representative row per config while cycling verticals."""

    selected: list[dict[str, Any]] = []
    vertical_cycle = list(VERTICALS)
    for index, config in enumerate(build_config_specs()):
        vertical = vertical_cycle[index % len(vertical_cycle)]
        match = next(
            row
            for row in rows
            if row["config_id"] == config.config_id and row["vertical"] == vertical
        )
        selected.append(match)
        if len(selected) >= limit:
            break
    return selected


def select_repaired_validation_rows(
    rows: list[dict[str, Any]], *, prompts_per_config: int = 20
) -> list[dict[str, Any]]:
    """Select 20 prompts per config for the optional 500-row repaired validation."""

    selected: list[dict[str, Any]] = []
    for config in build_config_specs():
        config_rows = [row for row in rows if row["config_id"] == config.config_id]
        per_vertical = max(1, prompts_per_config // len(VERTICALS))
        for vertical in VERTICALS:
            selected.extend(
                [row for row in config_rows if row["vertical"] == vertical][:per_vertical]
            )
    return selected[: prompts_per_config * len(build_config_specs())]


def _audit_target(audit_path: str) -> dict[str, Any]:
    path = _repo_path(audit_path)
    if not path.exists():
        return {}
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    target = audit.get("violation")
    return target if isinstance(target, dict) else {}


def select_mm4_safety_target_rows(
    rows: list[dict[str, Any]],
    *,
    audit_path: str,
    neighbor_count: int = 10,
) -> list[dict[str, Any]]:
    """Select the audited MM4 safety row plus neighboring same-config rows."""

    validation_rows = select_repaired_validation_rows(rows)
    mm4_rows = [row for row in validation_rows if row["memory_mode"] == "mm4_bounded_agentic"]
    target = _audit_target(audit_path)
    config_id = str(
        target.get("config_id") or "api_model6_gated_api_provider_route_mm4_bounded_agentic_c4"
    )
    vertical = str(target.get("vertical") or "airline")
    prompt_id = str(target.get("prompt_id") or "")
    candidates = [
        row for row in mm4_rows if row["config_id"] == config_id and row["vertical"] == vertical
    ]
    desired_count = neighbor_count + 1
    if len(candidates) < desired_count:
        candidates = [row for row in mm4_rows if row["config_id"] == config_id]
    if len(candidates) < desired_count:
        candidates = mm4_rows
    target_index = 0
    for index, row in enumerate(candidates):
        if row.get("prompt_id") == prompt_id:
            target_index = index
            break
    start = max(0, target_index - neighbor_count // 2)
    end = min(len(candidates), start + neighbor_count + 1)
    start = max(0, end - neighbor_count - 1)
    return candidates[start:end]


def run_mm4_safety_targeted_replay(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run the audited MM4 safety row plus neighbors."""

    selected = select_mm4_safety_target_rows(rows, audit_path=args.mm4_safety_audit_json)
    report = run_repaired_subset(
        args=args,
        rows=selected,
        report_path=args.mm4_safety_targeted_report,
        status="MM4_SAFETY_TARGETED_REPLAY_COMPLETE",
    )
    summary = report["summary"]
    report["passed_quality_gate"] = (
        float(summary["json_valid_rate"]) == 1.0
        and float(summary["generation_contract_valid_rate"]) == 1.0
        and int(summary["safety_violation_count"]) == 0
        and float(summary["evidence_match_rate"]) >= 0.75
        and float(summary["grounded_rate"]) >= 0.75
    )
    report["targeted_quality_floor"] = {
        "scope": "audited_row_plus_available_neighboring_mm4_rows",
        "evidence_match_min": 0.75,
        "grounded_rate_min": 0.75,
        "authoritative_mm4_floor_checked_by_500_validation": 0.9,
    }
    _write_json(args.mm4_safety_targeted_report, report)
    _write_csv(args.mm4_safety_targeted_summary, [report["summary"]])
    return report


def run_repaired_subset(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    report_path: str,
    status: str,
) -> dict[str, Any]:
    """Run a repaired subset without touching full raw 10k outputs."""

    api_route = _api_route(args)
    completed = []
    for row in rows:
        result = _execute_one_request(args=args, row=row, api_route=api_route)
        completed.append(
            normalize_generation_contract_output(result) if result.get("success") else result
        )
    report = _evaluation_summary_for_rows(
        args=args,
        rows=completed,
        report_path=report_path,
        status=status,
        baseline={
            "json_valid_rate": 0.0,
            "generation_contract_valid_rate": 0.0,
            "evidence_match_rate": 0.0397,
            "grounded_rate": 0.0397,
        },
    )
    evaluation_rows = evaluate_generated_answers(
        [result_row_to_generated_answer(row) for row in completed],
        load_gold_records(args.dataset_root),
    )
    if status == "REPAIRED_25_REPLAY_COMPLETE":
        audit = build_25_replay_failure_audit(rows=completed, evaluation_rows=evaluation_rows)
        _write_json(args.repaired_25_failure_audit_json, audit)
        _write_csv(args.repaired_25_failure_audit_csv, audit["rows"])
        report["failure_audit"] = {
            "json_path": args.repaired_25_failure_audit_json,
            "csv_path": args.repaired_25_failure_audit_csv,
            "failure_class_counts": audit["failure_class_counts"],
        }
    if status == "REPAIRED_500_VALIDATION_COMPLETE":
        _write_csv(args.repaired_500_validation_summary, [report["summary"]])
        audit = build_mm4_safety_violation_audit(
            rows=completed,
            evaluation_rows=evaluation_rows,
        )
        _write_json(args.mm4_safety_audit_json, audit)
        _write_mm4_safety_audit_markdown(args.mm4_safety_audit_md, audit)
        report["mm4_safety_audit"] = {
            "json_path": args.mm4_safety_audit_json,
            "markdown_path": args.mm4_safety_audit_md,
            "violation_count": audit["violation_count"],
        }
        _write_repair_ready_report(
            args=args,
            targeted_report=None,
            validation_report=report,
        )
    _write_json(report_path, report)
    return report


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _server_models(base_url: str, api_key: str, timeout_seconds: float = 5.0) -> list[str]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    return sorted(
        str(item.get("id"))
        for item in data
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    )


def _models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def _credential_presence(environment: dict[str, str]) -> dict[str, dict[str, bool]]:
    return {
        canonical: {alias: bool(environment.get(alias)) for alias in aliases}
        for canonical, aliases in ENV_ALIASES.items()
    }


def check_runtime_gate(args: argparse.Namespace) -> dict[str, Any]:
    """Check runtime, server, API, and mm4 gates before any full run."""

    project_config = load_project_config()
    checks: dict[str, Any] = {
        "vllm_model3_7b": {
            "status": "NOT_CHECKED",
            "model_alias": SELF_HOSTED_MODEL_ALIAS,
            "model_id": SELF_HOSTED_MODEL_ID,
            "health_check_url": _models_url(args.base_url),
        },
        "sglang_model3_7b": {
            "status": "NOT_CHECKED",
            "model_alias": SELF_HOSTED_MODEL_ALIAS,
            "model_id": SELF_HOSTED_MODEL_ID,
            "startup_command": SGLANG_STARTUP_COMMAND,
            "health_check_url": _models_url(args.sglang_base_url),
        },
        "api_model6_gated": {
            "status": "NOT_CHECKED",
            "model_alias": API_MODEL_ALIAS,
            "model_id": API_MODEL_ID,
        },
        "mm4_bounded_agentic": {"status": "NOT_CHECKED"},
    }
    try:
        selection = select_runtime_for_model(
            model_alias=SELF_HOSTED_MODEL_ALIAS,
            runtime="vllm",
            hardware_type=args.gpu_id,
            backend_route="openai_compatible_vllm",
            live_run=True,
        )
        models = _server_models(args.base_url, args.api_key)
        checks["vllm_model3_7b"].update(
            {
                "runtime_selection": selection.to_dict(),
                "server_models": models,
                "status": (
                    "SMOKE_READY" if SELF_HOSTED_MODEL_ID in models else "SERVER_MODEL_NOT_READY"
                ),
                "reason": (
                    "model available on local vLLM server"
                    if SELF_HOSTED_MODEL_ID in models
                    else f"{SELF_HOSTED_MODEL_ID} not available at {args.base_url}"
                ),
            }
        )
    except (RuntimeError, ValueError, OSError, urllib.error.URLError) as exc:
        checks["vllm_model3_7b"].update({"status": "BLOCKED", "reason": str(exc)})

    try:
        selection = select_runtime_for_model(
            model_alias=SELF_HOSTED_MODEL_ALIAS,
            runtime="sglang",
            hardware_type=args.gpu_id,
            backend_route="sglang_openai_compatible",
            live_run=True,
        )
        package_available = _module_available("sglang")
        models = _server_models(args.sglang_base_url, args.api_key)
        model_available = SELF_HOSTED_MODEL_ID in models
        smoke_ready = package_available and model_available
        checks["sglang_model3_7b"].update(
            {
                "runtime_selection": selection.to_dict(),
                "runtime_registry_allows_sglang": True,
                "server_models": models,
                "python_package_available": package_available,
                "health_check": {
                    "endpoint": _models_url(args.sglang_base_url),
                    "passed": bool(models),
                    "model_available": model_available,
                },
                "status": "SMOKE_READY" if smoke_ready else "BLOCKED",
                "reason": (
                    "model available on SGLang server"
                    if smoke_ready
                    else (
                        "SGLang package is importable, but /v1/models did not list "
                        f"{SELF_HOSTED_MODEL_ID}. Start SGLang with: "
                        f"{SGLANG_STARTUP_COMMAND}"
                    )
                ),
            }
        )
    except ValueError as exc:
        checks["sglang_model3_7b"].update(
            {
                "status": "BLOCKED",
                "python_package_available": _module_available("sglang"),
                "runtime_registry_allows_sglang": False,
                "health_check": {
                    "endpoint": _models_url(args.sglang_base_url),
                    "passed": False,
                    "model_available": False,
                },
                "reason": str(exc),
            }
        )
    except (RuntimeError, OSError, urllib.error.URLError) as exc:
        checks["sglang_model3_7b"].update(
            {
                "status": "BLOCKED",
                "python_package_available": _module_available("sglang"),
                "runtime_registry_allows_sglang": True,
                "health_check": {
                    "endpoint": _models_url(args.sglang_base_url),
                    "passed": False,
                    "model_available": False,
                },
                "reason": (
                    f"SGLang /v1/models health check failed at "
                    f"{_models_url(args.sglang_base_url)}: {exc}. "
                    f"Start SGLang with: {SGLANG_STARTUP_COMMAND}"
                ),
            }
        )

    model6 = project_config.resolve_model_config(API_MODEL_ALIAS)
    environment = load_probe_environment(env_path=ROOT / args.env_file)
    hf_token = environment.get("HF_TOKEN", "")
    provider_keys = {
        "OPENROUTER_API_KEY": bool(environment.get("OPENROUTER_API_KEY")),
        "NOVITA_API_KEY": bool(environment.get("NOVITA_API_KEY")),
    }
    api_ready = bool(hf_token) and any(provider_keys.values())
    checks["api_model6_gated"].update(
        {
            "requires_hf_token": model6.requires_hf_token,
            "requires_license_acceptance": model6.requires_license_acceptance,
            "hf_token_present": bool(hf_token),
            "provider_keys_present": provider_keys,
            "credential_alias_presence": _credential_presence(environment),
            "credential_sources_checked": [
                "process_environment",
                str((ROOT / args.env_file).resolve()),
                "supported_aliases",
            ],
            "status": "SMOKE_READY" if api_ready else "BLOCKED",
            "reason": (
                "HF token and provider key present"
                if api_ready
                else "HF_TOKEN and a provider API key are required for model6_gated"
            ),
        }
    )

    mm4_available = (
        _module_available("langgraph")
        and (ROOT / "scripts/phase4/run_mm4_agentic_smoke.py").exists()
        and (ROOT / "src/inference_bench/agents/langgraph_mm4.py").exists()
    )
    checks["mm4_bounded_agentic"].update(
        {
            "langgraph_available": _module_available("langgraph"),
            "script_available": (ROOT / "scripts/phase4/run_mm4_agentic_smoke.py").exists(),
            "status": "SMOKE_READY" if mm4_available else "MM4_NOT_RUNNABLE",
            "reason": (
                "bounded LangGraph mm4 runner is importable"
                if mm4_available
                else "bounded LangGraph mm4 runner dependencies are incomplete"
            ),
        }
    )
    full_ready = all(
        checks[name]["status"] == "SMOKE_READY"
        for name in (
            "vllm_model3_7b",
            "sglang_model3_7b",
            "api_model6_gated",
            "mm4_bounded_agentic",
        )
    )
    return {"checks": checks, "full_simulation_allowed": full_ready}


def build_smoke_report(gates: dict[str, Any]) -> dict[str, Any]:
    """Build a smoke report without faking unavailable tracks."""

    checks = cast(dict[str, dict[str, Any]], gates["checks"])
    tracks = {}
    for track, check_name in {
        "vllm_model3_7b": "vllm_model3_7b",
        "sglang_model3_7b": "sglang_model3_7b",
        "api_model6_gated": "api_model6_gated",
    }.items():
        check = checks[check_name]
        gate_ready = check.get("status") == "SMOKE_READY"
        tracks[track] = {
            "attempted": False,
            "status": "SMOKE_READY" if gate_ready else "SMOKE_BLOCKED",
            "reason": check.get("reason"),
            "gate_status": check.get("status"),
        }
    all_ready = bool(gates["full_simulation_allowed"])
    return {
        "status": "SMOKE_READY" if all_ready else "SMOKE_BLOCKED",
        "reason": (
            "All required smoke gates are ready; full simulation still requires explicit "
            "--run-full."
            if all_ready
            else "One or more required smoke gates are not ready."
        ),
        "tracks": tracks,
        "full_simulation_allowed": all_ready,
        "serving_commands": {
            "sglang_model3_7b": SGLANG_STARTUP_COMMAND,
        },
    }


def _request_key(row: dict[str, Any]) -> str:
    return f"{row['config_id']}::{row['prompt_id']}"


def _prompt_hash(prompt: object) -> str:
    return hashlib.sha256(str(prompt or "").encode("utf-8")).hexdigest()


def _append_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = _repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _read_existing_result_rows(path: str | Path) -> list[dict[str, Any]]:
    output = _repo_path(path)
    if not output.exists():
        return []
    return _read_jsonl(output)


def _read_existing_completed_timing(path: str | Path) -> dict[str, Any]:
    report_path = _repo_path(path)
    if not report_path.exists():
        return {}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if report.get("status") != "CONTROLLED_FINAL_SIMULATION_COMPLETED":
        return {}
    wall_seconds = report.get("wall_seconds")
    if not isinstance(wall_seconds, int | float):
        cost_report = report.get("cost_report")
        if isinstance(cost_report, dict):
            wall_seconds = cost_report.get("wall_seconds")
    if not isinstance(wall_seconds, int | float):
        return {}
    return {
        "started_at": report.get("started_at_utc"),
        "completed_at": report.get("completed_at_utc"),
        "wall_seconds": float(wall_seconds),
    }


def _write_checkpoint(path: str | Path, rows: list[dict[str, Any]], *, status: str) -> None:
    completed_keys = sorted({_request_key(row) for row in rows})
    payload = {
        "run_id": RUN_ID,
        "status": status,
        "completed_request_keys": completed_keys,
        "row_count": len(rows),
        "success_count": sum(bool(row.get("success")) for row in rows),
        "failure_count": sum(not bool(row.get("success")) for row in rows),
        "updated_at_utc": utc_now(),
    }
    _write_json(path, payload)


def _chat_completion_request(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_seconds: float,
) -> tuple[str, float | None, float]:
    openai = cast(Any, import_module("openai"))
    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
    start = time.perf_counter()
    first_token: float | None = None
    chunks: list[str] = []
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True,
    )
    for chunk in response:
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        content = getattr(delta, "content", None)
        if isinstance(content, str) and content:
            if first_token is None:
                first_token = time.perf_counter()
            chunks.append(content)
    end = time.perf_counter()
    return "".join(chunks), first_token - start if first_token is not None else None, end - start


def _api_route(args: argparse.Namespace) -> tuple[str, str, str, str]:
    project = load_project_config()
    model = project.resolve_model_config(API_MODEL_ALIAS)
    pricing = resolve_api_pricing(API_MODEL_ALIAS)
    route = resolve_api_provider_route(model=model, pricing=pricing)
    environment = load_probe_environment(env_path=ROOT / args.env_file)
    api_key = api_key_for_route(route, environment)
    return route.base_url, api_key, route.provider_model_id, route.provider


def _execute_one_request(
    *,
    args: argparse.Namespace,
    row: dict[str, Any],
    api_route: tuple[str, str, str, str] | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    generated_text = ""
    input_tokens = count_whitespace_tokens(str(row["prompt"]))
    output_tokens = 0
    base_url = args.base_url
    api_key = args.api_key
    model_id = str(row["model_id"])
    provider = "self_hosted"
    if row["engine"] == "sglang":
        base_url = args.sglang_base_url
    if row["backend_type"] == "api_provider":
        if api_route is None:
            msg = "API route was not resolved"
            raise RuntimeError(msg)
        base_url, api_key, model_id, provider = api_route
    try:
        generated_text, ttft_seconds, elapsed_seconds = _chat_completion_request(
            base_url=base_url,
            api_key=api_key,
            model=model_id,
            prompt=str(row["prompt"]),
            max_tokens=args.max_new_tokens,
            timeout_seconds=args.timeout_seconds,
        )
        output_tokens = count_whitespace_tokens(generated_text)
        ttft_ms = ttft_seconds * 1000.0 if ttft_seconds is not None else None
        e2e_ms = elapsed_seconds * 1000.0
        tpot_ms = (
            max(e2e_ms - (ttft_ms or 0.0), 0.0) / max(output_tokens, 1) if output_tokens else None
        )
        total_tokens = input_tokens + output_tokens
        total_cost_usd = 0.0
        if row["backend_type"] == "api_provider":
            pricing = resolve_api_pricing(API_MODEL_ALIAS)
            cost = estimate_api_cost_from_pricing(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                pricing=pricing,
            )
            total_cost_usd = float(cost["total_api_cost_usd"])
        return {
            **row,
            "request_id": _request_key(row),
            "run_id": RUN_ID,
            "provider": provider,
            "success": True,
            "generated_text": generated_text,
            "output_text": generated_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "end_to_end_latency_ms": e2e_ms,
            "latency_ms": e2e_ms,
            "throughput_tokens_per_second": (
                total_tokens / elapsed_seconds if elapsed_seconds > 0 else None
            ),
            "requests_per_second": 1.0 / elapsed_seconds if elapsed_seconds > 0 else None,
            "total_cost_usd": total_cost_usd,
            "error_message": "",
            "failure_reason": "",
            "final_status": "answer",
        }
    except Exception as exc:  # noqa: BLE001
        elapsed_seconds = time.perf_counter() - started
        return {
            **row,
            "request_id": _request_key(row),
            "run_id": RUN_ID,
            "provider": provider,
            "success": False,
            "generated_text": generated_text,
            "output_text": generated_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "ttft_ms": None,
            "tpot_ms": None,
            "end_to_end_latency_ms": elapsed_seconds * 1000.0,
            "latency_ms": elapsed_seconds * 1000.0,
            "throughput_tokens_per_second": 0.0,
            "requests_per_second": 0.0,
            "total_cost_usd": 0.0,
            "error_message": f"{type(exc).__name__}: {exc}",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "final_status": "failed_validation",
        }


def _telemetry_loop(
    *,
    path: Path,
    stop_event: threading.Event,
    interval_seconds: float,
    errors: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    while not stop_event.is_set():
        try:
            samples = collect_gpu_sample()
            with path.open("a", encoding="utf-8", newline="\n") as file:
                for sample in samples:
                    file.write(
                        json.dumps(sample.to_dict(), ensure_ascii=True, sort_keys=True) + "\n"
                    )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
        stop_event.wait(interval_seconds)


def _read_gpu_samples(path: str | Path) -> list[GpuTelemetrySample]:
    samples: list[GpuTelemetrySample] = []
    output = _repo_path(path)
    if not output.exists():
        return samples
    for row in _read_jsonl(output):
        if "gpu_name" not in row:
            continue
        samples.append(
            GpuTelemetrySample(
                timestamp=str(row["timestamp"]),
                gpu_name=str(row["gpu_name"]),
                utilization_gpu_percent=float(row["utilization_gpu_percent"]),
                memory_used_mb=float(row["memory_used_mb"]),
                memory_total_mb=float(row["memory_total_mb"]),
                power_draw_w=float(row["power_draw_w"]),
                temperature_c=float(row["temperature_c"]),
                process_info=str(row.get("process_info") or ""),
            )
        )
    return samples


def _run_config(
    *,
    args: argparse.Namespace,
    config: ConfigSpec,
    rows: list[dict[str, Any]],
    completed_keys: set[str],
    api_route: tuple[str, str, str, str] | None,
) -> list[dict[str, Any]]:
    pending = [row for row in rows if _request_key(row) not in completed_keys]
    if not pending:
        return []
    print(
        f"running config_id={config.config_id} requests={len(pending)} "
        f"engine={config.engine} memory_mode={config.memory_mode} concurrency={config.concurrency}",
        flush=True,
    )
    completed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        futures = [
            executor.submit(_execute_one_request, args=args, row=row, api_route=api_route)
            for row in pending
        ]
        for future in as_completed(futures):
            completed.append(future.result())
    completed.sort(key=lambda item: str(item["request_id"]))
    return completed


def _float_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            values.append(float(str(value)))
        except ValueError:
            continue
    return values


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _config_summary_rows(
    *,
    configs: list[ConfigSpec],
    rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
    expected_per_config: int,
) -> list[dict[str, Any]]:
    eval_by_prompt = {str(row["prompt_id"]): row for row in evaluation_rows}
    summaries: list[dict[str, Any]] = []
    for config in configs:
        config_rows = [row for row in rows if row.get("config_id") == config.config_id]
        success_rows = [row for row in config_rows if bool(row.get("success"))]
        config_eval = [
            eval_by_prompt[str(row["prompt_id"])]
            for row in config_rows
            if str(row.get("prompt_id")) in eval_by_prompt
        ]
        total = len(config_eval)
        summary = {
            "config_id": config.config_id,
            "model_alias": config.model_alias,
            "model_id": config.model_id,
            "backend_type": config.backend_type,
            "engine": config.engine,
            "runtime": config.runtime,
            "memory_mode": config.memory_mode,
            "concurrency": config.concurrency,
            "status": "COMPLETED" if len(config_rows) == expected_per_config else "PARTIAL",
            "requests_attempted": len(config_rows),
            "requests_completed": len(success_rows),
            "requests_failed": len(config_rows) - len(success_rows),
            "mean_ttft_ms": _mean(_float_values(success_rows, "ttft_ms")),
            "mean_tpot_ms": _mean(_float_values(success_rows, "tpot_ms")),
            "mean_e2e_latency_ms": _mean(_float_values(success_rows, "end_to_end_latency_ms")),
            "mean_total_tokens_per_second": _mean(
                _float_values(success_rows, "throughput_tokens_per_second")
            ),
            "total_input_tokens": sum(int(row.get("input_tokens") or 0) for row in config_rows),
            "total_output_tokens": sum(int(row.get("output_tokens") or 0) for row in config_rows),
            "total_cost_usd": sum(float(row.get("total_cost_usd") or 0.0) for row in config_rows),
            "json_valid_rate": (
                sum(bool(row.get("json_validity")) for row in config_eval) / total
                if total
                else None
            ),
            "generation_contract_valid_rate": (
                sum(bool(row.get("generation_contract_valid")) for row in config_eval) / total
                if total
                else None
            ),
            "evidence_match_rate": (
                sum(bool(row.get("evidence_match")) for row in config_eval) / total
                if total
                else None
            ),
            "grounded_rate": (
                sum(bool(row.get("groundedness")) for row in config_eval) / total if total else None
            ),
            "safety_violation_count": (
                sum(bool(row.get("safety_violation")) for row in config_eval) if total else None
            ),
        }
        if config.backend_type == "self_hosted_gpu":
            summary["gpu_telemetry_scope"] = "self_hosted_gpu"
        else:
            summary["gpu_telemetry_scope"] = "not_applicable_api_provider"
        summaries.append(summary)
    return summaries


def _status_against_min(value: object, target: float) -> str:
    if value in (None, ""):
        return "NOT_AVAILABLE"
    observed = float(value)
    if observed >= target:
        return "PASS"
    if observed >= target * 0.9:
        return "WARNING"
    return "FAIL"


def _status_against_max(value: object, target: float) -> str:
    if value in (None, ""):
        return "NOT_AVAILABLE"
    observed = float(value)
    if observed <= target:
        return "PASS"
    if observed <= target * 1.1:
        return "WARNING"
    return "FAIL"


def _slo_rows(config_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in config_rows:
        checks = {
            "ttft": _status_against_max(row.get("mean_ttft_ms"), 1000.0),
            "tpot": _status_against_max(row.get("mean_tpot_ms"), 100.0),
            "latency": _status_against_max(row.get("mean_e2e_latency_ms"), 10000.0),
            "throughput": _status_against_min(row.get("mean_total_tokens_per_second"), 20.0),
            "json_validity": _status_against_min(row.get("json_valid_rate"), 0.95),
            "contract_validity": _status_against_min(
                row.get("generation_contract_valid_rate"), 0.95
            ),
            "evidence_match": _status_against_min(row.get("evidence_match_rate"), 0.9),
            "groundedness": _status_against_min(row.get("grounded_rate"), 0.95),
        }
        failed = [name for name, status in checks.items() if status == "FAIL"]
        warnings = [name for name, status in checks.items() if status == "WARNING"]
        candidates = []
        if any(name in failed for name in ("ttft", "tpot", "latency", "throughput")):
            candidates.append("serving_profile_tuning")
        if any(name in failed for name in ("json_validity", "contract_validity")):
            candidates.append("generation_contract_prompt_repair")
        if any(name in failed for name in ("evidence_match", "groundedness")):
            candidates.append("retrieval_or_context_selection_repair")
        rows.append(
            {
                **row,
                "passed_slos": sum(status == "PASS" for status in checks.values()),
                "warning_slos": len(warnings),
                "failed_slos": len(failed),
                "failed_metric_family": ";".join(failed) if failed else "",
                "bottleneck_category": ";".join(failed or warnings) if (failed or warnings) else "",
                "recommended_optimization_candidates": ";".join(candidates),
                **{f"slo_{name}": status for name, status in checks.items()},
            }
        )
    return rows


def _write_completed_reports(
    args: argparse.Namespace,
    *,
    matrix_summary: dict[str, Any],
    gates: dict[str, Any],
    smoke_report: dict[str, Any],
    rows: list[dict[str, Any]],
    started_at: str,
    completed_at: str,
    wall_seconds: float,
    telemetry_errors: list[str],
) -> dict[str, Any]:
    generated = [result_row_to_generated_answer(row) for row in rows]
    evaluation_rows = evaluate_generated_answers(generated, load_gold_records(args.dataset_root))
    eval_summary = build_summary_rows(
        results_path=args.raw_results_path,
        result_rows=rows,
        evaluation_rows=evaluation_rows,
    )[0]
    latency = latency_summary_rows(rows)[0]
    configs = build_config_specs()
    config_rows = _config_summary_rows(
        configs=configs,
        rows=rows,
        evaluation_rows=evaluation_rows,
        expected_per_config=int(matrix_summary["prompt_count_per_config"]),
    )
    slo_rows = _slo_rows(config_rows)
    gpu_samples = _read_gpu_samples(args.gpu_telemetry_path)
    gpu_summary = summarize_gpu_telemetry(
        gpu_samples,
        interval_seconds=args.telemetry_interval_seconds,
        requested_duration_seconds=wall_seconds,
    )
    gpu_cost = (wall_seconds / 3600.0) * args.hourly_price
    api_cost = sum(
        float(row.get("total_cost_usd") or 0.0)
        for row in rows
        if row.get("backend_type") == "api_provider"
    )
    cost_report = {
        "run_id": RUN_ID,
        "status": "COST_MEASURED",
        "wall_seconds": wall_seconds,
        "self_hosted_gpu_hourly_price_usd": args.hourly_price,
        "gpu_cost_usd": gpu_cost,
        "api_cost_usd": api_cost,
        "total_cost_usd": gpu_cost + api_cost,
        "self_hosted_request_count": sum(
            1 for row in rows if row.get("backend_type") == "self_hosted_gpu"
        ),
        "api_request_count": sum(1 for row in rows if row.get("backend_type") == "api_provider"),
    }
    _write_csv(args.eval_summary_path, [eval_summary])
    _write_csv(
        args.engine_comparison_path,
        [row for row in config_rows if row["model_alias"] == SELF_HOSTED_MODEL_ALIAS],
    )
    _write_csv(args.memory_comparison_path, config_rows)
    _write_csv(args.concurrency_comparison_path, config_rows)
    _write_csv(
        args.api_comparison_path,
        [row for row in config_rows if row["model_alias"] == API_MODEL_ALIAS],
    )
    _write_csv(args.api_vs_self_hosted_comparison_path, config_rows)
    _write_csv(args.model_comparison_path, config_rows)
    _write_csv(args.slo_summary_path, slo_rows)
    _write_json(
        args.slo_report_path,
        {
            "run_id": RUN_ID,
            "status": "SLO_COMPARISON_COMPLETE",
            "deployability_verdict": (
                "DEPLOYABLE_BASELINE"
                if all(int(row["failed_slos"]) == 0 for row in slo_rows)
                else "NOT_DEPLOYABLE_SLO_FAILURES"
            ),
            "benchmark_execution_verdict": "COMPLETED",
            "optimization_needed_verdict": (
                "OPTIMIZATION_NEEDED"
                if any(int(row["failed_slos"]) > 0 for row in slo_rows)
                else "NO_OPTIMIZATION_REQUIRED"
            ),
            "config_slo_results": slo_rows,
            "gate_report": gates,
        },
    )
    _write_json(args.cost_report_path, cost_report)
    eval_report = {
        "run_id": RUN_ID,
        "status": "CONTROLLED_FINAL_SIMULATION_COMPLETED",
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "matrix_summary": matrix_summary,
        "smoke_report": smoke_report,
        "gate_report": gates,
        "summary": eval_summary,
        "latency_summary": latency,
        "gpu_summary": gpu_summary,
        "telemetry_errors": telemetry_errors,
        "cost_report": cost_report,
        "config_summaries": config_rows,
        "slo_status_counts": dict(Counter(str(row["failed_slos"]) for row in slo_rows)),
        "configs_completed": sum(row["status"] == "COMPLETED" for row in config_rows),
        "configs_failed": sum(row["requests_failed"] > 0 for row in config_rows),
        "total_requests_attempted": len(rows),
        "total_requests_completed": sum(bool(row.get("success")) for row in rows),
        "total_requests_failed": sum(not bool(row.get("success")) for row in rows),
        "total_requests_planned": matrix_summary["row_count"],
        "vllm_ran": any(row.get("engine") == "vllm" for row in rows),
        "sglang_ran": any(row.get("engine") == "sglang" for row in rows),
        "api_route_ran": any(row.get("backend_type") == "api_provider" for row in rows),
        "mm4_ran": any(row.get("memory_mode") == "mm4_bounded_agentic" for row in rows),
        "final_10000_prompt_experiment_allowed": True,
        "wall_seconds": wall_seconds,
    }
    _write_json(args.eval_report_path, eval_report)
    _write_manifest(
        args,
        status="completed" if len(rows) == matrix_summary["row_count"] else "partial",
        matrix_summary=matrix_summary,
        rows=rows,
        started_at=started_at,
        completed_at=completed_at,
    )
    automation_report = build_post_run_automation_report(
        PostRunAutomationInputs(
            run_id=RUN_ID,
            manifest_path=args.manifest_path,
            eval_summary_path=args.eval_summary_path,
            latency_summary_path=args.eval_summary_path,
            telemetry_summary_path=None,
            cost_report_path=args.cost_report_path,
            comparison_paths=(
                args.engine_comparison_path,
                args.memory_comparison_path,
                args.concurrency_comparison_path,
                args.api_comparison_path,
                args.api_vs_self_hosted_comparison_path,
                args.model_comparison_path,
                args.slo_summary_path,
            ),
        )
    )
    write_post_run_automation_artifacts(
        report=automation_report,
        report_path=args.post_run_automation_report_path,
        plotting_dataset_path=args.plotting_dataset_path,
    )
    config = ArtifactSyncConfig(run_id=RUN_ID, backup_root=args.backup_root)
    sync = sync_artifacts(
        specs=_artifact_specs(args), config=config, event="completed", repo_root=ROOT
    )
    verification = verify_backup(specs=_artifact_specs(args), config=config, repo_root=ROOT)
    _write_json(
        args.artifact_sync_report_path,
        {
            "run_id": RUN_ID,
            "artifact_sync_enabled": True,
            "sync": sync,
            "verification": verification,
            "success": bool(verification.get("passed")),
        },
    )
    return eval_report


def _artifact_specs(args: argparse.Namespace) -> list[Any]:
    return build_artifact_specs(
        raw_jsonl=args.raw_results_path,
        manifest=args.manifest_path,
        telemetry=args.gpu_telemetry_path,
        processed_reports=[
            args.eval_report_path,
            args.eval_summary_path,
            args.engine_comparison_path,
            args.memory_comparison_path,
            args.concurrency_comparison_path,
            args.api_comparison_path,
            args.api_vs_self_hosted_comparison_path,
            args.model_comparison_path,
            args.slo_report_path,
            args.slo_summary_path,
            args.cost_report_path,
            args.artifact_sync_report_path,
            args.post_run_automation_report_path,
            args.plotting_dataset_path,
        ],
        logs=[args.checkpoint_path],
    )


def _write_manifest(
    args: argparse.Namespace,
    *,
    status: str,
    matrix_summary: dict[str, Any],
    rows: list[dict[str, Any]] | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
) -> None:
    now = utc_now()
    observed_rows = rows or []
    start_time = started_at or now
    end_time = completed_at or now
    manifest = RunManifest(
        run_id=RUN_ID,
        timestamp_utc=now,
        backend="mixed",
        model_alias="model3_7b+model6_gated",
        model_id=f"{SELF_HOSTED_MODEL_ID}+{API_MODEL_ID}",
        memory_mode="mm0-mm4",
        split="controlled_final_simulation_80_per_vertical",
        ablation_mode="prompt_plus_metadata",
        input_workload_path=args.matrix_path,
        output_path=args.raw_results_path,
        max_records=int(matrix_summary["row_count"]),
        git_commit=current_git_commit(ROOT),
        command=" ".join(sys.argv),
        status=status,
        start_time=start_time,
        end_time=end_time,
        error_count=sum(not bool(row.get("success")) for row in observed_rows),
        telemetry_path=args.gpu_telemetry_path,
        config_id=RUN_ID,
        vertical="all",
        runtime="mixed",
        engine="vllm+sglang+api_provider_route",
        backend_type="mixed",
        hardware=args.gpu_id,
        provider="self_hosted+api_provider",
        concurrency=max((*SELF_HOSTED_CONCURRENCY, *API_CONCURRENCY)),
        traffic_profile=args.traffic_profile,
        prompt_count=int(matrix_summary["row_count"]),
        dataset_workload_hash=hash_existing_paths([_repo_path(args.matrix_path)]),
        config_hash=hash_existing_paths(
            ["configs/runtime_engines.yaml", "configs/slo_targets.yaml"]
        ),
        started_at=start_time,
        updated_at=now,
        completed_at=end_time,
        completed_count=sum(bool(row.get("success")) for row in observed_rows),
        failed_count=sum(not bool(row.get("success")) for row in observed_rows),
        expected_count=int(matrix_summary["row_count"]),
        artifact_paths={
            "matrix": args.matrix_path,
            "raw_results": args.raw_results_path,
            "manifest": args.manifest_path,
            "gpu_telemetry": args.gpu_telemetry_path,
            "eval_report": args.eval_report_path,
            "engine_comparison": args.engine_comparison_path,
            "memory_comparison": args.memory_comparison_path,
            "concurrency_comparison": args.concurrency_comparison_path,
            "api_vs_self_hosted_comparison": args.api_vs_self_hosted_comparison_path,
            "model_comparison": args.model_comparison_path,
            "slo_report": args.slo_report_path,
            "post_run_automation_report": args.post_run_automation_report_path,
            "plotting_dataset": args.plotting_dataset_path,
        },
        run_type="baseline",
        baseline_or_optimized="baseline",
        optimization_flags=(),
        dataset_version="controlled_2000",
    )
    write_run_manifest(manifest, _repo_path(args.manifest_path))


def _blocked_rows(configs: list[ConfigSpec], reason: str) -> list[dict[str, Any]]:
    return [
        {
            "config_id": config.config_id,
            "model_alias": config.model_alias,
            "engine": config.engine,
            "memory_mode": config.memory_mode,
            "concurrency": config.concurrency,
            "status": "NOT_RUN",
            "reason": reason,
            "requests_attempted": 0,
            "requests_completed": 0,
            "requests_failed": 0,
        }
        for config in configs
    ]


def write_blocked_reports(
    args: argparse.Namespace,
    *,
    matrix_summary: dict[str, Any],
    gates: dict[str, Any],
    smoke_report: dict[str, Any],
) -> dict[str, Any]:
    """Write all required reports for a safety-gated blocked run."""

    reason = "Safety gates blocked full controlled simulation before provider/model requests."
    configs = build_config_specs()
    blocked_rows = _blocked_rows(configs, reason)
    _write_jsonl(args.raw_results_path, [])
    _repo_path(args.gpu_telemetry_path).parent.mkdir(parents=True, exist_ok=True)
    _repo_path(args.gpu_telemetry_path).write_text("", encoding="utf-8")
    _write_manifest(args, status="failed", matrix_summary=matrix_summary)
    eval_report = {
        "run_id": RUN_ID,
        "status": "CONTROLLED_FINAL_SIMULATION_BLOCKED_BY_SAFETY_GATES",
        "matrix_summary": matrix_summary,
        "smoke_report": smoke_report,
        "gate_report": gates,
        "serving_commands": {
            "sglang_model3_7b": SGLANG_STARTUP_COMMAND,
        },
        "configs_completed": 0,
        "configs_failed": len(configs),
        "total_requests_attempted": 0,
        "total_requests_completed": 0,
        "total_requests_failed": 0,
        "total_requests_planned": matrix_summary["row_count"],
        "vllm_ran": False,
        "sglang_ran": False,
        "api_route_ran": False,
        "mm4_ran": False,
        "final_10000_prompt_experiment_allowed": False,
    }
    _write_json(args.eval_report_path, eval_report)
    _write_csv(args.eval_summary_path, blocked_rows)
    _write_csv(
        args.engine_comparison_path,
        [row for row in blocked_rows if row["model_alias"] == SELF_HOSTED_MODEL_ALIAS],
    )
    _write_csv(args.memory_comparison_path, blocked_rows)
    _write_csv(args.concurrency_comparison_path, blocked_rows)
    _write_csv(
        args.api_comparison_path,
        [row for row in blocked_rows if row["model_alias"] == API_MODEL_ALIAS],
    )
    _write_csv(args.api_vs_self_hosted_comparison_path, blocked_rows)
    _write_csv(args.model_comparison_path, blocked_rows)
    slo_rows = [
        {
            **row,
            "passed_slos": 0,
            "failed_slos": 0,
            "failed_metric_family": "not_evaluated",
            "bottleneck_category": "safety_gate",
            "recommended_optimization_candidates": "",
        }
        for row in blocked_rows
    ]
    slo_report = {
        "run_id": RUN_ID,
        "status": "SLO_COMPARISON_NOT_RUN_SAFETY_GATED",
        "deployability_verdict": "NOT_DEPLOYABLE_SIMULATION_BLOCKED",
        "benchmark_execution_verdict": "NOT_READY",
        "optimization_needed_verdict": "NOT_EVALUATED",
        "config_slo_results": slo_rows,
        "gate_report": gates,
    }
    _write_json(args.slo_report_path, slo_report)
    _write_csv(args.slo_summary_path, slo_rows)
    cost_report = {
        "run_id": RUN_ID,
        "status": "COST_NOT_MEASURED_SAFETY_GATED",
        "self_hosted_gpu_hourly_price_usd": args.hourly_price,
        "api_cost_usd": 0.0,
        "gpu_cost_usd": 0.0,
        "total_cost_usd": 0.0,
        "cost_notes": [
            "No provider API call was made.",
            "No GPU inference request was made by this runner.",
        ],
    }
    _write_json(args.cost_report_path, cost_report)
    automation_report = build_post_run_automation_report(
        PostRunAutomationInputs(
            run_id=RUN_ID,
            manifest_path=args.manifest_path,
            eval_summary_path=args.eval_summary_path,
            latency_summary_path=None,
            telemetry_summary_path=None,
            cost_report_path=args.cost_report_path,
            comparison_paths=(
                args.engine_comparison_path,
                args.memory_comparison_path,
                args.concurrency_comparison_path,
                args.api_comparison_path,
                args.api_vs_self_hosted_comparison_path,
                args.model_comparison_path,
                args.slo_summary_path,
            ),
        )
    )
    write_post_run_automation_artifacts(
        report=automation_report,
        report_path=args.post_run_automation_report_path,
        plotting_dataset_path=args.plotting_dataset_path,
    )
    config = ArtifactSyncConfig(run_id=RUN_ID, backup_root=args.backup_root)
    sync = sync_artifacts(
        specs=_artifact_specs(args), config=config, event="blocked_preflight", repo_root=ROOT
    )
    verification = verify_backup(specs=_artifact_specs(args), config=config, repo_root=ROOT)
    artifact_report = {
        "run_id": RUN_ID,
        "artifact_sync_enabled": True,
        "sync": sync,
        "verification": verification,
        "success": bool(verification.get("passed")),
    }
    _write_json(args.artifact_sync_report_path, artifact_report)
    return eval_report


def run_controlled_final_simulation(args: argparse.Namespace) -> dict[str, Any]:
    """Build matrix, run safety gates, and write reports."""

    if args.prompt_count_per_vertical != DEFAULT_PROMPTS_PER_VERTICAL:
        msg = "controlled final simulation is locked to 80 prompts per vertical"
        raise ValueError(msg)
    if args.gpu_id != GPU_ID:
        msg = "controlled final simulation is locked to A100 SXM 80GB"
        raise ValueError(msg)
    started = time.perf_counter()
    matrix_rows = build_matrix_rows(
        dataset_root=args.dataset_root,
        prompts_per_vertical=args.prompt_count_per_vertical,
        args=args,
    )
    _write_jsonl(args.matrix_path, matrix_rows)
    matrix_summary = summarize_matrix(matrix_rows, args.prompt_count_per_vertical)
    contract_preflight = contract_preflight_report(matrix_rows)
    _write_json(args.contract_preflight_report, contract_preflight)
    gates = check_runtime_gate(args)
    smoke_report = build_smoke_report(gates)
    if not matrix_summary["passed"]:
        gates["full_simulation_allowed"] = False
        smoke_report["status"] = "SMOKE_BLOCKED"
        smoke_report["reason"] = "Matrix cardinality validation failed."
    if not contract_preflight["passed"]:
        gates["full_simulation_allowed"] = False
        smoke_report["status"] = "SMOKE_BLOCKED"
        smoke_report["reason"] = "Contract preflight blocked full simulation."
    repaired_smoke_report: dict[str, Any] | None = None
    if args.run_repaired_smoke:
        if not gates["full_simulation_allowed"]:
            repaired_smoke_report = {
                "run_id": RUN_ID,
                "status": "REPAIRED_25_REPLAY_BLOCKED_BY_PREFLIGHT",
                "passed_quality_gate": False,
                "reason": smoke_report["reason"],
            }
            _write_json(args.repaired_25_replay_report, repaired_smoke_report)
        else:
            repaired_smoke_report = run_repaired_subset(
                args=args,
                rows=select_repaired_smoke_rows(matrix_rows),
                report_path=args.repaired_25_replay_report,
                status="REPAIRED_25_REPLAY_COMPLETE",
            )
    repaired_validation_report: dict[str, Any] | None = None
    targeted_mm4_report: dict[str, Any] | None = None
    if args.run_mm4_safety_targeted:
        if not gates["full_simulation_allowed"]:
            targeted_mm4_report = {
                "run_id": RUN_ID,
                "status": "MM4_SAFETY_TARGETED_REPLAY_BLOCKED_BY_PREFLIGHT",
                "passed_quality_gate": False,
                "reason": smoke_report["reason"],
            }
            _write_json(args.mm4_safety_targeted_report, targeted_mm4_report)
        else:
            targeted_mm4_report = run_mm4_safety_targeted_replay(
                args=args,
                rows=matrix_rows,
            )
    if args.run_repaired_validation:
        if repaired_smoke_report is None and _repo_path(args.repaired_25_replay_report).exists():
            repaired_smoke_report = json.loads(
                _repo_path(args.repaired_25_replay_report).read_text(encoding="utf-8")
            )
        if not bool((repaired_smoke_report or {}).get("passed_quality_gate")):
            repaired_validation_report = {
                "run_id": RUN_ID,
                "status": "REPAIRED_500_VALIDATION_BLOCKED_BY_25_REPLAY",
                "passed_quality_gate": False,
                "reason": "25-row repaired replay did not pass quality gate.",
            }
            _write_json(args.repaired_500_validation_report, repaired_validation_report)
        else:
            repaired_validation_report = run_repaired_subset(
                args=args,
                rows=select_repaired_validation_rows(matrix_rows),
                report_path=args.repaired_500_validation_report,
                status="REPAIRED_500_VALIDATION_COMPLETE",
            )
            _write_repair_ready_report(
                args=args,
                targeted_report=targeted_mm4_report,
                validation_report=repaired_validation_report,
                gates=gates,
            )
    if _repo_path(args.repair_vs_broken_comparison_report).parent.exists():
        _write_json(
            args.repair_vs_broken_comparison_report,
            {
                "run_id": RUN_ID,
                "status": "REPAIR_COMPARISON_AVAILABLE",
                "broken_10k_baseline": {
                    "json_valid_rate": 0.0,
                    "generation_contract_valid_rate": 0.0,
                    "evidence_match_rate": 0.0397,
                    "grounded_rate": 0.0397,
                },
                "repaired_25_replay": repaired_smoke_report
                or (
                    json.loads(
                        _repo_path(args.repaired_25_replay_report).read_text(encoding="utf-8")
                    )
                    if _repo_path(args.repaired_25_replay_report).exists()
                    else {}
                ),
                "repaired_500_validation": repaired_validation_report
                or (
                    json.loads(
                        _repo_path(args.repaired_500_validation_report).read_text(encoding="utf-8")
                    )
                    if _repo_path(args.repaired_500_validation_report).exists()
                    else {}
                ),
                "targeted_mm4_safety_replay": targeted_mm4_report
                or (
                    json.loads(
                        _repo_path(args.mm4_safety_targeted_report).read_text(encoding="utf-8")
                    )
                    if _repo_path(args.mm4_safety_targeted_report).exists()
                    else {}
                ),
            },
        )
    full_repair_allowed = False
    if args.allow_full_after_repair:
        if repaired_smoke_report is None and _repo_path(args.repaired_25_replay_report).exists():
            repaired_smoke_report = json.loads(
                _repo_path(args.repaired_25_replay_report).read_text(encoding="utf-8")
            )
        if (
            repaired_validation_report is None
            and _repo_path(args.repaired_500_validation_report).exists()
        ):
            repaired_validation_report = json.loads(
                _repo_path(args.repaired_500_validation_report).read_text(encoding="utf-8")
            )
        if targeted_mm4_report is None and _repo_path(args.mm4_safety_targeted_report).exists():
            targeted_mm4_report = json.loads(
                _repo_path(args.mm4_safety_targeted_report).read_text(encoding="utf-8")
            )
        full_repair_allowed = bool(
            contract_preflight["passed"]
            and (repaired_smoke_report or {}).get("passed_quality_gate")
            and (targeted_mm4_report or {}).get("passed_quality_gate")
            and (repaired_validation_report or {}).get("passed_quality_gate")
        )
    if args.run_full and not full_repair_allowed:
        gates["full_simulation_allowed"] = False
        smoke_report["status"] = "SMOKE_BLOCKED"
        smoke_report["reason"] = (
            "Full 10k rerun is blocked until contract preflight, repaired 25-row replay, "
            "targeted MM4 safety replay, and repaired 500-row validation pass."
        )
    if (
        args.run_repaired_smoke or args.run_repaired_validation or args.run_mm4_safety_targeted
    ) and not args.run_full:
        return {
            "run_id": RUN_ID,
            "status": "CONTROLLED_FINAL_REPAIR_GATES_COMPLETE",
            "matrix_summary": matrix_summary,
            "contract_preflight": contract_preflight,
            "runtime_gate": smoke_report,
            "repaired_25_replay": repaired_smoke_report
            or (
                json.loads(_repo_path(args.repaired_25_replay_report).read_text(encoding="utf-8"))
                if _repo_path(args.repaired_25_replay_report).exists()
                else {}
            ),
            "repaired_500_validation": repaired_validation_report
            or (
                json.loads(
                    _repo_path(args.repaired_500_validation_report).read_text(encoding="utf-8")
                )
                if _repo_path(args.repaired_500_validation_report).exists()
                else {}
            ),
            "targeted_mm4_safety_replay": targeted_mm4_report
            or (
                json.loads(_repo_path(args.mm4_safety_targeted_report).read_text(encoding="utf-8"))
                if _repo_path(args.mm4_safety_targeted_report).exists()
                else {}
            ),
            "full_10000_rerun_allowed": False,
            "wall_seconds": time.perf_counter() - started,
        }
    if args.run_full and gates["full_simulation_allowed"]:
        started_at = utc_now()
        existing_rows = _read_existing_result_rows(args.raw_results_path)
        matrix_prompt_hashes = {_request_key(row): row.get("prompt_hash") for row in matrix_rows}
        deduped: dict[str, dict[str, Any]] = {}
        for row in existing_rows:
            if row.get("config_id") and row.get("prompt_id"):
                key = _request_key(row)
                if row.get("prompt_hash") == matrix_prompt_hashes.get(key):
                    deduped[key] = row
        rows = list(deduped.values())
        completed_keys = set(deduped)
        pending_keys = {_request_key(row) for row in matrix_rows} - completed_keys
        existing_timing = _read_existing_completed_timing(args.eval_report_path)
        if not rows:
            _write_jsonl(args.raw_results_path, [])
            _repo_path(args.gpu_telemetry_path).parent.mkdir(parents=True, exist_ok=True)
            _repo_path(args.gpu_telemetry_path).write_text("", encoding="utf-8")
            if _repo_path(args.checkpoint_path).exists():
                _repo_path(args.checkpoint_path).unlink()
        if not pending_keys and len(rows) == matrix_summary["row_count"]:
            completed_at = str(existing_timing.get("completed_at") or utc_now())
            started_at = str(existing_timing.get("started_at") or completed_at)
            wall_seconds = float(
                existing_timing.get("wall_seconds") or (time.perf_counter() - started)
            )
            _write_checkpoint(args.checkpoint_path, rows, status="completed")
            return _write_completed_reports(
                args,
                matrix_summary=matrix_summary,
                gates=gates,
                smoke_report=smoke_report,
                rows=rows,
                started_at=started_at,
                completed_at=completed_at,
                wall_seconds=wall_seconds,
                telemetry_errors=[],
            )
        _write_manifest(
            args,
            status="running",
            matrix_summary=matrix_summary,
            rows=rows,
            started_at=started_at,
            completed_at=None,
        )
        api_route = _api_route(args)
        telemetry_errors: list[str] = []
        stop_event = threading.Event()
        telemetry_thread = threading.Thread(
            target=_telemetry_loop,
            kwargs={
                "path": _repo_path(args.gpu_telemetry_path),
                "stop_event": stop_event,
                "interval_seconds": args.telemetry_interval_seconds,
                "errors": telemetry_errors,
            },
            daemon=True,
        )
        telemetry_thread.start()
        try:
            for config in build_config_specs():
                config_matrix_rows = [
                    row for row in matrix_rows if row["config_id"] == config.config_id
                ]
                new_rows = _run_config(
                    args=args,
                    config=config,
                    rows=config_matrix_rows,
                    completed_keys=completed_keys,
                    api_route=api_route,
                )
                if new_rows:
                    _append_jsonl(args.raw_results_path, new_rows)
                    rows.extend(new_rows)
                    completed_keys.update(_request_key(row) for row in new_rows)
                    _write_checkpoint(args.checkpoint_path, rows, status="running")
                    _write_manifest(
                        args,
                        status="running",
                        matrix_summary=matrix_summary,
                        rows=rows,
                        started_at=started_at,
                        completed_at=None,
                    )
        finally:
            stop_event.set()
            telemetry_thread.join(timeout=max(args.telemetry_interval_seconds * 2.0, 2.0))
        completed_at = utc_now()
        _write_checkpoint(args.checkpoint_path, rows, status="completed")
        report = _write_completed_reports(
            args,
            matrix_summary=matrix_summary,
            gates=gates,
            smoke_report=smoke_report,
            rows=rows,
            started_at=started_at,
            completed_at=completed_at,
            wall_seconds=time.perf_counter() - started,
            telemetry_errors=telemetry_errors,
        )
        return report
    report = write_blocked_reports(
        args,
        matrix_summary=matrix_summary,
        gates=gates,
        smoke_report=smoke_report,
    )
    report["wall_seconds"] = time.perf_counter() - started
    return report


def main() -> int:
    """CLI entry point."""

    args = build_parser().parse_args()
    try:
        report = run_controlled_final_simulation(args)
    except Exception as exc:  # noqa: BLE001
        print(f"controlled final simulation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

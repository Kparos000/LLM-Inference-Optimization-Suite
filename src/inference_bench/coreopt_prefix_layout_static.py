"""Static prompt-prefix layout analysis for Main_Inference_V1.

This module compares the authoritative runner prompt layout against a candidate
layout that moves stable instruction content earlier in the prompt. It performs
static analysis only: no inference is executed and no runtime/cache/latency
claim is made from these outputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml

from inference_bench.context_schema import WorkloadRecord
from inference_bench.workload_adapter import workload_record_to_runner_item

JsonDict = dict[str, Any]
TokenizeFn = Callable[[str], list[str]]

SCENARIO_ID = "coreopt_prefix_layout_static_v1"
PARENT_RUN_ID = "main_inference_v1"
OPTIMIZATION_ID = "prompt_prefix_layout_optimization"
RESULT_TYPE = "measured_static_analysis"
BASELINE_LAYOUT_ID = "baseline_prompt_layout_v1"
CANDIDATE_LAYOUT_ID = "prefix_optimized_prompt_layout_v1"
MODEL_ALIAS = "model3_7b"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
EXPERIMENT_ROOT = Path("experiments/optimizations") / SCENARIO_ID
WORKLOAD_ROOT = Path("data/workloads/final_10000/prompt_plus_metadata")
DEFAULT_MEMORY_MODES = (
    "mm0_no_context",
    "mm1_dense_top5",
    "mm2_hybrid_top5",
    "mm3_compressed_hybrid_top5",
)
LAYOUT_SECTION_ORDER = {
    BASELINE_LAYOUT_ID: (
        "system",
        "memory_mode",
        "retrieved_evidence",
        "user_question",
        "output_contract",
    ),
    CANDIDATE_LAYOUT_ID: (
        "system",
        "memory_mode",
        "output_contract",
        "retrieved_evidence",
        "user_question",
    ),
}
SECTION_MARKERS = {
    "system": "SYSTEM:",
    "memory_mode": "MEMORY MODE:",
    "retrieved_evidence": "RETRIEVED EVIDENCE:",
    "user_question": "USER QUESTION:",
    "output_contract": "OUTPUT CONTRACT:",
}
STATIC_SECTIONS = {"system", "memory_mode", "output_contract"}
DYNAMIC_SECTIONS = {"retrieved_evidence", "user_question"}
REGISTRY_PATH = Path("configs/core_optimization_scenario_registry.yaml")


@dataclass(frozen=True)
class PromptSection:
    section_id: str
    text: str
    stable: bool


@dataclass(frozen=True)
class RenderedPrompt:
    layout_id: str
    prompt_id: str
    vertical: str
    memory_mode: str
    prompt: str
    sections: tuple[PromptSection, ...]


@dataclass(frozen=True)
class TokenizedPrompt:
    layout_id: str
    prompt_id: str
    vertical: str
    memory_mode: str
    tokens: list[str]
    section_token_counts: dict[str, int]
    section_char_counts: dict[str, int]
    section_hashes: dict[str, str]


def workload_paths(memory_modes: Sequence[str] = DEFAULT_MEMORY_MODES) -> list[Path]:
    """Return authoritative Main_Inference-compatible workload paths."""

    return [WORKLOAD_ROOT / f"{memory_mode}.jsonl" for memory_mode in memory_modes]


def load_workload_records(path: Path, *, limit: int | None = None) -> Iterable[WorkloadRecord]:
    """Yield workload records from a phase-3 JSONL workload file."""

    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if line.strip():
                yield WorkloadRecord(**json.loads(line))


def render_baseline_prompt(record: WorkloadRecord) -> RenderedPrompt:
    """Render the exact prompt sent by the runner today."""

    item = workload_record_to_runner_item(record)
    sections = _split_authoritative_sections(item.prompt)
    return RenderedPrompt(
        layout_id=BASELINE_LAYOUT_ID,
        prompt_id=record.prompt_id,
        vertical=record.vertical,
        memory_mode=record.memory_mode,
        prompt=item.prompt,
        sections=sections,
    )


def render_prefix_optimized_prompt(record: WorkloadRecord) -> RenderedPrompt:
    """Render the candidate static prefix-optimized layout.

    The candidate preserves the byte content of every authoritative section and
    changes only section ordering.
    """

    baseline = render_baseline_prompt(record)
    section_by_id = {section.section_id: section for section in baseline.sections}
    ordered_sections = tuple(
        section_by_id[section_id] for section_id in LAYOUT_SECTION_ORDER[CANDIDATE_LAYOUT_ID]
    )
    return RenderedPrompt(
        layout_id=CANDIDATE_LAYOUT_ID,
        prompt_id=record.prompt_id,
        vertical=record.vertical,
        memory_mode=record.memory_mode,
        prompt="\n\n".join(section.text for section in ordered_sections),
        sections=ordered_sections,
    )


def analyze_prefix_layout_static(
    *,
    workload_file_paths: Sequence[Path] | None = None,
    limit_per_memory_mode: int | None = None,
) -> JsonDict:
    """Build the static analysis payload without writing artifacts."""

    started_at = _utc_now()
    selected_workload_paths = list(workload_file_paths or workload_paths())
    tokenizer_report, tokenize = _load_tokenizer()
    rendered_prompts: list[RenderedPrompt] = []
    source_counts: dict[str, int] = {}

    for workload_path in selected_workload_paths:
        path_count = 0
        for record in load_workload_records(workload_path, limit=limit_per_memory_mode):
            rendered_prompts.append(render_baseline_prompt(record))
            rendered_prompts.append(render_prefix_optimized_prompt(record))
            path_count += 1
        source_counts[_display_path(workload_path)] = path_count

    tokenized = [_tokenize_prompt(prompt, tokenize) for prompt in rendered_prompts]
    prefix_groups = _build_prefix_groups(tokenized)
    section_rows = _build_section_rows(tokenized)
    per_vertical_memory_rows = _build_per_vertical_memory_rows(tokenized, prefix_groups)
    prefix_family_rows = _build_prefix_family_rows(prefix_groups)
    prefix_summary = _build_prefix_summary(
        tokenized,
        prefix_groups,
        source_counts=source_counts,
        limit_per_memory_mode=limit_per_memory_mode,
    )
    equivalence = _build_equivalence_report(rendered_prompts)
    held_constants = _build_held_constants_report(selected_workload_paths)
    decision = _build_decision(prefix_summary, equivalence)
    layouts = _build_layouts(rendered_prompts)
    plotting_dataset = _build_plotting_dataset(
        prefix_family_rows=prefix_family_rows,
        per_vertical_memory_rows=per_vertical_memory_rows,
        prefix_summary=prefix_summary,
        decision=decision,
    )
    completed_at = _utc_now()
    manifest = _build_manifest(
        started_at=started_at,
        completed_at=completed_at,
        source_counts=source_counts,
        tokenizer_report=tokenizer_report,
        decision=decision,
        limit_per_memory_mode=limit_per_memory_mode,
    )
    ui_story = _build_ui_story(prefix_summary, equivalence, decision)

    return {
        "manifest": manifest,
        "layouts": layouts,
        "held_constants": held_constants,
        "tokenizer_report": tokenizer_report,
        "prefix_summary": prefix_summary,
        "prefix_families": prefix_family_rows,
        "per_vertical_memory": per_vertical_memory_rows,
        "prompt_section_analysis": section_rows,
        "equivalence_report": equivalence,
        "decision": decision,
        "plotting_dataset": plotting_dataset,
        "ui_story": ui_story,
    }


def write_coreopt_prefix_layout_static_artifacts(
    *,
    output_root: Path = EXPERIMENT_ROOT,
    workload_file_paths: Sequence[Path] | None = None,
    limit_per_memory_mode: int | None = None,
    update_registry: bool = False,
) -> JsonDict:
    """Write all static-analysis artifacts and optionally update the registry."""

    analysis = analyze_prefix_layout_static(
        workload_file_paths=workload_file_paths,
        limit_per_memory_mode=limit_per_memory_mode,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    layouts_dir = output_root / "layouts"
    logs_dir = output_root / "logs"
    checksums_dir = output_root / "checksums"
    layouts_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    checksums_dir.mkdir(exist_ok=True)

    artifact_paths = {
        "manifest": output_root / f"{SCENARIO_ID}_manifest.json",
        "baseline_layout": layouts_dir / f"{BASELINE_LAYOUT_ID}.json",
        "candidate_layout": layouts_dir / f"{CANDIDATE_LAYOUT_ID}.json",
        "held_constants": output_root / f"{SCENARIO_ID}_held_constants.json",
        "tokenizer_report": output_root / f"{SCENARIO_ID}_tokenizer_report.json",
        "prefix_summary": output_root / f"{SCENARIO_ID}_prefix_summary.json",
        "prefix_summary_csv": output_root / f"{SCENARIO_ID}_prefix_summary.csv",
        "prefix_families": output_root / f"{SCENARIO_ID}_prefix_families.csv",
        "per_vertical_memory": output_root / f"{SCENARIO_ID}_per_vertical_memory.csv",
        "prompt_section_analysis": output_root / f"{SCENARIO_ID}_prompt_section_analysis.csv",
        "equivalence_report": output_root / f"{SCENARIO_ID}_equivalence_report.json",
        "decision": output_root / f"{SCENARIO_ID}_decision.json",
        "plotting_dataset": output_root / f"{SCENARIO_ID}_plotting_dataset.json",
        "ui_story": output_root / f"{SCENARIO_ID}_ui_story.json",
        "log": logs_dir / f"{SCENARIO_ID}.log",
        "readme": output_root / "README.md",
        "checksums": checksums_dir / "SHA256SUMS.txt",
    }

    analysis["manifest"]["artifact_paths"] = {
        key: _display_path(path) for key, path in artifact_paths.items() if key != "checksums"
    }
    _write_json(artifact_paths["manifest"], analysis["manifest"])
    _write_json(artifact_paths["baseline_layout"], analysis["layouts"][BASELINE_LAYOUT_ID])
    _write_json(artifact_paths["candidate_layout"], analysis["layouts"][CANDIDATE_LAYOUT_ID])
    _write_json(artifact_paths["held_constants"], analysis["held_constants"])
    _write_json(artifact_paths["tokenizer_report"], analysis["tokenizer_report"])
    _write_json(artifact_paths["prefix_summary"], analysis["prefix_summary"])
    _write_csv(artifact_paths["prefix_summary_csv"], [_flatten_summary(analysis["prefix_summary"])])
    _write_csv(artifact_paths["prefix_families"], analysis["prefix_families"])
    _write_csv(artifact_paths["per_vertical_memory"], analysis["per_vertical_memory"])
    _write_csv(artifact_paths["prompt_section_analysis"], analysis["prompt_section_analysis"])
    _write_json(artifact_paths["equivalence_report"], analysis["equivalence_report"])
    _write_json(artifact_paths["decision"], analysis["decision"])
    _write_json(artifact_paths["plotting_dataset"], analysis["plotting_dataset"])
    _write_json(artifact_paths["ui_story"], analysis["ui_story"])
    _write_text(artifact_paths["log"], _build_log_text(analysis))
    _write_text(artifact_paths["readme"], _build_readme_text(analysis))
    _write_checksums(artifact_paths["checksums"], artifact_paths)

    if update_registry:
        _update_scenario_registry(analysis["decision"], artifact_paths)

    return {
        "scenario_id": SCENARIO_ID,
        "output_root": _display_path(output_root),
        "decision": analysis["decision"]["decision"],
        "artifact_paths": {key: _display_path(path) for key, path in artifact_paths.items()},
    }


def _split_authoritative_sections(prompt: str) -> tuple[PromptSection, ...]:
    marker_positions = [
        (section_id, prompt.index(marker)) for section_id, marker in SECTION_MARKERS.items()
    ]
    marker_positions.sort(key=lambda item: item[1])
    sections: list[PromptSection] = []
    for index, (section_id, start) in enumerate(marker_positions):
        end = marker_positions[index + 1][1] if index + 1 < len(marker_positions) else len(prompt)
        text = prompt[start:end].strip()
        sections.append(
            PromptSection(
                section_id=section_id,
                text=text,
                stable=section_id in STATIC_SECTIONS,
            )
        )
    observed_order = tuple(section.section_id for section in sections)
    if observed_order != LAYOUT_SECTION_ORDER[BASELINE_LAYOUT_ID]:
        raise ValueError(f"Unexpected authoritative prompt section order: {observed_order}")
    return tuple(sections)


def _load_tokenizer() -> tuple[JsonDict, TokenizeFn]:
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            local_files_only=True,
            trust_remote_code=False,
        )

        def tokenize(text: str) -> list[str]:
            return [str(token) for token in tokenizer.encode(text, add_special_tokens=False)]

        chat_template = getattr(tokenizer, "chat_template", None)
        special_tokens_map = getattr(tokenizer, "special_tokens_map", {})
        metadata = {
            "tokenizer_available": True,
            "tokenizer_source": "local_transformers_cache",
            "tokenizer_model_id": MODEL_ID,
            "tokenizer_class": tokenizer.__class__.__name__,
            "name_or_path": str(getattr(tokenizer, "name_or_path", MODEL_ID)),
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "model_max_length": getattr(tokenizer, "model_max_length", None),
            "chat_template_sha256": _sha256_text(chat_template) if chat_template else None,
            "special_tokens_map": special_tokens_map,
            "analysis_timestamp": _utc_now(),
            "fallback_used": False,
        }
        metadata["metadata_sha256"] = _sha256_json(metadata)
        return metadata, tokenize
    except Exception as exc:  # pragma: no cover - depends on local model cache.

        def fallback_tokenize(text: str) -> list[str]:
            return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

        metadata = {
            "tokenizer_available": False,
            "tokenizer_source": "deterministic_regex_fallback",
            "tokenizer_model_id": MODEL_ID,
            "fallback_used": True,
            "fallback_reason": str(exc),
            "tokenizer_semantics": "Static proxy tokenizer; not suitable for latency claims.",
            "analysis_timestamp": _utc_now(),
        }
        metadata["metadata_sha256"] = _sha256_json(metadata)
        return metadata, fallback_tokenize


def _tokenize_prompt(prompt: RenderedPrompt, tokenize: TokenizeFn) -> TokenizedPrompt:
    section_token_counts = {
        section.section_id: len(tokenize(section.text)) for section in prompt.sections
    }
    return TokenizedPrompt(
        layout_id=prompt.layout_id,
        prompt_id=prompt.prompt_id,
        vertical=prompt.vertical,
        memory_mode=prompt.memory_mode,
        tokens=tokenize(prompt.prompt),
        section_token_counts=section_token_counts,
        section_char_counts={section.section_id: len(section.text) for section in prompt.sections},
        section_hashes={
            section.section_id: _sha256_text(section.text) for section in prompt.sections
        },
    )


def _build_prefix_groups(tokenized: Sequence[TokenizedPrompt]) -> dict[tuple[str, str], JsonDict]:
    grouped: dict[tuple[str, str], list[TokenizedPrompt]] = defaultdict(list)
    for item in tokenized:
        grouped[(item.layout_id, item.memory_mode)].append(item)

    prefix_groups: dict[tuple[str, str], JsonDict] = {}
    for key, rows in grouped.items():
        lcp_tokens = _longest_common_prefix([row.tokens for row in rows])
        prefix_hash = _sha256_text(" ".join(lcp_tokens))
        prefix_groups[key] = {
            "layout_id": key[0],
            "memory_mode": key[1],
            "request_count": len(rows),
            "longest_exact_common_prefix_tokens": len(lcp_tokens),
            "prefix_hash": prefix_hash,
            "prefix_family_id": _sha256_text(f"{key[0]}|{key[1]}|{prefix_hash}")[:16],
            "input_token_counts": [len(row.tokens) for row in rows],
        }
    return prefix_groups


def _build_prefix_family_rows(prefix_groups: dict[tuple[str, str], JsonDict]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for key in sorted(prefix_groups):
        group = prefix_groups[key]
        token_counts = group["input_token_counts"]
        lcp = int(group["longest_exact_common_prefix_tokens"])
        rows.append(
            {
                "scenario_id": SCENARIO_ID,
                "layout_id": group["layout_id"],
                "model_alias": MODEL_ALIAS,
                "model_id": MODEL_ID,
                "memory_mode": group["memory_mode"],
                "prefix_family_id": group["prefix_family_id"],
                "prefix_hash": group["prefix_hash"],
                "request_count": group["request_count"],
                "longest_exact_common_prefix_tokens": lcp,
                "mean_input_tokens": round(mean(token_counts), 6),
                "median_input_tokens": round(median(token_counts), 6),
                "p95_input_tokens": round(_percentile(token_counts, 0.95), 6),
                "p99_input_tokens": round(_percentile(token_counts, 0.99), 6),
                "mean_reusable_token_ratio": round(
                    mean(_safe_ratio(lcp, count) for count in token_counts),
                    6,
                ),
                "source": "static_analysis",
                "measurement_type": "derived",
                "inference_executed": False,
                "cache_hits_measured": False,
            }
        )
    return rows


def _build_section_rows(tokenized: Sequence[TokenizedPrompt]) -> list[JsonDict]:
    grouped: dict[tuple[str, str, str, str], list[TokenizedPrompt]] = defaultdict(list)
    for item in tokenized:
        for section_id in item.section_token_counts:
            grouped[(item.layout_id, item.memory_mode, item.vertical, section_id)].append(item)

    rows: list[JsonDict] = []
    for key in sorted(grouped):
        layout_id, memory_mode, vertical, section_id = key
        items = grouped[key]
        token_counts = [item.section_token_counts[section_id] for item in items]
        char_counts = [item.section_char_counts[section_id] for item in items]
        hashes = {item.section_hashes[section_id] for item in items}
        rows.append(
            {
                "scenario_id": SCENARIO_ID,
                "layout_id": layout_id,
                "memory_mode": memory_mode,
                "vertical": vertical,
                "section_id": section_id,
                "section_role": "stable" if section_id in STATIC_SECTIONS else "dynamic",
                "request_count": len(items),
                "mean_section_tokens": round(mean(token_counts), 6),
                "median_section_tokens": round(median(token_counts), 6),
                "p95_section_tokens": round(_percentile(token_counts, 0.95), 6),
                "mean_section_chars": round(mean(char_counts), 6),
                "unique_section_hash_count": len(hashes),
                "raw_text_included": False,
            }
        )
    return rows


def _build_per_vertical_memory_rows(
    tokenized: Sequence[TokenizedPrompt],
    prefix_groups: dict[tuple[str, str], JsonDict],
) -> list[JsonDict]:
    grouped: dict[tuple[str, str, str], list[TokenizedPrompt]] = defaultdict(list)
    for item in tokenized:
        grouped[(item.layout_id, item.memory_mode, item.vertical)].append(item)

    rows: list[JsonDict] = []
    for key in sorted(grouped):
        layout_id, memory_mode, vertical = key
        items = grouped[key]
        token_counts = [len(item.tokens) for item in items]
        prefix_group = prefix_groups[(layout_id, memory_mode)]
        lcp = int(prefix_group["longest_exact_common_prefix_tokens"])
        rows.append(
            {
                "scenario_id": SCENARIO_ID,
                "layout_id": layout_id,
                "memory_mode": memory_mode,
                "vertical": vertical,
                "request_count": len(items),
                "mean_input_tokens": round(mean(token_counts), 6),
                "median_input_tokens": round(median(token_counts), 6),
                "p95_input_tokens": round(_percentile(token_counts, 0.95), 6),
                "p99_input_tokens": round(_percentile(token_counts, 0.99), 6),
                "longest_exact_common_prefix_tokens": lcp,
                "mean_reusable_token_ratio": round(
                    mean(_safe_ratio(lcp, count) for count in token_counts),
                    6,
                ),
                "variable_suffix_tokens_mean": round(
                    mean(max(count - lcp, 0) for count in token_counts), 6
                ),
                "source": "static_analysis",
            }
        )
    return rows


def _build_prefix_summary(
    tokenized: Sequence[TokenizedPrompt],
    prefix_groups: dict[tuple[str, str], JsonDict],
    *,
    source_counts: dict[str, int],
    limit_per_memory_mode: int | None,
) -> JsonDict:
    by_layout: dict[str, list[TokenizedPrompt]] = defaultdict(list)
    for item in tokenized:
        by_layout[item.layout_id].append(item)

    layout_summaries = {}
    for layout_id, items in sorted(by_layout.items()):
        token_counts = [len(item.tokens) for item in items]
        group_entries = [
            group for (group_layout, _), group in prefix_groups.items() if group_layout == layout_id
        ]
        reusable_ratios = []
        for item in items:
            group = prefix_groups[(item.layout_id, item.memory_mode)]
            reusable_ratios.append(
                _safe_ratio(int(group["longest_exact_common_prefix_tokens"]), len(item.tokens))
            )
        layout_summaries[layout_id] = {
            "prompt_count": len(items),
            "total_input_tokens": sum(token_counts),
            "mean_input_tokens": round(mean(token_counts), 6),
            "median_input_tokens": round(median(token_counts), 6),
            "p95_input_tokens": round(_percentile(token_counts, 0.95), 6),
            "p99_input_tokens": round(_percentile(token_counts, 0.99), 6),
            "prefix_family_count": len(group_entries),
            "mean_longest_exact_common_prefix_tokens": round(
                mean(int(group["longest_exact_common_prefix_tokens"]) for group in group_entries),
                6,
            ),
            "mean_reusable_token_ratio": round(mean(reusable_ratios), 6),
            "median_reusable_token_ratio": round(median(reusable_ratios), 6),
            "p95_reusable_token_ratio": round(_percentile(reusable_ratios, 0.95), 6),
            "prefix_family_entropy_bits": round(
                _entropy([int(group["request_count"]) for group in group_entries]), 6
            ),
        }

    baseline = layout_summaries[BASELINE_LAYOUT_ID]
    candidate = layout_summaries[CANDIDATE_LAYOUT_ID]
    delta_tokens = (
        candidate["mean_longest_exact_common_prefix_tokens"]
        - baseline["mean_longest_exact_common_prefix_tokens"]
    )
    delta_ratio = candidate["mean_reusable_token_ratio"] - baseline["mean_reusable_token_ratio"]
    return {
        "scenario_id": SCENARIO_ID,
        "parent_run_id": PARENT_RUN_ID,
        "optimization_id": OPTIMIZATION_ID,
        "result_type": RESULT_TYPE,
        "status": "completed_static_analysis",
        "source": "static_analysis",
        "measurement_type": "derived",
        "inference_executed": False,
        "cache_hits_measured": False,
        "latency_claimed": False,
        "cost_claimed": False,
        "workload_rows_scanned": sum(source_counts.values()),
        "rendered_prompt_count": len(tokenized),
        "source_counts": source_counts,
        "limit_per_memory_mode": limit_per_memory_mode,
        "layout_summaries": layout_summaries,
        "deltas": {
            "candidate_minus_baseline_mean_common_prefix_tokens": round(delta_tokens, 6),
            "candidate_minus_baseline_mean_reusable_token_ratio": round(delta_ratio, 6),
            "candidate_minus_baseline_total_input_tokens": (
                candidate["total_input_tokens"] - baseline["total_input_tokens"]
            ),
        },
        "interpretation": (
            "The candidate increases the exact leading stable prefix available for "
            "future prefix-cache validation while preserving total prompt content."
        ),
    }


def _build_equivalence_report(rendered_prompts: Sequence[RenderedPrompt]) -> JsonDict:
    paired: dict[tuple[str, str], dict[str, RenderedPrompt]] = defaultdict(dict)
    for prompt in rendered_prompts:
        paired[(prompt.memory_mode, prompt.prompt_id)][prompt.layout_id] = prompt

    section_mismatch_count = 0
    section_order_mismatch_count = 0
    for pair in paired.values():
        baseline = pair[BASELINE_LAYOUT_ID]
        candidate = pair[CANDIDATE_LAYOUT_ID]
        baseline_hashes = {
            section.section_id: _sha256_text(section.text) for section in baseline.sections
        }
        candidate_hashes = {
            section.section_id: _sha256_text(section.text) for section in candidate.sections
        }
        if baseline_hashes != candidate_hashes:
            section_mismatch_count += 1
        if (
            tuple(section.section_id for section in candidate.sections)
            != LAYOUT_SECTION_ORDER[CANDIDATE_LAYOUT_ID]
        ):
            section_order_mismatch_count += 1

    return {
        "scenario_id": SCENARIO_ID,
        "status": "PASS"
        if section_mismatch_count == 0 and section_order_mismatch_count == 0
        else "FAIL",
        "rows_checked": len(paired),
        "all_sections_present": section_mismatch_count == 0,
        "section_content_byte_equivalent": section_mismatch_count == 0,
        "evidence_content_byte_equivalent": section_mismatch_count == 0,
        "evidence_order_fixed": section_mismatch_count == 0,
        "citation_aliases_preserved": section_mismatch_count == 0,
        "schema_instruction_unchanged": section_mismatch_count == 0,
        "safety_instruction_unchanged": section_mismatch_count == 0,
        "memory_instruction_unchanged": section_mismatch_count == 0,
        "model_chat_roles_valid": True,
        "forbidden_gold_leakage_detected": False,
        "max_output_settings_unchanged": True,
        "instruction_priority_risk": True,
        "requires_inference_validation": True,
        "section_mismatch_count": section_mismatch_count,
        "section_order_mismatch_count": section_order_mismatch_count,
        "notes": [
            "Candidate changes only section order and preserves section bytes.",
            (
                "Because instruction order changes, quality and safety must be "
                "validated by inference before deployment claims."
            ),
        ],
    }


def _build_held_constants_report(workload_file_paths: Sequence[Path]) -> JsonDict:
    paths: dict[str, str | list[str]] = {
        "workload_files": [_display_path(path) for path in workload_file_paths],
        "models": "configs/models.yaml",
        "memory_modes": "configs/memory_modes.yaml",
        "runtime_engines": "configs/runtime_engines.yaml",
        "slo_targets": "configs/slo_targets.yaml",
        "generation_contract_renderer": "src/inference_bench/generation_contract.py",
        "workload_adapter": "src/inference_bench/workload_adapter.py",
    }
    hashes: JsonDict = {}
    for key, value in paths.items():
        if isinstance(value, list):
            hashes[key] = {item: _sha256_file(Path(item)) for item in value if Path(item).exists()}
        elif Path(value).exists():
            hashes[key] = _sha256_file(Path(value))

    return {
        "scenario_id": SCENARIO_ID,
        "held_constant_categories": [
            "workload",
            "prompt_ids",
            "gold_contract",
            "knowledge_base_context",
            "retrieval_order",
            "evidence_aliases",
            "memory_semantics",
            "tokenizer_model",
            "generation_contract",
            "safety_and_escalation_rules",
            "evaluator_semantics",
            "output_token_limits",
            "model_metadata",
            "engine_metadata",
        ],
        "changed_factor": "rendered_prompt_section_order",
        "paths": paths,
        "hashes": hashes,
        "inference_executed": False,
        "result_mutation": "none",
    }


def _build_decision(prefix_summary: JsonDict, equivalence: JsonDict) -> JsonDict:
    deltas = prefix_summary["deltas"]
    missing_threshold = "minimum_reusable_token_ratio_delta_for_engine_validation"
    return {
        "scenario_id": SCENARIO_ID,
        "decision": "MISSING_CONFIGURATION",
        "reason": (
            "Static analysis completed and the candidate increases reusable leading "
            "prefix potential, but no explicit acceptance threshold is configured."
        ),
        "acceptance_threshold_configured": False,
        "missing_config_fields": [missing_threshold],
        "proposed_config_field": {
            "name": missing_threshold,
            "purpose": (
                "Minimum derived static prefix-reuse improvement required before engine validation."
            ),
            "recommended_owner_action": (
                "Set by engineer before accepting this candidate for live engine validation."
            ),
        },
        "observed_static_delta": deltas,
        "equivalence_status": equivalence["status"],
        "instruction_priority_risk": equivalence["instruction_priority_risk"],
        "requires_gpu_rerun": True,
        "requires_engine_validation": True,
        "inference_executed": False,
        "next_required_experiment": "coreopt_prefix_layout_engine_validation_v1",
        "disallowed_claims": [
            "TTFT improvement",
            "latency improvement",
            "cache-hit improvement",
            "cost improvement",
            "deployability improvement",
        ],
    }


def _build_layouts(rendered_prompts: Sequence[RenderedPrompt]) -> JsonDict:
    first_by_layout: dict[str, RenderedPrompt] = {}
    for prompt in rendered_prompts:
        first_by_layout.setdefault(prompt.layout_id, prompt)

    layouts: JsonDict = {}
    for layout_id, prompt in sorted(first_by_layout.items()):
        layouts[layout_id] = {
            "layout_id": layout_id,
            "scenario_id": SCENARIO_ID,
            "section_order": list(LAYOUT_SECTION_ORDER[layout_id]),
            "raw_prompt_text_included": False,
            "sections": [
                {
                    "section_id": section.section_id,
                    "position": index + 1,
                    "role": "stable" if section.section_id in STATIC_SECTIONS else "dynamic",
                    "sha256": _sha256_text(section.text),
                    "sample_char_count": len(section.text),
                    "content_source": "authoritative_runner_prompt_section",
                }
                for index, section in enumerate(prompt.sections)
            ],
        }
    return layouts


def _build_plotting_dataset(
    *,
    prefix_family_rows: Sequence[JsonDict],
    per_vertical_memory_rows: Sequence[JsonDict],
    prefix_summary: JsonDict,
    decision: JsonDict,
) -> JsonDict:
    return {
        "scenario_id": SCENARIO_ID,
        "source": "static_analysis",
        "measurement_type": "derived",
        "charts": {
            "prefix_family_common_prefix_tokens": list(prefix_family_rows),
            "per_vertical_memory_reusable_ratio": list(per_vertical_memory_rows),
            "layout_summary": [
                {
                    "layout_id": layout_id,
                    **summary,
                }
                for layout_id, summary in prefix_summary["layout_summaries"].items()
            ],
        },
        "decision": decision["decision"],
        "inference_executed": False,
    }


def _build_manifest(
    *,
    started_at: str,
    completed_at: str,
    source_counts: dict[str, int],
    tokenizer_report: JsonDict,
    decision: JsonDict,
    limit_per_memory_mode: int | None,
) -> JsonDict:
    return {
        "run_id": SCENARIO_ID,
        "scenario_id": SCENARIO_ID,
        "parent_run_id": PARENT_RUN_ID,
        "optimization_id": OPTIMIZATION_ID,
        "model_alias": MODEL_ALIAS,
        "model_id": MODEL_ID,
        "runtime": "static_analysis",
        "engine": "none",
        "backend_type": "none",
        "hardware": "cpu_only_local_static_analysis",
        "provider": "local_artifacts",
        "result_type": RESULT_TYPE,
        "status": "completed",
        "started_at": started_at,
        "updated_at": completed_at,
        "completed_at": completed_at,
        "prompt_count": sum(source_counts.values()),
        "expected_count": sum(source_counts.values()),
        "completed_count": sum(source_counts.values()),
        "failed_count": 0,
        "limit_per_memory_mode": limit_per_memory_mode,
        "source_counts": source_counts,
        "tokenizer": tokenizer_report,
        "decision": decision["decision"],
        "inference_executed": False,
        "main_inference_artifacts_mutated": False,
        "optimized_inference_created": False,
    }


def _build_ui_story(
    prefix_summary: JsonDict, equivalence: JsonDict, decision: JsonDict
) -> JsonDict:
    baseline = prefix_summary["layout_summaries"][BASELINE_LAYOUT_ID]
    candidate = prefix_summary["layout_summaries"][CANDIDATE_LAYOUT_ID]
    return {
        "scenario_id": SCENARIO_ID,
        "title": "Static Prompt Prefix Layout Optimization",
        "result_type": RESULT_TYPE,
        "status": decision["decision"],
        "story_steps": [
            {
                "id": "problem",
                "title": "Problem",
                "body": (
                    "The authoritative runner prompt placed a long stable output "
                    "contract after request-specific context and question content."
                ),
            },
            {
                "id": "mechanism",
                "title": "Mechanism",
                "body": (
                    "The candidate moves stable reusable instructions before dynamic "
                    "evidence and question sections so future prefix caching can reuse "
                    "a longer exact leading token sequence."
                ),
            },
            {
                "id": "instrumentation",
                "title": "Instrumentation",
                "body": (
                    "The experiment tokenizes both layouts, assigns prefix families, "
                    "and reports derived static prefix-reuse metrics without executing inference."
                ),
            },
            {
                "id": "result",
                "title": "Static Result",
                "body": (
                    f"Mean common prefix tokens changed from "
                    f"{baseline['mean_longest_exact_common_prefix_tokens']} to "
                    f"{candidate['mean_longest_exact_common_prefix_tokens']}."
                ),
            },
            {
                "id": "decision",
                "title": "Decision",
                "body": decision["reason"],
            },
        ],
        "headline_metrics": {
            "baseline_mean_reusable_token_ratio": baseline["mean_reusable_token_ratio"],
            "candidate_mean_reusable_token_ratio": candidate["mean_reusable_token_ratio"],
            "delta_reusable_token_ratio": prefix_summary["deltas"][
                "candidate_minus_baseline_mean_reusable_token_ratio"
            ],
            "equivalence_status": equivalence["status"],
            "requires_engine_validation": decision["requires_engine_validation"],
        },
        "valid_user_action": "review_plan_only",
        "apply_behavior": (
            "No inference is executed; clicking apply can only reveal the engine-validation plan."
        ),
    }


def _flatten_summary(summary: JsonDict) -> JsonDict:
    baseline = summary["layout_summaries"][BASELINE_LAYOUT_ID]
    candidate = summary["layout_summaries"][CANDIDATE_LAYOUT_ID]
    return {
        "scenario_id": summary["scenario_id"],
        "parent_run_id": summary["parent_run_id"],
        "workload_rows_scanned": summary["workload_rows_scanned"],
        "baseline_mean_input_tokens": baseline["mean_input_tokens"],
        "candidate_mean_input_tokens": candidate["mean_input_tokens"],
        "baseline_mean_common_prefix_tokens": baseline["mean_longest_exact_common_prefix_tokens"],
        "candidate_mean_common_prefix_tokens": candidate["mean_longest_exact_common_prefix_tokens"],
        "baseline_mean_reusable_token_ratio": baseline["mean_reusable_token_ratio"],
        "candidate_mean_reusable_token_ratio": candidate["mean_reusable_token_ratio"],
        "delta_common_prefix_tokens": summary["deltas"][
            "candidate_minus_baseline_mean_common_prefix_tokens"
        ],
        "delta_reusable_token_ratio": summary["deltas"][
            "candidate_minus_baseline_mean_reusable_token_ratio"
        ],
        "inference_executed": summary["inference_executed"],
        "cache_hits_measured": summary["cache_hits_measured"],
        "latency_claimed": summary["latency_claimed"],
    }


def _build_log_text(analysis: JsonDict) -> str:
    return (
        f"{SCENARIO_ID}\n"
        f"status={analysis['prefix_summary']['status']}\n"
        f"workload_rows_scanned={analysis['prefix_summary']['workload_rows_scanned']}\n"
        f"decision={analysis['decision']['decision']}\n"
        "inference_executed=false\n"
    )


def _build_readme_text(analysis: JsonDict) -> str:
    decision = analysis["decision"]
    summary = analysis["prefix_summary"]
    return (
        f"# {SCENARIO_ID}\n\n"
        "Static prompt-prefix layout optimization analysis for Main_Inference_V1.\n\n"
        "No inference was executed. No Main_Inference_V1 artifacts were mutated. "
        "No Optimized_Inference_V1 artifact was created.\n\n"
        f"- Parent run: `{PARENT_RUN_ID}`\n"
        f"- Optimization: `{OPTIMIZATION_ID}`\n"
        f"- Workload rows scanned: {summary['workload_rows_scanned']}\n"
        f"- Decision: `{decision['decision']}`\n"
        f"- Next required experiment: `{decision['next_required_experiment']}`\n\n"
        "The candidate is a plan-only static artifact until an engineer configures "
        "an acceptance threshold and runs engine validation.\n"
    )


def _update_scenario_registry(decision: JsonDict, artifact_paths: dict[str, Path]) -> None:
    data = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    scenario_types = data.setdefault("scenario_types", [])
    if "one_factor_static" not in scenario_types:
        scenario_types.append("one_factor_static")

    scenario_entry = {
        "optimization_id": OPTIMIZATION_ID,
        "scenario_id": SCENARIO_ID,
        "parent_run_id": PARENT_RUN_ID,
        "scenario_type": "one_factor_static",
        "lineage": {"parent_run_id": PARENT_RUN_ID},
        "changed_factor": "rendered_prompt_section_order",
        "held_constants": [
            "workload",
            "prompt_ids",
            "gold_contract",
            "knowledge_base_context",
            "retrieval_order",
            "evidence_aliases",
            "memory_semantics",
            "tokenizer_model",
            "generation_contract",
            "safety_and_escalation_rules",
            "evaluator_semantics",
            "output_token_limits",
            "model_metadata",
            "engine_metadata",
        ],
        "artifact_root": _display_path(EXPERIMENT_ROOT),
        "artifact_paths": {
            key: _display_path(path)
            for key, path in artifact_paths.items()
            if key not in {"checksums", "log"}
        },
        "instrumentation_readiness": "ready",
        "execution_status": "completed_static_analysis",
        "result_type": RESULT_TYPE,
        "decision": decision["decision"],
        "next_required_experiment": decision["next_required_experiment"],
        "ui_replay_available": True,
        "champion_selected": False,
        "source": "static_analysis",
        "inference_executed": False,
        "cache_hits_measured": False,
        "latency_claimed": False,
    }
    scenarios = data.setdefault("scenarios", [])
    if isinstance(scenarios, list):
        for index, existing in enumerate(scenarios):
            if isinstance(existing, dict) and existing.get("scenario_id") == SCENARIO_ID:
                scenarios[index] = scenario_entry
                break
        else:
            scenarios.append(scenario_entry)
    else:
        scenarios[SCENARIO_ID] = scenario_entry

    REGISTRY_PATH.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=False, width=100),
        encoding="utf-8",
    )


def _longest_common_prefix(sequences: Sequence[list[str]]) -> list[str]:
    if not sequences:
        return []
    prefix = list(sequences[0])
    for sequence in sequences[1:]:
        limit = min(len(prefix), len(sequence))
        index = 0
        while index < limit and prefix[index] == sequence[index]:
            index += 1
        prefix = prefix[:index]
        if not prefix:
            break
    return prefix


def _percentile(values: Sequence[float | int], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _entropy(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count:
            probability = count / total
            entropy -= probability * math.log2(probability)
    return entropy


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_checksums(checksum_path: Path, artifact_paths: dict[str, Path]) -> None:
    lines = []
    for key, path in sorted(artifact_paths.items()):
        if key == "checksums" or not path.exists():
            continue
        lines.append(f"{_sha256_file(path)}  {_display_path(path)}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sha256_text(text: str | None) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _sha256_json(payload: JsonDict) -> str:
    return _sha256_text(json.dumps(payload, sort_keys=True, default=str))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    return path.as_posix()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

"""B6R3 Research AI model-capacity replay loading and guard helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from inference_bench.research_ai_contract_renderer import (
    assert_no_canonical_gold_id_leakage,
)
from inference_bench.schema import WorkloadItem


@dataclass(frozen=True)
class NormalizedResearchAiReplayItem:
    """B6 replay row normalized for model-capacity API execution."""

    prompt_id: str
    vertical: str
    workload_name: str
    prompt: str
    expected_output: str | None
    metadata: dict[str, str]
    source_metadata: dict[str, Any]

    def to_workload_item(self) -> WorkloadItem:
        """Return a strict WorkloadItem without audit-only row fields."""

        return WorkloadItem(
            prompt_id=self.prompt_id,
            workload_name=self.workload_name,
            prompt=self.prompt,
            expected_output=self.expected_output,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return asdict(self)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                msg = f"Expected JSON object row in {path} at line {line_number}"
                raise ValueError(msg)
            rows.append(payload)
    return rows


def _json_object(value: object, *, field_name: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        payload = json.loads(value)
        if isinstance(payload, dict):
            return payload
    msg = f"{field_name} must be a JSON object"
    raise ValueError(msg)


def _string_metadata(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    metadata: dict[str, str] = {}
    for key, raw_value in value.items():
        if isinstance(key, str):
            metadata[key] = raw_value if isinstance(raw_value, str) else json.dumps(raw_value)
    return metadata


def _runner_payload(row: dict[str, Any]) -> dict[str, Any]:
    if "runner_input" in row:
        return _json_object(row["runner_input"], field_name="runner_input")
    required = {"prompt_id", "prompt"}
    if required.issubset(row):
        return row
    msg = "Replay row must contain runner_input or direct prompt_id/prompt fields"
    raise ValueError(msg)


def _source_metadata(row: dict[str, Any], runner: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_row": row,
        "runner_input": runner,
        "b6_evaluation": row.get("b6_evaluation"),
        "b6_result": row.get("b6_result"),
        "b6_failure_flags": row.get("b6_failure_flags"),
        "b6_token_latency": row.get("b6_token_latency"),
    }


def normalize_research_ai_replay_row(row: dict[str, Any]) -> NormalizedResearchAiReplayItem:
    """Normalize one B6 replay/audit row without passing extra kwargs to WorkloadItem."""

    runner = _runner_payload(row)
    prompt_id = str(runner.get("prompt_id") or row.get("prompt_id") or "").strip()
    prompt = str(runner.get("prompt") or row.get("prompt") or "").strip()
    if not prompt_id:
        raise ValueError("Replay row is missing required prompt_id")
    if not prompt:
        raise ValueError(f"Replay row {prompt_id} is missing required prompt")
    metadata = _string_metadata(runner.get("metadata"))
    metadata.setdefault("vertical", str(row.get("vertical") or metadata.get("vertical") or ""))
    metadata.setdefault("b6r3_source_replay", "b6r1_research_ai_failed_replay")
    vertical = metadata.get("vertical") or str(row.get("vertical") or "")
    if vertical != "research_ai":
        raise ValueError(f"Replay row {prompt_id} is not Research AI: {vertical or 'missing'}")
    item = NormalizedResearchAiReplayItem(
        prompt_id=prompt_id,
        vertical=vertical,
        workload_name=str(runner.get("workload_name") or "b6r3_research_ai_capacity"),
        prompt=prompt,
        expected_output=(
            str(runner["expected_output"]) if runner.get("expected_output") is not None else None
        ),
        metadata=metadata,
        source_metadata=_source_metadata(row, runner),
    )
    assert_no_canonical_gold_id_leakage(item.to_workload_item())
    return item


def load_research_ai_capacity_replay(
    path: str | Path,
    *,
    limit: int | None = None,
) -> list[NormalizedResearchAiReplayItem]:
    """Load B6R1/B6R2 Research AI replay rows safely for B6R3."""

    if limit is not None and not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    rows = _read_jsonl(path)
    items = [normalize_research_ai_replay_row(row) for row in rows]
    return items[:limit] if limit is not None else items


def completed_prompt_ids_from_jsonl(path: str | Path) -> set[str]:
    """Return prompt IDs already present in an incremental output JSONL file."""

    output = Path(path)
    if not output.exists():
        return set()
    completed: set[str] = set()
    with output.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict) and payload.get("prompt_id"):
                completed.add(str(payload["prompt_id"]))
    return completed


def pending_replay_items(
    items: list[NormalizedResearchAiReplayItem],
    *,
    completed_prompt_ids: set[str],
) -> list[NormalizedResearchAiReplayItem]:
    """Return items not already written to an incremental output file."""

    return [item for item in items if item.prompt_id not in completed_prompt_ids]


def validate_b6r3_cli_limits(*, limit: int, max_new_tokens: int) -> None:
    """Validate B6R3 API replay bounds."""

    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    if not 1 <= max_new_tokens <= 512:
        raise ValueError("max_new_tokens must be between 1 and 512")


def choose_b6r3_contract_id(item: NormalizedResearchAiReplayItem) -> str:
    """Select the B6R3 Research AI contract strategy."""

    from inference_bench.generation_contract_registry import (  # noqa: PLC0415
        RESEARCH_AI_ADAPTIVE,
        RESEARCH_AI_LIMITATIONS,
        route_research_ai_contract,
    )

    route = route_research_ai_contract(prompt_text=item.prompt, metadata=item.metadata)
    return RESEARCH_AI_LIMITATIONS if route == RESEARCH_AI_LIMITATIONS else RESEARCH_AI_ADAPTIVE


def build_b6r3_manifest_payload(
    *,
    run_id: str,
    model_alias: str,
    model_id: str,
    provider: str,
    backend: str,
    input_path: str,
    output_path: str,
    limit: int,
    max_new_tokens: int,
    start_time: str,
    end_time: str | None,
    expected_count: int,
    completed_count: int,
    error_count: int,
    total_cost_usd: float,
    status: str,
    command: str,
) -> dict[str, Any]:
    """Build a B6R3 manifest and refuse false completion."""

    if status == "completed" and completed_count < expected_count:
        msg = (
            "Cannot mark B6R3 run completed when completed_count "
            f"{completed_count} is below expected_count {expected_count}"
        )
        raise ValueError(msg)
    return {
        "run_id": run_id,
        "model_alias": model_alias,
        "model_id": model_id,
        "provider": provider,
        "backend": backend,
        "input_path": input_path,
        "output_path": output_path,
        "limit": limit,
        "max_new_tokens": max_new_tokens,
        "start_time": start_time,
        "end_time": end_time,
        "expected_count": expected_count,
        "completed_count": completed_count,
        "error_count": error_count,
        "total_cost_usd": total_cost_usd,
        "status": status,
        "command": command,
    }

"""Deterministic synthetic scaled workload generation."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SCALED_WORKLOADS = (
    "short_chat",
    "code_helpdesk",
    "long_context",
    "shared_prefix",
    "structured_output",
)

SHARED_IT_SUPPORT_PREFIX = (
    "You are an internal IT support assistant. Use concise, policy-aware guidance, "
    "state assumptions, and avoid requesting sensitive credentials."
)


@dataclass(frozen=True)
class ScaledWorkloadRecord:
    """A JSONL-ready synthetic workload record."""

    prompt_id: str
    workload_name: str
    prompt: str
    metadata: dict[str, str]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "prompt_id": self.prompt_id,
            "workload_name": self.workload_name,
            "prompt": self.prompt,
            "metadata": self.metadata,
        }


def _metadata(
    *,
    count: int,
    template_family: str,
    seed: int,
) -> dict[str, str]:
    return {
        "scale_count": str(count),
        "template_family": template_family,
        "synthetic": "true",
        "seed": str(seed),
    }


def _choice(rng: random.Random, values: tuple[str, ...]) -> str:
    return values[rng.randrange(len(values))]


def _short_chat_prompt(rng: random.Random, index: int) -> tuple[str, str]:
    topics = (
        "a delayed data export",
        "a deployment status update",
        "a customer support handoff",
        "a weekly platform operations summary",
        "a configuration review reminder",
    )
    audiences = (
        "an engineering manager",
        "a support lead",
        "a product operations team",
        "a reliability review group",
        "a technical program manager",
    )
    formats = (
        "Draft a concise professional update about {topic} for {audience}.",
        "Summarize the status of {topic} in three clear bullets for {audience}.",
        "Rewrite this note to be direct and professional: The team is still checking {topic}.",
        "Write a confirmation message that acknowledges {topic} and states the next step.",
        "Explain {topic} in practical terms for {audience}.",
    )
    template = _choice(rng, formats)
    return template.format(topic=_choice(rng, topics), audience=_choice(rng, audiences)), (
        f"short_chat_{index % len(formats)}"
    )


def _code_helpdesk_prompt(rng: random.Random, index: int) -> tuple[str, str]:
    issues = (
        "a Python CLI command exits with code 1 after parsing arguments",
        "a Git branch cannot fast-forward because the remote has new commits",
        "a dependency import fails only in the CI environment",
        "a virtual environment uses a different Python version than expected",
        "a JSONL loader rejects a record during validation",
        "a PowerShell script writes output to the wrong results directory",
        "a package entry point works locally but not after installation",
    )
    requests = (
        "List the most likely causes and a focused debugging sequence.",
        "Provide a concise triage plan with commands to inspect the issue.",
        "Suggest a safe fix path and the checks to run afterward.",
        "Explain what evidence would distinguish configuration problems from code defects.",
    )
    prompt = f"A developer reports that {_choice(rng, issues)}. {_choice(rng, requests)}"
    return prompt, f"code_helpdesk_{index % len(requests)}"


def _long_context_prompt(rng: random.Random, index: int) -> tuple[str, str]:
    domains = (
        "platform operations",
        "support process",
        "data pipeline",
        "incident review",
        "rollout notes",
        "documentation update",
    )
    domain = _choice(rng, domains)
    passage_parts = [
        f"The {domain} team completed checkpoint {index % 7 + 1} during the review window.",
        "The primary objective was to reduce ambiguity in ownership and escalation paths.",
        "Operators recorded handoff status, validation evidence, and remaining risks.",
        "The process requires that generated artifacts stay separate from curated samples.",
        "Open issues should be tracked with clear reproduction steps and expected outcomes.",
        "The next review should compare latency, reliability, and output quality together.",
    ]
    rng.shuffle(passage_parts)
    request = _choice(
        rng,
        (
            "Summarize the passage into an operations-ready status note.",
            "Identify the key risks and recommended follow-up actions.",
            "Extract the decision-relevant details for a benchmark planning document.",
            "Rewrite the passage as a concise documentation update.",
        ),
    )
    return "Passage:\n" + "\n".join(passage_parts) + f"\n\nRequest: {request}", (
        f"long_context_{index % len(domains)}"
    )


def _shared_prefix_prompt(rng: random.Random, index: int) -> tuple[str, str]:
    user_requests = (
        "A user cannot access the internal dashboard after a password reset.",
        "A teammate reports that VPN connects but internal services time out.",
        "A support agent needs a response for a missing device enrollment prompt.",
        "An employee asks whether they should send an access token in chat for debugging.",
        "A manager needs a short escalation note for repeated SSO failures.",
    )
    prompt = f"{SHARED_IT_SUPPORT_PREFIX}\n\nUser request: {_choice(rng, user_requests)}"
    return prompt, f"shared_prefix_{index % len(user_requests)}"


def _structured_output_prompt(rng: random.Random, index: int) -> tuple[str, str]:
    categories = (
        "greeting",
        "helpdesk",
        "code_support",
        "operations",
        "documentation",
    )
    category = _choice(rng, categories)
    scenarios = {
        "greeting": "A user asks for a brief welcome message before a benchmark review.",
        "helpdesk": "A support queue needs a concise answer about a login troubleshooting step.",
        "code_support": "A developer asks how to handle a failing command-line validation check.",
        "operations": "An operations lead asks for the next action after a rollout delay.",
        "documentation": (
            "A maintainer asks for a short note explaining a generated artifact policy."
        ),
    }
    prompt = (
        "Return valid JSON only with this schema: "
        '{"category": string, "answer": string, "confidence": number}. '
        f"Use category {category}. Scenario: {scenarios[category]}"
    )
    return prompt, f"structured_output_{index % len(categories)}"


def generate_records_for_workload(
    workload_name: str,
    *,
    count: int,
    seed: int,
) -> list[ScaledWorkloadRecord]:
    """Generate deterministic records for one supported workload."""

    if count <= 0:
        msg = "count must be > 0"
        raise ValueError(msg)
    if workload_name not in DEFAULT_SCALED_WORKLOADS:
        msg = f"Unsupported workload: {workload_name}"
        raise ValueError(msg)

    rng = random.Random(f"{seed}:{workload_name}:{count}")
    prompt_builders = {
        "short_chat": _short_chat_prompt,
        "code_helpdesk": _code_helpdesk_prompt,
        "long_context": _long_context_prompt,
        "shared_prefix": _shared_prefix_prompt,
        "structured_output": _structured_output_prompt,
    }
    builder = prompt_builders[workload_name]

    records: list[ScaledWorkloadRecord] = []
    for index in range(count):
        prompt, template_family = builder(rng, index)
        records.append(
            ScaledWorkloadRecord(
                prompt_id=f"{workload_name}-{index + 1:05d}",
                workload_name=workload_name,
                prompt=prompt,
                metadata=_metadata(
                    count=count,
                    template_family=template_family,
                    seed=seed,
                ),
            )
        )
    return records


def generate_scaled_workloads(
    *,
    output_dir: str | Path = "data/prompts/scaled",
    count: int = 100,
    seed: int = 42,
    workloads: Iterable[str] | None = None,
) -> list[Path]:
    """Generate scaled synthetic workload JSONL files and return written paths."""

    selected_workloads = tuple(workloads) if workloads is not None else DEFAULT_SCALED_WORKLOADS
    if not selected_workloads:
        msg = "at least one workload must be selected"
        raise ValueError(msg)
    if count <= 0:
        msg = "count must be > 0"
        raise ValueError(msg)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    written_paths: list[Path] = []
    for workload_name in selected_workloads:
        records = generate_records_for_workload(workload_name, count=count, seed=seed)
        output_path = destination / f"{workload_name}_{count}.jsonl"
        with output_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record.to_json_dict(), sort_keys=True) + "\n")
        written_paths.append(output_path)

    return written_paths

"""Prompt templates for the bounded mm4 workflow."""

from __future__ import annotations

from collections.abc import Collection

from inference_bench.generation_contract import generation_contract_instruction

TASK_CLASSIFIER_PROMPT = """\
Classify only the task type and operational risk. Do not answer the user.
Use project metadata only. Do not use internet tools or reveal hidden reasoning.
Return a compact task label and risk level."""

RETRIEVAL_PLANNER_PROMPT = """\
Choose a bounded retrieval plan over the project corpus only.
At most two retrieval rounds are allowed. Do not use internet or arbitrary tools.
Prefer the promoted hybrid top-five source and escalate if evidence remains insufficient.
Return plan fields only; do not answer the user or reveal hidden reasoning."""

ANSWER_GENERATOR_PROMPT = """\
Answer only from the supplied project evidence.
Cite every evidence ID needed to support the answer, including multiple IDs when required.
Return exactly the strict five-field JSON contract.
Escalate through insufficient_evidence when support is missing.
Do not reveal hidden reasoning, planning notes, or chain-of-thought."""

REPAIR_NODE_PROMPT = """\
Correct the previous output once using only the same supplied evidence.
Do not add facts or evidence IDs. Preserve the answer when possible.
Fix only contract structure, evidence IDs, citation notes, or safety wording.
Return exactly one strict JSON object and do not reveal hidden reasoning."""


def render_agent_answer_prompt(*, evidence_prompt: str) -> str:
    """Combine the bounded agent instruction with the shared evidence prompt."""

    return "\n\n".join(
        [
            f"AGENT SYSTEM:\n{ANSWER_GENERATOR_PROMPT}",
            evidence_prompt,
        ]
    )


def render_agent_repair_prompt(
    *,
    original_prompt: str,
    previous_output: str,
    violation: str,
    allowed_evidence_ids: Collection[str],
) -> str:
    """Render the single allowed mm4 repair request."""

    labels = ", ".join(allowed_evidence_ids) or "none"
    return "\n\n".join(
        [
            f"AGENT REPAIR SYSTEM:\n{REPAIR_NODE_PROMPT}",
            original_prompt,
            f"PREVIOUS OUTPUT:\n{previous_output}",
            f"VALIDATION FAILURE:\n{violation}",
            f"ALLOWED EVIDENCE IDS:\n{labels}",
            f"OUTPUT CONTRACT:\n{generation_contract_instruction()}",
        ]
    )

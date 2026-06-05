"""Shared structured generation contract for grounded inference outputs."""

from __future__ import annotations

import json
import re
from collections.abc import Collection
from dataclasses import asdict, dataclass
from typing import cast

from inference_bench.context_schema import ContextRecord

GENERATION_CONTRACT_FORMAT = "generation_contract_json"
GENERATION_CONTRACT_FIELDS = (
    "answer",
    "evidence_ids",
    "confidence",
    "insufficient_evidence",
    "citation_notes",
)
PARSE_ERROR_NO_JSON = "no_json_object"
PARSE_ERROR_TRUNCATED_JSON = "truncated_json"
PARSE_ERROR_INVALID_JSON = "invalid_json"
PARSE_ERROR_MISSING_FIELDS = "missing_fields"
PARSE_ERROR_INVALID_CONTRACT = "invalid_contract"
PARSE_ERROR_INVALID_EVIDENCE_ID = "invalid_evidence_id"


def _validate_non_empty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        msg = f"{field_name} must be a non-empty string"
        raise ValueError(msg)


@dataclass(frozen=True)
class GenerationContract:
    """Evaluator-friendly answer payload produced by all model runners."""

    answer: str
    evidence_ids: list[str]
    confidence: float
    insufficient_evidence: bool
    citation_notes: str

    def __post_init__(self) -> None:
        if not isinstance(self.answer, str):
            msg = "answer must be a string"
            raise ValueError(msg)
        if not isinstance(self.evidence_ids, list) or not all(
            isinstance(evidence_id, str) and evidence_id.strip()
            for evidence_id in self.evidence_ids
        ):
            msg = "evidence_ids must be a list of non-empty strings"
            raise ValueError(msg)
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            msg = "confidence must be a number between 0 and 1"
            raise ValueError(msg)
        if not 0.0 <= float(self.confidence) <= 1.0:
            msg = "confidence must be between 0 and 1"
            raise ValueError(msg)
        if not isinstance(self.insufficient_evidence, bool):
            msg = "insufficient_evidence must be boolean"
            raise ValueError(msg)
        if not isinstance(self.citation_notes, str):
            msg = "citation_notes must be a string"
            raise ValueError(msg)
        if not self.insufficient_evidence and not self.answer.strip():
            msg = "answer must be non-empty unless insufficient_evidence is true"
            raise ValueError(msg)
        if self.insufficient_evidence and self.evidence_ids:
            msg = "evidence_ids must be empty when insufficient_evidence is true"
            raise ValueError(msg)
        if self.insufficient_evidence and self.answer.strip():
            msg = "answer must be empty when insufficient_evidence is true"
            raise ValueError(msg)
        if not self.insufficient_evidence and not self.evidence_ids:
            msg = "evidence_ids must contain at least one label for an answer"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe contract payload."""

        return asdict(self)


@dataclass(frozen=True)
class GenerationContractParse:
    """Structured result of parsing model-generated text."""

    json_valid: bool
    contract_valid: bool
    missing_fields: list[str]
    error: str | None
    parse_error_type: str | None
    parse_repair_applied: bool
    truncation_detected: bool
    contract: GenerationContract | None
    parsed_payload: dict[str, object] | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe parse result."""

        payload = asdict(self)
        payload["contract"] = self.contract.to_dict() if self.contract else None
        return payload


def _first_json_object_text(text: str) -> tuple[str | None, bool]:
    """Return the first balanced JSON object substring and truncation signal."""

    for start_index, character in enumerate(text):
        if character != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for end_index in range(start_index, len(text)):
            current = text[end_index]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    return text[start_index : end_index + 1], False
        return None, depth > 0 or in_string
    return None, False


def detect_json_truncation(text: str) -> bool:
    """Return whether generated text starts a JSON object but does not close it."""

    candidate, truncated = _first_json_object_text(text)
    return candidate is None and truncated


def _repair_simple_json(candidate: str) -> tuple[str, bool]:
    """Remove only trailing commas before object/array closing delimiters."""

    repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
    return repaired, repaired != candidate


def _parse_json_candidate(text: str) -> tuple[dict[str, object] | None, bool, bool, str | None]:
    candidate, truncated = _first_json_object_text(text)
    if candidate is None:
        error_type = PARSE_ERROR_TRUNCATED_JSON if truncated else PARSE_ERROR_NO_JSON
        return None, False, truncated, error_type

    extracted = candidate.strip() != text.strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        repaired, repair_applied = _repair_simple_json(candidate)
        if not repair_applied:
            return None, extracted, False, PARSE_ERROR_INVALID_JSON
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError:
            return None, extracted, False, PARSE_ERROR_INVALID_JSON
        extracted = True

    if not isinstance(parsed, dict):
        return None, extracted, False, PARSE_ERROR_INVALID_JSON
    return parsed, extracted, False, None


def parse_generation_contract(
    text: str,
    *,
    allowed_evidence_ids: Collection[str] | None = None,
) -> GenerationContractParse:
    """Parse, safely repair, and validate the first JSON object in generated text."""

    parsed, repair_applied, truncation_detected, parse_error_type = _parse_json_candidate(text)
    if parsed is None:
        return GenerationContractParse(
            json_valid=False,
            contract_valid=False,
            missing_fields=list(GENERATION_CONTRACT_FIELDS),
            error=(
                "Generated JSON appears truncated."
                if truncation_detected
                else "No valid JSON object found in generated text."
            ),
            parse_error_type=parse_error_type,
            parse_repair_applied=repair_applied,
            truncation_detected=truncation_detected,
            contract=None,
            parsed_payload=None,
        )

    missing_fields = [field for field in GENERATION_CONTRACT_FIELDS if field not in parsed]
    if missing_fields:
        return GenerationContractParse(
            json_valid=True,
            contract_valid=False,
            missing_fields=missing_fields,
            error=f"Missing generation contract fields: {', '.join(missing_fields)}",
            parse_error_type=PARSE_ERROR_MISSING_FIELDS,
            parse_repair_applied=repair_applied,
            truncation_detected=False,
            contract=None,
            parsed_payload=parsed,
        )

    try:
        contract = GenerationContract(
            answer=cast(str, parsed["answer"]),
            evidence_ids=cast(list[str], parsed["evidence_ids"]),
            confidence=cast(float, parsed["confidence"]),
            insufficient_evidence=cast(bool, parsed["insufficient_evidence"]),
            citation_notes=cast(str, parsed["citation_notes"]),
        )
    except (TypeError, ValueError) as exc:
        return GenerationContractParse(
            json_valid=True,
            contract_valid=False,
            missing_fields=[],
            error=str(exc),
            parse_error_type=PARSE_ERROR_INVALID_CONTRACT,
            parse_repair_applied=repair_applied,
            truncation_detected=False,
            contract=None,
            parsed_payload=parsed,
        )
    if allowed_evidence_ids is not None:
        allowed = set(allowed_evidence_ids)
        invalid_evidence_ids = [
            evidence_id for evidence_id in contract.evidence_ids if evidence_id not in allowed
        ]
        if invalid_evidence_ids:
            return GenerationContractParse(
                json_valid=True,
                contract_valid=False,
                missing_fields=[],
                error=(
                    "evidence_ids contains labels not present in the retrieved evidence: "
                    + ", ".join(invalid_evidence_ids)
                ),
                parse_error_type=PARSE_ERROR_INVALID_EVIDENCE_ID,
                parse_repair_applied=repair_applied,
                truncation_detected=False,
                contract=None,
                parsed_payload=parsed,
            )
    return GenerationContractParse(
        json_valid=True,
        contract_valid=True,
        missing_fields=[],
        error=None,
        parse_error_type=None,
        parse_repair_applied=repair_applied,
        truncation_detected=False,
        contract=contract,
        parsed_payload=parsed,
    )


def generation_contract_result_fields(
    text: str,
    *,
    allowed_evidence_ids: Collection[str] | None = None,
) -> dict[str, object]:
    """Return normalized contract fields for a runner result row."""

    parsed = parse_generation_contract(text, allowed_evidence_ids=allowed_evidence_ids)
    contract = parsed.contract
    return {
        "generation_contract_valid": parsed.contract_valid,
        "generation_contract_error": parsed.error,
        "generation_contract_missing_fields": parsed.missing_fields,
        "parse_error_type": parsed.parse_error_type,
        "parse_repair_applied": parsed.parse_repair_applied,
        "truncation_detected": parsed.truncation_detected,
        "answer": contract.answer if contract else "",
        "evidence_ids": contract.evidence_ids if contract else [],
        "citations": contract.evidence_ids if contract else [],
        "confidence": contract.confidence if contract else None,
        "insufficient_evidence": contract.insufficient_evidence if contract else None,
        "citation_notes": contract.citation_notes if contract else "",
    }


def allowed_evidence_ids_from_aliases(value: object) -> list[str]:
    """Return allowed short labels from a citation-alias mapping or JSON string."""

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, dict):
        return []
    return [str(label) for label in value if str(label).strip()]


def citation_aliases(context: ContextRecord) -> list[str]:
    """Return stable evidence aliases exposed to the model and evaluator."""

    aliases: list[str] = [
        context.chunk_id,
        context.context_id,
        context.parent_id,
    ]
    for key in (
        "original_doc_id",
        "section_record_id",
        "source_manifest_record_id",
        "policy_family_id",
        "admin_procedure_family_id",
    ):
        value = context.metadata.get(key)
        if isinstance(value, str) and value:
            aliases.append(value)
    review_doc_ids = context.metadata.get("review_doc_ids")
    if isinstance(review_doc_ids, list):
        aliases.extend(str(value) for value in review_doc_ids if value)
    return list(dict.fromkeys(alias for alias in aliases if alias))


def primary_citation_id(context: ContextRecord) -> str:
    """Return the stable citation label models should emit."""

    for candidate in (
        context.chunk_id,
        str(context.metadata.get("original_doc_id") or ""),
        context.context_id,
    ):
        if candidate.strip():
            return candidate
    msg = "Context record has no stable citation identifier"
    raise ValueError(msg)


def citation_label(index: int) -> str:
    """Return the short stable label for one ranked evidence record."""

    if index <= 0:
        msg = "citation label index must be > 0"
        raise ValueError(msg)
    return f"E{index}"


def render_evidence_blocks(context_records: list[ContextRecord]) -> str:
    """Render context records with stable machine-readable citation labels."""

    if not context_records:
        return "No retrieved evidence was supplied."
    blocks: list[str] = []
    for index, context in enumerate(context_records, start=1):
        citation_id = citation_label(index)
        aliases = citation_aliases(context)
        blocks.append(
            "\n".join(
                [
                    f"[EVIDENCE {index}]",
                    f"evidence_id: {citation_id}",
                    f"title: {context.title}",
                    f"source_type: {context.source_type}",
                    f"citation_aliases: {json.dumps(aliases, ensure_ascii=True)}",
                    f"text: {context.text}",
                ]
            )
        )
    return "\n\n".join(blocks)


def generation_contract_instruction() -> str:
    """Return the exact output instruction shared by all runners."""

    return "\n".join(
        [
            "Return exactly one compact, single-line JSON object.",
            "Do not use markdown, code fences, headings, or prose outside the JSON object.",
            f"Required fields: {', '.join(GENERATION_CONTRACT_FIELDS)}.",
            "Use these exact JSON types: answer string, evidence_ids array of strings, "
            "confidence number, insufficient_evidence boolean, citation_notes string.",
            "Keep answer at or below 40 words and citation_notes at or below 12 words.",
            "Use only evidence_id labels shown in the retrieved evidence blocks, such as E1.",
            "For an answer, evidence_ids must contain at least one supporting provided label.",
            "Cite every provided evidence block directly used to support the answer.",
            "Inspect all evidence blocks before choosing labels; do not default to E1 only.",
            "If E1 and E2 provide distinct relevant support, include both E1 and E2.",
            "Do not cite a label whose evidence does not support the answer.",
            "Short evidence labels such as E1 map to canonical evidence IDs in runner metadata.",
            "confidence must be a number from 0 to 1.",
            "If the evidence is insufficient, set insufficient_evidence to true, "
            "answer to an empty string, evidence_ids to [], and explain why in citation_notes.",
            "Otherwise set insufficient_evidence to false and provide a non-empty answer.",
            "Do not copy wording from these instructions into answer or citation_notes.",
        ]
    )


def render_contract_retry_prompt(
    *,
    bad_output: str,
    violation: str,
    allowed_evidence_ids: Collection[str],
) -> str:
    """Render a one-time correction prompt without changing evidence or answer facts."""

    allowed_labels = ", ".join(allowed_evidence_ids) or "none"
    return "\n\n".join(
        [
            "CONTRACT CORRECTION TASK:",
            "Do not answer the original question again. Correct only the JSON structure and types.",
            f"PREVIOUS INVALID OUTPUT:\n{bad_output}",
            f"The previous output violated the contract: {violation}",
            f"Allowed evidence_id labels remain exactly: {allowed_labels}",
            "Preserve the previous factual answer and citation notes.",
            "Do not add facts or evidence labels that were not present above.",
            "Keep any allowed evidence labels already present in the previous output.",
            "confidence must be a number from 0 to 1.",
            "The corrected object must contain exactly these five fields: answer, evidence_ids, "
            "confidence, insufficient_evidence, citation_notes.",
            "Return only the corrected compact JSON object now.",
        ]
    )


def render_generation_contract_prompt(
    *,
    question: str,
    context_records: list[ContextRecord],
    memory_mode: str,
) -> str:
    """Render a model prompt with labeled evidence and the shared contract."""

    _validate_non_empty_string(question, "question")
    return "\n\n".join(
        [
            "SYSTEM:\nAnswer only from supplied evidence. Do not invent citations.",
            f"MEMORY MODE:\n{memory_mode}",
            f"RETRIEVED EVIDENCE:\n{render_evidence_blocks(context_records)}",
            f"USER QUESTION:\n{question.strip()}",
            f"OUTPUT CONTRACT:\n{generation_contract_instruction()}",
        ]
    )

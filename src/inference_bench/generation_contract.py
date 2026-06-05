"""Shared structured generation contract for grounded inference outputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import cast

from inference_bench.context_schema import ContextRecord
from inference_bench.quality import parse_json_object

GENERATION_CONTRACT_FORMAT = "generation_contract_json"
GENERATION_CONTRACT_FIELDS = (
    "answer",
    "evidence_ids",
    "confidence",
    "insufficient_evidence",
    "citation_notes",
)


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
    contract: GenerationContract | None
    parsed_payload: dict[str, object] | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe parse result."""

        payload = asdict(self)
        payload["contract"] = self.contract.to_dict() if self.contract else None
        return payload


def parse_generation_contract(text: str) -> GenerationContractParse:
    """Parse and validate the first JSON object in generated text."""

    parsed = parse_json_object(text)
    if parsed is None:
        return GenerationContractParse(
            json_valid=False,
            contract_valid=False,
            missing_fields=list(GENERATION_CONTRACT_FIELDS),
            error="No JSON object found in generated text.",
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
            contract=None,
            parsed_payload=parsed,
        )
    return GenerationContractParse(
        json_valid=True,
        contract_valid=True,
        missing_fields=[],
        error=None,
        contract=contract,
        parsed_payload=parsed,
    )


def generation_contract_result_fields(text: str) -> dict[str, object]:
    """Return normalized contract fields for a runner result row."""

    parsed = parse_generation_contract(text)
    contract = parsed.contract
    return {
        "generation_contract_valid": parsed.contract_valid,
        "generation_contract_error": parsed.error,
        "generation_contract_missing_fields": parsed.missing_fields,
        "answer": contract.answer if contract else "",
        "evidence_ids": contract.evidence_ids if contract else [],
        "citations": contract.evidence_ids if contract else [],
        "confidence": contract.confidence if contract else None,
        "insufficient_evidence": contract.insufficient_evidence if contract else None,
        "citation_notes": contract.citation_notes if contract else "",
    }


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

    example = {
        "answer": "Grounded answer in at most 40 words.",
        "evidence_ids": ["EXACT_EVIDENCE_ID"],
        "confidence": 0.8,
        "insufficient_evidence": False,
        "citation_notes": "Direct support.",
    }
    return "\n".join(
        [
            "Return exactly one compact, single-line JSON object and no markdown or extra prose.",
            f"Required fields: {', '.join(GENERATION_CONTRACT_FIELDS)}.",
            "Keep answer at or below 40 words and citation_notes at or below 12 words.",
            "Use only evidence_id values shown in the retrieved evidence blocks.",
            "Cite only the minimum evidence records needed to support the answer.",
            "Short evidence labels such as E1 map to canonical evidence IDs in runner metadata.",
            "confidence must be a number from 0 to 1.",
            "If the evidence is insufficient, set insufficient_evidence to true, "
            "answer to an empty string, evidence_ids to [], and explain why in citation_notes.",
            f"JSON shape example: {json.dumps(example, ensure_ascii=True)}",
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

"""Versioned vertical generation contracts and common evaluator mapping."""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from inference_bench.generation_contract import (
    GENERATION_CONTRACT_FIELDS,
    GenerationContract,
    parse_generation_contract,
)

DEFAULT_GENERATION_CONTRACT_ID = "default_grounded_json_v1"
RESEARCH_AI_MINIMAL_ANSWER = "research_ai_minimal_answer_v1"
RESEARCH_AI_FINDINGS = "research_ai_findings_v1"
RESEARCH_AI_LIMITATIONS = "research_ai_limitations_v1"
RESEARCH_AI_COMPARISON = "research_ai_comparison_v1"
RESEARCH_AI_ADAPTIVE = "research_ai_adaptive_v1"

RESEARCH_AI_DIRECT_CONTRACT_IDS = (
    RESEARCH_AI_MINIMAL_ANSWER,
    RESEARCH_AI_FINDINGS,
    RESEARCH_AI_LIMITATIONS,
    RESEARCH_AI_COMPARISON,
)
RESEARCH_AI_CONTRACT_IDS = (*RESEARCH_AI_DIRECT_CONTRACT_IDS, RESEARCH_AI_ADAPTIVE)
B6R2_CONTRACT_MAX_NEW_TOKENS = (224, 320)

DEFAULT_CONTRACT_VERTICALS = ("airline", "healthcare_admin", "retail", "finance")
CONFIDENCE_TO_FLOAT = {"low": 0.35, "medium": 0.65, "high": 0.9}


@dataclass(frozen=True)
class ContractDefinition:
    """Stable description for one model-facing generation contract."""

    contract_id: str
    version: str
    verticals: tuple[str, ...]
    required_fields: tuple[str, ...]
    schema: dict[str, Any]
    common_metric_mapping: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe definition."""

        return asdict(self)


@dataclass(frozen=True)
class ContractValidationResult:
    """Validation and common-contract mapping for one generated output."""

    contract_id: str
    effective_contract_id: str
    json_valid: bool
    contract_valid: bool
    missing_fields: list[str]
    error: str | None
    parse_error_type: str | None
    truncation_detected: bool
    parsed_payload: dict[str, Any] | None
    common_payload: dict[str, Any] | None

    @property
    def common_text(self) -> str:
        """Return the mapped common JSON text, or an empty string when invalid."""

        if self.common_payload is None:
            return ""
        return json.dumps(self.common_payload, ensure_ascii=True, separators=(",", ":"))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe validation payload."""

        return {
            **asdict(self),
            "common_text": self.common_text,
        }


def _schema(fields: tuple[str, ...]) -> dict[str, Any]:
    return {"type": "object", "required": list(fields), "additionalProperties": False}


CONTRACT_DEFINITIONS: dict[str, ContractDefinition] = {
    DEFAULT_GENERATION_CONTRACT_ID: ContractDefinition(
        contract_id=DEFAULT_GENERATION_CONTRACT_ID,
        version="v1",
        verticals=DEFAULT_CONTRACT_VERTICALS,
        required_fields=GENERATION_CONTRACT_FIELDS,
        schema=_schema(GENERATION_CONTRACT_FIELDS),
        common_metric_mapping=(
            "json_validity",
            "generation_contract_valid",
            "evidence_match",
            "groundedness",
            "safety_violation",
            "truncation_detected",
        ),
    ),
    RESEARCH_AI_MINIMAL_ANSWER: ContractDefinition(
        contract_id=RESEARCH_AI_MINIMAL_ANSWER,
        version="v1",
        verticals=("research_ai",),
        required_fields=("answer", "evidence", "insufficient_evidence", "confidence"),
        schema=_schema(("answer", "evidence", "insufficient_evidence", "confidence")),
        common_metric_mapping=(
            "answer -> answer",
            "evidence -> evidence_ids",
            "confidence enum -> confidence number",
        ),
    ),
    RESEARCH_AI_FINDINGS: ContractDefinition(
        contract_id=RESEARCH_AI_FINDINGS,
        version="v1",
        verticals=("research_ai",),
        required_fields=("summary", "findings", "insufficient_evidence", "confidence"),
        schema=_schema(("summary", "findings", "insufficient_evidence", "confidence")),
        common_metric_mapping=(
            "summary and findings[].claim -> answer",
            "findings[].evidence -> evidence_ids",
            "confidence enum -> confidence number",
        ),
    ),
    RESEARCH_AI_LIMITATIONS: ContractDefinition(
        contract_id=RESEARCH_AI_LIMITATIONS,
        version="v1",
        verticals=("research_ai",),
        required_fields=(
            "limitation",
            "why_it_matters",
            "evidence",
            "insufficient_evidence",
            "confidence",
        ),
        schema=_schema(
            (
                "limitation",
                "why_it_matters",
                "evidence",
                "insufficient_evidence",
                "confidence",
            )
        ),
        common_metric_mapping=(
            "limitation and why_it_matters -> answer",
            "evidence -> evidence_ids",
            "confidence enum -> confidence number",
        ),
    ),
    RESEARCH_AI_COMPARISON: ContractDefinition(
        contract_id=RESEARCH_AI_COMPARISON,
        version="v1",
        verticals=("research_ai",),
        required_fields=("comparison_summary", "items", "insufficient_evidence", "confidence"),
        schema=_schema(("comparison_summary", "items", "insufficient_evidence", "confidence")),
        common_metric_mapping=(
            "comparison_summary and items[].claim -> answer",
            "items[].evidence -> evidence_ids",
            "confidence enum -> confidence number",
        ),
    ),
    RESEARCH_AI_ADAPTIVE: ContractDefinition(
        contract_id=RESEARCH_AI_ADAPTIVE,
        version="v1",
        verticals=("research_ai",),
        required_fields=("adaptive_route",),
        schema={"type": "deterministic_router", "routes": list(RESEARCH_AI_DIRECT_CONTRACT_IDS)},
        common_metric_mapping=("routes to one Research AI direct contract before validation",),
    ),
}


def contract_for_vertical(
    vertical: str,
    *,
    selected_research_ai_contract: str | None = None,
) -> str:
    """Return the model-facing contract selected for a vertical."""

    if vertical == "research_ai":
        return selected_research_ai_contract or RESEARCH_AI_ADAPTIVE
    return DEFAULT_GENERATION_CONTRACT_ID


def selected_contracts_by_vertical(
    *,
    selected_research_ai_contract: str | None = None,
) -> dict[str, str]:
    """Return the current vertical contract map."""

    return {
        "airline": DEFAULT_GENERATION_CONTRACT_ID,
        "healthcare_admin": DEFAULT_GENERATION_CONTRACT_ID,
        "retail": DEFAULT_GENERATION_CONTRACT_ID,
        "finance": DEFAULT_GENERATION_CONTRACT_ID,
        "research_ai": contract_for_vertical(
            "research_ai",
            selected_research_ai_contract=selected_research_ai_contract,
        ),
    }


def _first_json_object_text(text: str) -> tuple[str | None, bool]:
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


def _repair_simple_json(candidate: str) -> tuple[str, bool]:
    repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
    return repaired, repaired != candidate


def _parse_json_payload(text: str) -> tuple[dict[str, Any] | None, bool, str | None]:
    candidate, truncated = _first_json_object_text(text)
    if candidate is None:
        return None, truncated, "truncated_json" if truncated else "no_json_object"
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        repaired, changed = _repair_simple_json(candidate)
        if not changed:
            return None, False, "invalid_json"
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError:
            return None, False, "invalid_json"
    if not isinstance(parsed, dict):
        return None, False, "invalid_json"
    return parsed, False, None


def _string(value: object, field: str) -> tuple[str | None, str | None]:
    if isinstance(value, str) and value.strip():
        return value.strip(), None
    return None, f"{field} must be a non-empty string"


def _bool(value: object, field: str) -> tuple[bool | None, str | None]:
    if isinstance(value, bool):
        return value, None
    return None, f"{field} must be boolean"


def _confidence(value: object) -> tuple[float | None, str | None]:
    if isinstance(value, str) and value.strip().lower() in CONFIDENCE_TO_FLOAT:
        return CONFIDENCE_TO_FLOAT[value.strip().lower()], None
    return None, "confidence must be one of low, medium, high"


def _evidence_list(
    value: object,
    *,
    field: str,
    allowed_evidence_ids: Collection[str] | None,
    require_non_empty: bool,
) -> tuple[list[str] | None, str | None]:
    if not isinstance(value, list) or not all(isinstance(label, str) for label in value):
        return None, f"{field} must be an array of evidence labels"
    evidence = list(dict.fromkeys(label.strip() for label in value if label.strip()))
    if require_non_empty and not evidence:
        return None, f"{field} must contain at least one evidence label"
    if allowed_evidence_ids is not None:
        allowed = set(allowed_evidence_ids)
        invalid = [label for label in evidence if label not in allowed]
        if invalid:
            return None, f"{field} contains labels not present in context: {', '.join(invalid)}"
    return evidence, None


def _unknown_fields(payload: Mapping[str, object], required_fields: tuple[str, ...]) -> list[str]:
    return sorted(set(payload).difference(required_fields))


def _notes(labels: list[str]) -> str:
    if not labels:
        return "Evidence was insufficient."
    return "; ".join(f"{label}: supports the Research AI answer" for label in labels)


def _common_payload(
    *,
    answer: str,
    evidence_ids: list[str],
    confidence: float,
    insufficient_evidence: bool,
) -> dict[str, Any]:
    if insufficient_evidence:
        return {
            "answer": "",
            "evidence_ids": [],
            "confidence": confidence,
            "insufficient_evidence": True,
            "citation_notes": "Evidence was insufficient.",
        }
    return GenerationContract(
        answer=answer,
        evidence_ids=evidence_ids,
        confidence=confidence,
        insufficient_evidence=False,
        citation_notes=_notes(evidence_ids),
    ).to_dict()


def _validate_required_payload(
    *,
    payload: dict[str, Any],
    contract_id: str,
) -> tuple[tuple[str, ...], list[str], str | None]:
    definition = CONTRACT_DEFINITIONS[contract_id]
    required = definition.required_fields
    missing = [field for field in required if field not in payload]
    unknown = _unknown_fields(payload, required)
    if unknown:
        return required, missing, f"Unexpected fields: {', '.join(unknown)}"
    return required, missing, None


def _invalid_result(
    *,
    contract_id: str,
    effective_contract_id: str,
    json_valid: bool,
    missing_fields: list[str],
    error: str,
    parse_error_type: str,
    truncation_detected: bool = False,
    parsed_payload: dict[str, Any] | None = None,
) -> ContractValidationResult:
    return ContractValidationResult(
        contract_id=contract_id,
        effective_contract_id=effective_contract_id,
        json_valid=json_valid,
        contract_valid=False,
        missing_fields=missing_fields,
        error=error,
        parse_error_type=parse_error_type,
        truncation_detected=truncation_detected,
        parsed_payload=parsed_payload,
        common_payload=None,
    )


def _validate_minimal(
    *,
    contract_id: str,
    effective_contract_id: str,
    payload: dict[str, Any],
    allowed_evidence_ids: Collection[str] | None,
) -> ContractValidationResult:
    _required, missing, unknown_error = _validate_required_payload(
        payload=payload,
        contract_id=effective_contract_id,
    )
    if missing or unknown_error:
        return _invalid_result(
            contract_id=contract_id,
            effective_contract_id=effective_contract_id,
            json_valid=True,
            missing_fields=missing,
            error=unknown_error or f"Missing fields: {', '.join(missing)}",
            parse_error_type="invalid_contract" if unknown_error else "missing_fields",
            parsed_payload=payload,
        )
    insufficient, error = _bool(payload["insufficient_evidence"], "insufficient_evidence")
    if error:
        return _invalid_result(
            contract_id=contract_id,
            effective_contract_id=effective_contract_id,
            json_valid=True,
            missing_fields=[],
            error=error,
            parse_error_type="invalid_contract",
            parsed_payload=payload,
        )
    confidence, error = _confidence(payload["confidence"])
    if error or confidence is None or insufficient is None:
        return _invalid_result(
            contract_id=contract_id,
            effective_contract_id=effective_contract_id,
            json_valid=True,
            missing_fields=[],
            error=error or "invalid insufficient_evidence",
            parse_error_type="invalid_contract",
            parsed_payload=payload,
        )
    if insufficient:
        return ContractValidationResult(
            contract_id=contract_id,
            effective_contract_id=effective_contract_id,
            json_valid=True,
            contract_valid=True,
            missing_fields=[],
            error=None,
            parse_error_type=None,
            truncation_detected=False,
            parsed_payload=payload,
            common_payload=_common_payload(
                answer="",
                evidence_ids=[],
                confidence=confidence,
                insufficient_evidence=True,
            ),
        )
    answer, error = _string(payload["answer"], "answer")
    if error or answer is None:
        return _invalid_result(
            contract_id=contract_id,
            effective_contract_id=effective_contract_id,
            json_valid=True,
            missing_fields=[],
            error=error or "invalid answer",
            parse_error_type="invalid_contract",
            parsed_payload=payload,
        )
    evidence, error = _evidence_list(
        payload["evidence"],
        field="evidence",
        allowed_evidence_ids=allowed_evidence_ids,
        require_non_empty=True,
    )
    if error or evidence is None:
        return _invalid_result(
            contract_id=contract_id,
            effective_contract_id=effective_contract_id,
            json_valid=True,
            missing_fields=[],
            error=error or "invalid evidence",
            parse_error_type="invalid_contract",
            parsed_payload=payload,
        )
    return ContractValidationResult(
        contract_id=contract_id,
        effective_contract_id=effective_contract_id,
        json_valid=True,
        contract_valid=True,
        missing_fields=[],
        error=None,
        parse_error_type=None,
        truncation_detected=False,
        parsed_payload=payload,
        common_payload=_common_payload(
            answer=answer,
            evidence_ids=evidence,
            confidence=confidence,
            insufficient_evidence=False,
        ),
    )


def _validate_findings_or_comparison(
    *,
    contract_id: str,
    effective_contract_id: str,
    payload: dict[str, Any],
    allowed_evidence_ids: Collection[str] | None,
) -> ContractValidationResult:
    _required, missing, unknown_error = _validate_required_payload(
        payload=payload,
        contract_id=effective_contract_id,
    )
    if missing or unknown_error:
        return _invalid_result(
            contract_id=contract_id,
            effective_contract_id=effective_contract_id,
            json_valid=True,
            missing_fields=missing,
            error=unknown_error or f"Missing fields: {', '.join(missing)}",
            parse_error_type="invalid_contract" if unknown_error else "missing_fields",
            parsed_payload=payload,
        )
    insufficient, error = _bool(payload["insufficient_evidence"], "insufficient_evidence")
    confidence, confidence_error = _confidence(payload["confidence"])
    if error or confidence_error or confidence is None or insufficient is None:
        return _invalid_result(
            contract_id=contract_id,
            effective_contract_id=effective_contract_id,
            json_valid=True,
            missing_fields=[],
            error=error or confidence_error or "invalid control fields",
            parse_error_type="invalid_contract",
            parsed_payload=payload,
        )
    if insufficient:
        return ContractValidationResult(
            contract_id=contract_id,
            effective_contract_id=effective_contract_id,
            json_valid=True,
            contract_valid=True,
            missing_fields=[],
            error=None,
            parse_error_type=None,
            truncation_detected=False,
            parsed_payload=payload,
            common_payload=_common_payload(
                answer="",
                evidence_ids=[],
                confidence=confidence,
                insufficient_evidence=True,
            ),
        )

    if effective_contract_id == RESEARCH_AI_FINDINGS:
        summary, summary_error = _string(payload["summary"], "summary")
        list_field = "findings"
        item_fields: tuple[str, ...] = ("claim", "evidence")
    else:
        summary, summary_error = _string(payload["comparison_summary"], "comparison_summary")
        list_field = "items"
        item_fields = ("item", "claim", "evidence")
    if summary_error or summary is None:
        return _invalid_result(
            contract_id=contract_id,
            effective_contract_id=effective_contract_id,
            json_valid=True,
            missing_fields=[],
            error=summary_error or "invalid summary",
            parse_error_type="invalid_contract",
            parsed_payload=payload,
        )
    entries = payload[list_field]
    if not isinstance(entries, list) or not entries:
        return _invalid_result(
            contract_id=contract_id,
            effective_contract_id=effective_contract_id,
            json_valid=True,
            missing_fields=[],
            error=f"{list_field} must be a non-empty array",
            parse_error_type="invalid_contract",
            parsed_payload=payload,
        )
    claims: list[str] = []
    evidence_ids: list[str] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            return _invalid_result(
                contract_id=contract_id,
                effective_contract_id=effective_contract_id,
                json_valid=True,
                missing_fields=[],
                error=f"{list_field}[{index}] must be an object",
                parse_error_type="invalid_contract",
                parsed_payload=payload,
            )
        unknown = sorted(set(entry).difference(item_fields))
        missing_entry = [field for field in item_fields if field not in entry]
        if unknown or missing_entry:
            return _invalid_result(
                contract_id=contract_id,
                effective_contract_id=effective_contract_id,
                json_valid=True,
                missing_fields=missing_entry,
                error=(
                    f"{list_field}[{index}] has unexpected fields: {', '.join(unknown)}"
                    if unknown
                    else f"{list_field}[{index}] missing fields: {', '.join(missing_entry)}"
                ),
                parse_error_type="invalid_contract",
                parsed_payload=payload,
            )
        claim, claim_error = _string(entry["claim"], f"{list_field}[{index}].claim")
        if claim_error or claim is None:
            return _invalid_result(
                contract_id=contract_id,
                effective_contract_id=effective_contract_id,
                json_valid=True,
                missing_fields=[],
                error=claim_error or "invalid claim",
                parse_error_type="invalid_contract",
                parsed_payload=payload,
            )
        evidence, evidence_error = _evidence_list(
            entry["evidence"],
            field=f"{list_field}[{index}].evidence",
            allowed_evidence_ids=allowed_evidence_ids,
            require_non_empty=True,
        )
        if evidence_error or evidence is None:
            return _invalid_result(
                contract_id=contract_id,
                effective_contract_id=effective_contract_id,
                json_valid=True,
                missing_fields=[],
                error=evidence_error or "invalid evidence",
                parse_error_type="invalid_contract",
                parsed_payload=payload,
            )
        claims.append(claim)
        evidence_ids.extend(evidence)
    unique_evidence = list(dict.fromkeys(evidence_ids))
    answer = " ".join([summary, *claims]).strip()
    return ContractValidationResult(
        contract_id=contract_id,
        effective_contract_id=effective_contract_id,
        json_valid=True,
        contract_valid=True,
        missing_fields=[],
        error=None,
        parse_error_type=None,
        truncation_detected=False,
        parsed_payload=payload,
        common_payload=_common_payload(
            answer=answer,
            evidence_ids=unique_evidence,
            confidence=confidence,
            insufficient_evidence=False,
        ),
    )


def _validate_limitations(
    *,
    contract_id: str,
    effective_contract_id: str,
    payload: dict[str, Any],
    allowed_evidence_ids: Collection[str] | None,
) -> ContractValidationResult:
    _required, missing, unknown_error = _validate_required_payload(
        payload=payload,
        contract_id=effective_contract_id,
    )
    if missing or unknown_error:
        return _invalid_result(
            contract_id=contract_id,
            effective_contract_id=effective_contract_id,
            json_valid=True,
            missing_fields=missing,
            error=unknown_error or f"Missing fields: {', '.join(missing)}",
            parse_error_type="invalid_contract" if unknown_error else "missing_fields",
            parsed_payload=payload,
        )
    insufficient, error = _bool(payload["insufficient_evidence"], "insufficient_evidence")
    confidence, confidence_error = _confidence(payload["confidence"])
    if error or confidence_error or confidence is None or insufficient is None:
        return _invalid_result(
            contract_id=contract_id,
            effective_contract_id=effective_contract_id,
            json_valid=True,
            missing_fields=[],
            error=error or confidence_error or "invalid control fields",
            parse_error_type="invalid_contract",
            parsed_payload=payload,
        )
    if insufficient:
        return ContractValidationResult(
            contract_id=contract_id,
            effective_contract_id=effective_contract_id,
            json_valid=True,
            contract_valid=True,
            missing_fields=[],
            error=None,
            parse_error_type=None,
            truncation_detected=False,
            parsed_payload=payload,
            common_payload=_common_payload(
                answer="",
                evidence_ids=[],
                confidence=confidence,
                insufficient_evidence=True,
            ),
        )
    limitation, limitation_error = _string(payload["limitation"], "limitation")
    why, why_error = _string(payload["why_it_matters"], "why_it_matters")
    if limitation_error or why_error or limitation is None or why is None:
        return _invalid_result(
            contract_id=contract_id,
            effective_contract_id=effective_contract_id,
            json_valid=True,
            missing_fields=[],
            error=limitation_error or why_error or "invalid limitation payload",
            parse_error_type="invalid_contract",
            parsed_payload=payload,
        )
    evidence, evidence_error = _evidence_list(
        payload["evidence"],
        field="evidence",
        allowed_evidence_ids=allowed_evidence_ids,
        require_non_empty=True,
    )
    if evidence_error or evidence is None:
        return _invalid_result(
            contract_id=contract_id,
            effective_contract_id=effective_contract_id,
            json_valid=True,
            missing_fields=[],
            error=evidence_error or "invalid evidence",
            parse_error_type="invalid_contract",
            parsed_payload=payload,
        )
    return ContractValidationResult(
        contract_id=contract_id,
        effective_contract_id=effective_contract_id,
        json_valid=True,
        contract_valid=True,
        missing_fields=[],
        error=None,
        parse_error_type=None,
        truncation_detected=False,
        parsed_payload=payload,
        common_payload=_common_payload(
            answer=f"{limitation} {why}",
            evidence_ids=evidence,
            confidence=confidence,
            insufficient_evidence=False,
        ),
    )


def _metadata_value(metadata: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = metadata.get(name)
        if value:
            return value
    return ""


def _visible_research_ai_request_text(prompt_text: str) -> str:
    """Extract the user-visible request from a rendered benchmark prompt."""

    text = prompt_text
    if "\nUSER QUESTION:\n" in text:
        text = text.split("\nUSER QUESTION:\n", maxsplit=1)[1]
    if "\n\nOUTPUT CONTRACT:\n" in text:
        text = text.split("\n\nOUTPUT CONTRACT:\n", maxsplit=1)[0]
    if "INTERNAL ANSWER PLAN:" in text:
        text = text.split("INTERNAL ANSWER PLAN:", maxsplit=1)[0]
    return text.strip()


def route_research_ai_contract(
    *,
    prompt_text: str,
    metadata: Mapping[str, str] | None = None,
) -> str:
    """Route Research AI prompts to a direct contract with deterministic rules."""

    meta = metadata or {}
    task_text = " ".join(
        [
            _metadata_value(meta, "task_type", "question_type", "expected_output_format"),
            _visible_research_ai_request_text(prompt_text),
        ]
    ).lower()
    if any(
        phrase in task_text
        for phrase in (
            "compare",
            "comparison",
            "contrast",
            "versus",
            " vs ",
            "between papers",
            "methods differ",
            "approaches differ",
        )
    ):
        return RESEARCH_AI_COMPARISON
    if any(
        phrase in task_text
        for phrase in (
            "limitation",
            "limitations",
            "assumption",
            "future work",
            "failure mode",
            "weakness",
            "threat",
        )
    ):
        return RESEARCH_AI_LIMITATIONS
    if any(
        phrase in task_text
        for phrase in (
            "finding",
            "findings",
            "results",
            "contributions",
            "main points",
            "key points",
            "evidence from",
        )
    ):
        return RESEARCH_AI_FINDINGS
    return RESEARCH_AI_MINIMAL_ANSWER


def effective_contract_id(
    contract_id: str,
    *,
    prompt_text: str = "",
    metadata: Mapping[str, str] | None = None,
) -> str:
    """Resolve adaptive contracts to the direct contract used for validation."""

    if contract_id == RESEARCH_AI_ADAPTIVE:
        return route_research_ai_contract(prompt_text=prompt_text, metadata=metadata)
    return contract_id


def validate_and_map_contract_text(
    *,
    text: str,
    contract_id: str,
    allowed_evidence_ids: Collection[str] | None = None,
    prompt_text: str = "",
    metadata: Mapping[str, str] | None = None,
) -> ContractValidationResult:
    """Validate one model output and map it to the common evaluator contract."""

    if contract_id not in CONTRACT_DEFINITIONS:
        raise ValueError(f"Unknown generation contract: {contract_id}")
    effective = effective_contract_id(
        contract_id,
        prompt_text=prompt_text,
        metadata=metadata,
    )
    if effective not in CONTRACT_DEFINITIONS:
        raise ValueError(f"Unknown effective generation contract: {effective}")
    if effective == DEFAULT_GENERATION_CONTRACT_ID:
        parsed = parse_generation_contract(text, allowed_evidence_ids=allowed_evidence_ids)
        return ContractValidationResult(
            contract_id=contract_id,
            effective_contract_id=effective,
            json_valid=parsed.json_valid,
            contract_valid=parsed.contract_valid,
            missing_fields=parsed.missing_fields,
            error=parsed.error,
            parse_error_type=parsed.parse_error_type,
            truncation_detected=parsed.truncation_detected,
            parsed_payload=parsed.parsed_payload,
            common_payload=parsed.contract.to_dict() if parsed.contract else None,
        )
    if effective not in RESEARCH_AI_DIRECT_CONTRACT_IDS:
        return _invalid_result(
            contract_id=contract_id,
            effective_contract_id=effective,
            json_valid=False,
            missing_fields=[],
            error=f"Contract {effective} is not directly model-facing",
            parse_error_type="invalid_contract",
        )
    payload, truncated, parse_error = _parse_json_payload(text)
    if payload is None:
        return _invalid_result(
            contract_id=contract_id,
            effective_contract_id=effective,
            json_valid=False,
            missing_fields=list(CONTRACT_DEFINITIONS[effective].required_fields),
            error="No valid JSON object found in Research AI output.",
            parse_error_type=parse_error or "invalid_json",
            truncation_detected=truncated,
        )
    if effective == RESEARCH_AI_MINIMAL_ANSWER:
        return _validate_minimal(
            contract_id=contract_id,
            effective_contract_id=effective,
            payload=payload,
            allowed_evidence_ids=allowed_evidence_ids,
        )
    if effective in {RESEARCH_AI_FINDINGS, RESEARCH_AI_COMPARISON}:
        return _validate_findings_or_comparison(
            contract_id=contract_id,
            effective_contract_id=effective,
            payload=payload,
            allowed_evidence_ids=allowed_evidence_ids,
        )
    return _validate_limitations(
        contract_id=contract_id,
        effective_contract_id=effective,
        payload=payload,
        allowed_evidence_ids=allowed_evidence_ids,
    )

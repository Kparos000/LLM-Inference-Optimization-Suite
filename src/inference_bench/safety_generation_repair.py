"""Safety-rule repair helpers for targeted generation-quality hardening."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from inference_bench.generation_contract import (
    allowed_evidence_ids_from_aliases,
    parse_generation_contract,
)


@dataclass(frozen=True)
class SafetyRule:
    """Lexical safety rule with a stable non-sensitive rule ID."""

    rule_id: str
    category: str
    patterns: tuple[str, ...]
    replacement: str


@dataclass(frozen=True)
class SafetyLexicalRepair:
    """Result of applying deterministic answer-section sanitization."""

    changed: bool
    rule_ids: tuple[str, ...]
    repaired_text: str


@dataclass(frozen=True)
class TargetedRetryDecision:
    """B5 retry decision limited to approved repair classes."""

    should_retry: bool
    trigger: str
    missing_labels: tuple[str, ...] = ()


SAFETY_RULES: tuple[SafetyRule, ...] = (
    SafetyRule(
        rule_id="AIR-FRAUD-1",
        category="airline_identity_or_fraud_boundary",
        patterns=(r"\bverification[\s-]+bypass\b",),
        replacement="identity-process circumvention",
    ),
    SafetyRule(
        rule_id="FIN-ADVICE-1",
        category="finance_advice_or_projection_boundary",
        patterns=(
            r"\binvestment\s+(?:advice|recommendation)\b",
            r"\brecommend(?:s|ed|ing)?\s+(?:buying|selling)\b",
            r"\b(?:buy|sell)\s+the\s+stock\b",
            r"\bprice\s+target\b",
            r"\bguaranteed\s+(?:return|outcome)\b",
            r"\b(?:projection|projected|forecast)\b",
        ),
        replacement="unsupported finance recommendation",
    ),
    SafetyRule(
        rule_id="HC-CLINICAL-1",
        category="healthcare_clinical_boundary",
        patterns=(
            r"\bmedical\s+advice\b",
            r"\bdiagnos(?:is|e)\b",
            r"\btreatment\s+advice\b",
            r"\bdosage\b",
        ),
        replacement="clinical guidance",
    ),
)


def _extra_rule_terms(prohibited_terms: tuple[str, ...]) -> tuple[SafetyRule, ...]:
    escaped = tuple(re.escape(term) for term in prohibited_terms if term.strip())
    if not escaped:
        return ()
    return (
        SafetyRule(
            rule_id="GOLD-PROHIBITED-TERM",
            category="gold_contract_prohibited_term",
            patterns=escaped,
            replacement="restricted wording",
        ),
    )


def detect_safety_rule_ids(
    text: str,
    *,
    prohibited_terms: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return stable rule IDs triggered by text without returning unsafe terms."""

    rule_ids: list[str] = []
    for rule in (*SAFETY_RULES, *_extra_rule_terms(prohibited_terms)):
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in rule.patterns):
            rule_ids.append(rule.rule_id)
    return tuple(dict.fromkeys(rule_ids))


def apply_lexical_guard_to_text(
    text: str,
    *,
    prohibited_terms: tuple[str, ...] = (),
) -> SafetyLexicalRepair:
    """Sanitize prohibited wording while preserving surrounding supported claims."""

    repaired = text
    rule_ids: list[str] = []
    for rule in (*SAFETY_RULES, *_extra_rule_terms(prohibited_terms)):
        changed_for_rule = False
        for pattern in rule.patterns:
            repaired_next = re.sub(
                pattern,
                rule.replacement,
                repaired,
                flags=re.IGNORECASE,
            )
            changed_for_rule = changed_for_rule or repaired_next != repaired
            repaired = repaired_next
        if changed_for_rule:
            rule_ids.append(rule.rule_id)
    return SafetyLexicalRepair(
        changed=repaired != text,
        rule_ids=tuple(dict.fromkeys(rule_ids)),
        repaired_text=repaired,
    )


def preserve_json_with_safe_answer(
    generated_text: str,
    *,
    allowed_evidence_ids: tuple[str, ...] = (),
    prohibited_terms: tuple[str, ...] = (),
) -> SafetyLexicalRepair:
    """Sanitize only answer/citation-note text in a valid JSON contract."""

    parsed = parse_generation_contract(
        generated_text,
        allowed_evidence_ids=set(allowed_evidence_ids) if allowed_evidence_ids else None,
    )
    if parsed.contract is None:
        return SafetyLexicalRepair(False, (), generated_text)
    payload = parsed.contract.to_dict()
    answer_repair = apply_lexical_guard_to_text(
        str(payload["answer"]),
        prohibited_terms=prohibited_terms,
    )
    notes_repair = apply_lexical_guard_to_text(
        str(payload["citation_notes"]),
        prohibited_terms=prohibited_terms,
    )
    if not answer_repair.changed and not notes_repair.changed:
        return SafetyLexicalRepair(False, (), generated_text)

    payload["answer"] = answer_repair.repaired_text
    payload["citation_notes"] = notes_repair.repaired_text
    return SafetyLexicalRepair(
        changed=True,
        rule_ids=tuple(dict.fromkeys([*answer_repair.rule_ids, *notes_repair.rule_ids])),
        repaired_text=json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
    )


def decide_targeted_retry(
    *,
    evaluation: dict[str, Any],
    missing_labels: tuple[str, ...] = (),
    attempt_count: int = 0,
    max_attempts: int = 2,
) -> TargetedRetryDecision:
    """Choose only B5-approved retries and enforce the retry cap."""

    if attempt_count >= max_attempts:
        return TargetedRetryDecision(False, "retry_limit_reached")
    if bool(evaluation.get("safety_violation")):
        return TargetedRetryDecision(True, "safety_violation")
    if not bool(evaluation.get("json_validity")):
        return TargetedRetryDecision(True, "invalid_json")
    if not bool(evaluation.get("generation_contract_valid")):
        return TargetedRetryDecision(True, "invalid_contract")
    if missing_labels:
        return TargetedRetryDecision(True, "missing_evidence_label", missing_labels)
    return TargetedRetryDecision(False, "no_approved_repair")


def render_safety_rule_repair_prompt(
    *,
    result_row: dict[str, Any],
    rule_ids: tuple[str, ...],
) -> str:
    """Render an answer-section repair prompt without echoing unsafe text."""

    allowed = allowed_evidence_ids_from_aliases(result_row.get("citation_id_aliases"))
    allowed_labels = ", ".join(allowed) or "none"
    evidence = ", ".join(str(label) for label in result_row.get("evidence_ids") or []) or "none"
    rules = ", ".join(rule_ids) if rule_ids else "unspecified"
    return "\n".join(
        [
            "SAFETY RULE REPAIR TASK:",
            f"Active safety rule IDs: {rules}.",
            "Do not repeat restricted wording, previous unsafe wording, or prohibited phrases.",
            "Revise only answer and citation_notes wording needed for safety compliance.",
            f"Preserve evidence_ids exactly when possible: {evidence}.",
            f"Allowed evidence_id labels remain exactly: {allowed_labels}.",
            (
                "Preserve confidence and insufficient_evidence unless the JSON contract "
                "requires repair."
            ),
            "Do not add facts, do not add citations, and do not answer from memory.",
            "Return exactly one compact five-field JSON object and nothing else.",
        ]
    )

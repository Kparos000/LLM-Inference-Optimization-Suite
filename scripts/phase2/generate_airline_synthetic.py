"""Generate deterministic Airline Phase 2A synthetic seed samples."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_PROMPTS = Path("data/real_world_samples/airline_sample.jsonl")
DEFAULT_OUTPUT_KB = Path("data/kb/airline/kb_sample.jsonl")
DEFAULT_OUTPUT_GOLD = Path("data/eval/gold/airline_gold_sample.jsonl")
DEFAULT_REPORT_PATH = Path("data/generated/airline/airline_synthetic_report.json")

VERTICAL = "airline"
AIRLINE_NAME = "Canada Air"
GENERATOR_NAME = "phase2a_4_deterministic_airline"

AIRLINE_POLICIES: list[dict[str, Any]] = [
    {
        "doc_id": "CA-POL-001",
        "title": "24-Hour Cancellation Policy",
        "document_type": "policy",
        "body": (
            "Canada Air permits cancellation within 24 hours of ticket purchase when "
            "the scheduled departure is at least seven days away. Eligible bookings "
            "may be refunded to the original form of payment after identity and fare "
            "ownership checks."
        ),
        "tags": ["airline", "cancellation", "refund", "24-hour"],
    },
    {
        "doc_id": "CA-POL-002",
        "title": "Refundable Fare Policy",
        "document_type": "policy",
        "body": (
            "Refundable Canada Air fares may be cancelled before departure for a "
            "refund subject to fare rules, used coupon status, and payment "
            "verification. Taxes and carrier-imposed charges follow the purchased "
            "fare conditions."
        ),
        "tags": ["airline", "refundable", "refund"],
    },
    {
        "doc_id": "CA-POL-003",
        "title": "Non-Refundable Fare Credit Policy",
        "document_type": "policy",
        "body": (
            "Non-refundable fares are generally not cash refundable after the "
            "24-hour window. When fare rules allow, Canada Air may issue a travel "
            "credit after applicable change fees and no-show restrictions."
        ),
        "tags": ["airline", "non-refundable", "travel-credit"],
    },
    {
        "doc_id": "CA-POL-004",
        "title": "Same-Day Change Policy",
        "document_type": "procedure",
        "body": (
            "Same-day confirmed changes may be offered on eligible Canada Air "
            "operated flights when seats are available in the same cabin. Fare "
            "differences, route restrictions, and airport cutoff times still apply."
        ),
        "tags": ["airline", "same-day-change", "ticket-change"],
    },
    {
        "doc_id": "CA-POL-005",
        "title": "Involuntary Disruption Policy",
        "document_type": "policy",
        "body": (
            "When Canada Air cancels or substantially delays a flight for controllable "
            "operational reasons, agents should offer rebooking options and review "
            "refund or travel credit eligibility under the disruption workflow."
        ),
        "tags": ["airline", "disruption", "rebook"],
    },
    {
        "doc_id": "CA-POL-006",
        "title": "Missed Connection Policy",
        "document_type": "procedure",
        "body": (
            "For protected itineraries, missed connections caused by a prior delayed "
            "Canada Air segment should be handled through rebooking. Separate-ticket "
            "connections may require manual review and may not be protected."
        ),
        "tags": ["airline", "missed-flight", "connection"],
    },
    {
        "doc_id": "CA-POL-007",
        "title": "Baggage Delay Policy",
        "document_type": "policy",
        "body": (
            "Delayed baggage should be documented with a baggage claim file. Canada "
            "Air may reimburse reasonable interim expenses when receipts and trip "
            "details are supplied within the stated claim window."
        ),
        "tags": ["airline", "baggage", "delay"],
    },
    {
        "doc_id": "CA-POL-008",
        "title": "Baggage Damage Policy",
        "document_type": "policy",
        "body": (
            "Visible baggage damage should be reported promptly after arrival. Agents "
            "should collect bag tag details, photos, and a description before routing "
            "the case to baggage claims."
        ),
        "tags": ["airline", "baggage", "damage"],
    },
    {
        "doc_id": "CA-POL-009",
        "title": "Partner Airline Responsibility Policy",
        "document_type": "policy",
        "body": (
            "When another airline operates or controls the affected segment, Canada "
            "Air agents should identify the validating carrier and operating carrier "
            "before advising whether Canada Air can resolve the case directly."
        ),
        "tags": ["airline", "partner", "manual-review"],
    },
    {
        "doc_id": "CA-POL-010",
        "title": "Codeshare Responsibility Policy",
        "document_type": "policy",
        "body": (
            "Codeshare itineraries require checking the marketing carrier, operating "
            "carrier, and ticketing carrier. Edge cases involving compensation, "
            "schedule control, or irregular operations should be escalated."
        ),
        "tags": ["airline", "codeshare", "partner"],
    },
    {
        "doc_id": "CA-POL-011",
        "title": "Passport and Visa Responsibility Policy",
        "document_type": "compliance_note",
        "body": (
            "Travelers are responsible for required passport, visa, transit, and entry "
            "documents. Canada Air can provide general reminders but cannot guarantee "
            "admissibility or replace official government guidance."
        ),
        "tags": ["airline", "visa", "passport", "documentation"],
    },
    {
        "doc_id": "CA-POL-012",
        "title": "Accessibility Assistance Policy",
        "document_type": "policy",
        "body": (
            "Accessibility assistance may include wheelchair support, boarding "
            "assistance, accessible seating coordination, and communication support. "
            "Requests should be recorded before travel when possible."
        ),
        "tags": ["airline", "accessibility", "assistance"],
    },
    {
        "doc_id": "CA-POL-013",
        "title": "Medical Equipment Assistance Policy",
        "document_type": "procedure",
        "body": (
            "Portable medical equipment requests require review of battery, cabin, "
            "and safety rules. Agents should provide administrative routing and avoid "
            "medical suitability advice."
        ),
        "tags": ["airline", "accessibility", "medical-equipment"],
    },
    {
        "doc_id": "CA-POL-014",
        "title": "Unaccompanied Minor Policy",
        "document_type": "policy",
        "body": (
            "Unaccompanied minor service is limited by age, route, connection type, "
            "and operating carrier. International and partner-operated itineraries "
            "may require manual review before confirmation."
        ),
        "tags": ["airline", "minor", "assistance"],
    },
    {
        "doc_id": "CA-POL-015",
        "title": "Loyalty Points Credit Policy",
        "document_type": "faq",
        "body": (
            "Missing loyalty points may be requested after travel once the flown "
            "segment is eligible and the traveler provides ticket, flight, and loyalty "
            "account details through the secure loyalty workflow."
        ),
        "tags": ["airline", "loyalty", "points"],
    },
    {
        "doc_id": "CA-POL-016",
        "title": "Chargeback and Fraud Review Policy",
        "document_type": "compliance_note",
        "body": (
            "Messages requesting charge reversal, verification bypass, or action on "
            "unrelated bookings should be routed to fraud review or ignored when they "
            "lack legitimate itinerary details."
        ),
        "tags": ["airline", "fraud", "chargeback"],
    },
    {
        "doc_id": "CA-POL-017",
        "title": "Payment Verification Policy",
        "document_type": "procedure",
        "body": (
            "Payment corrections, duplicate charges, or ownership disputes require "
            "manual account review. Agents should not expose payment details or modify "
            "charges without verification."
        ),
        "tags": ["airline", "payment", "verification"],
    },
    {
        "doc_id": "CA-POL-018",
        "title": "Schedule Change Policy",
        "document_type": "policy",
        "body": (
            "Canada Air schedule changes may allow rebooking or refund options when "
            "the revised itinerary meets defined time, connection, or routing impact "
            "thresholds."
        ),
        "tags": ["airline", "schedule-change", "rebook"],
    },
    {
        "doc_id": "CA-POL-019",
        "title": "Weather Disruption Policy",
        "document_type": "policy",
        "body": (
            "Weather disruptions are safety-related events. Agents should offer "
            "available rebooking options and explain that compensation eligibility "
            "may differ from controllable operational disruptions."
        ),
        "tags": ["airline", "weather", "disruption"],
    },
    {
        "doc_id": "CA-POL-020",
        "title": "Compensation Eligibility Policy",
        "document_type": "policy",
        "body": (
            "Compensation eligibility depends on disruption cause, arrival delay, "
            "notice period, jurisdiction, and itinerary control. Ambiguous cases "
            "should be escalated for manual review."
        ),
        "tags": ["airline", "compensation", "manual-review"],
    },
    {
        "doc_id": "CA-POL-021",
        "title": "Refund Processing Timeline Policy",
        "document_type": "faq",
        "body": (
            "Eligible refunds are processed after validation and may take several "
            "business days to appear depending on payment method and financial "
            "institution processing."
        ),
        "tags": ["airline", "refund", "timeline"],
    },
    {
        "doc_id": "CA-POL-022",
        "title": "Customer Identity Verification Policy",
        "document_type": "procedure",
        "body": (
            "Account-specific changes require verification of traveler identity and "
            "booking ownership. Agents should ask for secure-channel verification "
            "rather than collecting sensitive details in free text."
        ),
        "tags": ["airline", "identity", "verification"],
    },
    {
        "doc_id": "CA-POL-023",
        "title": "Airport Check-In Cutoff Policy",
        "document_type": "policy",
        "body": (
            "Airport check-in and bag-drop cutoffs vary by route type. Late arrivals "
            "may be handled under missed flight or standby procedures depending on "
            "fare rules and operational availability."
        ),
        "tags": ["airline", "check-in", "missed-flight"],
    },
    {
        "doc_id": "CA-POL-024",
        "title": "International Travel Documentation Policy",
        "document_type": "compliance_note",
        "body": (
            "International travelers should verify entry, transit, and return "
            "documentation before departure. Canada Air agents may direct travelers "
            "to official sources but cannot approve entry to a country."
        ),
        "tags": ["airline", "international", "documentation"],
    },
    {
        "doc_id": "CA-POL-025",
        "title": "Escalation to Manual Review Policy",
        "document_type": "procedure",
        "body": (
            "Manual review is required for account access, payment disputes, identity "
            "uncertainty, partner irregular operations, and compensation edge cases "
            "that cannot be resolved from policy alone."
        ),
        "tags": ["airline", "escalation", "manual-review"],
    },
]

SUPPORT_POLICY_IDS = {
    "ticket_purchase": ["CA-POL-001", "CA-POL-002"],
    "ticket_change": ["CA-POL-004", "CA-POL-017"],
    "cancellation_refund": ["CA-POL-001", "CA-POL-003", "CA-POL-021"],
    "missed_flight": ["CA-POL-006", "CA-POL-023"],
    "baggage_delay": ["CA-POL-007"],
    "baggage_damage": ["CA-POL-008"],
    "partner_airline": ["CA-POL-009", "CA-POL-025"],
    "codeshare": ["CA-POL-010", "CA-POL-025"],
    "visa_passport": ["CA-POL-011", "CA-POL-024"],
    "accessibility": ["CA-POL-012", "CA-POL-013"],
    "disruption": ["CA-POL-005", "CA-POL-018", "CA-POL-019", "CA-POL-020"],
    "loyalty": ["CA-POL-015"],
    "fraud_or_chargeback": ["CA-POL-016", "CA-POL-022"],
}

AIRLINE_PROMPT_CATEGORIES = (
    ["ticket_purchase"] * 4
    + ["ticket_change"] * 4
    + ["cancellation_refund"] * 4
    + ["missed_flight"] * 4
    + ["baggage_delay"] * 4
    + ["baggage_damage"] * 3
    + ["partner_airline"] * 3
    + ["codeshare"] * 3
    + ["visa_passport"] * 3
    + ["accessibility"] * 3
    + ["disruption"] * 2
    + ["loyalty"] * 2
    + ["fraud_or_chargeback"]
)

ROUTES = ("YYZ-YVR", "YUL-LAX", "YVR-NRT", "YYC-MEX", "YOW-YYZ")
TRAVEL_TYPES = (
    "domestic Canada",
    "regional US/Mexico",
    "international",
)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n")


def build_airline_kb_records() -> list[dict[str, Any]]:
    return [
        {
            **policy,
            "vertical": VERTICAL,
            "source_type": "synthetic_public_inspired",
            "version": "1.0",
            "allowed_to_commit": True,
            "metadata": {
                "fictional_airline": AIRLINE_NAME,
                "generator": GENERATOR_NAME,
            },
        }
        for policy in AIRLINE_POLICIES
    ]


def _status_for_category(category: str, occurrence: int) -> str:
    if category == "fraud_or_chargeback":
        return "spam_or_fraud"
    if (category, occurrence) in {
        ("ticket_change", 4),
        ("partner_airline", 3),
        ("codeshare", 3),
    }:
        return "escalate"
    return "answer"


def _action_for(category: str, status: str) -> str:
    if status == "spam_or_fraud":
        return "ignore_spam_or_fraud"
    if status == "escalate":
        return "escalate_manual_review"
    if category in {"ticket_change", "missed_flight", "disruption"}:
        return "rebook"
    if category == "cancellation_refund":
        return "refund"
    if category in {"baggage_delay", "baggage_damage"}:
        return "baggage_claim"
    if category == "loyalty":
        return "ask_for_more_info"
    if category == "ticket_purchase":
        return "answer_policy"
    if category in {"partner_airline", "codeshare"}:
        return "ask_for_more_info"
    return "answer_policy"


def _issue_for(category: str, occurrence: int, status: str) -> str:
    if status == "spam_or_fraud":
        return (
            "Message asks Canada Air to bypass verification and reverse charges for "
            "several unrelated itineraries without legitimate trip details."
        )
    if status == "escalate":
        return {
            "ticket_change": (
                "Traveler reports a duplicate charge after a manual itinerary change "
                "and asks Canada Air to correct the account-specific payment record."
            ),
            "partner_airline": (
                "Traveler requests compensation for a partner-operated segment after "
                "an irregular operation that Canada Air did not control."
            ),
            "codeshare": (
                "Traveler asks which carrier must provide compensation on a codeshare "
                "trip where the operating carrier changed the schedule."
            ),
        }[category]

    issue_templates = {
        "ticket_purchase": (
            "Traveler asks what fare and identity checks apply before buying a Canada "
            "Air ticket for itinerary option {occurrence}."
        ),
        "ticket_change": (
            "Traveler wants to move a Canada Air flight to a later departure on the "
            "same travel day and asks what policy applies."
        ),
        "cancellation_refund": (
            "Traveler asks whether a recently purchased Canada Air ticket can be "
            "cancelled or refunded under the published fare rules."
        ),
        "missed_flight": (
            "Traveler missed a connection after an inbound Canada Air delay and asks "
            "whether rebooking is available."
        ),
        "baggage_delay": (
            "Traveler arrived without a checked bag and asks what claim steps and "
            "interim expense rules apply."
        ),
        "baggage_damage": (
            "Traveler reports visible damage to checked baggage after arrival and asks "
            "how to open a claim."
        ),
        "partner_airline": (
            "Traveler asks whether Canada Air or a partner airline should handle a "
            "support request for the operated segment."
        ),
        "codeshare": (
            "Traveler booked a Canada Air marketed flight operated by another carrier "
            "and asks who owns schedule-change support."
        ),
        "visa_passport": (
            "Traveler asks whether Canada Air can confirm travel documentation for an "
            "international itinerary."
        ),
        "accessibility": (
            "Traveler asks how to request wheelchair assistance and equipment handling "
            "for an upcoming Canada Air trip."
        ),
        "disruption": (
            "Traveler asks what options are available after a Canada Air disruption "
            "changed the planned arrival time."
        ),
        "loyalty": (
            "Traveler asks how to request missing loyalty points after completing a "
            "Canada Air flight."
        ),
    }
    return issue_templates[category].format(occurrence=occurrence)


def build_airline_prompt_records(seed: int) -> list[dict[str, Any]]:
    occurrences: dict[str, int] = defaultdict(int)
    records: list[dict[str, Any]] = []
    for index, category in enumerate(AIRLINE_PROMPT_CATEGORIES, start=1):
        occurrences[category] += 1
        occurrence = occurrences[category]
        status = _status_for_category(category, occurrence)
        action = _action_for(category, status)
        output_format = "json" if status == "answer" and index % 5 == 0 else "text"
        if status == "escalate":
            output_format = "escalation_response"
        if status == "spam_or_fraud":
            output_format = "spam_fraud_ignore"
        record = {
            "prompt_id": f"airline_seed_{index:04d}",
            "ticket_id": f"CA-TKT-{index:04d}",
            "vertical": VERTICAL,
            "airline": AIRLINE_NAME,
            "support_type": category,
            "issue": _issue_for(category, occurrence, status),
            "expected_status": status,
            "required_policy_ids": SUPPORT_POLICY_IDS[category],
            "route": ROUTES[(index + seed) % len(ROUTES)],
            "travel_type": TRAVEL_TYPES[(index + seed) % len(TRAVEL_TYPES)],
            "partner_airline_involved": category in {"partner_airline", "codeshare"},
            "expected_action": action,
            "expected_output_format": output_format,
            "metadata": {
                "prompt_category": category,
                "generator": GENERATOR_NAME,
                "seed": seed,
            },
        }
        if status == "escalate":
            record["escalation_reason"] = (
                "account_payment_partner_or_irregular_operations_manual_review_required"
            )
            record["required_follow_up_questions"] = [
                "Verify itinerary ownership through a secure Canada Air channel.",
                "Confirm the operating carrier or payment-review context.",
            ]
        records.append(record)

    if len(records) != 40:
        msg = f"Expected 40 airline records, built {len(records)}"
        raise RuntimeError(msg)
    return records


def _task_type_for(record: dict[str, Any]) -> str:
    if record["expected_status"] == "escalate":
        return "escalation_response"
    if record["expected_status"] == "spam_or_fraud":
        return "recommend_action"
    if record["expected_action"] in {"rebook", "refund", "baggage_claim"}:
        return "recommend_action"
    return "policy_lookup"


def build_airline_gold_records(prompt_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gold_records: list[dict[str, Any]] = []
    for record in prompt_records:
        action = str(record["expected_action"])
        policy_ids = list(record["required_policy_ids"])
        gold = {
            "prompt_id": record["prompt_id"],
            "vertical": VERTICAL,
            "task_type": _task_type_for(record),
            "expected_status": record["expected_status"],
            "reference_answer": (
                f"Apply Canada Air policy records {', '.join(policy_ids)} and "
                f"return action label {action}."
            ),
            "expected_action": action,
            "required_doc_ids": policy_ids,
            "must_include": [AIRLINE_NAME, action, *policy_ids],
            "must_not_include": [
                "unsupported compensation promise",
                "verification bypass",
                "guaranteed refund outside policy",
            ],
            "metadata": {
                "ticket_id": record["ticket_id"],
                "support_type": record["support_type"],
                "required_policy_ids": policy_ids,
                "expected_action": action,
                "prompt_category": record["metadata"]["prompt_category"],
            },
        }
        if record["expected_status"] == "escalate":
            gold["expected_escalation"] = True
            gold["escalation_reason"] = record.get("escalation_reason", "")
            gold["must_include"] = ["manual review", action, *policy_ids]
        if record["expected_status"] == "spam_or_fraud":
            gold["must_include"] = ["fraud review", action, "do not bypass verification"]
            gold["must_not_include"] = [
                "process reversal",
                "bypass verification",
                "act on unrelated bookings",
            ]
        gold_records.append(gold)

    if len(gold_records) != len(prompt_records):
        msg = "Airline gold records must match prompt records one-to-one"
        raise RuntimeError(msg)
    return gold_records


def build_report(
    prompt_records: list[dict[str, Any]],
    kb_records: list[dict[str, Any]],
    gold_records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "phase": "2A-4",
        "vertical": VERTICAL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_record_count": len(prompt_records),
        "kb_record_count": len(kb_records),
        "gold_record_count": len(gold_records),
        "counts_by_expected_status": dict(
            Counter(str(record["expected_status"]) for record in prompt_records)
        ),
        "counts_by_prompt_category": dict(
            Counter(str(record["metadata"]["prompt_category"]) for record in prompt_records)
        ),
        "counts_by_expected_action": dict(
            Counter(str(record["expected_action"]) for record in prompt_records)
        ),
        "warnings": [
            "This is a curated synthetic seed dataset, not the full 10,000-prompt dataset.",
            (
                "RAG, retrieval, prompt assembly, and inference are deferred until all "
                "five Phase 2A vertical datasets are prepared."
            ),
        ],
        "next_step": (
            "Review the Airline synthetic seed records, then continue Phase 2A sample "
            "pilots for the remaining verticals."
        ),
    }


def build_samples(args: argparse.Namespace) -> dict[str, Any]:
    kb_records = build_airline_kb_records()
    prompt_records = build_airline_prompt_records(seed=args.seed)
    gold_records = build_airline_gold_records(prompt_records)
    report = build_report(prompt_records, kb_records, gold_records)

    status_counts = Counter(str(record["expected_status"]) for record in prompt_records)
    expected_status_counts = {"answer": 36, "escalate": 3, "spam_or_fraud": 1}
    if dict(status_counts) != expected_status_counts:
        msg = f"Unexpected airline status counts: {dict(status_counts)}"
        raise RuntimeError(msg)
    if len(kb_records) < 25 or len(gold_records) != 40:
        msg = "Airline curated samples failed count validation"
        raise RuntimeError(msg)

    write_jsonl(args.output_prompts, prompt_records)
    write_jsonl(args.output_kb, kb_records)
    write_jsonl(args.output_gold, gold_records)
    write_json(args.report_path, report)

    return {
        "mode": "build_samples",
        "phase": "2A-4",
        "vertical": VERTICAL,
        "output_prompts": str(args.output_prompts),
        "output_kb": str(args.output_kb),
        "output_gold": str(args.output_gold),
        "report_path": str(args.report_path),
        "prompt_record_count": len(prompt_records),
        "kb_record_count": len(kb_records),
        "gold_record_count": len(gold_records),
        "counts_by_expected_status": dict(status_counts),
        "warnings": report["warnings"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-samples", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-prompts", type=Path, default=DEFAULT_OUTPUT_PROMPTS)
    parser.add_argument("--output-kb", type=Path, default=DEFAULT_OUTPUT_KB)
    parser.add_argument("--output-gold", type=Path, default=DEFAULT_OUTPUT_GOLD)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.build_samples:
        print("Pass --build-samples to generate Airline Phase 2A seed data.", file=sys.stderr)
        return 2
    try:
        summary = build_samples(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

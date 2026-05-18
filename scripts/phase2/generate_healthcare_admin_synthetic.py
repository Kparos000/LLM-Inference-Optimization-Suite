"""Generate deterministic Healthcare Administrative Phase 2A synthetic seed samples."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_PROMPTS = Path("data/real_world_samples/healthcare_admin_sample.jsonl")
DEFAULT_OUTPUT_KB = Path("data/kb/healthcare_admin/kb_sample.jsonl")
DEFAULT_OUTPUT_GOLD = Path("data/eval/gold/healthcare_admin_gold_sample.jsonl")
DEFAULT_REPORT_PATH = Path("data/generated/healthcare_admin/healthcare_admin_synthetic_report.json")

VERTICAL = "healthcare_admin"
PROVIDER_NAME = "MapleCare Health"
GENERATOR_NAME = "phase2a_4_deterministic_healthcare_admin"

HEALTHCARE_POLICIES: list[dict[str, Any]] = [
    {
        "doc_id": "MCH-POL-001",
        "title": "Appointment Booking Policy",
        "document_type": "healthcare_admin_policy",
        "body": (
            "MapleCare Health appointment booking requests should capture visit "
            "reason category, preferred clinic, preferred date range, and patient "
            "type. Administrative staff do not provide clinical triage."
        ),
        "tags": ["healthcare-admin", "appointment", "scheduling"],
    },
    {
        "doc_id": "MCH-POL-002",
        "title": "Appointment Rescheduling Policy",
        "document_type": "healthcare_admin_policy",
        "body": (
            "Routine appointment rescheduling may be handled by the scheduling queue "
            "when the requester can verify the appointment context through an approved "
            "administrative channel."
        ),
        "tags": ["healthcare-admin", "reschedule", "scheduling"],
    },
    {
        "doc_id": "MCH-POL-003",
        "title": "Cancellation and No-Show Policy",
        "document_type": "policy",
        "body": (
            "Cancellations should be recorded before the appointment window when "
            "possible. Repeated no-shows may require staff follow-up before another "
            "routine appointment is booked."
        ),
        "tags": ["healthcare-admin", "cancellation", "no-show"],
    },
    {
        "doc_id": "MCH-POL-004",
        "title": "Referral Status Policy",
        "document_type": "procedure",
        "body": (
            "Referral status requests should be routed to the referrals queue. Staff "
            "may provide administrative status only and should not interpret clinical "
            "priority or medical necessity."
        ),
        "tags": ["healthcare-admin", "referral", "routing"],
    },
    {
        "doc_id": "MCH-POL-005",
        "title": "Insurance Verification Policy",
        "document_type": "procedure",
        "body": (
            "Insurance verification confirms plan details, eligibility date, and "
            "coverage workflow status. Complex payer discrepancies require manual "
            "review by the insurance queue."
        ),
        "tags": ["healthcare-admin", "insurance", "verification"],
    },
    {
        "doc_id": "MCH-POL-006",
        "title": "Billing Question Policy",
        "document_type": "policy",
        "body": (
            "Billing staff may explain charges, statement timing, and payment options "
            "from administrative records. Disputed balances or payer adjustments may "
            "require manual billing review."
        ),
        "tags": ["healthcare-admin", "billing", "insurance"],
    },
    {
        "doc_id": "MCH-POL-007",
        "title": "Payment Plan Policy",
        "document_type": "procedure",
        "body": (
            "Payment plan requests are routed to billing. Staff may describe available "
            "administrative options but should not promise approval until the billing "
            "team reviews account eligibility."
        ),
        "tags": ["healthcare-admin", "billing", "payment-plan"],
    },
    {
        "doc_id": "MCH-POL-008",
        "title": "Medical Records Request Policy",
        "document_type": "healthcare_admin_policy",
        "body": (
            "Medical records release requests require identity and authorization "
            "verification. Requests involving another person's records must be routed "
            "to records or privacy review."
        ),
        "tags": ["healthcare-admin", "medical-records", "privacy"],
    },
    {
        "doc_id": "MCH-POL-009",
        "title": "Patient Portal Access Policy",
        "document_type": "procedure",
        "body": (
            "Portal access support covers login recovery, account activation, and "
            "secure-message routing. Staff should not ask for passwords in free text."
        ),
        "tags": ["healthcare-admin", "portal", "access"],
    },
    {
        "doc_id": "MCH-POL-010",
        "title": "Telehealth Setup Guide",
        "document_type": "troubleshooting_guide",
        "body": (
            "Telehealth setup support may cover device readiness, appointment link "
            "access, and check-in timing. Clinical suitability questions must be "
            "routed to clinical staff."
        ),
        "tags": ["healthcare-admin", "telehealth", "portal"],
    },
    {
        "doc_id": "MCH-POL-011",
        "title": "Prior Authorization Status Policy",
        "document_type": "procedure",
        "body": (
            "Prior authorization status requests should be routed to the insurance "
            "queue. Staff may provide administrative status but not guarantee payer "
            "approval."
        ),
        "tags": ["healthcare-admin", "prior-authorization", "insurance"],
    },
    {
        "doc_id": "MCH-POL-012",
        "title": "Prescription Refill Routing Policy",
        "document_type": "healthcare_admin_policy",
        "body": (
            "Prescription refill messages are administrative routing requests only. "
            "Staff should route the message to clinical staff and avoid dosage, "
            "treatment, or medication advice."
        ),
        "tags": ["healthcare-admin", "prescription", "clinical-routing"],
    },
    {
        "doc_id": "MCH-POL-013",
        "title": "Lab Result Availability Routing Policy",
        "document_type": "healthcare_admin_policy",
        "body": (
            "Administrative staff may state whether lab results are available in the "
            "portal or routed for follow-up. They must not interpret results or provide "
            "medical advice."
        ),
        "tags": ["healthcare-admin", "lab-results", "clinical-boundary"],
    },
    {
        "doc_id": "MCH-POL-014",
        "title": "Transportation and Accessibility Request Policy",
        "document_type": "policy",
        "body": (
            "Transportation and accessibility support requests should capture service "
            "need, clinic location, appointment date, and accommodation type for "
            "administrative coordination."
        ),
        "tags": ["healthcare-admin", "accessibility", "transportation"],
    },
    {
        "doc_id": "MCH-POL-015",
        "title": "Interpreter Request Policy",
        "document_type": "procedure",
        "body": (
            "Interpreter requests should capture preferred language, appointment "
            "context, and communication format. Requests should be documented before "
            "the visit when possible."
        ),
        "tags": ["healthcare-admin", "interpreter", "accessibility"],
    },
    {
        "doc_id": "MCH-POL-016",
        "title": "New Patient Registration Policy",
        "document_type": "procedure",
        "body": (
            "New patient registration requires administrative intake, identity "
            "verification, insurance workflow setup when applicable, and consent-form "
            "completion through approved channels."
        ),
        "tags": ["healthcare-admin", "registration", "intake"],
    },
    {
        "doc_id": "MCH-POL-017",
        "title": "Clinic Hours and Location Policy",
        "document_type": "faq",
        "body": (
            "Clinic hours, location details, parking notes, and holiday closures may "
            "be answered from the current administrative schedule. Urgent symptoms "
            "should not be handled as a location question."
        ),
        "tags": ["healthcare-admin", "clinic-hours", "location"],
    },
    {
        "doc_id": "MCH-POL-018",
        "title": "Complaint and Grievance Policy",
        "document_type": "policy",
        "body": (
            "Complaints and grievances should be acknowledged, categorized, and routed "
            "to the appropriate administrative review queue without making unsupported "
            "outcome promises."
        ),
        "tags": ["healthcare-admin", "complaint", "grievance"],
    },
    {
        "doc_id": "MCH-POL-019",
        "title": "Privacy Request Policy",
        "document_type": "compliance_note",
        "body": (
            "Privacy requests, proxy access questions, and concerns about disclosure "
            "must be routed to the privacy office when identity, authorization, or "
            "scope is uncertain."
        ),
        "tags": ["healthcare-admin", "privacy", "compliance"],
    },
    {
        "doc_id": "MCH-POL-020",
        "title": "Identity Verification Policy",
        "document_type": "procedure",
        "body": (
            "Identity verification is required before account-specific scheduling, "
            "records, billing, or portal changes. Staff should direct verification "
            "through approved secure workflows."
        ),
        "tags": ["healthcare-admin", "identity", "verification"],
    },
    {
        "doc_id": "MCH-POL-021",
        "title": "Proxy Access Policy",
        "document_type": "compliance_note",
        "body": (
            "Proxy access requests require authorization review and may require legal "
            "or guardianship documentation. Staff should route uncertain requests to "
            "privacy review."
        ),
        "tags": ["healthcare-admin", "proxy-access", "privacy"],
    },
    {
        "doc_id": "MCH-POL-022",
        "title": "Records Release Timeline Policy",
        "document_type": "faq",
        "body": (
            "Records release timing depends on request completeness, verification, "
            "format, and receiving party. Staff may provide administrative timeline "
            "ranges but not release records without approval."
        ),
        "tags": ["healthcare-admin", "records", "timeline"],
    },
    {
        "doc_id": "MCH-POL-023",
        "title": "Urgent Symptom Boundary Policy",
        "document_type": "compliance_note",
        "body": (
            "Messages describing urgent symptoms must be redirected to emergency or "
            "urgent clinical channels. Administrative staff must not diagnose, provide "
            "treatment instructions, or reassure the patient clinically."
        ),
        "tags": ["healthcare-admin", "urgent", "boundary"],
    },
    {
        "doc_id": "MCH-POL-024",
        "title": "Emergency Redirect Policy",
        "document_type": "procedure",
        "body": (
            "If a message suggests immediate danger, severe symptoms, or urgent safety "
            "concern, the response should direct the person to emergency services or "
            "urgent care channels without giving medical advice."
        ),
        "tags": ["healthcare-admin", "emergency", "boundary"],
    },
    {
        "doc_id": "MCH-POL-025",
        "title": "Spam, Fraud, and Irrelevant Message Handling Policy",
        "document_type": "procedure",
        "body": (
            "Irrelevant, abusive, or suspicious administrative messages should be "
            "ignored or routed to fraud review. Staff should not open attachments or "
            "act on unverifiable account-change demands."
        ),
        "tags": ["healthcare-admin", "spam", "fraud"],
    },
]

HEALTHCARE_PROMPT_CATEGORIES = (
    ["appointment_booking"] * 3
    + ["appointment_reschedule"] * 4
    + ["appointment_cancellation"] * 3
    + ["referral_status"] * 3
    + ["insurance_verification"] * 3
    + ["billing_question"] * 3
    + ["payment_plan_request"] * 2
    + ["medical_records_request"] * 2
    + ["portal_access"] * 2
    + ["telehealth_setup"] * 2
    + ["prior_authorization_status"] * 2
    + ["prescription_refill_routing"] * 2
    + ["lab_result_availability"]
    + ["transportation_or_accessibility_request"]
    + ["language_interpreter_request"]
    + ["new_patient_registration"]
    + ["clinic_location_hours"]
    + ["complaint_or_grievance"]
    + ["privacy_request"]
    + ["urgent_clinical_redirect"]
    + ["spam_or_fraud"]
)

SUPPORT_TYPE_FOR_CATEGORY = {
    "urgent_clinical_redirect": "lab_result_availability",
    "spam_or_fraud": "portal_access",
}

POLICY_IDS_BY_CATEGORY = {
    "appointment_booking": ["MCH-POL-001", "MCH-POL-020"],
    "appointment_reschedule": ["MCH-POL-002", "MCH-POL-020"],
    "appointment_cancellation": ["MCH-POL-003"],
    "referral_status": ["MCH-POL-004"],
    "insurance_verification": ["MCH-POL-005", "MCH-POL-020"],
    "billing_question": ["MCH-POL-006"],
    "payment_plan_request": ["MCH-POL-007"],
    "medical_records_request": ["MCH-POL-008", "MCH-POL-020", "MCH-POL-022"],
    "portal_access": ["MCH-POL-009", "MCH-POL-020"],
    "telehealth_setup": ["MCH-POL-010"],
    "prior_authorization_status": ["MCH-POL-011"],
    "prescription_refill_routing": ["MCH-POL-012"],
    "lab_result_availability": ["MCH-POL-013"],
    "transportation_or_accessibility_request": ["MCH-POL-014"],
    "language_interpreter_request": ["MCH-POL-015"],
    "new_patient_registration": ["MCH-POL-016", "MCH-POL-020"],
    "clinic_location_hours": ["MCH-POL-017"],
    "complaint_or_grievance": ["MCH-POL-018"],
    "privacy_request": ["MCH-POL-019", "MCH-POL-021"],
    "urgent_clinical_redirect": ["MCH-POL-023", "MCH-POL-024"],
    "spam_or_fraud": ["MCH-POL-025"],
}

QUEUE_BY_SUPPORT_TYPE = {
    "appointment_booking": "scheduling",
    "appointment_reschedule": "scheduling",
    "appointment_cancellation": "scheduling",
    "referral_status": "referrals",
    "insurance_verification": "insurance",
    "billing_question": "billing",
    "payment_plan_request": "billing",
    "medical_records_request": "records",
    "portal_access": "portal_support",
    "telehealth_setup": "portal_support",
    "prior_authorization_status": "insurance",
    "prescription_refill_routing": "clinical_staff_review",
    "lab_result_availability": "clinical_staff_review",
    "transportation_or_accessibility_request": "general_admin",
    "language_interpreter_request": "general_admin",
    "new_patient_registration": "general_admin",
    "clinic_location_hours": "general_admin",
    "complaint_or_grievance": "general_admin",
    "privacy_request": "privacy_office",
}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n")


def build_healthcare_kb_records() -> list[dict[str, Any]]:
    return [
        {
            **policy,
            "vertical": VERTICAL,
            "source_type": "synthetic_public_inspired",
            "version": "1.0",
            "allowed_to_commit": True,
            "metadata": {
                "fictional_provider": PROVIDER_NAME,
                "generator": GENERATOR_NAME,
            },
        }
        for policy in HEALTHCARE_POLICIES
    ]


def _status_for_category(category: str, occurrence: int) -> str:
    if category == "urgent_clinical_redirect":
        return "boundary_response"
    if category == "spam_or_fraud":
        return "spam_or_fraud"
    if (category, occurrence) in {
        ("insurance_verification", 3),
        ("medical_records_request", 2),
        ("privacy_request", 1),
    }:
        return "escalate"
    return "answer"


def _support_type_for(category: str) -> str:
    return SUPPORT_TYPE_FOR_CATEGORY.get(category, category)


def _action_for(category: str, status: str) -> str:
    if status == "boundary_response":
        return "urgent_clinical_redirect"
    if status == "spam_or_fraud":
        return "ignore_spam_or_fraud"
    if status == "escalate":
        if category in {"privacy_request", "medical_records_request"}:
            return "escalate_privacy_review"
        return "escalate_manual_review"
    if category in {
        "referral_status",
        "prescription_refill_routing",
        "lab_result_availability",
        "prior_authorization_status",
    }:
        return "route_to_queue"
    if category in {
        "appointment_booking",
        "appointment_reschedule",
        "appointment_cancellation",
        "billing_question",
        "clinic_location_hours",
    }:
        return "answer_policy"
    if category in {"medical_records_request", "portal_access", "new_patient_registration"}:
        return "ask_for_more_info"
    return "route_to_queue"


def _issue_for(category: str, occurrence: int, status: str) -> str:
    if status == "boundary_response":
        return (
            "Patient reports severe chest pain and asks the administrative assistant "
            "to interpret lab results and say whether it is safe to wait."
        )
    if status == "spam_or_fraud":
        return (
            "Message demands immediate portal access changes for unrelated accounts "
            "and includes suspicious instructions to ignore verification."
        )
    if status == "escalate":
        return {
            "insurance_verification": (
                "Requester says payer eligibility does not match clinic records and "
                "asks MapleCare Health to override the insurance status."
            ),
            "medical_records_request": (
                "Requester asks for another person's medical records without clear "
                "authorization and asks staff to send the files directly."
            ),
            "privacy_request": (
                "Requester disputes proxy access and asks MapleCare Health to change "
                "privacy permissions without verified authorization."
            ),
        }[category]

    issue_templates = {
        "appointment_booking": (
            "Patient asks how to book a routine administrative appointment at "
            "MapleCare Health for option {occurrence}."
        ),
        "appointment_reschedule": (
            "Patient asks how to reschedule a routine visit to a later date through "
            "the scheduling workflow."
        ),
        "appointment_cancellation": (
            "Patient asks how to cancel a routine appointment and whether the no-show "
            "policy applies."
        ),
        "referral_status": (
            "Patient asks where a referral status question should be routed and what "
            "administrative information is needed."
        ),
        "insurance_verification": (
            "Patient asks how MapleCare Health verifies insurance before a scheduled "
            "administrative visit."
        ),
        "billing_question": (
            "Patient asks how to get an explanation of a recent statement and payment options."
        ),
        "payment_plan_request": (
            "Patient asks whether a payment plan request can be started through "
            "administrative support."
        ),
        "medical_records_request": (
            "Patient asks how to request copies of their own medical records through "
            "the approved records workflow."
        ),
        "portal_access": (
            "Patient asks how to recover portal access without sharing a password in the message."
        ),
        "telehealth_setup": (
            "Patient asks what administrative steps are needed before a telehealth appointment."
        ),
        "prior_authorization_status": (
            "Patient asks where to route a prior authorization status request."
        ),
        "prescription_refill_routing": (
            "Patient asks where to route a prescription refill message without asking "
            "for medication advice."
        ),
        "lab_result_availability": (
            "Patient asks whether lab results are available in the portal and who can "
            "answer interpretation questions."
        ),
        "transportation_or_accessibility_request": (
            "Patient asks how to request transportation or accessibility support for "
            "a scheduled visit."
        ),
        "language_interpreter_request": (
            "Patient asks how to request an interpreter for a future appointment."
        ),
        "new_patient_registration": (
            "Prospective patient asks what administrative steps are needed for new "
            "patient registration."
        ),
        "clinic_location_hours": (
            "Patient asks where to find current clinic hours and location details."
        ),
        "complaint_or_grievance": (
            "Patient asks how to submit a complaint about an administrative service experience."
        ),
    }
    return issue_templates[category].format(occurrence=occurrence)


def _safety_boundary_for(category: str, status: str) -> str:
    if status == "boundary_response":
        return "urgent_clinical_redirect"
    if status == "escalate" and category in {"privacy_request", "medical_records_request"}:
        return "privacy_sensitive"
    return "administrative_only"


def build_healthcare_prompt_records(seed: int) -> list[dict[str, Any]]:
    occurrences: dict[str, int] = defaultdict(int)
    records: list[dict[str, Any]] = []
    channels = ("portal", "web_form", "clinic_desk", "secure_message")
    patient_types = ("existing_patient", "new_patient", "care_partner")
    for index, category in enumerate(HEALTHCARE_PROMPT_CATEGORIES, start=1):
        occurrences[category] += 1
        occurrence = occurrences[category]
        support_type = _support_type_for(category)
        status = _status_for_category(category, occurrence)
        action = _action_for(category, status)
        safety_boundary = _safety_boundary_for(category, status)
        expected_queue = (
            "clinical_staff_review"
            if status == "boundary_response"
            else QUEUE_BY_SUPPORT_TYPE[support_type]
        )
        privacy_sensitive = safety_boundary == "privacy_sensitive"
        record = {
            "prompt_id": f"healthcare_admin_seed_{index:04d}",
            "ticket_id": f"MCH-TKT-{index:04d}",
            "vertical": VERTICAL,
            "support_type": support_type,
            "issue": _issue_for(category, occurrence, status),
            "expected_status": status,
            "expected_queue": expected_queue,
            "required_policy_ids": POLICY_IDS_BY_CATEGORY[category],
            "safety_boundary": safety_boundary,
            "patient_type": patient_types[(index + seed) % len(patient_types)],
            "channel": channels[(index + seed) % len(channels)],
            "department": expected_queue,
            "expected_action": action,
            "expected_output_format": (
                "boundary_response"
                if status == "boundary_response"
                else "escalation_response"
                if status == "escalate"
                else "spam_fraud_ignore"
                if status == "spam_or_fraud"
                else "json"
                if index % 6 == 0
                else "text"
            ),
            "privacy_sensitive": privacy_sensitive,
            "metadata": {
                "prompt_category": category,
                "generator": GENERATOR_NAME,
                "seed": seed,
            },
        }
        if status == "escalate":
            record["escalation_reason"] = "identity_privacy_payer_or_manual_review_required"
        records.append(record)

    if len(records) != 40:
        msg = f"Expected 40 healthcare admin records, built {len(records)}"
        raise RuntimeError(msg)
    return records


def _task_type_for(record: dict[str, Any]) -> str:
    if record["expected_status"] == "boundary_response":
        return "boundary_response"
    if record["expected_status"] == "escalate":
        return "escalation_response"
    if record["expected_action"] in {"route_to_queue", "ask_for_more_info"}:
        return "recommend_action"
    return "policy_lookup"


def build_healthcare_gold_records(prompt_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gold_records: list[dict[str, Any]] = []
    for record in prompt_records:
        action = str(record["expected_action"])
        policy_ids = list(record["required_policy_ids"])
        must_not_include = [
            "diagnosis",
            "treatment instructions",
            "medication dosage advice",
            "clinical reassurance",
            "medical advice",
        ]
        gold = {
            "prompt_id": record["prompt_id"],
            "vertical": VERTICAL,
            "task_type": _task_type_for(record),
            "expected_status": record["expected_status"],
            "reference_answer": (
                f"Apply MapleCare Health policy records {', '.join(policy_ids)} and "
                f"return action label {action}."
            ),
            "expected_queue": record["expected_queue"],
            "expected_action": action,
            "required_doc_ids": policy_ids,
            "must_include": [PROVIDER_NAME, action, *policy_ids],
            "must_not_include": must_not_include,
            "privacy_sensitive": bool(record["privacy_sensitive"]),
            "metadata": {
                "ticket_id": record["ticket_id"],
                "support_type": record["support_type"],
                "required_policy_ids": policy_ids,
                "expected_queue": record["expected_queue"],
                "expected_action": action,
                "safety_boundary": record["safety_boundary"],
                "privacy_sensitive": bool(record["privacy_sensitive"]),
                "prompt_category": record["metadata"]["prompt_category"],
            },
        }
        if record["expected_status"] == "escalate":
            gold["expected_escalation"] = True
            gold["escalation_reason"] = record.get("escalation_reason", "")
            gold["must_include"] = ["manual review", action, *policy_ids]
        if record["expected_status"] == "boundary_response":
            gold["reference_answer"] = (
                "Redirect to emergency or urgent clinical channels without diagnosis, "
                "treatment instructions, or reassurance."
            )
            gold["must_include"] = [
                "urgent clinical channels",
                "emergency",
                action,
                *policy_ids,
            ]
        if record["expected_status"] == "spam_or_fraud":
            gold["must_include"] = ["ignore", "fraud review", action]
            gold["must_not_include"] = [
                "open attachments",
                "change account access",
                "bypass verification",
            ]
        gold_records.append(gold)

    if len(gold_records) != len(prompt_records):
        msg = "Healthcare admin gold records must match prompt records one-to-one"
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
            "Review the Healthcare Administrative synthetic seed records, then continue "
            "Phase 2A sample pilots for the remaining verticals."
        ),
    }


def build_samples(args: argparse.Namespace) -> dict[str, Any]:
    kb_records = build_healthcare_kb_records()
    prompt_records = build_healthcare_prompt_records(seed=args.seed)
    gold_records = build_healthcare_gold_records(prompt_records)
    report = build_report(prompt_records, kb_records, gold_records)

    status_counts = Counter(str(record["expected_status"]) for record in prompt_records)
    expected_status_counts = {
        "answer": 35,
        "escalate": 3,
        "boundary_response": 1,
        "spam_or_fraud": 1,
    }
    if dict(status_counts) != expected_status_counts:
        msg = f"Unexpected healthcare admin status counts: {dict(status_counts)}"
        raise RuntimeError(msg)
    if len(kb_records) < 25 or len(gold_records) != 40:
        msg = "Healthcare admin curated samples failed count validation"
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
        print(
            "Pass --build-samples to generate Healthcare Administrative Phase 2A seed data.",
            file=sys.stderr,
        )
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

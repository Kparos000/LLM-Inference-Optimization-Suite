# Phase 2A-4 Airline and Healthcare Synthetic Data Pilots

## Purpose

Phase 2A-4 creates deterministic synthetic data assets for Airline Customer
Support and Healthcare Administrative Support. These pilots provide curated seed
data for source/prompt records, KB/policy/context records, and gold/eval records
before any context engineering or inference work.

This phase does not implement RAG, run inference, call models, build retrieval
indexes, create embeddings, or generate the full 10,000-prompt datasets. It is
limited to committed seed fixtures and local generation reports.

## Airline Customer Support

The Airline vertical simulates a fictional airline named Canada Air. The seed
dataset contains:

- 40 prompt/source records
- 25+ KB/policy records
- 40 matching gold/eval records
- 36 answerable records
- 3 escalation records
- 1 spam/fraud record

The workflow is policy-grounded customer support. Records cover ticket purchase,
ticket changes, cancellation/refund questions, missed flights, baggage issues,
partner and codeshare routing, travel documentation, accessibility support,
disruptions, loyalty points, and fraud/chargeback handling.

Expected action labels include:

- `answer_policy`
- `ask_for_more_info`
- `refund`
- `issue_travel_credit`
- `rebook`
- `baggage_claim`
- `escalate_manual_review`
- `ignore_spam_or_fraud`

## Healthcare Administrative Support

The Healthcare Administrative vertical simulates a fictional non-clinical
provider named MapleCare Health. The seed dataset contains:

- 40 prompt/source records
- 25+ KB/policy records
- 40 matching gold/eval records
- 35 answerable records
- 3 escalation records
- 1 urgent/safety-boundary record
- 1 spam/fraud record

This vertical is administrative only. It covers appointment workflows,
referrals, insurance, billing, payment plans, records requests, portal access,
telehealth setup, prior authorization status, refill routing, lab-result
availability routing, transportation/accessibility requests, interpreter
requests, registration, clinic hours, grievances, and privacy requests.

Healthcare records include privacy and safety-boundary labels. They explicitly
avoid diagnosis, treatment instructions, medication dosage advice, clinical
reassurance, or emergency medical advice. Urgent boundary records redirect to
emergency or urgent clinical channels without interpreting symptoms.

## Generated Assets

Airline committed seed assets:

- `data/real_world_samples/airline_sample.jsonl`
- `data/kb/airline/kb_sample.jsonl`
- `data/eval/gold/airline_gold_sample.jsonl`

Healthcare Administrative committed seed assets:

- `data/real_world_samples/healthcare_admin_sample.jsonl`
- `data/kb/healthcare_admin/kb_sample.jsonl`
- `data/eval/gold/healthcare_admin_gold_sample.jsonl`

Local generation reports:

- `data/generated/airline/airline_synthetic_report.json`
- `data/generated/healthcare_admin/healthcare_admin_synthetic_report.json`

## Example Commands

```text
python scripts/phase2/generate_airline_synthetic.py --build-samples
```

```text
python scripts/phase2/generate_healthcare_admin_synthetic.py --build-samples
```

## Next Step

After reviewing these curated samples, proceed to:

- Phase 2A-5 AI Research paper registry/sample pilot
- Phase 2A-6 Retail Amazon Reviews exploration pilot

Full RAG/context engineering remains deferred until all five verticals have
data, KB/context, and gold/eval assets.

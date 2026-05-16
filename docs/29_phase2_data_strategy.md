# Phase 2 Data Strategy

## Purpose

This document freezes the approved Phase 2A data strategy. It clarifies what data
each vertical uses, whether the data is real, synthetic, or hybrid, what knowledge
base is needed, what gold/eval layer is needed, what gets committed, what stays
local/ignored, and what happens before RAG.

## Data Strategy Principles

- Data before RAG.
- KB before retrieval.
- Gold/eval before scaled inference.
- Clean data before prompt generation.
- Provenance and licensing must be recorded.
- Small curated samples may be committed.
- Large raw datasets should stay ignored/local unless explicitly curated.
- No GPU spend until data, KB, and eval plans are ready.

## Approved Vertical Data Strategies

| vertical | data type | source strategy | KB strategy | gold/eval strategy | first pilot action |
| --- | --- | --- | --- | --- | --- |
| Finance / insurance document QA | Real public filings plus optional later QA seed. | SEC EDGAR filings and XBRL company facts for selected public tech companies; DocFinQA/FinQA-style examples only if useful later. | Filing sections, tables, XBRL facts, document registry, and filing metadata. | XBRL numeric gold, filing evidence spans, required document IDs, citation checks, formulas, and 100 insufficient-evidence prompts out of 10,000. | Implement finance SEC acquisition pilot. |
| Airline customer support | Synthetic tickets plus synthetic/public-inspired policy KB. | Canada Air global support tickets from January 2024 to June 30, 2026. | Airline policy corpus for purchases, cancellations, refunds, changes, baggage, partner airlines, documentation, accessibility, disruptions, loyalty, and fraud. | 90% answerable, 8% escalation, 2% spam/fraud/ignore with documented escalation split. | Implement airline ticket and policy schemas. |
| Retail / e-commerce support | Real reviews and metadata plus synthetic policy overlay. | McAuley-Lab/Amazon-Reviews-2023, starting with All_Beauty after timestamp validation. | Product metadata, review-derived facts, common complaints, and synthetic return/warranty/shipping policies. | Metadata facts, review facts, product IDs, category, sentiment/issue labels, support actions, and 90/8/2 stratification. | Implement Amazon Reviews exploration pilot. |
| AI Research Assistant / Education-Research | Real research papers. | arXiv PDFs/HTML for AI inference optimization, reinforcement learning, and LLMs/agentic systems, with optional Semantic Scholar/OpenAlex metadata. | Section-aware paper corpus with citation metadata and chunk IDs. | Required paper IDs, chunk IDs, citations, must-include and must-not-include fields, answerability, and escalation categories. | Create paper registry and ingestion plan. |
| Healthcare Administrative Support | Synthetic administrative tickets plus synthetic/public-inspired policy KB. | MapleCare or NorthBridge Health Administrative Support Desk tickets from January 2024 to June 30, 2026. | Administrative healthcare policy corpus for scheduling, referrals, billing, records, portal access, telehealth, authorization, privacy, accessibility, grievances, and emergency boundaries. | 88% answerable, 8% escalation, 2% urgent clinical/safety boundary, 2% spam/fraud/irrelevant with safety and privacy fields. | Implement healthcare administrative ticket and policy schemas. |

## Finance / Insurance Document QA

The finance strategy uses SEC EDGAR filings and XBRL company facts for the
NorthBridge Equity Research AI Desk scenario. The initial company set is NVDA,
MSFT, AAPL, AMZN, GOOGL, META, TSLA, and AMD.

The document plan is 2 latest 10-Ks per company, 4 latest 10-Qs per company, 2
latest earnings 8-Ks per company, optional investor presentations, and XBRL
company facts. The KB contains SEC filings, filing sections, extracted financial
tables, XBRL facts, a document registry, and filing metadata.

Gold/eval should use XBRL numeric gold for factual numeric answers, filing
evidence spans for text answers, required document IDs, citation checks, and
calculation formulas. The plan includes 100 escalation/insufficient-evidence
prompts out of 10,000.

Output families include short factual answers, long analytical answers,
calculation answers, JSON extraction, markdown tables, comparison memos, risk
summaries, trend summaries, citation-grounded answers, and escalation responses.

## Airline Customer Support

The airline strategy uses synthetic Canada Air global support tickets from
January 2024 to June 30, 2026. Travel types cover domestic Canada, regional
US/Mexico, and international support scenarios.

The KB is a public-inspired airline policy corpus covering ticket purchase,
24-hour cancellation, refunds, ticket changes, missed flights/no-show, same-day
change, standby, baggage allowance, delayed/lost/damaged baggage, travel credits,
partner airlines, codeshare responsibility, visa/passport documentation,
accessibility support, weather disruption, crew/operational cancellation, medical
emergency exception, loyalty points, and chargeback/fraud policy.

Gold/eval stratification is 90% answerable, 8% escalation, and 2%
spam/fraud/ignore. The escalation split is 70% account/payment/identity/manual
booking access required and 30% partner airline/irregular operations/compensation
edge case requiring human review.

Output families include short support responses, structured classification,
grounded RAG answers, policy citations, and escalation/action recommendations.

## Retail / E-commerce Support

The retail strategy uses McAuley-Lab/Amazon-Reviews-2023 and metadata. The first
pilot starts with the All_Beauty category, expands only after exploration, and
uses the last 18 months available after timestamp validation.

The KB combines product metadata and reviews: title, features, description,
details, category, price if available, average rating if available, review
summaries, common complaints, and a synthetic return/warranty/shipping policy
overlay.

Gold/eval uses metadata facts, review facts, product category,
sentiment/issue labels, required product IDs, and support action labels. The
target stratification is 90% answerable, 8% escalation, and 2%
spam/fraud/irrelevant.

Output families include product support answers, structured issue
classification, grounded product answers, product attribute extraction, review
summaries, and action recommendations.

## AI Research Assistant / Education-Research

The AI research strategy uses real papers. The v1 target is 30 papers each for
AI inference optimization, reinforcement learning, and LLMs/agentic systems, for
90 papers total. Sources are arXiv PDFs/HTML where available, optional Semantic
Scholar/OpenAlex metadata enrichment, and manually added relevant PDFs when
needed.

The KB is a section-aware paper corpus. Papers should be parsed into abstract,
introduction, method, system architecture, experiments, results, limitations,
related work, conclusion, and tables/captions where extractable. Chunks should
preserve paper ID, section, title, topic, and citation metadata.

Gold/eval should use citation and evidence requirements: required paper IDs,
required chunk IDs, required citations, must-include content, must-not-include
content, topic, task type, and escalation flags. Answerability categories should
separate answerable records from insufficient-corpus-evidence prompts and prompts
that require expert research judgment. Malformed or out-of-scope prompts should
be tracked separately when included.

Output families include `answer_short`, `answer_grounded`,
`long_context_analysis`, `compare_papers`, `extract_structured`,
`literature_table`, `method_classification`, `limitation_extraction`,
`research_gap_identification`, and `escalation_response`.

## Healthcare Administrative Support

The healthcare administrative strategy uses synthetic administrative support
tickets, not clinical diagnosis data. The scenario is MapleCare Administrative
Services or NorthBridge Health Administrative Support Desk from January 2024 to
June 30, 2026.

Support classes include appointment booking, appointment reschedule, appointment
cancellation, referral status, insurance verification, billing question, payment
plan request, medical records request, portal access, telehealth setup, provider
schedule change, prior authorization status, prescription refill routing, lab
result availability, transportation or accessibility request, language
interpreter request, new patient registration, clinic location/hours, complaint
or grievance, and privacy request.

The KB is a synthetic public-inspired administrative healthcare policy corpus:
appointment scheduling, no-show/cancellation, referral processing, insurance
verification, billing/payment plan, medical records release, portal access,
telehealth setup, prior authorization workflow, prescription refill routing, lab
result notification, patient privacy, interpreter/accessibility,
complaints/grievances, and emergency boundary policies.

The stratification target is 88% answerable administrative support, 8%
escalation, 2% urgent clinical/safety boundary, and 2% spam/fraud/irrelevant.
The safety boundary is explicit: the assistant must not diagnose, interpret lab
results, or provide medical advice. It should redirect urgent symptoms to
appropriate clinical or emergency channels.

Output families include `answer_short`, `answer_grounded`, `extract_structured`,
`classification_routing`, `recommend_action`, `escalation_response`, and
`boundary_response`.

## Shared Source Registry

Proposed file: `data/sources/source_registry.yaml`

Fields:

- `source_id`
- `vertical`
- `source_name`
- `source_type`
- `access_method`
- `license`
- `provenance_url`
- `local_raw_path`
- `local_processed_path`
- `allowed_to_commit`
- `notes`

## Shared KB Registry

Proposed file: `data/kb/kb_registry.jsonl`

Fields:

- `doc_id`
- `vertical`
- `title`
- `document_type`
- `source_type`
- `body_path`
- `effective_date`
- `version`
- `tags`
- `license`
- `allowed_to_commit`

## Shared Gold/Eval Schema

Proposed file: `data/eval/schema/gold_record_schema.json`

Fields:

- `prompt_id`
- `vertical`
- `task_type`
- `reference_answer`
- `expected_category`
- `required_doc_ids`
- `required_chunk_ids`
- `required_citations`
- `must_include`
- `must_not_include`
- `expected_escalation`
- `risk_level`
- `safety_boundary`
- `notes`

## Commit Policy

The shared source registry, KB registry, and commit policy should make data
provenance and public-repo suitability explicit before any prompt files or
benchmark artifacts are promoted.

Commit:

- Source registry.
- Schemas.
- Small curated samples.
- Synthetic KB samples.
- Small gold samples.

Do not commit by default:

- Large raw PDFs.
- Large SEC filings.
- Full Amazon review dumps.
- Large generated 10,000-prompt files.
- Private or license-restricted data.
- Secrets or API keys.

## Phase 2A Completion Criteria

Phase 2A is complete when:

- Each vertical has a validated data plan.
- Each vertical has a KB plan.
- Each vertical has an eval/gold plan.
- Shared schemas exist.
- Small samples exist.
- Repo audit passes.
- Then Phase 2B can begin.

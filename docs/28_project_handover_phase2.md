# Project Handover: Phase 2 Start

## Project Identity

LLM-Inference-Optimization-Suite is an AI inference engineering benchmark suite
for measuring and optimizing LLM serving performance, correctness, groundedness,
memory behavior, and future model/retrieval routing.

The repository should remain employer-facing and professional. Continue to use
committed source, committed curated artifacts, explicit validation checks, and
clear limitations instead of unsupported results.

## Phase 1 Completed

Phase 1 built the benchmark foundation:

- Synthetic workloads.
- Hugging Face baseline.
- vLLM OpenAI-compatible serving.
- Concurrency/load testing.
- 75,000-request Qwen 0.5B vLLM benchmark.
- TTFT, TPOT, latency, throughput, p95/p99, and success/failure metrics.
- Chunked/resumable runs.
- Logs, checkpoints, and metadata.
- Curated artifact preservation.
- Phase 1 inventory, report, plots, and analysis.

## Why Phase 2 Changed

Phase 2 starts with data because realistic enterprise inference benchmarks need
auditable sources, vertical-specific knowledge base design, and evaluation records
before RAG or GPU experiments. The context engineering layer is central: it determines which
evidence reaches the model, how much context is sent, and how that affects TTFT,
TPOT, memory pressure, correctness, and groundedness.

Memory-aware inference matters because RAG adds context tokens, KV cache pressure,
longer prefill, and new throughput/tail-latency tradeoffs. The long-term direction
is an enterprise router that can choose model, retrieval mode, context budget,
serving configuration, and escalation path.

The previous Developer/Code Helpdesk and Enterprise IT choices were revised. The
approved Phase 2 verticals now include AI Research Assistant / Education-Research
and Healthcare Administrative Support.

## Approved Phase 2 Structure

- Phase 2A: Data, KB, Gold/Eval Foundation.
- Phase 2B: Context Engineering, RAG, Evaluation.
- Phase 2C: Memory-Aware GPU Benchmark and Optimization.

## Approved Vertical Strategy

| vertical | data/source strategy | KB strategy | eval/gold strategy | first pilot action |
| --- | --- | --- | --- | --- |
| Finance / insurance document QA | SEC EDGAR filings and XBRL company facts for NVDA, MSFT, AAPL, AMZN, GOOGL, META, TSLA, and AMD; DocFinQA/FinQA-style QA can be a later seed. | SEC filing sections, tables, XBRL facts, document registry, and filing metadata. | XBRL numeric facts, filing evidence spans, document IDs, citation checks, formulas, and 100 insufficient-evidence prompts out of 10,000. | Implement the finance SEC acquisition pilot. |
| Airline customer support | Synthetic Canada Air support tickets from January 2024 to June 30, 2026 across domestic Canada, regional US/Mexico, and international travel. | Public-inspired and synthetic airline policy corpus covering ticketing, refunds, changes, baggage, travel credits, partner airline issues, accessibility, disruptions, loyalty, and fraud. | 90% answerable, 8% escalation, 2% spam/fraud/ignore; escalation split between account/payment/identity/manual access and partner/irregular operations edge cases. | Implement airline ticket and policy schemas. |
| Retail / e-commerce support | McAuley-Lab/Amazon-Reviews-2023 and metadata, starting with All_Beauty after timestamp validation. | Product metadata, descriptions, details, category, price/rating where available, review summaries, common complaints, and synthetic return/warranty/shipping policies. | Metadata facts, review facts, product category, sentiment/issue labels, product IDs, support action labels, and 90/8/2 stratification. | Implement Amazon Reviews exploration pilot. |
| AI Research Assistant / Education-Research | Real AI papers from arXiv PDFs/HTML, with 30 papers each for inference optimization, reinforcement learning, and LLMs/agentic systems. | Section-aware paper corpus covering abstracts, methods, systems, experiments, results, limitations, related work, conclusions, and extractable table/caption text. | Citation/evidence gold with required paper IDs, chunk IDs, citations, must-include terms, must-not-include terms, escalation flags, topic, and task type. | Create paper registry and ingestion plan. |
| Healthcare Administrative Support | Synthetic administrative support tickets from January 2024 to June 30, 2026; non-clinical administrative support only. | Synthetic public-inspired administrative policy corpus for scheduling, referrals, billing, records, portal access, telehealth, prior authorization, refill routing, privacy, accessibility, grievances, and emergency boundaries. | 88% answerable, 8% escalation, 2% urgent clinical/safety boundary, 2% spam/fraud/irrelevant; includes policy IDs, queue, status, escalation, safety boundary, and privacy flags. | Implement healthcare administrative ticket and policy schemas. |

## Important User Decisions

- Finance uses the SEC/DocFinQA direction.
- Finance escalation is reduced to 100 insufficient-evidence prompts out of
  10,000.
- Airline uses synthetic Canada Air support tickets with a public-inspired policy
  KB.
- Retail uses Amazon Reviews 2023 and metadata.
- AI research uses real AI papers from arXiv/PDF/HTML rather than Stack Overflow
  or BigQuery.
- AI research covers inference optimization, reinforcement learning, and
  LLMs/agentic systems.
- Healthcare admin replaces Enterprise IT support.
- Healthcare admin uses synthetic administrative tickets and a public-inspired
  administrative policy KB.
- Healthcare admin is not clinical diagnosis or medical advice.
- No GPU until data, KB, and gold/eval are ready.
- Old messy IT ticket data is not preferred.

## Current Next Step

Create the data-source validation matrix and then implement the finance SEC
acquisition pilot first.

## Files To Create Next

- `docs/30_phase2_data_source_validation_matrix.md`
- `data/sources/source_registry.yaml`
- `data/kb/schema/kb_document_schema.json`
- `data/eval/schema/gold_record_schema.json`
- Finance SEC pilot script or CLI.
- Small curated sample files.

## Local vs RunPod

Current work is local VS Code only. RunPod returns only after Phase 2A and Phase
2B have sufficient data, KB, retrieval, and eval foundation.

## Immediate New-Window Instruction

Do not start GPU work. Do not implement RAG before data/KB/gold schemas. Start by
creating the data-source validation matrix and shared schemas.

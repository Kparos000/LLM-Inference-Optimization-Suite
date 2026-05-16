# Phase 2 Vertical Data Contracts

## Purpose

Phase 2A-2 defines the concrete data contracts for the five approved verticals.
Every vertical must provide three assets:

1. source/prompt records.
2. KB/policy/context records.
3. gold/eval records.

These records are tiny fixtures only. They are not the full benchmark datasets,
they do not represent completed benchmark results, and they should not be used as
evidence of scaled model behavior.

## Shared Data Flow

```text
source documents or records
-> cleaned/normalized vertical records
-> KB/policy/context records
-> prompt records
-> gold/eval records
-> Phase 2B retrieval/evaluation
-> Phase 2C inference optimization
```

## Vertical Contracts Summary

| Vertical | Source/Prompt Record | KB Record | Gold/Eval Record | Main Format | Main Risk | Next Pilot |
| --- | --- | --- | --- | --- | --- | --- |
| Finance Document QA | Finance prompt record with company, ticker, filing, period, and task type. | SEC filing section / XBRL fact table. | Numeric/evidence/citation gold. | SEC HTML/TXT, XBRL JSON, derived JSONL. | Numeric correctness and evidence span alignment. | SEC/XBRL acquisition and exploration. |
| Airline Customer Support | Synthetic Canada Air support ticket. | Synthetic/public-inspired airline policy. | Policy/action/escalation labels. | JSONL. | Unrealistic synthetic records if templates are weak. | Synthetic generator and policy KB expansion. |
| Retail / E-commerce Support | Product/review-derived support record. | Product metadata, review summary, support policy. | Product ID, issue label, action label. | Dataset records and derived JSONL. | Large source dataset and timestamp/category filtering. | Amazon Reviews exploration only. |
| AI Research Assistant / Education-Research | Paper-grounded research question. | Research paper section/chunk. | Required paper/chunk/citation/concept checks. | PDF/HTML/metadata/parsed JSONL. | Citation correctness and long-context parsing quality. | Paper registry and tiny curated sample. |
| Healthcare Administrative Support | Synthetic healthcare admin ticket. | Synthetic public-inspired admin policy. | Queue/policy/privacy/safety labels. | JSONL. | Accidentally drifting into clinical advice. | Synthetic generator and policy KB expansion. |

## Validation Expectations

Tests must confirm:

- Schemas parse as JSON.
- JSONL samples parse line-by-line.
- Every sample has the correct vertical.
- Every KB sample conforms to the shared KB schema at a basic required-field
  level.
- Every gold sample conforms to the shared gold schema at a basic required-field
  level.
- Sample records include stable IDs.

# Phase 2 Data Source Validation Matrix

## Purpose

Phase 2A starts by validating data sources before downloading, transforming,
chunking, retrieving, or running GPU inference. The project will not implement
RAG, chunking experiments, dense retrieval, or GPU inference until each vertical
has:

- Data/source strategy.
- KB/policy strategy.
- Gold/eval strategy.
- License/provenance notes.
- Storage/commit policy.
- First-pilot action.

Phase 2A exists to make the benchmark real before Phase 2B context engineering
and Phase 2C GPU optimization. This matrix is a planning and validation artifact;
it does not claim that any dataset has already been downloaded or benchmarked.

## Validation Criteria

| criterion | validation question |
| --- | --- |
| Source availability | Is the source public, synthetic, or user-provided, and is it reachable or generatable in a controlled later pilot? |
| Access method | Will the pilot use a local generator, official download path, Hugging Face dataset access, user-provided files, or another documented route? |
| License/provenance clarity | Can the project record license, source name, provenance URL or generation method, and commit permissions? |
| Usable fields | Does the source expose fields needed for prompts, KB records, citations, labels, and evaluation? |
| Data cleanliness | Is the expected cleaning burden manageable before prompt generation? |
| KB suitability | Can source or companion policy documents become a vertical knowledge base? |
| Gold/eval suitability | Can deterministic reference fields, labels, required documents, or evidence spans be created? |
| Prompt generation suitability | Can the source produce realistic prompts across multiple output families? |
| Risk/privacy concerns | Does the source avoid non-public personal data, secrets, clinical diagnosis content, and sensitive raw records? |
| Source format | Can Phase 2A handle the source format without premature RAG implementation? |
| Local vs committed storage policy | Are large raw files kept local/ignored while schemas, manifests, and tiny curated samples may be committed? |
| First pilot action | Is there a narrow first action that validates the source without running scaled inference? |

For source format, Phase 2A must handle different formats, including:

- SEC HTML/TXT filings.
- XBRL JSON facts.
- Synthetic JSONL support tickets.
- Product/review metadata records.
- PDF/HTML research papers.
- Parsed text sections.
- Policy KB JSONL documents.
- Gold/eval JSONL records.

## Approved Verticals

The approved Phase 2A verticals are:

1. Finance Document QA.
2. Airline Customer Support.
3. Retail / E-commerce Support.
4. AI Research Assistant / Education-Research.
5. Healthcare Administrative Support.

Developer / Code Helpdesk and Enterprise IT Support are not approved Phase 2A
verticals.

## Validation Matrix

| Vertical | Source Strategy | Data Type | Source Format | Access Method | KB Strategy | Gold/Eval Strategy | Expected Prompt Count | Commit Policy | First Pilot Action | Status |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| Finance Document QA | SEC EDGAR filings and XBRL company facts; optional DocFinQA/FinQA-style seed later. | Real public documents and structured financial facts. | SEC HTML/TXT filings, XBRL JSON, derived filing-section text, derived manifest rows. | SEC APIs/downloads in later pilot. | SEC filing sections, tables, XBRL facts, document registry. | XBRL facts, evidence spans, citations, formulas, 100 escalation prompts out of 10,000. | 10,000 | Commit schemas, manifests, and tiny curated samples only; large filings stay local/ignored unless curated. | Implement finance SEC/XBRL acquisition and exploration pilot. | Approved for first pilot. |
| Airline Customer Support | Synthetic Canada Air global support tickets. | Synthetic support tickets plus public-inspired/synthetic policy KB. | Generated JSONL support records, JSONL policy KB records, JSONL gold/eval records. | Generated locally. | Airline policy corpus covering ticket purchase, refunds, changes, missed flights, baggage, partner airlines, codeshare, visa/passport, accessibility, disruption, loyalty, fraud. | 90% answerable, 8% escalation, 2% spam/fraud/ignore with policy IDs and action labels. | 10,000 | Commit schemas and tiny synthetic samples; generated large synthetic corpora stay local/ignored unless curated. | Create Canada Air ticket/KB/gold schema and tiny sample. | Approved synthetic vertical. |
| Retail / E-commerce Support | McAuley-Lab/Amazon-Reviews-2023 and metadata. | Real public review/metadata dataset with synthetic support-policy overlay. | Dataset records from Hugging Face exploration, derived JSONL metadata/review samples, JSONL policy KB, JSONL gold/eval. | Hugging Face datasets in later exploration pilot. | Product metadata, review summaries, common complaints, synthetic return/warranty/shipping policy. | Metadata facts, review facts, product IDs, category/issue labels, action labels. | 10,000 | Commit schemas and tiny curated samples only; large dataset stays ignored/local. | Run Amazon Reviews exploration pilot for All_Beauty and timestamp/category validation. | Approved for exploration pilot. |
| AI Research Assistant / Education-Research | Real AI research papers from arXiv/PDF/HTML; optional metadata from Semantic Scholar/OpenAlex later. | Real research papers. | PDF, HTML, metadata JSON, parsed text sections, section/chunk JSONL. | User-provided PDFs or later source acquisition. | Paper sections/chunks: abstract, introduction, method, architecture, experiments, results, limitations, related work, conclusion, tables/captions. | Required paper IDs, required chunk IDs, citations, must_include, must_not_include, expert-judgment escalation. | 10,000 | Commit registry and tiny samples; large PDFs may remain local/ignored unless explicitly curated. | Create AI research paper registry schema and tiny sample manifest. | Approved real-document vertical. |
| Healthcare Administrative Support | Synthetic healthcare administrative support tickets. | Synthetic non-clinical administrative tickets plus public-inspired policy KB. | Generated JSONL support tickets, JSONL policy KB, JSONL gold/eval. | Generated locally. | Appointment, referral, insurance, billing, medical records, portal, telehealth, prior authorization, refill routing, privacy, accessibility, grievances, emergency-boundary policies. | 88% answerable, 8% escalation, 2% urgent/safety boundary, 2% spam/fraud/irrelevant; queue/policy/privacy/safety labels. | 10,000 | Commit schema and tiny synthetic sample. | Create healthcare admin ticket/KB/gold schema and tiny sample. | Approved synthetic vertical. |

## Rejected or Deferred Sources

- Old messy IT ticket dataset is not approved for Phase 2A.
- Stack Overflow/BigQuery is deferred because the project currently avoids BigQuery
  complexity and replaced developer/code helpdesk with AI Research Assistant /
  Education-Research.
- Enterprise IT Support is deferred because Healthcare Administrative Support
  gives a clearer safety/privacy/evaluation story for Phase 2A.
- Clinical medical-note datasets are not the target because the healthcare
  vertical is administrative, not diagnosis or treatment.
- Paid GPU data generation is deferred until Phase 2C.

## Phase 2A Completion Gate

Phase 2A is complete only when:

- Each vertical has a source registry entry.
- Each vertical has a KB/policy plan.
- Each vertical has a gold/eval plan.
- Shared schemas exist.
- Vertical-specific schemas exist.
- Tiny prompt/source samples exist.
- Tiny KB samples exist.
- Tiny gold/eval samples exist.
- Storage/commit policy is documented.
- Repo audit passes.

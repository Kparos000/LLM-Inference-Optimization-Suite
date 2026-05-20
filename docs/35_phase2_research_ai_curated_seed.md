# Phase 2A-5B Research AI Curated Seed

Phase 2A-5B creates the reviewed seed data for the Research AI vertical. It turns
the approved ICLR paper registry, enriched metadata, extracted text, and section
quality manifests into a small set of prompt/source, KB/context, and gold/eval
records.

This is a curated seed only. It is not the final 5,000-10,000 prompt dataset, and
it does not run RAG, retrieval, embeddings, prompt assembly, model calls, GPU
runs, or benchmark inference.

## Inputs

- `data/sources/research_ai_approved_papers.jsonl`
- `data/generated/research_ai/enriched_paper_registry.jsonl`
- `data/processed/research_ai/paper_text_manifest.jsonl`
- `data/processed/research_ai/paper_sections_manifest.jsonl`
- `data/generated/research_ai/research_ai_paper_preparation_report.json`
- `data/generated/research_ai/research_ai_section_quality_report.json`

The generated metadata, text manifests, section manifests, and reports remain
local and ignored. They are inputs for this curated seed step, not committed
full-text artifacts.

## Outputs

- `data/real_world_samples/research_ai_sample.jsonl`
- `data/kb/research_ai/kb_sample.jsonl`
- `data/eval/gold/research_ai_gold_sample.jsonl`
- `data/generated/research_ai/research_ai_curation_report.json`

The curation report is local and ignored. The three curated seed JSONL files are
reviewed sample assets and are committed.

## Prompt Distribution

| Category | Count | Task type | Output format | Status |
|---|---:|---|---|---|
| concept_explanation | 6 | answer_grounded | text | answer |
| paper_method | 7 | answer_grounded | text | answer |
| results_evaluation | 6 | answer_grounded | text | answer |
| structured_extraction | 6 | extract_structured | json | answer |
| compare_papers | 5 | compare_papers | markdown_table | answer |
| literature_table | 4 | literature_table | markdown_table | answer |
| evidence_citation_lookup | 3 | answer_grounded | text | answer |
| insufficient_evidence_or_escalation | 2 | escalation_response | text | insufficient_evidence or escalate |
| out_of_scope | 1 | boundary_response | text | out_of_scope |

The seed contains 40 prompts total: 37 answerable records, 2 insufficient
evidence or escalation records, and 1 out-of-scope record.

## KB Strategy

The KB sample contains derived paper context records:

- paper abstracts from clean enriched metadata
- selected paper sections from extracted text
- compact metadata records for source grounding

Section records prefer abstract, introduction, method, approach, experiments,
evaluation, results, limitations, and conclusion evidence. References and
appendices are avoided unless a later phase explicitly needs them. Suspicious
oversized sections are not used as single precise evidence spans.

## Gold/Eval Strategy

Each prompt has exactly one matching gold record. Answerable gold records include
required document IDs, section or KB chunk IDs, provenance-backed citation
strings, required terms, and guardrails against unsupported claims.

Structured extraction records require JSON keys such as `paper_title`, `method`,
`task_or_benchmark`, `result_or_claim`, `limitation`, and `evidence_id`.
Comparison and literature-table records require evidence from multiple papers.
Insufficient-evidence records require the answer to acknowledge missing evidence
or escalate for expert review.

## Boundary Behavior

The out-of-scope seed record is intentionally unrelated to the Research AI paper
corpus. Its gold record prohibits answering from general model memory and
requires a boundary response that says the question is outside the Research AI
corpus.

## Command

```text
python scripts/phase2/curate_research_ai_seed.py --build-curated-samples
```

Inspect the local curation report:

```text
python -m json.tool data/generated/research_ai/research_ai_curation_report.json
```

## Next Step

After reviewing the Research AI curated samples, proceed to Phase 2A-6 Retail
Amazon Reviews exploration and seed creation.

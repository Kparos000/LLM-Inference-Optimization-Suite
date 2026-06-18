# Phase 3 Context And Memory Mode Foundation

This document describes the Phase 3 Block 1 foundation for model aliases,
memory modes, context records, and generated workload records.

This work does not implement retrieval, embeddings, inference, GPU execution,
or a new benchmark harness.

## Model Aliases

The existing canonical model keys remain valid:

- `qwen2_5_0_5b_instruct`
- `qwen2_5_1_5b_instruct`
- `qwen2_5_7b_instruct`
- `qwen2_5_32b_instruct`
- `large_model_placeholder`

Phase 3 adds public aliases:

- `model1_0_5b`
- `model2_1_5b`
- `model3_7b`
- `model4_32b`
- `model5_large_placeholder`

Current production aliases have since been frozen in `configs/models.yaml` as
`model1_0_5b`, `model2_3b`, `model3_7b`, `model4_32b`, `model5_gated`,
`model6_gated`, and `model7_gated`. The Phase 3 aliases above remain historical
context or deprecated compatibility aliases where applicable.

Aliases are safer than directly renaming keys because current configs, tests,
docs, and sample artifacts already reference the old names. The alias resolver
lets future configs use clearer public names while old configs keep working.
The canonical model registry still has five model records; aliases resolve to
those same records instead of duplicating model definitions.

## Memory Modes

Memory modes define how a prompt should be assembled before inference. They make
experiments modular because the runner can execute a workload without needing
to know whether context came from no context, dense retrieval, hybrid retrieval,
compression, or a bounded workflow.

The memory modes are configured in `configs/memory_modes.yaml`:

| mode | purpose | retrieval | max context tokens |
| --- | --- | --- | ---: |
| `mm0_no_context` | prompt only, no retrieved evidence | none | 0 |
| `mm1_dense_top5` | dense semantic retrieval, top 5 | dense | 4096 |
| `mm2_hybrid_top5` | dense plus BM25 hybrid retrieval, top 5 | hybrid | 4096 |
| `mm3_compressed_hybrid_top5` | hybrid retrieval plus deterministic compression | hybrid | 2048 |
| `mm4_bounded_agentic` | bounded retrieval, validation, one repair attempt, and escalation | adaptive | 4096 |

`mm4_bounded_agentic` is contract-only for now. It defines the future workflow
shape without implementing autonomous agent behavior.

## Context Records

A context record is a normalized chunk of evidence or candidate evidence. It is
the unit that future retrieval and context packing will pass into workload
assembly.

Required fields:

- `context_id`
- `vertical`
- `source_id`
- `parent_id`
- `chunk_id`
- `chunk_strategy`
- `source_type`
- `title`
- `text`
- `metadata`
- `token_estimate`
- `provenance`
- `is_gold_linked`

Validation requires a non-empty `context_id`, one of the five supported
verticals, non-empty text, non-negative token estimate, object metadata, and a
boolean gold-link flag.

## Workload Records

A workload record is the generated prompt package that the existing HF, vLLM, or
mock harness can run later. It preserves prompt identity, memory mode, assembled
messages, selected context records, gold evidence IDs, and source prompt
metadata.

Required fields:

- `workload_id`
- `prompt_id`
- `vertical`
- `memory_mode`
- `messages`
- `context_records`
- `context_token_estimate`
- `retrieval_metadata`
- `expected_output_format`
- `gold_evidence_ids`
- `dataset_split`
- `source_prompt_record`

Validation requires a configured memory mode, non-empty messages, valid context
records, non-negative context-token estimate, and one of these dataset splits:

- `smoke_500`
- `controlled_2000`
- `final_10000`
- `test_fixture`

## Harness Reuse

This foundation is intentionally separate from model execution. The existing
benchmark harness already has:

- mock runner
- Hugging Face runner
- OpenAI-compatible runner
- OpenAI load runner
- result CSV schema
- generation JSONL traces
- summary and comparison utilities

Phase 3 workload records should be converted into the existing runner input
shape when inference begins. That avoids rebuilding the HF/vLLM/mock harness and
keeps the new context layer focused on data assembly, validation, and
evaluation readiness.

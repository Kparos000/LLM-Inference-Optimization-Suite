# Repo-Aware Phase 3 To 6 Inference Plan

This report is a repository-aware implementation audit and planning document.
It does not add inference code, refactor the harness, rename model keys, call
external APIs, run GPU experiments, build embeddings, or implement RAG.

The goal is to identify what already exists and what should be reused before
Phase 3-6 implementation starts.

## Executive Finding

The repository is not starting from scratch. It already has a working benchmark
harness foundation, result schemas, local Hugging Face execution, an
OpenAI-compatible client path, an async OpenAI-compatible load runner with
chunking/checkpoint/resume support, Phase 1 vLLM sample artifacts, and a
promoted 10,000-record benchmark dataset with public EDA.

The current public EDA path is:

```text
data/generated/dataset_10000/
```

Older EDA paths such as `data/generated/eda/dataset_10000/` and
`data/generated/phase2a/eda/` are legacy paths and should not be used for public
planning. Research AI full retrieval corpus export artifacts still live under
`data/generated/phase2a/retrieval_corpus/research_ai/` because that export was
created during the internal data-preparation stage; it is a future Phase 3 input,
not a public EDA output.

## 1. Existing Models And Model Config

### Config Files

- `configs/models.yaml`: checked-in model registry.
- `configs/experiments.yaml`: active default experiments, currently mock-only.
- `configs/vllm_baseline_experiments.yaml`: planned OpenAI-compatible vLLM
  baseline experiments.
- `configs/stress_plan.yaml`: planned scale, backend, optimization, model role,
  and concurrency grid.
- `configs/workloads.yaml`: current synthetic harness workloads.
- `configs/scaled_workloads.yaml`: scaled synthetic workload generation plan.

### Current Model Keys

| key | model id | current role |
| --- | --- | --- |
| `qwen2_5_0_5b_instruct` | `Qwen/Qwen2.5-0.5B-Instruct` | local smoke/dev model |
| `qwen2_5_1_5b_instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | small baseline candidate |
| `qwen2_5_7b_instruct` | `Qwen/Qwen2.5-7B-Instruct` | first serious GPU benchmark candidate |
| `qwen2_5_32b_instruct` | `Qwen/Qwen2.5-32B-Instruct` | later scale comparison candidate |
| `large_model_placeholder` | `placeholder/large-model` | future large model placeholder |

### Required Model Fields

`src/inference_bench/config.py` defines `ModelConfig` with these fields:

- `name`: required non-empty string.
- `provider`: required non-empty string.
- `model_id`: required non-empty string.
- `parameter_count`: optional positive integer.
- `default_dtype`: optional string.
- `notes`: optional string.

The current loader does not support an alias field inside a model record. Adding
unexpected keys to a model entry would raise a `TypeError` through
`ModelConfig(**value)`.

### Code And Config That Depends On Current Keys

Direct model-key references exist in:

- `configs/models.yaml`
- `configs/experiments.yaml`
- `configs/vllm_baseline_experiments.yaml`
- `configs/stress_plan.yaml`
- `tests/test_config.py`

`ProjectConfig.__post_init__` validates that every experiment model key exists
in `models`. Directly renaming keys in `configs/models.yaml` without updating
the experiment configs and tests would break `inference-bench validate-config`
and `tests/test_config.py`.

### Safe Model Key Migration

Requested public key migration:

| current key | proposed key |
| --- | --- |
| `qwen2_5_0_5b_instruct` | `model1_0_5b` |
| `qwen2_5_1_5b_instruct` | `model2_1_5b` |
| `qwen2_5_7b_instruct` | `model3_7b` |
| `qwen2_5_32b_instruct` | `model4_32b` |
| `large_model_placeholder` | `model5_large_placeholder` |

Recommended migration:

1. Add alias support before renaming any active key.
2. Keep old keys working for at least one release window.
3. Add new public keys as aliases to the same `model_id` values.
4. Update `configs/stress_plan.yaml` and future Phase 3-6 configs to use the
   new public keys.
5. Update tests to assert alias compatibility instead of only old key presence.
6. Update documentation and command examples.
7. After all checked-in configs and docs use the new keys, mark old keys as
   deprecated aliases.

Yes, aliases should be supported. Without aliases, the rename is a high-risk
breaking change because existing configs and tests use the current keys. Since
the current `ModelConfig` cannot accept arbitrary alias fields, the cleanest
future implementation is either a separate alias resolver or a documented
duplicate-key transition in `configs/models.yaml`. The resolver is preferable
because it avoids duplicating model records and lets old configs keep working.

## 2. Existing Backend And Harness State

### Existing Commands And Runners

The CLI in `src/inference_bench/cli.py` exposes:

- `inference-bench doctor`
- `inference-bench system-info`
- `inference-bench validate-config`
- `inference-bench generate-workloads`
- `inference-bench mock-run`
- `inference-bench hf-run`
- `inference-bench openai-compatible-run`
- `inference-bench openai-load-run`
- `inference-bench report-summary`
- `inference-bench score-structured-jsonl`
- `inference-bench compare-results`
- `inference-bench make-plots`
- `inference-bench make-phase1-plots`
- `inference-bench explain`

Reusable runner modules:

- `src/inference_bench/runners/mock_runner.py`
- `src/inference_bench/runners/hf_runner.py`
- `src/inference_bench/runners/openai_compatible_runner.py`
- `src/inference_bench/runners/openai_load_runner.py`

### Current Backend Support

| backend path | current state | reuse |
| --- | --- | --- |
| mock | implemented and active in `configs/experiments.yaml` | keep for schema and pipeline tests |
| Hugging Face local | implemented through `hf-run` | use for local plumbing and small CPU/GPU validation |
| vLLM | accessed through OpenAI-compatible server/client path | reuse `openai-compatible-run` and `openai-load-run` |
| SGLang | planned only, no runnable backend adapter yet | implement later as comparable OpenAI-compatible or native client |

### What `hf-run` Measures

`hf-run` loads a local Hugging Face causal LM and records:

- `input_tokens` with the model tokenizer.
- `output_tokens`.
- end-to-end latency.
- TTFT when `--use-streaming` is enabled.
- TPOT.
- token throughput.
- success/failure and error message.
- generation JSONL traces when `generation_output_path` is set.

It does not support concurrency, chunked checkpoint/resume, server queue timing,
backend batch metrics, or GPU telemetry.

### What OpenAI-Compatible Runs Measure

`openai-compatible-run` records:

- request-level end-to-end latency.
- TTFT when streaming is enabled.
- TPOT.
- approximate input/output tokens using whitespace token counts.
- token throughput.
- success/failure and generation traces.

It can point at a local vLLM OpenAI-compatible server with:

```text
base_url=http://localhost:8000/v1
api_key=EMPTY
```

`openai-load-run` adds:

- async concurrency.
- `--concurrency`.
- optional run-level metadata JSON.
- optional `--chunk-size`.
- optional checkpoint JSON.
- `--resume`.
- optional progress log.
- aggregate request/sec and output-token/sec in metadata.

It is the right reuse point for Phase 4/5 load tests.

### Current Harness Capability Summary

| capability | current state |
| --- | --- |
| TTFT | partial: available with streaming, client-observed |
| TPOT | yes |
| latency | yes |
| request throughput | partial: aggregate metadata in `openai-load-run` |
| token throughput | yes per request, aggregate output tokens/sec in load metadata |
| tokens | partial: exact in HF, whitespace estimates in OpenAI-compatible clients |
| failures | yes |
| streaming | yes: HF optional, OpenAI-compatible default true |
| concurrency | yes: `openai-load-run` |
| checkpoint/resume | yes: `openai-load-run` chunked mode only |
| output CSV | yes |
| generation JSONL | yes for HF/OpenAI-compatible paths |
| results/raw integration | yes by default paths |
| results/processed integration | yes through `compare-results` |
| backend-native vLLM metrics | no |
| SGLang runner | no |

## 3. Existing Telemetry And Hardware Metrics

`src/inference_bench/system_info.py` captures lightweight static metadata:

- timestamp.
- platform and platform release.
- Python version.
- processor.
- logical CPU count.
- physical CPU count if `psutil` is available.
- total RAM if `psutil` is available.
- torch version.
- CUDA availability.
- CUDA device count.
- CUDA device names.
- transformers version.

The repo does not currently collect live GPU telemetry:

- no live GPU utilization time series.
- no GPU memory used/free/headroom time series.
- no power draw.
- no temperature.
- no clocks or throttling reasons.
- no `nvidia-smi` sampler.
- no `pynvml` integration.
- no DCGM integration.
- no Prometheus metrics ingestion.
- no backend log parser for scheduler/KV/prefix metrics.

Before GPU experiments, add a run-level telemetry collector and a run manifest
that aligns telemetry timestamps with request timestamps.

## 4. Existing Result Schema

`BenchmarkResult.csv_fieldnames()` defines the current result CSV schema:

```text
run_id,timestamp_utc,backend,model_name,optimization,workload_name,prompt_id,
input_tokens,output_tokens,ttft_ms,tpot_ms,end_to_end_latency_ms,
throughput_tokens_per_second,peak_memory_mb,estimated_cost_usd,success,
error_message
```

`GenerationRecord` mirrors the run/request metadata and adds:

- `prompt`
- `generated_text`

Existing result folders:

- `results/raw/`: raw mock and HF local smoke/baseline outputs plus generation
  JSONL traces and `system_info.json`.
- `results/processed/`: processed comparison CSVs such as
  `hf_workload_comparison.csv`.
- `results/figures/`: early smoke/harness plots.
- `results/samples/`: curated sample artifacts, including vLLM concurrency
  results, metadata, checkpoints, processed comparison tables, and Phase 1
  figures.

The top-level `results/figures/cost_by_optimization.png`,
`results/figures/latency_by_optimization.png`, and
`results/figures/throughput_by_optimization.png` should be treated as smoke or
early harness artifacts unless their source CSV is tied to a documented promoted
benchmark result.

### Standard Future Result Schema

Do not replace the current schema. Extend it for Phase 3-5:

- `vertical`
- `dataset_id`
- `context_mode`
- `memory_mode`
- `retrieval_mode`
- `context_tokens`
- `total_input_tokens_exact`
- `tokenizer_name`
- `concurrency`
- `queue_time_ms`
- `prefill_ms`
- `decode_ms`
- `server_ttft_ms`
- `batch_size_observed`
- `active_sequence_count`
- `kv_cache_used_blocks`
- `kv_cache_free_blocks`
- `kv_cache_evictions`
- `prefix_cache_hit`
- `prefix_cache_hit_rate`
- `telemetry_run_id`
- `backend_metrics_id`
- `run_manifest_path`
- `eval_correct`
- `format_valid`
- `grounded`
- `safety_violation`
- `gpu_cost_usd`
- `cost_per_1000_requests`
- `cost_per_1m_tokens`

## 5. Existing Evaluation Support

The repo has `src/inference_bench/quality.py`, which can:

- parse generated JSON objects.
- validate required JSON fields.
- score structured output validity.

The repo does not yet have an evaluator that joins model generations to the
promoted gold/eval records.

The promoted data can be joined by `prompt_id`: every vertical has prompt JSONL
records and gold JSONL records with matching `prompt_id`. Gold records include
`reference_answer`, `must_include`, `must_not_include`, `required_doc_ids`,
`required_chunk_ids`, and `required_citations`.

Before correctness, groundedness, format validity, and safety can be measured,
Phase 3 needs:

- a generated workload schema that carries `prompt_id` and `vertical`.
- a gold loader keyed by `prompt_id`.
- an evidence loader keyed by `doc_id`.
- an evaluator for `must_include` and `must_not_include`.
- an expected-output-format validator.
- citation/evidence matching against `required_doc_ids` and `required_citations`.
- a generation safety/domain-boundary scanner by vertical.
- aggregate evaluation reports by vertical, model, backend, memory mode, and
  concurrency.

## 6. Existing Context And RAG Support

No runnable RAG system exists yet. The repo currently has:

- promoted prompt/gold/KB JSONL files under `data/scaleup_2000_full/`.
- public EDA reports under `data/generated/dataset_10000/`.
- Research AI full section corpus export under
  `data/generated/phase2a/retrieval_corpus/research_ai/`.
- a Research AI mapping/quality report for future retrieval use.
- synthetic harness workloads under `data/prompts/` and `data/prompts/scaled/`.

The repo does not currently have:

- a shared context schema.
- a corpus registry.
- BM25 retrieval.
- dense retrieval.
- hybrid retrieval.
- reranking.
- contextual compression.
- vector indexes.
- a context packer.
- workload JSONL builders for the promoted 10,000-record dataset.
- retrieval recall@k evaluation.

Reuse the data loaders, schema style, result writers, and CLI patterns. Do not
build a separate benchmark harness.

## 7. Memory Modes

| mode | purpose | inputs | workload record output | retrieval required | model calls required before GPU | local-only before GPU | required metrics |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mm0_no_context` | baseline without external evidence | prompt row and gold link | prompt only plus gold/eval metadata | no | no | yes | latency, tokens, format, safety, correctness against gold where possible |
| `mm1_dense_top5` | dense semantic retrieval baseline | prompt, normalized corpus, embedding/index artifact | prompt plus top-5 retrieved chunks and scores | yes | embedding model or precomputed embeddings | yes after local index is built | recall@5, context tokens, latency, quality, retrieval latency |
| `mm2_hybrid_top5` | combine lexical and dense recall | prompt, BM25 index, dense index, fusion settings | prompt plus top-5 hybrid chunks and scores | yes | embeddings if not precomputed | yes | recall@5, rank fusion diagnostics, context tokens, quality |
| `mm3_compressed_hybrid_top5` | reduce context while preserving evidence | hybrid retrieval output plus compression config | prompt plus compressed context chunks | yes | no if deterministic compression first; yes if model compression later | deterministic compression should be local first | compression ratio, retained evidence coverage, quality, latency |
| `mm4_bounded_agentic` | bounded tool/retrieval workflow for harder tasks | prompt, corpus, allowed tools, max-step contract | initial prompt plus replayable step trace | yes | yes during experiment | fixtures and dry-run validation local only | per-step latency/tokens, retrieval calls, final quality, budget adherence |

`mm4_bounded_agentic` should be a separate post-main experiment. It must have a
fixed maximum step count, no autonomous external side effects, deterministic
replay artifacts, and explicit per-step cost/latency accounting.

## 8. Chunking Strategy By Vertical

These strategies are based on actual fields observed in
`data/scaleup_2000_full/`.

### Airline

Available fields:

- KB: `doc_id`, `title`, `body`, `document_type`, `tags`, `metadata.base_doc_id`,
  `allowed_to_commit`.
- Prompts: `support_type`, `route`, `travel_type`,
  `partner_airline_involved`, `required_policy_ids`, `required_evidence_ids`.

Recommended chunking:

- Use `doc_id` as the primary citation ID.
- Treat `policy`, `procedure`, `faq`, `compliance_note`, and
  `airline_scaleup_policy_note` records as policy/procedure chunks.
- Use `metadata.base_doc_id` to keep derived policy notes grouped under the
  source policy family.
- If a `body` grows beyond the token budget, split by paragraph/sentence while
  preserving `doc_id`, `title`, and `tags`.
- Preserve `support_type`, `route`, and partner-airline metadata on prompt-side
  workload records for stratified evaluation.

### Healthcare Admin

Available fields:

- KB: `doc_id`, `title`, `body`, `document_type`, `tags`,
  `metadata.base_doc_id`.
- Prompts: `department`, `expected_queue`, `privacy_sensitive`,
  `safety_boundary`, `support_type`, `required_policy_ids`.

Recommended chunking:

- Use admin-procedure chunks keyed by `doc_id`.
- Keep safety-boundary and privacy chunks explicit when `tags`,
  `document_type`, or prompt metadata indicate identity, privacy, or clinical
  boundary risk.
- Group derived policy notes by `metadata.base_doc_id`.
- Preserve `expected_queue`, `privacy_sensitive`, and `safety_boundary` in the
  workload record so generation evaluation can check routing and clinical
  boundary compliance.

### Retail

Available fields:

- KB document types: `product_metadata`, `review_evidence`,
  `review_summary`, `retail_multicategory_review_evidence`, `support_policy`.
- KB metadata: `category`, `parent_asin`, `asin`, `product_title`, `rating`,
  `average_rating`, `rating_number`, `issue_terms`, `verified_purchase`,
  `helpful_vote`, `synthetic_policy_not_amazon_policy`.
- Prompts: `category`, `product_id`, `product_title`, `source_parent_asins`,
  `source_product_ids`, `issue_type`.

Recommended chunking:

- Use parent-child chunking.
- Parent: product/category node from `parent_asin`, `category`, and
  `product_title`.
- Children: sanitized review snippets, review summaries, product metadata, and
  synthetic support policies.
- Keep `support_policy` chunks separate from review evidence so support answers
  do not confuse product sentiment with policy permission.
- Preserve `issue_terms`, ratings, and verified-purchase metadata for retrieval
  filters and diagnostics.

### Finance

Available fields:

- KB document types: `sec_filing_section`, `sec_filing_event`,
  `xbrl_fact_evidence`, `xbrl_fact_table`, `xbrl_concept_inventory`.
- KB metadata: `ticker`, `company_name`, `form`, `accession_number`,
  `filing_date`, `report_date`, `section_type`, `section_title`, `concept`,
  `concepts`, `units`, `fiscal_years_present`, `fiscal_periods_present`.
- Prompts: `ticker`, `company`, `filing_form`, `required_doc_ids`,
  `required_chunk_ids`, `required_citations`.

Recommended chunking:

- Use SEC/XBRL-aware structured chunks.
- Keep `xbrl_fact_evidence` atomic because the concept/value/unit/time period is
  the evidence.
- Use `xbrl_concept_inventory` as compact coverage metadata chunks, not as
  answer evidence for financial claims.
- Use filing-section chunks for `sec_filing_section`, preserving `ticker`,
  `form`, `accession_number`, and `section_type`.
- Use sentence-window fallback for long narrative filing sections such as MD&A.
- Keep `sec_filing_event` as filing metadata evidence, not as a basis for
  financial conclusions beyond the filing event.

### Research AI

Available fields:

- KB document types: `paper_section_evidence`, `paper_section`,
  `paper_abstract`, `paper_metadata`.
- KB metadata: `paper_id`, `section_record_id`, `section_type`,
  `section_title`, `title`, `topic`, `topics`, `venue`, `year`, `authors`.
- Prompts: `topic`, `source_paper_ids`, `required_paper_ids`,
  `required_chunk_ids`, `required_citations`.
- Full retrieval corpus rows: `corpus_id`, `paper_id`, `section_id`,
  `section_title`, `section_type`, `text`, `word_count`, `char_count`.

Recommended chunking:

- Use paper-section chunking keyed by `paper_id` and `section_id`.
- Keep benchmark KB evidence separate from the broader full retrieval corpus.
- For full corpus retrieval, split long method/results/discussion sections by
  sentence windows while preserving section and paper metadata.
- Use `paper_metadata` for filtering/faceting, not as the only evidence for
  method/result claims.
- Preserve `topic` and `topics` for stratified retrieval evaluation.

## 9. Final Experiment Design

### Backends

- HF baseline: local plumbing, tokenizer correctness, and small validation.
- vLLM: first serious GPU backend through the existing OpenAI-compatible path.
- SGLang: add after vLLM metrics and result schema are stable.
- TensorRT-LLM: future work only for larger models such as 70B or
  NVIDIA-specific maximum-performance experiments.

### Models

- 0.5B: local plumbing and server smoke tests.
- 1.5B: small baseline if useful after plumbing is stable.
- 7B: first serious GPU benchmark.
- 32B: later scale comparison after telemetry, memory, and sharding plans are
  validated.
- larger models: future, hardware-dependent.

### Dataset Sizes

- local tiny sample: 5-25 prompts for schema validation.
- GPU smoke: 100 prompts per vertical, 500 total.
- controlled subset: 2,000 prompts stratified across vertical, task type,
  output format, evidence count, and context length.
- final run: 10,000 prompts.

### Concurrency

- `1`: single-request baseline; isolates model/runtime latency without queue
  pressure.
- `4`: small realistic load; detects early scheduler and KV cache behavior.
- `8`: moderate continuous batching pressure.
- `16`: main throughput/latency tradeoff point for controlled and final runs.
- `32`: stress/future level; use after memory, telemetry, and stop conditions
  are validated.

Small GPU smoke should use concurrency `1` and `4`. Controlled and main runs
should use `1`, `4`, `8`, and `16`. Concurrency `32` should be a stress/future
test, not a first-pass requirement.

### Main Memory Modes

Run `mm0`, `mm1`, `mm2`, and `mm3` in the main experiment once Phase 3 retrieval
is ready. Keep `mm4_bounded_agentic` as a separate post-main experiment.

## 10. Cost Tracking

For open-source self-hosted models:

- API token cost is zero.
- Infrastructure cost is not zero.

Track these run inputs:

- RunPod GPU hourly price.
- wall-clock runtime hours.
- total requests.
- successful requests.
- total input tokens.
- total output tokens.
- total tokens.
- successful answers.
- grounded correct answers.

Formulas:

```text
gpu_cost_usd = runpod_gpu_hourly_price_usd * runtime_hours
gpu_dollars_per_1000_requests = gpu_cost_usd / total_requests * 1000
gpu_dollars_per_1m_tokens = gpu_cost_usd / total_tokens * 1000000
gpu_dollars_per_successful_answer = gpu_cost_usd / successful_answers
gpu_dollars_per_grounded_correct_answer = gpu_cost_usd / grounded_correct_answers
```

If a denominator is zero, report `null` and include the failure reason.

## 11. API Key Requirements

- `HF_TOKEN`: needed for gated Hugging Face models or rate-limited downloads.
- `HUGGINGFACE_HUB_TOKEN`: accepted equivalent in some HF tooling.
- local vLLM OpenAI-compatible server: can use `EMPTY`.
- local SGLang OpenAI-compatible server: can use a local dummy key if the server
  accepts it.
- RunPod API key: only needed if automating pods.
- external/gated model APIs: only needed if later added.

No external model API key is required for the planned local/self-hosted open
model path.

## 12. Phase 3-6 Roadmap

### Phase 3: Context Engineering And RAG Foundation

Reuse:

- `data/scaleup_2000_full/`.
- `data/generated/dataset_10000/` EDA reports.
- Research AI full retrieval corpus export.
- existing loader/schema/result-writing style.
- `quality.py` structured-output utilities as a small component.

Implement:

- promoted dataset workload builder.
- shared context/corpus/evidence schema.
- normalized corpus registry.
- `mm0_no_context` and gold-context assembly.
- BM25 first, then dense/hybrid retrieval.
- context packer with token budgets.
- retrieval recall@k and evidence coverage evaluator.
- generation evaluator for must-include, must-not-include, format validity,
  groundedness, citations, and safety boundaries.
- bounded agentic contract and fixture format, without autonomous execution.

Expected commands:

```text
inference-bench generate-dataset-workloads --dataset-root data/scaleup_2000_full --output-dir data/generated/workloads/dataset_10000
inference-bench evaluate-retrieval --workload data/generated/workloads/dataset_10000/mm2_hybrid_top5.jsonl
```

Outputs:

- generated workload JSONL files.
- corpus registry.
- retrieval result JSONL files.
- evaluation reports.

Success criteria:

- all 10,000 prompts can be converted to workload records.
- every workload record preserves `prompt_id`, vertical, task type, expected
  output format, and gold references.
- retrieval can be scored before generation.
- no inference is required to validate the Phase 3 data path.

Risks/blockers:

- exact tokenizer choice affects context budgets.
- dense retrieval requires embedding dependency and index artifact decisions.
- Research AI full corpus is broader than gold-linked benchmark KB and must not
  be confused with evaluation ground truth.

### Phase 4: Harness/Plumbing Validation And Small GPU Smoke

Reuse:

- `hf-run` for local baseline.
- `openai-compatible-run` for single-request vLLM smoke.
- `openai-load-run` for concurrency, chunking, checkpoint, resume, metadata.
- `system-info`.
- `compare-results`, `report-summary`, and Phase 1 plotting patterns.

Implement:

- run manifest schema.
- exact token counting for OpenAI-compatible runs.
- telemetry sampler and alignment.
- vLLM metrics capture path.
- promoted dataset smoke workload commands.
- stop conditions for OOM, schema failure, telemetry failure, and high error
  rate.

Expected commands:

```text
inference-bench system-info --output-path results/raw/system_info_gpu_smoke.json
inference-bench hf-run --workload-path data/generated/workloads/dataset_10000/mm0_no_context_tiny.jsonl --max-prompts 25 --use-streaming
inference-bench openai-compatible-run --workload-path data/generated/workloads/dataset_10000/mm0_no_context_tiny.jsonl --base-url http://localhost:8000/v1 --api-key EMPTY --stream
inference-bench openai-load-run --workload-path data/generated/workloads/dataset_10000/mm0_no_context_500.jsonl --concurrency 4 --run-metadata-path results/raw/gpu_smoke_metadata.json
```

Outputs:

- raw CSV and generation JSONL.
- run manifest.
- telemetry time series.
- backend metrics snapshots.
- smoke evaluation report.

Success criteria:

- 100 prompts per vertical complete at concurrency 1 and 4.
- TTFT, TPOT, E2E latency, tokens, failures, run metadata, telemetry, and
  generation traces are present.
- no-context and gold-context modes are validated before retrieval modes run.

Risks/blockers:

- vLLM server metrics may require endpoint/config-specific ingestion.
- exact token counts may need tokenizer dependency alignment with served model.
- GPU environment can fail independently of client harness.

### Phase 5: Main GPU Experiment, Diagnosis, Optimization, Rerun

Reuse:

- `openai-load-run` as the load client.
- existing result/generation writers.
- `compare-results` summary pipeline.
- Phase 1 sample plot patterns.
- new Phase 3 workload/eval artifacts.
- new Phase 4 telemetry/manifest layer.

Implement:

- vLLM baseline run matrix for 7B.
- SGLang comparable backend path.
- bottleneck diagnosis labels.
- memory pressure sweeps.
- variable-length request subsets.
- prefix-cache A/B and prefix-aware routing tests.
- KV cache and PagedAttention metrics ingestion.
- estimated MFU calculation.
- optimization pass and controlled rerun.

Expected commands:

```text
inference-bench openai-load-run --workload-path data/generated/workloads/dataset_10000/mm0_no_context_2000.jsonl --concurrency 1
inference-bench openai-load-run --workload-path data/generated/workloads/dataset_10000/mm2_hybrid_top5_2000.jsonl --concurrency 16
inference-bench compare-results --input-csv results/raw/... --output-csv results/processed/main_gpu_comparison.csv
```

Outputs:

- backend comparison raw/processed results.
- telemetry series.
- backend metrics series.
- bottleneck diagnosis report.
- optimization comparison report.
- cost report.

Success criteria:

- identical workloads can run on HF baseline, vLLM, and SGLang where feasible.
- results are grouped by backend, model, memory mode, vertical, and concurrency.
- prefill/decode, queue, KV, prefix-cache, GPU, throughput, latency, quality,
  and cost metrics are available for the main comparison.
- optimization rerun shows measured tradeoffs, not just configuration changes.

Risks/blockers:

- SGLang metrics parity may lag vLLM.
- 32B may require tensor parallelism and larger GPU budget.
- queue/prefill/decode disaggregation depends on backend metrics availability.

### Phase 6: Plots, Dashboard, Deployment, Technical Paper

Reuse:

- current `reporting/summary.py`, `reporting/compare.py`,
  `reporting/phase1_plots.py`, and dataset EDA visual style.
- `results/samples/figures/phase1` as a pattern for report-ready figures.
- existing docs structure.

Implement:

- final inference dashboard.
- paper-ready figures.
- backend comparison charts.
- context mode quality/cost/latency tradeoff charts.
- MFU and hardware utilization plots.
- bottleneck diagnosis visuals.
- deployment/demo packaging.
- technical paper and public summary.

Expected commands:

```text
inference-bench make-phase3-6-plots --input-csv results/processed/main_gpu_comparison.csv --output-dir results/figures/phase3_6
inference-bench build-inference-dashboard --input-dir results/processed --output-dir data/generated/inference_dashboard
```

Outputs:

- final plots.
- interactive dashboard.
- paper/report markdown.
- public demo screenshots.
- LinkedIn/X summary material.

Success criteria:

- public report distinguishes dataset EDA from inference results.
- all figures trace to raw and processed result artifacts.
- cost, quality, latency, throughput, hardware, and bottleneck conclusions are
  reproducible from committed code and documented generated artifacts.

Risks/blockers:

- too many metrics can dilute the story; final dashboard needs a small set of
  primary conclusions.
- generated artifacts must not expose private paths, raw secrets, or stale smoke
  figures as final results.

## Exact Next Implementation Steps

1. Add model alias support or a documented duplicate-key transition before any
   model-key rename.
2. Define Phase 3 workload/context/evidence/eval schemas.
3. Build `mm0_no_context` and gold-context workload generation for the promoted
   dataset.
4. Add exact tokenizer counting for generated workload records.
5. Implement lexical retrieval and retrieval recall@k before dense retrieval.
6. Add generation evaluation against gold records by `prompt_id`.
7. Extend result schema with context mode, retrieval mode, concurrency, exact
   token fields, and run manifest IDs.
8. Add run manifest and telemetry collection.
9. Validate HF and mock paths locally.
10. Validate vLLM OpenAI-compatible smoke with 0.5B.
11. Run 500-prompt GPU smoke only after telemetry and manifests are present.
12. Add SGLang only after vLLM path and schema are stable.

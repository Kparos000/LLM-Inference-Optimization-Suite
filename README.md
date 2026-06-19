# LLM Inference Optimization Suite

A reproducible AI inference engineering project for learning, measuring, and explaining LLM inference optimization techniques.

## Authoritative Technical Reference

For the current end-to-end architecture, implemented versus planned
capabilities, dataset and retrieval details, model/provider registry, metrics,
SLOs, infrastructure status, project history, and glossary, see the
[Definitive Technical Briefing](docs/95_definitive_technical_briefing.md).

## Project Goal

This project benchmarks and explains how modern LLM inference optimizations affect:

- Time to First Token
- Time Per Output Token
- End-to-end latency
- Throughput
- Memory usage
- Cost per token
- Output quality

## Engineering Principle

Measure -> Understand -> Optimize -> Scale

Paid GPU will not be used until the local harness, CI/CD, metrics, workload loader, and dry-run experiment plan are correct.

## Current Status

- The promoted benchmark contains 10,000 prompts, 10,000 gold/eval rows, and 4,740 KB records across Airline, Healthcare Admin, Retail, Finance, and Research AI.
- Public dataset EDA is available under `data/generated/dataset_10000/`; Finance-specific assets are mirrored under `data/generated/finance/`.
- Vertical context builders, normalized corpora, canonical retrieval keys, local Qdrant collections, BM25, hybrid reranking, and deterministic compression are implemented.
- All five verticals pass the promoted retrieval SLOs in `data/generated/context_engineering/retrieval_source_of_truth_manifest.json`.
- Memory modes mm0 through mm3 provide single-pass workloads; mm4 is an executable bounded LangGraph inference mode with one optional repair.
- Mock, local Hugging Face, OpenAI-compatible, concurrent load, Hugging Face provider, and OpenRouter execution paths are implemented.
- The production model registry is frozen with active aliases `model1_0_5b`, `model2_3b`, `model3_7b`, `model4_32b`, `model5_gated`, `model6_gated`, and `model7_gated`. Historical `model2_1_5b` and placeholder aliases remain deprecated but resolvable.
- The production runtime registry separates Runtime -> Infrastructure -> Tooling -> Evaluation. Hugging Face Transformers, vLLM, SGLang, and API provider routes are selectable when compatible; TensorRT-LLM is registered only as a planned, unsmoked engine.
- Production workload profiles, ISL/OSL distributions, cache-readiness metrics, optional profiling hooks, post-SLO negative optimization rules, and deployment readiness guardrails are implemented as pre-run controls.
- Repository hygiene and CI validation are hardened: local temp/cache folders are ignored, public-content auditing runs in CI, and `validate-config` now covers model/runtime registries, SLO targets/profiles, load profiles, optimization negative rules, and the unified result-track schema.
- The grounded generation contract, short evidence labels, deterministic evaluator, streaming metrics, API cost accounting, run manifests, checkpointing, and resume controls are implemented.
- Historical curated Phase 1 samples document RunPod L40S vLLM calibration and concurrency behavior. They are not a hardware-equal comparison with local CPU results.
- Current local and API smoke tests have produced real model output. Model6 currently leads the API smoke on quality and cost; Model5 remains a provider/model-size comparison.
- The remote RTX 3070 is now a validated development GPU backend. Matched 50-prompt vLLM and SGLang Qwen 0.5B smokes completed with full request success and live GPU telemetry.
- The A1 serving path passed, but quality did not: JSON validity was 98%, contract validity 72%, evidence match 30%, and deterministic groundedness 28%.
- The matched SGLang smoke reached 100% JSON validity, 58% contract validity, 36% evidence match, and 24% groundedness. It remains a secondary engine; vLLM remains the default RTX 3070 backend.
- The matched mm4 smoke reached 94% contract validity, 44% evidence match, and 42% groundedness with a 6% repair/escalation rate. It remains opt-in because mean E2E latency and normalized token use exceeded mm2/mm3.
- Phase B1 loaded Qwen2.5-1.5B on the RTX 3070 and completed 100/100 requests without OOM. It reached 92% contract validity but only 35% evidence match and groundedness, 93% JSON validity, and two safety violations, so the result is `QUALITY_BLOCKED`.
- Phase B2 adds modular SLO profiles, 51 structured bottlenecks, 57 structured optimizations, failed-SLO-only diagnosis, deterministic compatibility filtering, and one-factor next-experiment recommendations. No LLM is used as a decision source.
- Phase B3 audited all 65 failed B1 rows without new inference. At least one required gold ID was absent from the frozen E1-E5 context in 52 failures; Finance accounted for 18 of those 52. Evidence was available but not cited in 18 failures.
- Phase B4 executed the context-alignment repair on the exact 100 B1 prompt IDs. All required gold evidence now maps to E1-E5, including Finance 20/20. The rerun improved evidence match and groundedness from 35% to 76%, but safety violations remained 2.
- Phase B5 repaired safety wording and multi-evidence citation selection on the frozen B4 matrix. The targeted 25 failed-row replay reached 100% JSON and contract validity, 92% evidence match and groundedness, and zero safety violations. The triggered full frozen 100 rerun reached 99% JSON and contract validity, 96% evidence match and groundedness, and zero safety violations.
- Phase B6 ran the controlled 500-prompt concurrency-one quality gate. It completed 500/500 requests with 91.2% evidence match, 90.8% groundedness, and zero safety violations, but JSON validity, contract validity, truncation, and Research AI vertical quality failed the B6 gate.
- Phase B6R1 replayed the 26 failed/truncated/invalid Research AI rows with two targeted repair strategies. Neither strategy passed: the better 224-token strategy reached 92.31% JSON validity, 84.62% contract validity, 73.08% evidence match, 65.38% groundedness, 7.69% truncation, and zero safety violations. The decision is `B6R1_BLOCKED`; the full 500-row rerun was not triggered and full-run readiness remains `NOT_READY`.
- Phase B6R2 added a versioned vertical generation-contract registry and tested five Research AI-specific contracts at 224 and 320 tokens on the same 26-row replay set. No candidate passed; the best `research_ai_limitations_v1` result reached 96.15% JSON/contract validity and 80.77% evidence/groundedness with zero truncation and zero safety violations. The decision is `B6R2_BLOCKED`; the full 500-row rerun was not triggered.
- Phase B6R3 replayed the same frozen 26 Research AI failed rows through `model6_gated` / Llama 3.1 8B on the existing Hugging Face provider route. The targeted gate passed with 100% JSON and contract validity, 96.15% evidence match and groundedness, zero safety violations, and zero truncation. This indicates Qwen2.5-1.5B model capacity is the likely Research AI blocker, but it does not replace the failed full B6 500-row gate.
- Result tracks are explicitly separated: API provider runs (`model5`/`model6`/`model7` through OpenRouter, Novita, or HF provider routes) use API token cost and no provider GPU telemetry; self-hosted GPU runs (`model2`/`model3`/`model4` through Hugging Face local, vLLM, SGLang, or RunPod) use GPU telemetry/hourly infrastructure cost when configured and no API token price.
- The next step is `B6R4_STRONGER_MODEL_PATH_AND_500_GATE_DECISION`. Choose the stronger-model path before rerunning the frozen 500-row gate. Do not run a 1,000-prompt terminal run, concurrency sweep, SGLang comparison, mm4 comparison, RunPod execution, or 2,000/10,000-prompt benchmark from the current state.
- The authoritative current-state explanation is [docs/95_definitive_technical_briefing.md](docs/95_definitive_technical_briefing.md).

## Documentation

### Project setup and reproducibility

- [Project scope](docs/00_project_scope.md)
- [Reproducibility](docs/01_reproducibility.md)
- [Dry-run plan](docs/02_dry_run_plan.md)
- [Decision log](docs/03_decision_log.md)
- [Result promotion policy](docs/06_result_promotion_policy.md)

### Benchmark methodology

- [Benchmark methodology](docs/08_benchmark_methodology.md)
- [Scaled workload generation](docs/14_scaled_workload_generation.md)

### vLLM/GPU execution

- [Hugging Face smoke test](docs/05_hf_smoke_test.md)
- [vLLM baseline preparation plan](docs/07_vllm_baseline_plan.md)
- [vLLM execution environment decision](docs/10_vllm_environment_decision.md)
- [vLLM smoke-test procedure](docs/11_vllm_smoke_test.md)
- [Resumable benchmarking plan](docs/15_resumable_benchmarking_plan.md)

### Experiment results

- [Experiment log](docs/12_experiment_log.md)
- [HF vs vLLM calibration comparison](docs/13_hf_vs_vllm_calibration_comparison.md)
- [Hugging Face baseline findings](docs/16_hf_baseline_findings.md)

### Phase 1 reporting

- [Phase 1 experiment inventory](docs/19_phase1_experiment_inventory.md)
- [Phase 1 project report](docs/20_phase1_project_report.md)
- [Phase 1 plot interpretation](docs/21_phase1_plot_interpretation.md)

### Phase 2 planning

- [Publication notes](docs/04_publication_notes.md)
- [Scaled benchmark plan](docs/09_scaled_benchmark_plan.md)
- [Project handover source pack](docs/24_project_handover_source_pack.md)
- [Phase 2 master plan](docs/27_phase2_master_plan.md)
- [Project handover: Phase 2 start](docs/28_project_handover_phase2.md)
- [Phase 2 data strategy](docs/29_phase2_data_strategy.md)
- [Phase 2 data source validation matrix](docs/30_phase2_data_source_validation_matrix.md)
- [Phase 2 vertical data contracts](docs/31_phase2_vertical_data_contracts.md)
- [Phase 2 finance SEC/XBRL pilot](docs/32_phase2_finance_sec_xbrl_pilot.md)
- [Phase 2A-4 airline and healthcare synthetic pilots](docs/33_phase2_airline_healthcare_synthetic_pilots.md)
- [Phase 2A-5A AI research paper discovery](docs/34_phase2_research_ai_paper_discovery.md)
- [Phase 2A-5B Research AI curated seed](docs/35_phase2_research_ai_curated_seed.md)
- [Phase 2A-6A Retail Amazon Reviews exploration](docs/36_phase2_retail_amazon_reviews_exploration.md)
- [Phase 2A-6C Retail curated seed](docs/37_phase2_retail_curated_seed.md)
- [Phase 2A-7 cross-vertical data QA](docs/38_phase2a_cross_vertical_data_qa.md)
- [Phase 2A progressive scale-up plan](docs/39_phase2a_progressive_scaleup_plan.md)
- [Phase 2A 250-scale generator foundation](docs/40_phase2a_250_scaleup_generator.md)
- [Phase 2A-9A Airline 250 candidate review](docs/41_phase2a_airline_250_candidate_review.md)
- [Phase 2A-10 250-scale cross-vertical QA](docs/42_phase2a_250_cross_vertical_qa.md)
- [Phase 2A-11 250-scale dataset promotion](docs/43_phase2a_250_scaleup_promotion.md)
- [Phase 2A-12A 1,000-scale readiness plan](docs/44_phase2a_1000_scaleup_plan.md)
- [Phase 2A-13A/13B/13C/13G 1,000-scale generator](docs/45_phase2a_1000_scaleup_generator.md)
- [Phase 2A-13D/13E partial 1,000-scale QA and promotion](docs/46_phase2a_1000_partial_qa_promotion.md)
- [Phase 2A-13G/13H full 1,000-scale QA and promotion](docs/47_phase2a_1000_full_qa_promotion.md)
- [Phase 2A-14 2,000-scale generator](docs/49_phase2a_2000_scaleup_generator.md)
- [Phase 2A-15 2,000-scale QA and promotion](docs/50_phase2a_2000_full_qa_promotion.md)
- [Phase 2A-16A large-scale scaffolding](docs/51_phase2a_large_scale_scaffolding.md)
- [Phase 2A-16B Research AI retrieval corpus](docs/52_phase2a_research_ai_retrieval_corpus.md)
- [10,000-record dataset EDA](docs/53_phase2a_10000_dataset_eda.md)
- [Inference readiness inventory](docs/54_inference_readiness_inventory.md)
- [Repo-aware Phase 3 to 6 inference plan](docs/55_repo_aware_phase3_to_6_plan.md)
- [Phase 3 context and memory mode foundation](docs/56_phase3_context_memory_mode_foundation.md)
- [Phase 3 corpus registry and chunking](docs/57_phase3_corpus_registry_and_chunking.md)
- [Phase 3 retrieval and memory-mode workloads](docs/58_phase3_retrieval_and_memory_mode_workloads.md)
- [Phase 3 completion and Phase 4 handoff](docs/59_phase3_completion_and_phase4_handoff.md)
- [Phase 3 retrieval hardening and run safety](docs/61_phase3_retrieval_hardening_and_run_safety.md)
- [Qdrant vector retrieval and ablation](docs/62_qdrant_vector_retrieval_and_ablation.md)
- [API-priced gated models and cost tracking](docs/63_api_priced_gated_models_and_cost_tracking.md)
- [Phase 4 runner adapter and smoke readiness](docs/64_phase4_runner_adapter_and_smoke_readiness.md)
- [Phase 4 handoff and retrieval promotion](docs/79_phase4_handoff_and_retrieval_promotion.md)
- [Phase 4 vLLM validation and telemetry](docs/81_phase4_vllm_validation_and_telemetry.md)
- [Phase 4 grounded generation contract](docs/82_phase4_generation_contract.md)
- [Phase 4 generation contract hardening](docs/83_phase4_generation_contract_hardening.md)
- [Phase 4 pre-GPU readiness](docs/84_phase4_pre_gpu_readiness.md)
- [Phase 4 stronger-model contract validation](docs/85_phase4_stronger_model_contract_validation.md)
- [Phase 4 API-priced gated-model smoke](docs/86_phase4_api_priced_model_smoke.md)
- [Phase 4 API-versus-local GPU readiness](docs/87_phase4_api_vs_local_comparison.md)
- [Phase 4 streaming API, pricing, and grounding diagnostics](docs/88_phase4_streaming_api_pricing_and_grounding.md)
- [Model5 pricing and provider routing](docs/89_model5_pricing_and_provider_routing.md)
- [Model5 streaming API smoke](docs/90_model5_streaming_api_smoke.md)
- [Multi-evidence grounding repair](docs/91_multi_evidence_grounding_repair.md)
- [Model registry and Ministral model5 switch](docs/92_model_registry_and_model5_switch.md)
- [Model5 OpenRouter streaming smoke](docs/93_model5_openrouter_streaming_smoke.md)
- [Controlled inference readiness audit](docs/94_controlled_inference_readiness_audit.md)
- [Definitive technical briefing](docs/95_definitive_technical_briefing.md)
- [Remote RTX 3070 vLLM smoke](docs/96_remote_rtx3070_vllm_smoke.md)
- [Remote RTX 3070 SGLang smoke](docs/96_remote_rtx3070_sglang_smoke.md)
- [LangGraph mm4 bounded agent](docs/97_langgraph_mm4_bounded_agent.md)
- [mm4 agentic smoke](docs/98_mm4_agentic_smoke.md)
- [Qwen2.5-1.5B vLLM quality smoke](docs/summaries/blockB1_vllm_1_5b_quality_smoke_summary.md)
- [Modular SLO diagnosis and optimization catalog](docs/99_modular_slo_diagnosis_and_optimization_catalog.md)
- [Block B2 summary](docs/summaries/blockB2_slo_diagnosis_optimization_catalog_summary.md)
- [Generation quality root-cause audit](docs/100_generation_quality_root_cause_audit.md)
- [Block B3 summary](docs/summaries/blockB3_generation_quality_root_cause_summary.md)
- [Context alignment and generation quality repair](docs/101_context_alignment_and_generation_quality_repair.md)
- [Block B4 summary](docs/summaries/blockB4_context_alignment_quality_repair_summary.md)
- [Final generation quality hardening](docs/102_final_generation_quality_hardening.md)
- [Block B5 summary](docs/summaries/blockB5_final_generation_quality_hardening_summary.md)
- [B6 500-prompt quality scale gate](docs/103_b6_500_prompt_quality_scale_gate.md)
- [Full-run AI engineering readiness](docs/104_full_run_ai_engineering_readiness.md)
- [Block B6 summary](docs/summaries/blockB6_500_prompt_quality_and_readiness_summary.md)
- [B6R1 Research AI truncation and contract repair](docs/105_b6r1_research_ai_truncation_contract_repair.md)
- [Block B6R1 summary](docs/summaries/blockB6R1_research_ai_truncation_contract_repair_summary.md)
- [B6R2 Research AI vertical generation contract](docs/106_research_ai_vertical_generation_contract.md)
- [Block B6R2 summary](docs/summaries/blockB6R2_research_ai_vertical_contract_summary.md)
- [B6R3 Research AI model capacity validation](docs/107_b6r3_research_ai_model_capacity_validation.md)
- [Block B6R3 summary](docs/summaries/blockB6R3_research_ai_model_capacity_summary.md)
- [Production runtime registry](docs/108_production_runtime_registry.md)
- [Production workload profiles](docs/109_production_workload_profiles.md)
- [Cache-readiness metrics](docs/110_cache_readiness_metrics.md)
- [Profiling hooks](docs/111_profiling_hooks.md)
- [Post-SLO optimization principle](docs/112_post_slo_optimization_principle.md)
- [Deployment readiness guardrails](docs/113_deployment_readiness_guardrails.md)
- [Repository cleanup and CI hardening](docs/114_repository_cleanup_ci_hardening.md)
- [Current project state](PROJECT_STATE.md)
- [Data directory policy](data/README.md)

## Environment Variables

Copy `.env.example` to `.env` for local secrets. Never commit `.env`. `HF_TOKEN`
and `HUGGINGFACE_HUB_TOKEN` may be used for Hugging Face model access.
`OPENROUTER_API_KEY` is required only for an explicitly authorized OpenRouter
smoke. Real Hugging Face execution requires installing the `hf` extra.

## Quality Checks

Local and CI validation use the same ordered gates:

```text
pytest tests/test_config_validation.py
pytest tests/test_repo_hygiene.py
pytest tests/test_ci_config_audit.py
mypy src tests
pytest
ruff check .
ruff format --check .
python scripts/audit_repo_public_content.py
inference-bench doctor
inference-bench validate-config
```

## Initial Development Model

The default development model is:

```text
Qwen/Qwen2.5-0.5B-Instruct


import type {
  Chapter,
  CoreOptimizationState,
  DeployabilityRepair,
  ExperimentStage,
  MetricCard,
  OptimizationState,
  OptimizationStory,
  PrefixLayoutStaticExperiment,
  RepairGate
} from "./types";

export const chapters: Chapter[] = [
  {
    id: "about",
    path: "/",
    title: "AI Inference Engineering Platform",
    shortTitle: "About",
    resultType: "measured",
    purpose: "Introduce the project as a guided replay of a full inference experiment.",
    sourceArtifacts: ["docs/main_inference_V1.md", "docs/95_definitive_technical_briefing.md"]
  },
  {
    id: "slo-metrics",
    path: "/slo-metrics",
    title: "SLO & Metrics",
    shortTitle: "SLOs",
    resultType: "planned",
    purpose: "Explain the pre-run production SLOs and metric families.",
    sourceArtifacts: ["configs/slo_targets.yaml", "configs/slo_profiles.yaml"]
  },
  {
    id: "data",
    path: "/data",
    title: "Data & Workflow Explorer",
    shortTitle: "Data",
    resultType: "measured",
    purpose: "Explore the 10,000-prompt benchmark and workflow shape.",
    sourceArtifacts: ["data/generated/dataset_10000/dataset_10000_eda_summary.csv"]
  },
  {
    id: "preparation",
    path: "/preparation",
    title: "Inference Experiment Preparation",
    shortTitle: "Preparation",
    resultType: "measured",
    purpose: "Inspect retrieval, memory modes, model registry, engines, and SLO setup.",
    sourceArtifacts: ["data/generated/context_engineering/retrieval_source_of_truth_manifest.json"]
  },
  {
    id: "main-inference",
    path: "/main-inference",
    title: "Main Inference Simulation",
    shortTitle: "Main Run",
    resultType: "measured",
    purpose: "Replay the measured A100 Main_Inference_V1 run without running inference.",
    sourceArtifacts: ["experiments/main/main_inference_v1/processed/main_inference_v1_eval_report.json"]
  },
  {
    id: "optimization",
    path: "/optimization",
    title: "Inference Optimization Lab",
    shortTitle: "Optimization",
    resultType: "planned",
    purpose: "Convert failed SLOs into valid mandatory repairs and educational core strategies.",
    sourceArtifacts: [
      "experiments/main/main_inference_v1/processed/main_inference_v1_ui_optimization_options.json"
    ]
  },
  {
    id: "optimized-inference",
    path: "/optimized-inference",
    title: "Optimized Inference Simulation",
    shortTitle: "Optimized",
    resultType: "planned",
    purpose: "Show the selected recipe and future optimized-run artifact contract.",
    sourceArtifacts: ["experiments/optimized/optimized_inference_v1/"]
  },
  {
    id: "comparison",
    path: "/comparison",
    title: "Before/After Comparison",
    shortTitle: "Compare",
    resultType: "planned",
    purpose: "Define the measured comparison target without fabricating optimized deltas.",
    sourceArtifacts: ["experiments/main/main_inference_v1/processed/main_inference_v1_slo_scorecard.csv"]
  },
  {
    id: "conclusions",
    path: "/conclusions",
    title: "Conclusions & Recommendations",
    shortTitle: "Conclusions",
    resultType: "planned",
    purpose: "Summarize measured lessons and the next measured optimized-run phase.",
    sourceArtifacts: ["docs/125_optimization_intelligence_ui_layer.md"]
  }
];

export const headlineMetrics: MetricCard[] = [
  {
    label: "Completed requests",
    value: "250,000",
    tone: "pass",
    detail: "Official Main_Inference_V1 completion count"
  },
  {
    label: "Request failures",
    value: "0",
    tone: "pass",
    detail: "Operational execution completed cleanly"
  },
  {
    label: "Quality verdict",
    value: "FAIL",
    tone: "fail",
    detail: "Contract, evidence, groundedness, and safety blocked deployability"
  },
  {
    label: "A100 runtime",
    value: "11.82 h",
    tone: "neutral",
    detail: "Measured wall time from saved manifest"
  }
];

export const verticalRows = [
  { vertical: "airline", prompts: 2000, gold: 2000, kb: 300, coverage: 1, pressure: 185.198 },
  {
    vertical: "healthcare_admin",
    prompts: 2000,
    gold: 2000,
    kb: 300,
    coverage: 1,
    pressure: 153.021
  },
  { vertical: "retail", prompts: 2000, gold: 2000, kb: 1000, coverage: 1, pressure: 218.097 },
  { vertical: "finance", prompts: 2000, gold: 2000, kb: 1540, coverage: 1, pressure: 272.933 },
  {
    vertical: "research_ai",
    prompts: 2000,
    gold: 2000,
    kb: 1600,
    coverage: 0.98,
    pressure: 615.77
  }
];

export const sloRows = [
  { metric: "JSON validity", target: ">= 0.95", observed: 0.99822, status: "PASS" },
  { metric: "Contract validity", target: ">= 0.95", observed: 0.805388, status: "FAIL" },
  { metric: "Format validity", target: ">= 0.95", observed: 0.805388, status: "FAIL" },
  { metric: "Evidence match", target: ">= 0.95", observed: 0.589724, status: "FAIL" },
  { metric: "Groundedness", target: ">= 0.98", observed: 0.567204, status: "FAIL" },
  { metric: "Safety findings", target: "= 0", observed: 2757, status: "FAIL" },
  { metric: "Runtime", target: "configured", observed: 42538.856, status: "PASS" },
  { metric: "Cost", target: "configured", observed: 18.4613, status: "PASS" }
];

export const fallbackOptimizationStates: OptimizationState[] = [
  {
    optimization_id: "prompt_contract_repair",
    display_name: "Prompt Contract Repair",
    category: "mandatory_repair",
    state: "applicable_measured",
    definition: "Tighten generation contract instructions and repair prompts.",
    mechanism: "code_change",
    affected_metrics: ["contract", "evidence_match", "safety"],
    possible_regressions: ["quality"],
    implementation_status: "implemented",
    current_project_support: "existing Main_Inference UI apply plan",
    requires_gpu_or_api_rerun: true,
    reason: "Contract validity and safety failed in the measured baseline."
  },
  {
    optimization_id: "improve_evidence_formatting",
    display_name: "Improve Evidence Formatting",
    category: "mandatory_repair",
    state: "applicable_measured",
    definition: "Reformat evidence blocks and citation instructions.",
    mechanism: "code_change",
    affected_metrics: ["evidence_match", "groundedness"],
    possible_regressions: ["latency"],
    implementation_status: "planned",
    current_project_support: "catalog-backed plan only",
    requires_gpu_or_api_rerun: true,
    reason: "Evidence match and groundedness failed while retrieval SLOs passed."
  },
  {
    optimization_id: "use_mm4_agentic_repair",
    display_name: "Use MM4 Bounded Agentic Repair",
    category: "mandatory_repair",
    state: "applicable_measured",
    definition: "Route eligible rows through bounded repair.",
    mechanism: "agent_mode_change",
    affected_metrics: ["quality", "safety"],
    possible_regressions: ["latency", "cost"],
    implementation_status: "implemented",
    current_project_support: "bounded LangGraph mm4 available",
    requires_gpu_or_api_rerun: true,
    reason: "Quality and safety failed in measured Main_Inference_V1."
  },
  {
    optimization_id: "use_quantized_model",
    display_name: "Quantization",
    category: "core_inference",
    state: "blocked_by_negative_rule",
    definition: "Use lower precision weights to reduce memory or improve speed.",
    mechanism: "model_precision_change",
    affected_metrics: ["latency", "memory", "cost"],
    possible_regressions: ["quality"],
    implementation_status: "planned",
    current_project_support: "cataloged but not valid for the current failed SLO profile",
    requires_gpu_or_api_rerun: true,
    reason: "Quality already failed, so quantization is educational but disabled.",
    negative_rule: "quantization"
  },
  {
    optimization_id: "enable_prefix_cache",
    display_name: "Prefix Caching",
    category: "core_inference",
    state: "blocked_by_negative_rule",
    definition: "Reuse KV state for repeated prompt prefixes.",
    mechanism: "serving_runtime_toggle",
    affected_metrics: ["TTFT", "throughput"],
    possible_regressions: ["memory"],
    implementation_status: "planned",
    current_project_support: "needs cache-hit telemetry",
    requires_gpu_or_api_rerun: true,
    reason: "Prefix reuse/cache-hit telemetry was not diagnosed as failed evidence.",
    negative_rule: "prefix_caching"
  }
];

export const fallbackDeployabilityRepairs: DeployabilityRepair[] = [
  {
    repair_id: "prompt_contract_repair",
    display_name: "Prompt Contract Repair",
    track: "deployability_repairs",
    state: "required_for_failed_deployability_slo",
    selectable_now: true,
    definition: "Tighten generation contract instructions and repair prompts.",
    why_it_is_a_repair: "It fixes failed quality and safety behavior before serving tuning.",
    why_it_applies: "Contract validity failed in the measured Main_Inference_V1 scorecard.",
    affected_failed_slos: [
      {
        slo_id: "main_inference_v1.contract_validity",
        metric_id: "contract_validity",
        metric_label: "Contract validity",
        target: ">= 0.95",
        observed: 0.805388,
        bottleneck_id: "low_contract_validity"
      }
    ],
    exact_changes: ["Tighten generation contract instructions and repair prompts."],
    implementation_status: "implemented",
    requires_gpu_or_api_rerun: true
  },
  {
    repair_id: "improve_evidence_formatting",
    display_name: "Improve Evidence Formatting",
    track: "deployability_repairs",
    state: "required_for_failed_deployability_slo",
    selectable_now: true,
    definition: "Reformat evidence blocks and citation instructions.",
    why_it_is_a_repair: "It helps the model use already-retrieved evidence correctly.",
    why_it_applies: "Evidence match and groundedness failed in Main_Inference_V1.",
    affected_failed_slos: [
      {
        slo_id: "main_inference_v1.evidence_match",
        metric_id: "evidence_match",
        metric_label: "Evidence match",
        target: ">= 0.95",
        observed: 0.589724,
        bottleneck_id: "low_evidence_match"
      }
    ],
    exact_changes: ["Reformat model-facing evidence blocks and citation instructions."],
    implementation_status: "planned",
    requires_gpu_or_api_rerun: true
  }
];

export const fallbackRepairGate: RepairGate = {
  gate_status: "NOT_MEASURED",
  core_optimization_eligible: false,
  blocking_reason:
    "Main_Inference_V1 failed quality and safety, and measured repaired artifacts are not present yet.",
  minimum_not_optimal_principle:
    "A PASS means the configured minimum target was met. It does not mean serving is optimal.",
  checks: [
    {
      check_id: "quality_slo",
      label: "Quality SLO",
      status: "FAIL",
      observed: "FAIL",
      target: "PASS",
      source_artifact: "main_inference_v1_slo_report.json"
    },
    {
      check_id: "safety_slo",
      label: "Safety SLO",
      status: "FAIL",
      observed: "FAIL",
      target: "PASS",
      source_artifact: "main_inference_v1_slo_report.json"
    },
    {
      check_id: "optimized_repair_validation_artifacts",
      label: "Measured repair validation artifacts",
      status: "NOT_MEASURED",
      observed: "missing",
      target: "optimized repair scorecard",
      source_artifact: "experiments/optimized/optimized_inference_v1"
    }
  ]
};

export const fallbackCoreOptimizationStates: CoreOptimizationState[] = [
  {
    optimization_id: "enable_prefix_cache",
    display_name: "Enable Prefix Cache",
    category: "serving_engine",
    track: "core_inference_optimizations",
    state: "blocked_by_negative_rule",
    selectable_now: false,
    definition: "Reuse KV state for repeated prompt prefixes.",
    mechanism: "config_toggle",
    affected_metrics: ["latency", "throughput", "memory"],
    possible_regressions: ["latency", "quality", "compatibility"],
    implementation_status: "config_only",
    current_project_support: "varies_by_engine",
    requires_gpu_or_api_rerun: true,
    reason: "No prefix-reuse or cache-hit telemetry was measured for this baseline.",
    compatible_config_count: 20,
    compatible_engines: ["sglang", "vllm"],
    compatible_memory_modes: ["mm0_no_context", "mm1_dense_top5", "mm2_hybrid_top5"],
    compatible_hardware: ["a100_sxm_80gb"],
    compatible_models: ["model3_7b"],
    negative_rule_triggered: "prefix_caching"
  },
  {
    optimization_id: "concurrency_sweep",
    display_name: "Concurrency Sweep",
    category: "concurrency_capacity",
    track: "core_inference_optimizations",
    state: "blocked_by_negative_rule",
    selectable_now: false,
    definition: "Measure bounded concurrency to improve GPU occupancy.",
    mechanism: "config_toggle",
    affected_metrics: ["throughput", "gpu_utilization"],
    possible_regressions: ["ttft", "tail_latency", "memory"],
    implementation_status: "implemented",
    current_project_support: "concurrent_runner_available",
    requires_gpu_or_api_rerun: true,
    reason: "Quality has not passed, so concurrency increase is locked.",
    compatible_config_count: 25,
    compatible_engines: ["api_provider", "sglang", "vllm"],
    compatible_memory_modes: ["mm0_no_context", "mm1_dense_top5", "mm2_hybrid_top5"],
    compatible_hardware: ["a100_sxm_80gb", "provider_managed"],
    compatible_models: ["model3_7b", "model6_gated"],
    negative_rule_triggered: "concurrency_increase"
  }
];

export const fallbackExperimentStage: ExperimentStage = {
  current_stage: "DEPLOYABILITY_REPAIR_PLANNED",
  stage_sequence: [
    {
      stage: "MAIN_INFERENCE_MEASURED",
      state: "complete",
      description: "Official Main_Inference_V1 artifacts are present."
    },
    {
      stage: "DEPLOYABILITY_REPAIR_REQUIRED",
      state: "complete",
      description: "Quality and safety failed, so repair comes first."
    },
    {
      stage: "DEPLOYABILITY_REPAIR_PLANNED",
      state: "current",
      description: "A deterministic plan-only repair track exists."
    },
    {
      stage: "CORE_OPTIMIZATION_ELIGIBLE",
      state: "blocked",
      description: "Core optimization waits for measured repair validation."
    }
  ],
  gates: {
    failed_slo_count: 5,
    repair_required: true,
    repair_plan_available: true,
    repair_gate_status: "NOT_MEASURED",
    core_optimization_eligible: false,
    optimized_inference_ready: false
  }
};

export const fallbackOptimizationStory: OptimizationStory = {
  title: "Two-Track Inference Optimization Story",
  summary:
    "Main_Inference_V1 proves the system can run at scale, but deployability repairs must be validated before core inference optimization starts.",
  principles: [
    "Passing an SLO means the minimum target was met, not that serving is optimal.",
    "Failed quality and safety SLOs create repair plans.",
    "Core optimizations remain educational until repair validation passes."
  ],
  interaction_flow: [
    {
      step: "Plan deployability repair",
      user_action: "Select repair-track changes.",
      system_response: "Show exact changes and constants held fixed."
    },
    {
      step: "Validate repair gate",
      user_action: "Inspect measured repaired artifacts when available.",
      system_response: "Unlock or block core optimization."
    }
  ]
};

export const fallbackPrefixLayoutStaticExperiment: PrefixLayoutStaticExperiment = {
  summary: {
    scenario_id: "coreopt_prefix_layout_static_v1",
    parent_run_id: "main_inference_v1",
    optimization_id: "prompt_prefix_layout_optimization",
    result_type: "measured_static_analysis",
    status: "completed_static_analysis",
    workload_rows_scanned: 40000,
    inference_executed: false,
    cache_hits_measured: false,
    latency_claimed: false,
    layout_summaries: {
      baseline_prompt_layout_v1: {
        prompt_count: 40000,
        mean_input_tokens: 822.333875,
        median_input_tokens: 904,
        p95_input_tokens: 1208,
        p99_input_tokens: 1519,
        mean_longest_exact_common_prefix_tokens: 29,
        mean_reusable_token_ratio: 0.041903,
        prefix_family_count: 4
      },
      prefix_optimized_prompt_layout_v1: {
        prompt_count: 40000,
        mean_input_tokens: 822.333875,
        median_input_tokens: 904,
        p95_input_tokens: 1208,
        p99_input_tokens: 1519,
        mean_longest_exact_common_prefix_tokens: 358,
        mean_reusable_token_ratio: 0.517288,
        prefix_family_count: 4
      }
    },
    deltas: {
      candidate_minus_baseline_mean_common_prefix_tokens: 329,
      candidate_minus_baseline_mean_reusable_token_ratio: 0.475385,
      candidate_minus_baseline_total_input_tokens: 0
    }
  },
  layouts: {
    baseline: {
      layout_id: "baseline_prompt_layout_v1",
      section_order: [
        "system",
        "memory_mode",
        "retrieved_evidence",
        "user_question",
        "output_contract"
      ],
      raw_prompt_text_included: false
    },
    candidate: {
      layout_id: "prefix_optimized_prompt_layout_v1",
      section_order: [
        "system",
        "memory_mode",
        "output_contract",
        "retrieved_evidence",
        "user_question"
      ],
      raw_prompt_text_included: false
    }
  },
  metrics: {
    prefix_families: [],
    per_vertical_memory: [],
    section_analysis: []
  },
  equivalence: {
    status: "PASS",
    rows_checked: 40000,
    section_content_byte_equivalent: true,
    evidence_order_fixed: true,
    instruction_priority_risk: true,
    requires_inference_validation: true
  },
  decision: {
    scenario_id: "coreopt_prefix_layout_static_v1",
    decision: "MISSING_CONFIGURATION",
    reason:
      "Static analysis completed and the candidate increases reusable leading prefix potential, but no explicit acceptance threshold is configured.",
    requires_gpu_rerun: true,
    requires_engine_validation: true,
    next_required_experiment: "coreopt_prefix_layout_engine_validation_v1",
    disallowed_claims: [
      "TTFT improvement",
      "latency improvement",
      "cache-hit improvement",
      "cost improvement",
      "deployability improvement"
    ]
  },
  story: {
    title: "Static Prompt Prefix Layout Optimization",
    story_steps: [
      {
        id: "problem",
        title: "Problem",
        body:
          "The authoritative runner prompt placed a long stable output contract after request-specific context and question content."
      },
      {
        id: "mechanism",
        title: "Mechanism",
        body:
          "The candidate moves stable reusable instructions before dynamic evidence and question sections so future prefix caching can reuse a longer exact leading token sequence."
      },
      {
        id: "decision",
        title: "Decision",
        body:
          "The result is plan-only until an engineer configures an acceptance threshold and runs engine validation."
      }
    ],
    headline_metrics: {
      baseline_mean_reusable_token_ratio: 0.041903,
      candidate_mean_reusable_token_ratio: 0.517288,
      delta_reusable_token_ratio: 0.475385,
      equivalence_status: "PASS",
      requires_engine_validation: true
    },
    apply_behavior:
      "No inference is executed; clicking apply can only reveal the engine-validation plan."
  },
  source_artifacts: [
    "experiments/optimizations/coreopt_prefix_layout_static_v1/coreopt_prefix_layout_static_v1_prefix_summary.json",
    "experiments/optimizations/coreopt_prefix_layout_static_v1/coreopt_prefix_layout_static_v1_equivalence_report.json",
    "experiments/optimizations/coreopt_prefix_layout_static_v1/coreopt_prefix_layout_static_v1_decision.json"
  ]
};

export const replayFallback = [
  { completed_requests: 0, failure_count: 0, compressed_second: 0, engine: "vLLM" },
  { completed_requests: 50000, failure_count: 0, compressed_second: 22, engine: "vLLM" },
  { completed_requests: 100000, failure_count: 0, compressed_second: 44, engine: "SGLang" },
  { completed_requests: 150000, failure_count: 0, compressed_second: 66, engine: "SGLang" },
  { completed_requests: 200000, failure_count: 0, compressed_second: 88, engine: "API route" },
  { completed_requests: 250000, failure_count: 0, compressed_second: 110, engine: "API route" }
];

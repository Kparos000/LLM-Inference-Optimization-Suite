import type { Chapter, MetricCard, OptimizationState } from "./types";

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

export const replayFallback = [
  { completed_requests: 0, failure_count: 0, compressed_second: 0, engine: "vLLM" },
  { completed_requests: 50000, failure_count: 0, compressed_second: 22, engine: "vLLM" },
  { completed_requests: 100000, failure_count: 0, compressed_second: 44, engine: "SGLang" },
  { completed_requests: 150000, failure_count: 0, compressed_second: 66, engine: "SGLang" },
  { completed_requests: 200000, failure_count: 0, compressed_second: 88, engine: "API route" },
  { completed_requests: 250000, failure_count: 0, compressed_second: 110, engine: "API route" }
];


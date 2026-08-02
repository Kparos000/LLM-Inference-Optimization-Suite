export type ResultType = "measured" | "modeled" | "planned";

export type ChapterId =
  | "about"
  | "slo-metrics"
  | "data"
  | "preparation"
  | "main-inference"
  | "optimization"
  | "optimized-inference"
  | "comparison"
  | "conclusions";

export type Chapter = {
  id: ChapterId;
  path: string;
  title: string;
  shortTitle: string;
  resultType: ResultType;
  sourceArtifacts: string[];
  purpose: string;
};

export type PlatformResponse<T> = {
  status: string;
  result_type: ResultType;
  source_artifacts: string[];
  data: T;
};

export type ExperimentSession = {
  baselineRunId: string;
  currentChapter: ChapterId;
  selectedMandatoryRepairs: string[];
  selectedCoreOptimizations: string[];
  validatedRecipe: string | null;
  resultType: ResultType;
  selectedScenarioId: string;
  selectedOptimizedRunId: string | null;
};

export type MetricCard = {
  label: string;
  value: string;
  tone: "neutral" | "pass" | "fail" | "warn";
  detail: string;
};

export type OptimizationState = {
  optimization_id: string;
  display_name: string;
  category: string;
  state:
    | "applicable_measured"
    | "applicable_planned"
    | "not_applicable"
    | "blocked_by_negative_rule"
    | "future_architecture";
  definition: string;
  mechanism: string;
  affected_metrics: string[];
  possible_regressions: string[];
  implementation_status: string;
  current_project_support: string;
  requires_gpu_or_api_rerun: boolean;
  reason: string;
  negative_rule?: string | null;
};

export type DeployabilityRepair = {
  repair_id: string;
  display_name: string;
  track: "deployability_repairs";
  state:
    | "required_for_failed_deployability_slo"
    | "available_supporting_repair_not_selected"
    | "blocked_by_negative_rule";
  selectable_now: boolean;
  definition: string;
  why_it_is_a_repair: string;
  why_it_applies: string;
  affected_failed_slos: Array<{
    slo_id: string;
    metric_id: string;
    metric_label: string;
    target: string;
    observed: number;
    bottleneck_id: string;
  }>;
  exact_changes: string[];
  implementation_status: string;
  requires_gpu_or_api_rerun: boolean;
};

export type RepairGate = {
  gate_status: "PASS" | "FAIL" | "NOT_MEASURED" | "MISSING_CONFIGURATION";
  core_optimization_eligible: boolean;
  blocking_reason: string;
  minimum_not_optimal_principle: string;
  checks: Array<{
    check_id: string;
    label: string;
    status: "PASS" | "FAIL" | "NOT_MEASURED" | "MISSING_CONFIGURATION";
    observed: string | number | null;
    target: string;
    source_artifact: string;
  }>;
};

export type CoreOptimizationState = {
  optimization_id: string;
  display_name: string;
  category: string;
  track: "core_inference_optimizations";
  state:
    | "already_measured_in_baseline"
    | "blocked_by_negative_rule"
    | "locked_until_deployability_repair_validated"
    | "planned_not_ready"
    | "not_compatible_with_measured_matrix"
    | "eligible_after_repair_gate";
  selectable_now: boolean;
  definition: string;
  mechanism: string;
  affected_metrics: string[];
  possible_regressions: string[];
  implementation_status: string;
  current_project_support: string;
  requires_gpu_or_api_rerun: boolean;
  reason: string;
  compatible_config_count: number;
  compatible_engines: string[];
  compatible_memory_modes: string[];
  compatible_hardware: string[];
  compatible_models: string[];
  negative_rule_triggered?: string | null;
};

export type ExperimentStage = {
  current_stage: string;
  stage_sequence: Array<{
    stage: string;
    state: string;
    description: string;
  }>;
  gates: {
    failed_slo_count: number;
    repair_required: boolean;
    repair_plan_available: boolean;
    repair_gate_status: string;
    core_optimization_eligible: boolean;
    optimized_inference_ready: boolean;
  };
};

export type OptimizationStory = {
  title: string;
  summary: string;
  principles: string[];
  interaction_flow: Array<{
    step: string;
    user_action: string;
    system_response: string;
  }>;
};

export type CoreObservabilityCard = {
  optimization_id: string;
  display_name: string;
  optimization_domain: string;
  difficulty_tier: string;
  instrumentation_state: string;
  problem: string;
  mechanism: string;
  experiment: string;
  required_instrumentation: string[];
  missing_instrumentation: string[];
  primary_metrics: string[];
  visualization: {
    hero: string;
    workload_grounded: string;
    live_experiment: string;
    final_result: string;
    empty_state: string;
  };
  source_label: string;
};

export type CoreObservabilityCards = {
  result_type: "planned";
  semantics: Record<string, string | boolean>;
  cards: CoreObservabilityCard[];
  readiness_summary: Record<string, number>;
  prefix_summary: Record<string, number | string | Record<string, number>>;
  source_artifacts: string[];
};

export type PrefixLayoutStaticExperiment = {
  summary: {
    scenario_id: string;
    parent_run_id: string;
    optimization_id: string;
    result_type: string;
    status: string;
    workload_rows_scanned: number;
    inference_executed: boolean;
    cache_hits_measured: boolean;
    latency_claimed: boolean;
    layout_summaries: Record<
      string,
      {
        prompt_count: number;
        mean_input_tokens: number;
        median_input_tokens: number;
        p95_input_tokens: number;
        p99_input_tokens: number;
        mean_longest_exact_common_prefix_tokens: number;
        mean_reusable_token_ratio: number;
        prefix_family_count: number;
      }
    >;
    deltas: {
      candidate_minus_baseline_mean_common_prefix_tokens: number;
      candidate_minus_baseline_mean_reusable_token_ratio: number;
      candidate_minus_baseline_total_input_tokens: number;
    };
  };
  layouts: {
    baseline: {
      layout_id: string;
      section_order: string[];
      raw_prompt_text_included: boolean;
    };
    candidate: {
      layout_id: string;
      section_order: string[];
      raw_prompt_text_included: boolean;
    };
  };
  metrics: {
    prefix_families: Array<Record<string, string>>;
    per_vertical_memory: Array<Record<string, string>>;
    section_analysis: Array<Record<string, string>>;
  };
  equivalence: {
    status: string;
    rows_checked: number;
    section_content_byte_equivalent: boolean;
    evidence_order_fixed: boolean;
    instruction_priority_risk: boolean;
    requires_inference_validation: boolean;
  };
  decision: {
    scenario_id: string;
    decision: string;
    reason: string;
    requires_gpu_rerun: boolean;
    requires_engine_validation: boolean;
    next_required_experiment: string;
    disallowed_claims: string[];
  };
  story: {
    title: string;
    story_steps: Array<{
      id: string;
      title: string;
      body: string;
    }>;
    headline_metrics: Record<string, string | number | boolean>;
    apply_behavior: string;
  };
  source_artifacts: string[];
};

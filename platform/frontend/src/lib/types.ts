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

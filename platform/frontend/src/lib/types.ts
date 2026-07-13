export type ResultType = "measured" | "modeled" | "planned";

export type ChapterId =
  | "about"
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


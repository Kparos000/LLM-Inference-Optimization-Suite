"use client";

import { motion } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  ClipboardList,
  Cpu,
  Database,
  FlaskConical,
  Gauge,
  Layers3,
  Play,
  ShieldAlert,
  Split,
  TerminalSquare
} from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { fetchPlatform } from "@/lib/api";
import {
  chapters,
  fallbackOptimizationStates,
  headlineMetrics,
  replayFallback,
  sloRows,
  verticalRows
} from "@/lib/facts";
import { useExperimentSession } from "@/lib/session";
import type { ChapterId, OptimizationState } from "@/lib/types";
import { MissionShell } from "./MissionShell";
import { MetricCard } from "./MetricCard";
import { StatusBadge } from "./StatusBadge";
import {
  CostSplitChart,
  LatencyPercentileChart,
  QualityRateChart,
  ReplayLineChart,
  RequestSplitChart,
  SloChart,
  TelemetryAreaChart,
  VerticalPressureChart
} from "./Charts";
import {
  StoryAboutPage,
  StoryDataPage,
  StoryMainInferencePage,
  StoryPreparationPage,
  StorySloMetricsPage
} from "./StoryRoutes";

type ReplayEvent = {
  completed_requests?: number;
  failure_count?: number;
  compressed_second?: number;
  engine?: string;
  runtime?: string;
  memory_mode?: string;
  current_config_id?: string;
  vertical?: string;
  approximate_cost_so_far_usd?: number;
  [key: string]: unknown;
};

type TelemetrySample = {
  timestamp?: string;
  utilization_gpu_percent?: number;
  temperature_c?: number;
  memory_used_mb?: number;
  power_draw_w?: number;
  [key: string]: unknown;
};

type MainResults = {
  eval_report?: {
    latency_summary?: Record<string, unknown>;
    matrix_summary?: Record<string, unknown>;
    summary?: Record<string, unknown>;
    wall_seconds?: number;
  };
  cost_report?: Record<string, unknown>;
  slo_scorecard?: Array<Record<string, unknown>>;
};

const fallbackMainResults: MainResults = {
  eval_report: {
    latency_summary: {
      mean_e2e_latency_ms: 1555.083189807053,
      p50_e2e_latency_ms: 1357.596322428435,
      p95_e2e_latency_ms: 2997.51268675318,
      p99_e2e_latency_ms: 3990.831322551244,
      mean_ttft_ms: 316.182404887296,
      p50_ttft_ms: 177.4042589822784,
      p95_ttft_ms: 881.2036044429988,
      p99_ttft_ms: 1206.8634418747388,
      mean_tpot_ms: 40.81780315907682,
      p50_tpot_ms: 37.353379669200095,
      p95_tpot_ms: 83.32632407512622,
      p99_tpot_ms: 116.49905651574956,
      mean_total_tokens_per_second: 574.1231592602473
    },
    matrix_summary: {
      self_hosted_request_count: 200000,
      api_request_count: 50000
    },
    summary: {
      json_valid_rate: 0.99822,
      generation_contract_valid_rate: 0.805388,
      evidence_match_rate: 0.589724,
      grounded_rate: 0.567204,
      safety_violation_rate: 0.011028
    }
  },
  cost_report: {
    gpu_cost_usd: 17.606359966432798,
    api_cost_usd: 0.854936759999998
  },
  slo_scorecard: []
};

function numericValue(value: unknown, fallback = 0) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function ratePercent(value: unknown) {
  return Number((numericValue(value) * 100).toFixed(2));
}

function Panel({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <section className={`rounded-3xl border border-white/10 bg-graphite-900/75 p-6 shadow-mission backdrop-blur-xl ${className}`}>
      {children}
    </section>
  );
}

function SectionTitle({
  icon,
  title,
  children
}: {
  icon: React.ReactNode;
  title: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="mb-5 flex items-start justify-between gap-4">
      <div className="flex gap-3">
        <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-signal-cyan/12 text-signal-cyan">
          {icon}
        </div>
        <div>
          <h2 className="text-xl font-semibold text-white">{title}</h2>
          {children ? <p className="mt-1 max-w-3xl text-sm leading-6 text-slate-400">{children}</p> : null}
        </div>
      </div>
    </div>
  );
}

function SourceList({ sources }: { sources: string[] }) {
  return (
    <div className="mt-6 rounded-2xl border border-white/10 bg-black/20 p-4">
      <p className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
        Source artifacts
      </p>
      <div className="space-y-2">
        {sources.map((source) => (
          <p key={source} className="truncate rounded-lg bg-white/[0.035] px-3 py-2 font-mono text-xs text-slate-300">
            {source}
          </p>
        ))}
      </div>
    </div>
  );
}

function useApiData<T>(path: string, fallback: T): T {
  const [data, setData] = useState<T>(fallback);
  useEffect(() => {
    let active = true;
    void fetchPlatform<T>(path).then((payload) => {
      if (active && payload) {
        setData(payload);
      }
    });
    return () => {
      active = false;
    };
  }, [path]);
  return data;
}

export function ChapterScreen({ chapterId }: { chapterId: ChapterId }) {
  const chapter = chapters.find((item) => item.id === chapterId) ?? chapters[0];
  const { setChapter } = useExperimentSession();
  const storyRouteIds: ChapterId[] = [
    "about",
    "slo-metrics",
    "data",
    "preparation",
    "main-inference"
  ];
  const usesStoryHeader = storyRouteIds.includes(chapter.id);

  useEffect(() => {
    setChapter(chapter.id, chapter.resultType);
  }, [chapter.id, chapter.resultType, setChapter]);

  return (
    <MissionShell>
      <motion.div
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35 }}
        className="space-y-5"
      >
        {!usesStoryHeader ? (
          <Panel>
            <div className="flex flex-col justify-between gap-6 xl:flex-row xl:items-end">
              <div>
                <div className="mb-4 flex flex-wrap gap-2">
                  <StatusBadge tone={chapter.resultType === "measured" ? "pass" : "warn"}>
                    {chapter.resultType}
                  </StatusBadge>
                  <StatusBadge tone="neutral">repo sourced</StatusBadge>
                  <StatusBadge tone="neutral">no GPU</StatusBadge>
                </div>
                <h1 className="max-w-5xl text-4xl font-semibold tracking-normal text-white md:text-6xl">
                  {chapter.title}
                </h1>
                <p className="mt-4 max-w-3xl text-base leading-7 text-slate-300">{chapter.purpose}</p>
              </div>
              {chapter.id === "about" ? (
                <Link
                  href="/data"
                  className="inline-flex w-fit items-center gap-2 rounded-2xl bg-signal-cyan px-5 py-3 text-sm font-semibold text-graphite-950"
                >
                  Start Experiment <ArrowRight size={18} />
                </Link>
              ) : null}
            </div>
          </Panel>
        ) : null}
        {chapter.id === "about" && <StoryAboutPage sources={chapter.sourceArtifacts} />}
        {chapter.id === "slo-metrics" && <StorySloMetricsPage sources={chapter.sourceArtifacts} />}
        {chapter.id === "data" && <StoryDataPage sources={chapter.sourceArtifacts} />}
        {chapter.id === "preparation" && <StoryPreparationPage sources={chapter.sourceArtifacts} />}
        {chapter.id === "main-inference" && <StoryMainInferencePage sources={chapter.sourceArtifacts} />}
        {chapter.id === "optimization" && <OptimizationPage sources={chapter.sourceArtifacts} />}
        {chapter.id === "optimized-inference" && <OptimizedPage sources={chapter.sourceArtifacts} />}
        {chapter.id === "comparison" && <ComparisonPage sources={chapter.sourceArtifacts} />}
        {chapter.id === "conclusions" && <ConclusionsPage sources={chapter.sourceArtifacts} />}
      </motion.div>
    </MissionShell>
  );
}

// Legacy first-pass page retained temporarily while the enhanced story routes settle.
// eslint-disable-next-line @typescript-eslint/no-unused-vars
function AboutPage({ sources }: { sources: string[] }) {
  return (
    <>
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {headlineMetrics.map((metric) => (
          <MetricCard key={metric.label} metric={metric} />
        ))}
      </div>
      <div className="grid gap-5 xl:grid-cols-[1.2fr_.8fr]">
        <Panel>
          <SectionTitle icon={<Activity size={22} />} title="What this product demonstrates">
            The platform replays a completed inference experiment end to end. Users inspect saved
            data, retrieval, serving, SLO, GPU, cost, and optimization-planning artifacts without
            running a GPU.
          </SectionTitle>
          <div className="grid gap-3 md:grid-cols-3">
            {[
              ["Data construction", "10,000 source prompts across five verticals."],
              ["Serving matrix", "25 configs across vLLM, SGLang, and API route."],
              ["Optimization logic", "Failed SLOs map to catalog-backed, filtered strategies."]
            ].map(([title, body]) => (
              <div key={title} className="rounded-2xl border border-white/10 bg-white/[0.035] p-4">
                <p className="font-semibold text-white">{title}</p>
                <p className="mt-2 text-sm leading-6 text-slate-400">{body}</p>
              </div>
            ))}
          </div>
        </Panel>
        <Panel>
          <SectionTitle icon={<ShieldAlert size={22} />} title="Measured verdict">
            Operationally complete, not deployable.
          </SectionTitle>
          <div className="space-y-3">
            <StatusBadge tone="pass">runtime pass</StatusBadge>
            <StatusBadge tone="pass">cost pass</StatusBadge>
            <StatusBadge tone="fail">quality fail</StatusBadge>
            <StatusBadge tone="fail">safety fail</StatusBadge>
          </div>
          <p className="mt-5 text-sm leading-6 text-slate-400">
            The central product lesson is that speed and cost are not enough. Deployability depends
            on grounded, safe, contract-valid answers.
          </p>
        </Panel>
      </div>
      <SourceList sources={sources} />
    </>
  );
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
function DataPage({ sources }: { sources: string[] }) {
  return (
    <>
      <div className="grid gap-5 xl:grid-cols-[.9fr_1.1fr]">
        <Panel>
          <SectionTitle icon={<Database size={22} />} title="Five-vertical benchmark">
            The workflow ties prompt records, gold/evaluation records, and vertical KB evidence
            into inference-ready workloads.
          </SectionTitle>
          <div className="space-y-3">
            {verticalRows.map((row) => (
              <div key={row.vertical} className="rounded-2xl border border-white/10 bg-white/[0.035] p-4">
                <div className="flex items-center justify-between gap-3">
                  <p className="font-semibold capitalize text-white">{row.vertical.replace("_", " ")}</p>
                  <StatusBadge tone={row.coverage >= 1 ? "pass" : "warn"}>
                    evidence {Math.round(row.coverage * 100)}%
                  </StatusBadge>
                </div>
                <p className="mt-2 text-sm text-slate-400">
                  {row.prompts.toLocaleString()} prompts, {row.gold.toLocaleString()} gold records,
                  {" "}{row.kb.toLocaleString()} KB rows.
                </p>
              </div>
            ))}
          </div>
        </Panel>
        <Panel>
          <SectionTitle icon={<Gauge size={22} />} title="Workload pressure">
            Higher pressure usually means more context and evidence coordination, which can affect
            TTFT, contract validity, evidence match, groundedness, and safety.
          </SectionTitle>
          <VerticalPressureChart />
        </Panel>
      </div>
      <Panel>
        <SectionTitle icon={<Split size={22} />} title="Prompt / gold / KB relationship">
          Each prompt has expected gold behavior and required evidence. The UI should teach that
          retrieval and context determine what the model can cite.
        </SectionTitle>
        <div className="grid gap-4 md:grid-cols-4">
          {["Prompt", "Gold contract", "Knowledge base", "Evaluation"].map((item, index) => (
            <div key={item} className="rounded-2xl border border-white/10 bg-white/[0.035] p-4">
              <p className="text-sm text-slate-400">Step {index + 1}</p>
              <p className="mt-2 text-lg font-semibold text-white">{item}</p>
            </div>
          ))}
        </div>
      </Panel>
      <SourceList sources={sources} />
    </>
  );
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
function PreparationPage({ sources }: { sources: string[] }) {
  const pipeline = [
    "Dataset",
    "query construction",
    "dense retrieval",
    "BM25",
    "hybrid fusion",
    "reranking",
    "context selection",
    "compression",
    "prompt assembly",
    "generation contract",
    "memory mode",
    "model",
    "engine",
    "concurrency",
    "A100",
    "telemetry",
    "evaluation",
    "SLO scoring"
  ];
  return (
    <>
      <Panel>
        <SectionTitle icon={<Layers3 size={22} />} title="End-to-end preparation pipeline">
          This is the controlled path from repository data to measured SLO outcomes.
        </SectionTitle>
        <div className="grid gap-3 md:grid-cols-3 xl:grid-cols-6">
          {pipeline.map((stage, index) => (
            <div key={`${stage}-${index}`} className="rounded-2xl border border-white/10 bg-white/[0.035] p-4">
              <p className="text-xs text-slate-500">{String(index + 1).padStart(2, "0")}</p>
              <p className="mt-2 text-sm font-semibold text-white">{stage}</p>
            </div>
          ))}
        </div>
      </Panel>
      <div className="grid gap-5 xl:grid-cols-3">
        {[
          ["MM0-MM4", "No context, dense, hybrid, compressed hybrid, and bounded agentic repair."],
          ["Models", "model3_7b and model6_gated were active in Main_Inference_V1."],
          ["Engines", "vLLM, SGLang, and API provider route were measured."]
        ].map(([title, body]) => (
          <Panel key={title}>
            <SectionTitle icon={<Cpu size={22} />} title={title}>
              {body}
            </SectionTitle>
          </Panel>
        ))}
      </div>
      <Panel>
        <SectionTitle icon={<CheckCircle2 size={22} />} title="Matrix construction">
          The full baseline measured 25 configs x 10,000 prompts = 250,000 requests.
        </SectionTitle>
        <div className="grid gap-4 md:grid-cols-3">
          <MetricCard metric={{ label: "Configs", value: "25", tone: "neutral", detail: "Main matrix rows" }} />
          <MetricCard metric={{ label: "Prompts/config", value: "10,000", tone: "neutral", detail: "2,000 per vertical" }} />
          <MetricCard metric={{ label: "Total requests", value: "250,000", tone: "pass", detail: "Completed without request failures" }} />
        </div>
      </Panel>
      <SourceList sources={sources} />
    </>
  );
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
function MainInferencePage({ sources }: { sources: string[] }) {
  const replay = useApiData<{ events: ReplayEvent[]; final_completed: number; final_failed: number }>(
    "/api/main-inference/replay-events",
    { events: replayFallback, final_completed: 250000, final_failed: 0 }
  );
  const telemetry = useApiData<{ samples: TelemetrySample[]; summary: Record<string, unknown> }>(
    "/api/main-inference/telemetry",
    { samples: [], summary: {} }
  );
  const results = useApiData<MainResults>("/api/main-inference/results", fallbackMainResults);
  const [running, setRunning] = useState(false);
  const [eventIndex, setEventIndex] = useState(0);
  const events: ReplayEvent[] = replay.events.length ? replay.events : replayFallback;
  const current = events[Math.min(eventIndex, events.length - 1)] ?? replayFallback[0];
  const visibleEvents = eventIndex === 0 && !running ? events.slice(0, 1) : events.slice(0, eventIndex + 1);
  const latency = results.eval_report?.latency_summary ?? {};
  const quality = results.eval_report?.summary ?? {};
  const matrix = results.eval_report?.matrix_summary ?? {};
  const cost = results.cost_report ?? {};
  const progressPct = Math.min(100, (numericValue(current.completed_requests) / 250000) * 100);
  const latencyChartData = [
    {
      metric: "TTFT",
      p50: numericValue(latency.p50_ttft_ms),
      p95: numericValue(latency.p95_ttft_ms),
      p99: numericValue(latency.p99_ttft_ms)
    },
    {
      metric: "TPOT",
      p50: numericValue(latency.p50_tpot_ms),
      p95: numericValue(latency.p95_tpot_ms),
      p99: numericValue(latency.p99_tpot_ms)
    },
    {
      metric: "E2E",
      p50: numericValue(latency.p50_e2e_latency_ms),
      p95: numericValue(latency.p95_e2e_latency_ms),
      p99: numericValue(latency.p99_e2e_latency_ms)
    }
  ];
  const qualityChartData = [
    { metric: "JSON", rate: ratePercent(quality.json_valid_rate) },
    { metric: "Contract", rate: ratePercent(quality.generation_contract_valid_rate) },
    { metric: "Evidence", rate: ratePercent(quality.evidence_match_rate) },
    { metric: "Grounded", rate: ratePercent(quality.grounded_rate) },
    { metric: "Safety clean", rate: ratePercent(1 - numericValue(quality.safety_violation_rate)) }
  ];
  const requestSplitData = [
    { track: "Self-hosted GPU", requests: numericValue(matrix.self_hosted_request_count) },
    { track: "API provider", requests: numericValue(matrix.api_request_count) }
  ];
  const costSplitData = [
    { track: "A100 GPU", cost: Number(numericValue(cost.gpu_cost_usd).toFixed(2)) },
    { track: "API provider", cost: Number(numericValue(cost.api_cost_usd).toFixed(2)) }
  ];

  useEffect(() => {
    if (!running) {
      return;
    }
    const interval = window.setInterval(() => {
      setEventIndex((value) => {
        if (value >= events.length - 1) {
          window.clearInterval(interval);
          setRunning(false);
          return value;
        }
        return value + 1;
      });
    }, Math.max(900, 110000 / Math.max(events.length, 1)));
    return () => window.clearInterval(interval);
  }, [events.length, running]);

  return (
    <>
      <Panel>
        <SectionTitle icon={<Play size={22} />} title="Measured run replay">
          Click Run Main Inference to time-compress the saved 11.82-hour run into roughly 90-120
          seconds. This uses saved progress events and does not run inference.
        </SectionTitle>
        <div className="flex flex-wrap items-center gap-3">
          <button
            className="inline-flex items-center gap-2 rounded-2xl bg-signal-cyan px-5 py-3 text-sm font-semibold text-graphite-950"
            onClick={() => {
              setEventIndex(0);
              setRunning(true);
            }}
          >
            <Play size={18} /> Run Main Inference Replay
          </button>
          <StatusBadge tone={running ? "warn" : eventIndex === events.length - 1 ? "pass" : "neutral"}>
            {running ? "replaying" : eventIndex === events.length - 1 ? "completed" : "ready"}
          </StatusBadge>
          <StatusBadge tone="pass">measured artifact</StatusBadge>
        </div>
        <div className="mt-6 rounded-2xl border border-white/10 bg-white/[0.035] p-4">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-3 text-sm">
            <span className="font-mono text-signal-cyan">
              {String(current.current_config_id ?? "waiting_for_replay_start")}
            </span>
            <span className="text-slate-400">{progressPct.toFixed(1)}% complete</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-white/10">
            <div
              className="h-full rounded-full bg-signal-cyan transition-all duration-700"
              style={{ width: `${progressPct}%` }}
            />
          </div>
          <div className="mt-4 grid gap-3 text-sm text-slate-400 md:grid-cols-4">
            <span>engine: {String(current.engine ?? current.runtime ?? "pending")}</span>
            <span>memory: {String(current.memory_mode ?? "pending")}</span>
            <span>vertical: {String(current.vertical ?? "pending")}</span>
            <span>concurrency: {String(current.concurrency ?? "pending")}</span>
          </div>
        </div>
        <div className="mt-6 grid gap-4 md:grid-cols-4">
          <MetricCard
            metric={{
              label: "Completed",
              value: Number(current.completed_requests ?? 0).toLocaleString(),
              tone: Number(current.completed_requests ?? 0) === 250000 ? "pass" : "neutral",
              detail: "Saved progress event count"
            }}
          />
          <MetricCard
            metric={{
              label: "Failed",
              value: String(current.failure_count ?? 0),
              tone: "pass",
              detail: "Final replay must remain zero"
            }}
          />
          <MetricCard
            metric={{
              label: "Engine",
              value: String(current.engine ?? current.runtime ?? "mixed"),
              tone: "neutral",
              detail: "vLLM, SGLang, or API route"
            }}
          />
          <MetricCard
            metric={{
              label: "Cost so far",
              value:
                typeof current.approximate_cost_so_far_usd === "number"
                  ? `$${current.approximate_cost_so_far_usd.toFixed(2)}`
                  : "$0.00",
              tone: "neutral",
              detail: "Measured final total cost is $18.46"
            }}
          />
        </div>
      </Panel>
      <div className="grid gap-5 xl:grid-cols-2">
        <Panel>
          <SectionTitle icon={<Activity size={22} />} title="Replay progress">
            This line grows from saved progress checkpoints as the replay advances. It should end
            at exactly 250,000 completed requests with zero request failures.
          </SectionTitle>
          <ReplayLineChart data={visibleEvents} />
        </Panel>
        <Panel>
          <SectionTitle icon={<Gauge size={22} />} title="GPU telemetry sample">
            The telemetry sample shows how the A100 behaved while self-hosted vLLM/SGLang configs
            were running. Utilization and temperature dip during API-provider phases because those
            requests do not use the rented GPU.
          </SectionTitle>
          <TelemetryAreaChart data={telemetry.samples.length ? telemetry.samples : []} />
        </Panel>
      </div>
      <div className="grid gap-5 xl:grid-cols-[1.1fr_.9fr]">
        <Panel>
          <SectionTitle icon={<Gauge size={22} />} title="Latency percentiles">
            TTFT measures time to first token, TPOT measures decode cadence per output token, and
            E2E measures full request latency. The percentile spread shows how tail latency differs
            from the median user experience.
          </SectionTitle>
          <LatencyPercentileChart data={latencyChartData} />
        </Panel>
        <Panel>
          <SectionTitle icon={<Activity size={22} />} title="Throughput snapshot">
            Throughput tells whether the serving stack is moving tokens efficiently, but it is only
            useful with quality and safety beside it. A fast system that misses evidence is still
            not deployable.
          </SectionTitle>
          <div className="grid gap-4 sm:grid-cols-2">
            <MetricCard
              metric={{
                label: "Mean E2E",
                value: `${numericValue(latency.mean_e2e_latency_ms).toFixed(0)} ms`,
                tone: "neutral",
                detail: "Average end-to-end request latency"
              }}
            />
            <MetricCard
              metric={{
                label: "p95 E2E",
                value: `${numericValue(latency.p95_e2e_latency_ms).toFixed(0)} ms`,
                tone: "neutral",
                detail: "Tail latency experienced by slower requests"
              }}
            />
            <MetricCard
              metric={{
                label: "Mean TTFT",
                value: `${numericValue(latency.mean_ttft_ms).toFixed(0)} ms`,
                tone: "pass",
                detail: "Time until the first streamed token"
              }}
            />
            <MetricCard
              metric={{
                label: "Tokens/sec",
                value: numericValue(latency.mean_total_tokens_per_second).toFixed(1),
                tone: "pass",
                detail: "Mean total token throughput"
              }}
            />
          </div>
        </Panel>
      </div>
      <div className="grid gap-5 xl:grid-cols-2">
        <Panel>
          <SectionTitle icon={<ShieldAlert size={22} />} title="Quality and safety rates">
            This graph separates quality rates from raw runtime and cost units. JSON validity was
            strong, while contract validity, evidence match, groundedness, and safety are the
            deployability blockers the optimization lab must explain.
          </SectionTitle>
          <QualityRateChart data={qualityChartData} />
        </Panel>
        <Panel>
          <SectionTitle icon={<Split size={22} />} title="Request and cost split">
            The experiment combined self-hosted A100 inference with API-provider inference. Keeping
            those tracks separate helps the user understand why GPU cost, API cost, latency, and
            quality do not move together.
          </SectionTitle>
          <div className="grid gap-4 lg:grid-cols-2">
            <RequestSplitChart data={requestSplitData} />
            <CostSplitChart data={costSplitData} />
          </div>
        </Panel>
      </div>
      <Panel>
        <SectionTitle icon={<ShieldAlert size={22} />} title="Deployability verdict">
          This is not a completion-count chart. It is an SLO attainment chart normalized to percent
          of target, so rates, counts, runtime, and cost can be interpreted together without hiding
          the failed quality bars.
        </SectionTitle>
        <div className="mb-5 flex flex-wrap gap-2">
          <StatusBadge tone="pass">runtime pass</StatusBadge>
          <StatusBadge tone="pass">cost pass</StatusBadge>
          <StatusBadge tone="fail">quality fail</StatusBadge>
          <StatusBadge tone="fail">safety fail</StatusBadge>
          <StatusBadge tone="fail">not deployable</StatusBadge>
        </div>
        <SloChart />
      </Panel>
      <SourceList sources={sources} />
    </>
  );
}

function OptimizationPage({ sources }: { sources: string[] }) {
  const { session, toggleMandatoryRepair, toggleCoreOptimization, applyAllSelected } =
    useExperimentSession();
  const applicability = useApiData<{ states: OptimizationState[] }>(
    "/api/optimizations/applicability",
    { states: fallbackOptimizationStates }
  );
  const states = applicability.states.length ? applicability.states : fallbackOptimizationStates;
  const mandatory = states.filter((item) =>
    ["prompt_contract_repair", "improve_evidence_formatting", "use_mm4_agentic_repair", "enable_escalation_path"].includes(
      item.optimization_id
    )
  );
  const core = states.filter((item) => !mandatory.some((repair) => repair.optimization_id === item.optimization_id));
  const selectedCount =
    session.selectedMandatoryRepairs.length + session.selectedCoreOptimizations.length;

  return (
    <>
      <div className="grid gap-5 xl:grid-cols-[1fr_360px]">
        <Panel>
          <SectionTitle icon={<FlaskConical size={22} />} title="Two-lane optimization logic">
            Mandatory repairs target failed deployability SLOs. Core strategies remain educational
            and disabled when negative rules block them.
          </SectionTitle>
          <div className="grid gap-5 xl:grid-cols-2">
            <OptimizationLane
              title="A. Mandatory System Quality & Workflow Repairs"
              items={mandatory}
              selected={session.selectedMandatoryRepairs}
              onToggle={toggleMandatoryRepair}
            />
            <OptimizationLane
              title="B. Core Inference Engineering Optimizations"
              items={core.slice(0, 12)}
              selected={session.selectedCoreOptimizations}
              onToggle={toggleCoreOptimization}
            />
          </div>
        </Panel>
        <Panel>
          <SectionTitle icon={<ClipboardList size={22} />} title="Selected recipe">
            Apply All means mandatory repairs plus selected compatible core strategies.
          </SectionTitle>
          <p className="text-5xl font-semibold text-white">{selectedCount}</p>
          <p className="mt-2 text-sm text-slate-400">selected changes, plan-only</p>
          <button
            className="mt-6 w-full rounded-2xl bg-signal-cyan px-4 py-3 text-sm font-semibold text-graphite-950"
            onClick={() => applyAllSelected(mandatory.map((item) => item.optimization_id))}
          >
            Apply All Selected Optimizations
          </button>
          <div className="mt-5 rounded-2xl border border-signal-amber/30 bg-signal-amber/10 p-4 text-sm leading-6 text-signal-amber">
            This produces a validated plan only. It does not create Optimized_Inference_V1.
          </div>
        </Panel>
      </div>
      <Panel>
        <SectionTitle icon={<AlertTriangle size={22} />} title="Failed SLOs driving the lab">
          The UI must expose only valid selectable options for the selected failure while preserving
          disabled strategies as educational explanations.
        </SectionTitle>
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          {sloRows.map((row) => (
            <div key={row.metric} className="rounded-2xl border border-white/10 bg-white/[0.035] p-4">
              <p className="font-semibold text-white">{row.metric}</p>
              <p className="mt-1 text-sm text-slate-400">target {row.target}</p>
              <StatusBadge tone={row.status === "PASS" ? "pass" : "fail"}>{row.status}</StatusBadge>
            </div>
          ))}
        </div>
      </Panel>
      <SourceList sources={sources} />
    </>
  );
}

function OptimizationLane({
  title,
  items,
  selected,
  onToggle
}: {
  title: string;
  items: OptimizationState[];
  selected: string[];
  onToggle: (id: string) => void;
}) {
  return (
    <div>
      <h3 className="mb-3 text-sm font-semibold uppercase tracking-[0.16em] text-slate-400">{title}</h3>
      <div className="max-h-[640px] space-y-3 overflow-auto pr-2 scrollbar-thin">
        {items.map((item) => {
          const selectable = item.state === "applicable_measured" || item.state === "applicable_planned";
          return (
            <button
              key={item.optimization_id}
              onClick={() => selectable && onToggle(item.optimization_id)}
              className="w-full rounded-2xl border border-white/10 bg-white/[0.035] p-4 text-left transition hover:border-signal-cyan/40 disabled:cursor-not-allowed"
              disabled={!selectable}
            >
              <div className="flex items-start justify-between gap-3">
                <p className="font-semibold text-white">{item.display_name}</p>
                <StatusBadge
                  tone={
                    item.state === "applicable_measured"
                      ? "pass"
                      : item.state === "blocked_by_negative_rule"
                        ? "fail"
                        : "warn"
                  }
                >
                  {item.state.replaceAll("_", " ")}
                </StatusBadge>
              </div>
              <p className="mt-2 text-sm leading-6 text-slate-400">{item.definition}</p>
              <p className="mt-3 text-xs text-slate-500">{item.reason}</p>
              {selected.includes(item.optimization_id) ? (
                <p className="mt-3 text-sm font-semibold text-signal-cyan">selected</p>
              ) : null}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function OptimizedPage({ sources }: { sources: string[] }) {
  const { session } = useExperimentSession();
  return (
    <>
      <Panel>
        <SectionTitle icon={<TerminalSquare size={22} />} title="Optimized run target">
          The selected recipe is ready to become a measured run, but the official
          Optimized_Inference_V1 artifact set is not present yet.
        </SectionTitle>
        <div className="grid gap-4 md:grid-cols-3">
          <MetricCard metric={{ label: "Result type", value: "planned", tone: "warn", detail: "No optimized artifact is loaded." }} />
          <MetricCard
            metric={{
              label: "Mandatory repairs",
              value: String(session.selectedMandatoryRepairs.length),
              tone: "neutral",
              detail: "Persisted browser session selection"
            }}
          />
          <MetricCard
            metric={{
              label: "Core strategies",
              value: String(session.selectedCoreOptimizations.length),
              tone: "neutral",
              detail: "Selectable only when compatible"
            }}
          />
        </div>
      </Panel>
      <Panel>
        <SectionTitle icon={<ClipboardList size={22} />} title="Required future artifact contract">
          This page will switch from planned to measured only after these files are saved.
        </SectionTitle>
        <div className="grid gap-3 md:grid-cols-2">
          {[
            "optimized_inference_v1_manifest.json",
            "optimized_inference_v1_eval_report.json",
            "optimized_inference_v1_slo_scorecard.csv",
            "optimized_inference_v1_cost_report.json",
            "optimized_inference_v1_gpu_telemetry.jsonl",
            "main_vs_optimized_inference_v1_ui_comparison.json"
          ].map((item) => (
            <p key={item} className="rounded-xl bg-white/[0.035] px-4 py-3 font-mono text-sm text-slate-300">
              {item}
            </p>
          ))}
        </div>
      </Panel>
      <SourceList sources={sources} />
    </>
  );
}

function ComparisonPage({ sources }: { sources: string[] }) {
  return (
    <>
      <Panel>
        <SectionTitle icon={<Split size={22} />} title="Comparison is intentionally blocked">
          Main_Inference_V1 is measured. Optimized_Inference_V1 is planned until exact saved
          optimized artifacts exist, so the UI cannot show optimized deltas yet.
        </SectionTitle>
        <SloChart />
      </Panel>
      <Panel>
        <SectionTitle icon={<AlertTriangle size={22} />} title="No fabricated before/after claims">
          The future comparison will report quality, safety, runtime, cost, model, engine, memory
          mode, and regression deltas only from saved artifacts.
        </SectionTitle>
        <div className="grid gap-4 md:grid-cols-3">
          <MetricCard metric={{ label: "Baseline", value: "measured", tone: "pass", detail: "Main_Inference_V1 artifacts exist." }} />
          <MetricCard metric={{ label: "Optimized", value: "planned", tone: "warn", detail: "Awaiting saved optimized run." }} />
          <MetricCard metric={{ label: "Delta", value: "blocked", tone: "warn", detail: "No optimized deltas are fabricated." }} />
        </div>
      </Panel>
      <SourceList sources={sources} />
    </>
  );
}

function ConclusionsPage({ sources }: { sources: string[] }) {
  return (
    <>
      <Panel>
        <SectionTitle icon={<CheckCircle2 size={22} />} title="Measured engineering lessons">
          The full run proved operational scale and exposed deployability failures. The next phase
          is a controlled optimized rerun, not a UI-only success claim.
        </SectionTitle>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {[
            ["Scale", "250,000 requests completed with zero request failures."],
            ["Runtime", "Runtime SLO passed on the saved A100 run."],
            ["Cost", "Total measured cost was $18.46."],
            ["Deployability", "Quality and safety failures block deployment."]
          ].map(([title, body]) => (
            <div key={title} className="rounded-2xl border border-white/10 bg-white/[0.035] p-4">
              <p className="font-semibold text-white">{title}</p>
              <p className="mt-2 text-sm leading-6 text-slate-400">{body}</p>
            </div>
          ))}
        </div>
      </Panel>
      <Panel>
        <SectionTitle icon={<TerminalSquare size={22} />} title="Future project-grounded assistant">
          The conclusions page is wired as a target for pre-generated interpretations and
          artifact-cited follow-up chat, but no conclusion endpoint is configured yet.
        </SectionTitle>
        <StatusBadge tone="warn">unavailable until saved conclusion artifacts exist</StatusBadge>
      </Panel>
      <SourceList sources={sources} />
    </>
  );
}

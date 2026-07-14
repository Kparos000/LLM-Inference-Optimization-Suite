"use client";

import {
  Activity,
  Archive,
  Brain,
  CheckCircle2,
  ChevronRight,
  Clock,
  Cpu,
  Database,
  FileSearch,
  Gauge,
  GitBranch,
  Layers3,
  Pause,
  Play,
  Route,
  Search,
  ShieldAlert,
  SlidersHorizontal,
  Sparkles,
  Target,
  Zap
} from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { fetchPlatform } from "@/lib/api";
import { chapters, replayFallback } from "@/lib/facts";
import {
  CostSplitChart,
  LatencyPercentileChart,
  QualityRateChart,
  ReplayLineChart,
  RequestSplitChart,
  SloChart,
  TelemetryAreaChart
} from "./Charts";
import { MetricCard } from "./MetricCard";
import { StatusBadge } from "./StatusBadge";

type JsonRecord = Record<string, unknown>;

type OverviewData = {
  story?: string;
  headline_metrics?: JsonRecord;
  matrix_snapshot?: {
    self_hosted_formula?: string;
    api_formula?: string;
    total_formula?: string;
    self_hosted_model?: string;
    api_model?: string;
    serving_routes?: string[];
    memory_modes?: string[];
    hardware?: string;
    verticals?: string[];
  };
};

type SloMetricsData = {
  opening_explanation?: string;
  families?: Array<{
    id: string;
    label: string;
    chronology?: string;
    explanation?: string;
    metrics?: Array<{
      id: string;
      label: string;
      definition: string;
      user_experience: string;
      influences: string[];
      common_optimizations: string[];
      tradeoffs: string;
      targets: Array<JsonRecord>;
      target_varies_by_vertical: boolean;
    }>;
  }>;
  evaluation_flow?: string[];
  applicability_notes?: string[];
};

type DatasetExplorerData = {
  opening_explanation?: string;
  totals?: JsonRecord;
  research_ai_coverage_explanation?: JsonRecord;
  verticals?: Array<JsonRecord>;
  pressure_simulator?: Array<JsonRecord>;
};

type DatasetCasesData = {
  total_matches?: number;
  cases?: Array<{
    prompt: JsonRecord;
    gold_contract: JsonRecord;
    knowledge_base: {
      required_evidence?: JsonRecord[];
      distractor_evidence?: JsonRecord[];
    };
    evaluation_rubric: JsonRecord;
  }>;
  public_safety?: JsonRecord;
};

type PreparationModulesData = {
  modules?: Array<JsonRecord>;
};

type MainReplayDetailData = {
  hero?: JsonRecord;
  run_contract?: JsonRecord;
  phases?: Array<JsonRecord>;
  matrix?: {
    rows?: Array<JsonRecord>;
    totals?: JsonRecord;
    formula?: JsonRecord;
  };
  replay?: {
    events?: Array<JsonRecord>;
    replay_duration_seconds?: number;
  };
  telemetry?: {
    samples?: Array<JsonRecord>;
    summary?: JsonRecord;
  };
  latency_throughput?: {
    summary?: JsonRecord;
    trend?: Array<JsonRecord>;
    chart_resolution_rule?: string;
  };
  cost?: JsonRecord;
  deployability_gates?: Array<JsonRecord>;
  quality_safety?: {
    summary?: JsonRecord;
    definitions?: JsonRecord;
  };
  comparisons?: JsonRecord;
  artifact_reliability?: JsonRecord;
  engineering_lessons?: JsonRecord;
};

const statusTone = (status: unknown) => String(status).toUpperCase() === "PASS" ? "pass" : "fail";

function numberValue(value: unknown, fallback = 0) {
  return typeof value === "number" && Number.isFinite(value) ? value : Number.parseFloat(String(value ?? fallback)) || fallback;
}

function percent(value: unknown) {
  return `${(numberValue(value) * 100).toFixed(1)}%`;
}

function labelize(value: unknown) {
  return String(value ?? "").replaceAll("_", " ");
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
  chronology,
  children
}: {
  icon: React.ReactNode;
  title: string;
  chronology?: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="mb-5 flex items-start justify-between gap-4">
      <div className="flex gap-3">
        <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-signal-cyan/12 text-signal-cyan">
          {icon}
        </div>
        <div>
          {chronology ? (
            <p className="mb-1 text-xs font-semibold uppercase tracking-[0.18em] text-signal-green">
              {chronology}
            </p>
          ) : null}
          <h2 className="text-xl font-semibold text-white">{title}</h2>
          {children ? <p className="mt-1 max-w-4xl text-sm leading-6 text-slate-400">{children}</p> : null}
        </div>
      </div>
    </div>
  );
}

function SourceList({ sources }: { sources: string[] }) {
  return (
    <details className="rounded-2xl border border-white/10 bg-black/20 p-4">
      <summary className="cursor-pointer text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
        Source artifacts
      </summary>
      <div className="mt-3 space-y-2">
        {sources.map((source) => (
          <p key={source} className="truncate rounded-lg bg-white/[0.035] px-3 py-2 font-mono text-xs text-slate-300">
            {source}
          </p>
        ))}
      </div>
    </details>
  );
}

function InfoDisclosure({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <details className="rounded-2xl border border-white/10 bg-white/[0.035] p-4">
      <summary className="cursor-pointer text-sm font-semibold text-white">{title}</summary>
      <div className="mt-3 text-sm leading-6 text-slate-400">{children}</div>
    </details>
  );
}

function ProgressBar({ value, label }: { value: number; label?: string }) {
  return (
    <div>
      {label ? <div className="mb-1 text-xs text-slate-400">{label}</div> : null}
      <div className="h-2 overflow-hidden rounded-full bg-white/10">
        <div
          className="h-full rounded-full bg-gradient-to-r from-signal-cyan to-signal-green"
          style={{ width: `${Math.max(0, Math.min(100, value))}%` }}
        />
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

const sloSnapshot = [
  ["Quality", "Can the answer be parsed, validated, cited, and grounded?"],
  ["Safety", "Does the answer stay inside policy and domain boundaries?"],
  ["TTFT", "How quickly does the first token arrive?"],
  ["TPOT", "How fast does the answer stream after it starts?"],
  ["E2E latency", "How long does the full request take?"],
  ["Throughput", "How much useful work happens per second?"],
  ["Resource usage", "How much GPU, VRAM, CPU, RAM, power, and temperature pressure exists?"],
  ["Cost", "How much does GPU and provider execution cost?"]
];

export function StoryAboutPage({ sources }: { sources: string[] }) {
  const overview = useApiData<OverviewData>("/api/project/overview", {});
  const matrix = overview.matrix_snapshot ?? {};
  const metrics = overview.headline_metrics ?? {};
  return (
    <>
      <Panel>
        <div className="grid gap-8 xl:grid-cols-[1.1fr_.9fr] xl:items-center">
          <div>
            <h1 className="max-w-4xl text-4xl font-semibold tracking-normal text-white md:text-6xl">
              AI Inference Engineering Platform
            </h1>
            <div className="mt-5 max-w-4xl space-y-4 text-base leading-8 text-slate-300">
              <p>
                Inference engineering is the discipline of making AI systems useful after a model
                has been chosen: serving it reliably, giving it the right context, measuring speed
                and cost, and proving that answers are grounded, safe, and deployable.
              </p>
              <p>
                This product replays the full experiment journey: dataset construction, workflow
                formulation, retrieval engineering, context engineering, memory modes, model and
                engine selection, SLO design, Main_Inference_V1, optimization intelligence, an
                optimized rerun contract, before/after comparison, and final lessons.
              </p>
            </div>
            <div className="mt-7 flex flex-wrap gap-3">
              <Link className="rounded-2xl bg-signal-cyan px-5 py-3 text-sm font-semibold text-graphite-950" href="/data">
                Start Experiment
              </Link>
              <Link className="rounded-2xl border border-white/10 px-5 py-3 text-sm font-semibold text-white" href="/slo-metrics">
                Explore SLOs
              </Link>
              <Link className="rounded-2xl border border-white/10 px-5 py-3 text-sm font-semibold text-white" href="/main-inference">
                Replay Main Inference
              </Link>
            </div>
          </div>
          <div className="rounded-3xl border border-signal-cyan/20 bg-signal-cyan/8 p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-signal-cyan">
              Experiment snapshot
            </p>
            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              {[
                ["Prompts", "10,000"],
                ["Verticals", "5"],
                ["Configurations", "25"],
                ["Requests", "250,000"],
                ["Hardware", matrix.hardware ?? "NVIDIA A100-SXM4-80GB"],
                ["Wall time", "11.82 h"]
              ].map(([label, value]) => (
                <div key={label} className="rounded-2xl bg-black/25 p-4">
                  <p className="text-xs text-slate-400">{label}</p>
                  <p className="mt-1 text-lg font-semibold text-white">{value}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Panel>

      <div className="grid gap-5 xl:grid-cols-3">
        <Panel className="xl:col-span-2">
          <SectionTitle icon={<GitBranch size={22} />} title="Matrix construction" chronology="Designed before the run">
            The experiment was not a fake full Cartesian product. The self-hosted and API tracks
            were intentionally separated.
          </SectionTitle>
          <div className="space-y-3">
            {[matrix.self_hosted_formula, matrix.api_formula, matrix.total_formula].map((formula) => (
              <div key={String(formula)} className="rounded-2xl border border-white/10 bg-white/[0.035] p-4 font-mono text-sm text-slate-200">
                {String(formula)}
              </div>
            ))}
          </div>
          <div className="mt-5 grid gap-3 md:grid-cols-3">
            <InfoDisclosure title="Models">
              Self-hosted: <strong>{matrix.self_hosted_model}</strong>. API route:{" "}
              <strong>{matrix.api_model}</strong>.
            </InfoDisclosure>
            <InfoDisclosure title="Serving routes">
              {(matrix.serving_routes ?? []).join(", ")}
            </InfoDisclosure>
            <InfoDisclosure title="Memory modes">
              {(matrix.memory_modes ?? []).join(", ")}
            </InfoDisclosure>
          </div>
        </Panel>
        <Panel>
          <SectionTitle icon={<Database size={22} />} title="Five workloads">
            The dataset is balanced across domains so quality and serving behavior can be compared.
          </SectionTitle>
          <div className="space-y-2">
            {(matrix.verticals ?? []).map((vertical) => (
              <div key={vertical} className="rounded-xl border border-white/10 bg-white/[0.035] px-3 py-2 text-sm text-slate-200">
                {vertical}
              </div>
            ))}
          </div>
        </Panel>
      </div>

      <Panel>
        <SectionTitle icon={<Target size={22} />} title="What we planned to measure" chronology="Designed before the run">
          The platform introduces metric families before showing run outcomes, because production
          targets must exist before experiment results are judged.
        </SectionTitle>
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          {sloSnapshot.map(([label, body]) => (
            <Link href="/slo-metrics" key={label} className="rounded-2xl border border-white/10 bg-white/[0.035] p-4 transition hover:border-signal-cyan/50">
              <p className="font-semibold text-white">{label}</p>
              <p className="mt-2 text-sm leading-6 text-slate-400">{body}</p>
            </Link>
          ))}
        </div>
      </Panel>

      <Panel>
        <SectionTitle icon={<ShieldAlert size={22} />} title="Measured outcome teaser" chronology="Learned after the run">
          The full run completed operationally, but deployability failed because answer usefulness
          and safety did not clear the SLO gate.
        </SectionTitle>
        <div className="grid gap-3 md:grid-cols-3 xl:grid-cols-5">
          {[
            ["Execution", "PASS"],
            ["Reliability", "PASS"],
            ["Latency", "PASS"],
            ["Throughput", "PASS"],
            ["Cost", "PASS"],
            ["Contract/format", "FAIL"],
            ["Evidence/grounding", "FAIL"],
            ["Safety", "FAIL"],
            ["Overall", "Not deployable yet"]
          ].map(([label, status]) => (
            <Link href="/main-inference" key={label} className="rounded-2xl border border-white/10 bg-white/[0.035] p-4">
              <p className="text-sm text-slate-400">{label}</p>
              <p className={`mt-1 font-semibold ${status === "PASS" ? "text-signal-green" : "text-rose-300"}`}>
                {status}
              </p>
            </Link>
          ))}
        </div>
        <p className="mt-4 text-sm text-slate-500">
          Completed: {Number(metrics.completed_requests ?? 250000).toLocaleString()} requests.
          Failed requests: {String(metrics.failed_requests ?? 0)}.
        </p>
      </Panel>
      <SourceList sources={sources} />
    </>
  );
}

export function StorySloMetricsPage({ sources }: { sources: string[] }) {
  const data = useApiData<SloMetricsData>("/api/slo-metrics", { families: [] });
  const families = data.families ?? [];
  const [familyId, setFamilyId] = useState("user_experience");
  const selected = families.find((family) => family.id === familyId) ?? families[0];
  const [metricId, setMetricId] = useState<string | null>(null);
  const selectedMetric = selected?.metrics?.find((metric) => metric.id === metricId) ?? selected?.metrics?.[0];

  useEffect(() => {
    setMetricId(null);
  }, [familyId]);

  return (
    <>
      <Panel>
        <SectionTitle icon={<Target size={22} />} title="SLO & Metrics" chronology="Designed before the run">
          {data.opening_explanation ?? "SLOs convert metrics into deployability decisions."}
        </SectionTitle>
        <div className="grid gap-3 md:grid-cols-3 xl:grid-cols-7">
          {families.map((family) => (
            <button
              key={family.id}
              onClick={() => setFamilyId(family.id)}
              className={`rounded-2xl border p-4 text-left text-sm transition ${
                family.id === selected?.id ? "border-signal-cyan bg-signal-cyan/10 text-white" : "border-white/10 bg-white/[0.035] text-slate-400"
              }`}
            >
              {family.label}
            </button>
          ))}
        </div>
      </Panel>

      <div className="grid gap-5 xl:grid-cols-[360px_1fr]">
        <Panel>
          <SectionTitle icon={<SlidersHorizontal size={22} />} title={selected?.label ?? "Metric family"} chronology={selected?.chronology}>
            {selected?.explanation}
          </SectionTitle>
          <div className="space-y-2">
            {(selected?.metrics ?? []).map((metric) => (
              <button
                key={metric.id}
                onClick={() => setMetricId(metric.id)}
                className={`w-full rounded-xl border px-3 py-2 text-left text-sm ${
                  selectedMetric?.id === metric.id ? "border-signal-cyan bg-signal-cyan/10 text-white" : "border-white/10 bg-white/[0.035] text-slate-400"
                }`}
              >
                {metric.label}
              </button>
            ))}
          </div>
        </Panel>
        <Panel>
          {selectedMetric ? (
            <>
              <SectionTitle icon={<Gauge size={22} />} title={selectedMetric.label}>
                {selectedMetric.definition}
              </SectionTitle>
              <div className="grid gap-4 md:grid-cols-2">
                <InfoDisclosure title="What the user experiences">{selectedMetric.user_experience}</InfoDisclosure>
                <InfoDisclosure title="System layers that influence it">
                  {(selectedMetric.influences ?? []).join(", ")}
                </InfoDisclosure>
                <InfoDisclosure title="Common optimization strategies">
                  {(selectedMetric.common_optimizations ?? []).join(", ")}
                </InfoDisclosure>
                <InfoDisclosure title="Tradeoff">{selectedMetric.tradeoffs}</InfoDisclosure>
              </div>
              <div className="mt-5 rounded-2xl border border-white/10 bg-white/[0.035] p-4">
                <p className="text-sm font-semibold text-white">Repo target applicability</p>
                <p className="mt-2 text-sm text-slate-400">
                  {selectedMetric.targets.length
                    ? `${selectedMetric.targets.length} vertical/group target entries. Varies by vertical: ${selectedMetric.target_varies_by_vertical ? "yes" : "no"}.`
                    : "No direct configured target; shown as an educational supporting metric."}
                </p>
                <div className="mt-3 grid gap-2 md:grid-cols-2">
                  {selectedMetric.targets.slice(0, 6).map((target, index) => (
                    <div key={`${selectedMetric.id}-${index}`} className="rounded-xl bg-black/20 px-3 py-2 text-xs text-slate-300">
                      {labelize(target.vertical)} / {labelize(target.group)}: {String(target.target)}
                    </div>
                  ))}
                </div>
              </div>
            </>
          ) : null}
        </Panel>
      </div>

      <Panel>
        <SectionTitle icon={<ChevronRight size={22} />} title="SLO evaluation flow">
          The platform uses this same reasoning pattern after the measured run finishes.
        </SectionTitle>
        <div className="grid gap-3 md:grid-cols-6">
          {(data.evaluation_flow ?? []).map((step, index) => (
            <div key={step} className="rounded-2xl border border-white/10 bg-white/[0.035] p-4">
              <p className="text-xs text-slate-500">{String(index + 1).padStart(2, "0")}</p>
              <p className="mt-2 font-semibold text-white">{step}</p>
            </div>
          ))}
        </div>
      </Panel>

      <Panel>
        <SectionTitle icon={<Archive size={22} />} title="Applicability rules">
          Some SLOs apply only when the route, hardware, memory mode, or telemetry exists.
        </SectionTitle>
        <div className="grid gap-3 md:grid-cols-2">
          {(data.applicability_notes ?? []).map((note) => (
            <div key={note} className="rounded-2xl border border-white/10 bg-white/[0.035] p-4 text-sm text-slate-300">
              {note}
            </div>
          ))}
        </div>
      </Panel>
      <SourceList sources={sources} />
    </>
  );
}

export function StoryDataPage({ sources }: { sources: string[] }) {
  const explorer = useApiData<DatasetExplorerData>("/api/dataset/explorer", { verticals: [] });
  const verticals = explorer.verticals ?? [];
  const [vertical, setVertical] = useState("airline");
  const [search, setSearch] = useState("");
  const [paused, setPaused] = useState(false);
  const [caseIndex, setCaseIndex] = useState(0);
  const casePath = `/api/dataset/cases?vertical=${encodeURIComponent(vertical)}&search=${encodeURIComponent(search)}&limit=12`;
  const casesData = useApiData<DatasetCasesData>(casePath, { cases: [] });
  const cases = casesData.cases ?? [];
  const selectedCase = cases[Math.min(caseIndex, Math.max(0, cases.length - 1))];

  useEffect(() => {
    setCaseIndex(0);
  }, [casePath]);

  useEffect(() => {
    if (paused || cases.length <= 1) {
      return;
    }
    const interval = window.setInterval(() => {
      setCaseIndex((value) => (value + 1) % cases.length);
    }, 6500);
    return () => window.clearInterval(interval);
  }, [cases.length, paused]);

  return (
    <>
      <Panel>
        <SectionTitle icon={<Database size={22} />} title="Data & Workflow Explorer" chronology="Designed before the run">
          {explorer.opening_explanation}
        </SectionTitle>
        <div className="grid gap-4 md:grid-cols-5">
          {[
            ["Prompts", explorer.totals?.prompt_count ?? 10000],
            ["Gold records", explorer.totals?.gold_count ?? 10000],
            ["KB rows", explorer.totals?.kb_count ?? 4740],
            ["Verticals", explorer.totals?.vertical_count ?? 5],
            ["Per vertical", explorer.totals?.prompts_per_vertical ?? 2000]
          ].map(([label, value]) => (
            <MetricCard key={String(label)} metric={{ label: String(label), value: Number(value).toLocaleString(), tone: "neutral", detail: "Pre-run dataset artifact" }} />
          ))}
        </div>
      </Panel>

      <Panel>
        <SectionTitle icon={<FileSearch size={22} />} title="Evidence coverage">
          Research AI is 98% by design: 1,960 prompts require evidence and 40 out-of-scope prompts
          intentionally require none. There are no answerable Research AI prompts missing evidence.
        </SectionTitle>
        <div className="grid gap-3 md:grid-cols-5">
          {verticals.map((row) => (
            <button
              key={String(row.vertical)}
              onClick={() => setVertical(String(row.vertical))}
              className={`rounded-2xl border p-4 text-left transition ${
                vertical === row.vertical ? "border-signal-cyan bg-signal-cyan/10" : "border-white/10 bg-white/[0.035]"
              }`}
            >
              <p className="font-semibold text-white">{String(row.label)}</p>
              <p className="mt-1 text-sm text-slate-400">{String(row.prompt_count)} prompts</p>
              <p className="mt-3 text-lg font-semibold text-signal-green">
                {percent(row.evidence_coverage_rate)}
              </p>
            </button>
          ))}
        </div>
      </Panel>

      <div className="grid gap-5 xl:grid-cols-[420px_1fr]">
        <Panel>
          <SectionTitle icon={<SlidersHorizontal size={22} />} title="Workload pressure simulator">
            These controls explain qualitative effects. They are not exact predictions.
          </SectionTitle>
          <div className="space-y-4">
            {(explorer.pressure_simulator ?? []).map((control) => (
              <div key={String(control.control)} className="rounded-2xl border border-white/10 bg-white/[0.035] p-4">
                <p className="font-semibold capitalize text-white">{String(control.control)}</p>
                <input className="mt-3 w-full accent-cyan-300" type="range" min="0" max="100" defaultValue="45" />
                <p className="mt-2 text-sm leading-6 text-slate-400">{String(control.increase_effect)}</p>
              </div>
            ))}
          </div>
        </Panel>
        <Panel>
          <SectionTitle icon={<Activity size={22} />} title="Multi-dimensional pressure view">
            Compare input pressure, output pressure, evidence complexity, retrieval difficulty, and
            contract/safety complexity by vertical.
          </SectionTitle>
          <div className="space-y-4">
            {verticals.map((row) => {
              const dims = row.pressure_dimensions as JsonRecord | undefined;
              return (
                <div key={String(row.vertical)} className="rounded-2xl border border-white/10 bg-white/[0.035] p-4">
                  <div className="mb-3 flex items-center justify-between">
                    <p className="font-semibold text-white">{String(row.label)}</p>
                    <StatusBadge tone={String(row.cost_pressure) === "high" ? "warn" : "neutral"}>
                      cost {String(row.cost_pressure)}
                    </StatusBadge>
                  </div>
                  {Object.entries(dims ?? {}).map(([key, value]) => (
                    <ProgressBar key={key} value={numberValue(value)} label={labelize(key)} />
                  ))}
                </div>
              );
            })}
          </div>
        </Panel>
      </div>

      <Panel>
        <SectionTitle icon={<Search size={22} />} title="Prompt -> Gold -> KB -> Evaluation case viewer">
          The panels stay synchronized on one prompt ID. The viewer auto-advances through compact,
          public-safe linked examples and never loads a raw 10,000-record dropdown.
        </SectionTitle>
        <div className="mb-5 flex flex-wrap gap-3">
          <select className="rounded-xl border border-white/10 bg-graphite-900 px-3 py-2 text-sm text-white" value={vertical} onChange={(event) => setVertical(event.target.value)}>
            {verticals.map((row) => (
              <option key={String(row.vertical)} value={String(row.vertical)}>{String(row.label)}</option>
            ))}
          </select>
          <input
            className="min-w-[240px] rounded-xl border border-white/10 bg-graphite-900 px-3 py-2 text-sm text-white"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search prompt ID or text"
          />
          <button className="rounded-xl border border-white/10 px-3 py-2 text-sm text-white" onClick={() => setPaused((value) => !value)}>
            {paused ? <Play size={16} /> : <Pause size={16} />}
          </button>
          <button className="rounded-xl border border-white/10 px-3 py-2 text-sm text-white" onClick={() => setCaseIndex((value) => Math.max(0, value - 1))}>Previous</button>
          <button className="rounded-xl border border-white/10 px-3 py-2 text-sm text-white" onClick={() => setCaseIndex((value) => Math.min(cases.length - 1, value + 1))}>Next</button>
        </div>
        {selectedCase ? (
          <div className="grid gap-4 xl:grid-cols-4">
            <CasePanel title="Prompt" data={selectedCase.prompt} />
            <CasePanel title="Gold Contract" data={selectedCase.gold_contract} />
            <CasePanel title="Knowledge Base" data={selectedCase.knowledge_base.required_evidence?.[0] ?? {}} />
            <CasePanel title="Evaluation Rubric" data={selectedCase.evaluation_rubric} />
          </div>
        ) : (
          <p className="rounded-2xl border border-white/10 bg-white/[0.035] p-4 text-sm text-slate-400">
            No public-safe case matched the current filter.
          </p>
        )}
      </Panel>
      <SourceList sources={sources} />
    </>
  );
}

function CasePanel({ title, data }: { title: string; data: JsonRecord }) {
  return (
    <div className="min-h-[280px] rounded-2xl border border-white/10 bg-white/[0.035] p-4">
      <p className="mb-3 text-sm font-semibold text-white">{title}</p>
      <div className="space-y-2 text-xs leading-5 text-slate-400">
        {Object.entries(data).slice(0, 8).map(([key, value]) => (
          <div key={key}>
            <span className="text-slate-500">{labelize(key)}: </span>
            <span>{Array.isArray(value) ? value.slice(0, 4).join(", ") : String(value ?? "")}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function StoryPreparationPage({ sources }: { sources: string[] }) {
  const data = useApiData<PreparationModulesData>("/api/preparation/modules", { modules: [] });
  const modules = data.modules ?? [];
  const matrixModule = modules.find((item) => item.id === "slo_matrix");
  const matrix = (matrixModule?.matrix as { rows?: JsonRecord[]; totals?: JsonRecord; formula?: JsonRecord } | undefined) ?? {};
  const [moduleId, setModuleId] = useState("retrieval_engineering");
  const selected = modules.find((module) => module.id === moduleId) ?? modules[0];
  const [matrixFilter, setMatrixFilter] = useState("all");
  const matrixRows = (matrix.rows ?? []).filter((row) => {
    if (matrixFilter === "all") return true;
    if (matrixFilter === "api") return row.track === "API";
    if (matrixFilter === "self_hosted") return row.track === "Self-hosted";
    return row.engine === matrixFilter || row.memory_mode === matrixFilter;
  });
  return (
    <>
      <Panel>
        <SectionTitle icon={<Layers3 size={22} />} title="Inference Experiment Preparation" chronology="Designed before the run">
          The preparation page shows the engineering choices made before Main_Inference_V1: retrieval,
          context, memory modes, models, serving hardware, SLOs, and matrix construction.
        </SectionTitle>
        <div className="grid gap-3 md:grid-cols-3 xl:grid-cols-6">
          {modules.map((module) => (
            <button
              key={String(module.id)}
              onClick={() => setModuleId(String(module.id))}
              className={`rounded-2xl border p-4 text-left text-sm transition ${
                moduleId === module.id ? "border-signal-cyan bg-signal-cyan/10 text-white" : "border-white/10 bg-white/[0.035] text-slate-400"
              }`}
            >
              {String(module.title)}
            </button>
          ))}
        </div>
      </Panel>

      <Panel>
        <SectionTitle icon={<Route size={22} />} title={String(selected?.title ?? "Module")}>
          {String(selected?.purpose ?? "")}
        </SectionTitle>
        <div className="grid gap-4 xl:grid-cols-[1fr_380px]">
          <div className="rounded-2xl border border-white/10 bg-black/20 p-5">
            <p className="mb-4 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">Visual flow</p>
            <p className="text-lg leading-8 text-white">{String(selected?.visual ?? "Select a module to inspect the flow.")}</p>
            <div className="mt-5 grid gap-2 md:grid-cols-4">
              {((selected?.stages as string[] | undefined) ?? (selected?.controls as string[] | undefined) ?? []).map((stage, index) => (
                <div key={`${stage}-${index}`} className="rounded-xl bg-white/[0.035] p-3 text-sm text-slate-300">
                  <span className="text-slate-500">{String(index + 1).padStart(2, "0")}</span> {stage}
                </div>
              ))}
            </div>
          </div>
          <div className="space-y-3">
            {Object.entries(selected ?? {})
              .filter(([key]) => !["id", "title", "purpose", "visual", "stages", "controls"].includes(key))
              .slice(0, 6)
              .map(([key, value]) => (
                <InfoDisclosure key={key} title={labelize(key)}>
                  <pre className="whitespace-pre-wrap break-words font-sans text-sm">
                    {typeof value === "string" ? value : JSON.stringify(value, null, 2).slice(0, 1400)}
                  </pre>
                </InfoDisclosure>
              ))}
          </div>
        </div>
      </Panel>

      <Panel>
        <SectionTitle icon={<GitBranch size={22} />} title="Exact 25-cell matrix">
          Filter by track, engine, or memory mode. Each row represents 10,000 requests, with 2,000
          prompts per vertical per config.
        </SectionTitle>
        <div className="mb-4 flex flex-wrap gap-2">
          {["all", "self_hosted", "api", "vllm", "sglang", "api_provider_route", ...Object.keys(MAIN_MEMORY_MODE_LABELS)].map((filter) => (
            <button
              key={filter}
              onClick={() => setMatrixFilter(filter)}
              className={`rounded-xl border px-3 py-2 text-xs font-semibold uppercase tracking-[0.12em] ${
                matrixFilter === filter ? "border-signal-cyan bg-signal-cyan/10 text-white" : "border-white/10 text-slate-400"
              }`}
            >
              {labelize(filter)}
            </button>
          ))}
        </div>
        <div className="grid gap-2 md:grid-cols-5">
          {matrixRows.map((row) => (
            <div key={String(row.config_id)} className="rounded-xl border border-white/10 bg-white/[0.035] p-3">
              <p className="truncate text-xs font-mono text-signal-cyan">{String(row.config_id)}</p>
              <p className="mt-2 text-sm font-semibold text-white">{String(row.track)}</p>
              <p className="text-xs text-slate-400">{String(row.engine)} / {String(row.memory_mode)}</p>
              <p className="text-xs text-slate-400">c{String(row.concurrency)} / {Number(row.requests_per_config).toLocaleString()} requests</p>
            </div>
          ))}
        </div>
      </Panel>
      <SourceList sources={sources} />
    </>
  );
}

const MAIN_MEMORY_MODE_LABELS: Record<string, string> = {
  mm0_no_context: "MM0 no context",
  mm1_dense_top5: "MM1 dense top-5",
  mm2_hybrid_top5: "MM2 hybrid top-5",
  mm3_compressed_hybrid_top5: "MM3 compressed hybrid top-5",
  mm4_bounded_agentic: "MM4 bounded agentic"
};

export function StoryMainInferencePage({ sources }: { sources: string[] }) {
  const detail = useApiData<MainReplayDetailData>("/api/main-inference/replay-detail", {});
  const events: JsonRecord[] = detail.replay?.events?.length
    ? detail.replay.events
    : (replayFallback as JsonRecord[]);
  const telemetry = detail.telemetry?.samples ?? [];
  const [running, setRunning] = useState(false);
  const [eventIndex, setEventIndex] = useState(0);
  const [comparisonTab, setComparisonTab] = useState("engine");
  const current = events[Math.min(eventIndex, events.length - 1)] ?? {};
  const visibleEvents = eventIndex === 0 && !running ? events.slice(0, 1) : events.slice(0, eventIndex + 1);
  const latency = detail.latency_throughput?.summary ?? {};
  const quality = detail.quality_safety?.summary ?? {};
  const cost = detail.cost ?? {};
  const progressPct = Math.min(100, (numberValue(current.completed_requests) / 250000) * 100);
  const latencyChartData = [
    { metric: "TTFT", p50: numberValue(latency.p50_ttft_ms), p95: numberValue(latency.p95_ttft_ms), p99: numberValue(latency.p99_ttft_ms) },
    { metric: "TPOT", p50: numberValue(latency.p50_tpot_ms), p95: numberValue(latency.p95_tpot_ms), p99: numberValue(latency.p99_tpot_ms) },
    { metric: "E2E", p50: numberValue(latency.p50_e2e_latency_ms), p95: numberValue(latency.p95_e2e_latency_ms), p99: numberValue(latency.p99_e2e_latency_ms) }
  ];
  const qualityChartData = [
    { metric: "JSON", rate: numberValue(quality.json_valid_rate) * 100 },
    { metric: "Contract", rate: numberValue(quality.generation_contract_valid_rate) * 100 },
    { metric: "Format", rate: numberValue(quality.format_valid_rate) * 100 },
    { metric: "Evidence", rate: numberValue(quality.evidence_match_rate) * 100 },
    { metric: "Grounded", rate: numberValue(quality.grounded_rate) * 100 },
    { metric: "Safety clean", rate: (1 - numberValue(quality.safety_violation_rate)) * 100 }
  ];
  const requestSplitData = [
    { track: "Self-hosted GPU", requests: numberValue(cost.self_hosted_request_count) },
    { track: "API provider", requests: numberValue(cost.api_request_count) }
  ];
  const costSplitData = [
    { track: "A100 GPU", cost: Number(numberValue(cost.gpu_cost_usd).toFixed(2)) },
    { track: "API provider", cost: Number(numberValue(cost.api_cost_usd).toFixed(2)) }
  ];

  useEffect(() => {
    if (!running) return;
    const interval = window.setInterval(() => {
      setEventIndex((value) => {
        if (value >= events.length - 1) {
          window.clearInterval(interval);
          setRunning(false);
          return value;
        }
        return value + 1;
      });
    }, Math.max(850, 110000 / Math.max(events.length, 1)));
    return () => window.clearInterval(interval);
  }, [events.length, running]);

  const comparisonRows = ((detail.comparisons?.[comparisonTab] as JsonRecord[] | undefined) ?? []).slice(0, 8);

  return (
    <>
      <Panel>
        <div className="grid gap-6 xl:grid-cols-[1fr_420px] xl:items-center">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-signal-green">Measured during the run</p>
            <h1 className="mt-2 text-4xl font-semibold text-white md:text-6xl">{String(detail.hero?.title ?? "Main_Inference_V1")}</h1>
            <p className="mt-4 max-w-4xl text-base leading-8 text-slate-300">
              {String(detail.hero?.explanation ?? "This is a time-compressed replay of saved artifacts.")}
            </p>
            <div className="mt-6 flex flex-wrap gap-2">
              {((detail.hero?.facts as string[] | undefined) ?? []).map((fact) => (
                <span key={fact} className="rounded-full bg-white/5 px-3 py-1 text-xs text-slate-300">{fact}</span>
              ))}
            </div>
          </div>
          <div className="rounded-3xl border border-signal-cyan/20 bg-signal-cyan/8 p-5">
            <button
              className="inline-flex items-center gap-2 rounded-2xl bg-signal-cyan px-5 py-3 text-sm font-semibold text-graphite-950"
              onClick={() => {
                setEventIndex(0);
                setRunning(true);
              }}
            >
              <Play size={18} /> Run Main Inference Replay
            </button>
            <div className="mt-5">
              <ProgressBar value={progressPct} label={`${progressPct.toFixed(1)}% complete`} />
            </div>
            <div className="mt-4 grid gap-2 text-sm text-slate-300">
              <span>phase: {String(current.phase ?? "Preflight")}</span>
              <span>config: {String(current.current_config_id ?? "manifest_loaded")}</span>
              <span>engine: {String(current.engine ?? "not_started")}</span>
              <span>memory: {String(current.memory_mode ?? "not_started")}</span>
            </div>
          </div>
        </div>
      </Panel>

      <Panel>
        <SectionTitle icon={<Archive size={22} />} title="Run contract">
          The replay is tied to the manifest, not live inference.
        </SectionTitle>
        <div className="grid gap-3 md:grid-cols-4">
          {Object.entries(detail.run_contract ?? {})
            .filter(([key]) => key !== "artifact_paths")
            .slice(0, 12)
            .map(([key, value]) => (
              <div key={key} className="rounded-2xl border border-white/10 bg-white/[0.035] p-4">
                <p className="text-xs text-slate-500">{labelize(key)}</p>
                <p className="mt-1 break-words text-sm font-semibold text-white">{String(value)}</p>
              </div>
            ))}
        </div>
      </Panel>

      <div className="grid gap-5 xl:grid-cols-2">
        <Panel>
          <SectionTitle icon={<Activity size={22} />} title="Completed requests over simulated time">
            The line reveals saved progress checkpoints as the replay advances. It starts at zero
            and ends at 250,000 completed requests.
          </SectionTitle>
          <ReplayLineChart data={visibleEvents} />
        </Panel>
        <Panel>
          <SectionTitle icon={<Cpu size={22} />} title="GPU telemetry">
            Local A100 telemetry applies only to self-hosted vLLM/SGLang phases; API-provider work
            runs on provider-managed infrastructure.
          </SectionTitle>
          <TelemetryAreaChart data={telemetry} />
        </Panel>
      </div>

      <div className="grid gap-5 xl:grid-cols-[1.1fr_.9fr]">
        <Panel>
          <SectionTitle icon={<Clock size={22} />} title="Latency percentiles">
            TTFT is first-token delay, TPOT is decode cadence, and E2E is full request latency.
            Percentiles show tail behavior rather than just averages.
          </SectionTitle>
          <LatencyPercentileChart data={latencyChartData} />
        </Panel>
        <Panel>
          <SectionTitle icon={<Zap size={22} />} title="Throughput and economics">
            Throughput and cost passed, but they are not sufficient for deployability without
            grounded, safe, contract-valid answers.
          </SectionTitle>
          <div className="grid gap-4 sm:grid-cols-2">
            <MetricCard metric={{ label: "Mean E2E", value: `${numberValue(latency.mean_e2e_latency_ms).toFixed(0)} ms`, tone: "neutral", detail: "Average full request latency" }} />
            <MetricCard metric={{ label: "Tokens/sec", value: numberValue(latency.mean_total_tokens_per_second).toFixed(1), tone: "pass", detail: "Mean total token throughput" }} />
            <MetricCard metric={{ label: "Cost/request", value: `$${numberValue(cost.cost_per_request_usd).toFixed(5)}`, tone: "pass", detail: "Total cost / completed requests" }} />
            <MetricCard metric={{ label: "Cost/1k", value: `$${numberValue(cost.cost_per_1000_requests_usd).toFixed(2)}`, tone: "pass", detail: "Normalized scale cost" }} />
          </div>
        </Panel>
      </div>

      <div className="grid gap-5 xl:grid-cols-2">
        <Panel>
          <SectionTitle icon={<ShieldAlert size={22} />} title="Quality and safety breakdown">
            Parseable JSON is not the same as a contract-valid, evidence-correct, grounded, safe
            answer.
          </SectionTitle>
          <QualityRateChart data={qualityChartData} />
        </Panel>
        <Panel>
          <SectionTitle icon={<Database size={22} />} title="Request and cost split">
            The experiment measured self-hosted GPU and provider API routes through one schema but
            keeps their economics separate.
          </SectionTitle>
          <div className="grid gap-4 lg:grid-cols-2">
            <RequestSplitChart data={requestSplitData} />
            <CostSplitChart data={costSplitData} />
          </div>
        </Panel>
      </div>

      <Panel>
        <SectionTitle icon={<CheckCircle2 size={22} />} title="Deployability gate sequence">
          The system completed the workload, but deployability requires all relevant SLO gates to
          pass together.
        </SectionTitle>
        <div className="grid gap-3 md:grid-cols-4 xl:grid-cols-6">
          {(detail.deployability_gates ?? []).map((gate) => (
            <div key={String(gate.label)} className="rounded-2xl border border-white/10 bg-white/[0.035] p-4">
              <p className="text-sm text-slate-400">{String(gate.label)}</p>
              <StatusBadge tone={statusTone(gate.status)}>{String(gate.status)}</StatusBadge>
            </div>
          ))}
        </div>
        <div className="mt-5 rounded-2xl border border-rose-400/30 bg-rose-500/10 p-4 text-sm font-semibold text-rose-200">
          NOT_DEPLOYABLE_SLO_FAILURES
        </div>
        <SloChart />
      </Panel>

      <Panel>
        <SectionTitle icon={<GitBranch size={22} />} title="Comparison tabs">
          Charts come first in the UI; this compact table preserves auditability from saved CSVs.
        </SectionTitle>
        <div className="mb-4 flex flex-wrap gap-2">
          {["engine", "memory_mode", "concurrency", "api_vs_self_hosted", "model", "slo_scorecard"].map((tab) => (
            <button
              key={tab}
              onClick={() => setComparisonTab(tab)}
              className={`rounded-xl border px-3 py-2 text-xs font-semibold uppercase tracking-[0.12em] ${
                comparisonTab === tab ? "border-signal-cyan bg-signal-cyan/10 text-white" : "border-white/10 text-slate-400"
              }`}
            >
              {labelize(tab)}
            </button>
          ))}
        </div>
        <div className="overflow-x-auto rounded-2xl border border-white/10">
          <table className="w-full min-w-[900px] text-left text-xs">
            <thead className="bg-white/[0.035] text-slate-400">
              <tr>
                {Object.keys(comparisonRows[0] ?? {}).slice(0, 8).map((key) => (
                  <th key={key} className="px-3 py-2 font-semibold">{labelize(key)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {comparisonRows.map((row, index) => (
                <tr key={index} className="border-t border-white/10 text-slate-300">
                  {Object.entries(row).slice(0, 8).map(([key, value]) => (
                    <td key={key} className="px-3 py-2">{String(value).slice(0, 72)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <div className="grid gap-5 xl:grid-cols-2">
        <Panel>
          <SectionTitle icon={<Archive size={22} />} title="Operational safety and reproducibility">
            {String(detail.artifact_reliability?.message ?? "The run was designed to be auditable.")}
          </SectionTitle>
          <div className="grid gap-2 md:grid-cols-2">
            {((detail.artifact_reliability?.items as string[] | undefined) ?? []).map((item) => (
              <div key={item} className="rounded-xl bg-white/[0.035] px-3 py-2 text-sm text-slate-300">{item}</div>
            ))}
          </div>
        </Panel>
        <Panel>
          <SectionTitle icon={<Brain size={22} />} title="Engineering lessons" chronology="Learned after the run">
            Main_Inference_V1 became the measured baseline for optimization intelligence.
          </SectionTitle>
          <div className="space-y-3 text-sm leading-6 text-slate-300">
            {((detail.engineering_lessons?.taught_technically as string[] | undefined) ?? []).map((lesson) => (
              <p key={lesson} className="rounded-xl bg-white/[0.035] px-3 py-2">{lesson}</p>
            ))}
          </div>
        </Panel>
      </div>

      <SourceList sources={sources} />
    </>
  );
}

export function StoryRouteHeader({ title, purpose }: { title: string; purpose: string }) {
  return (
    <Panel>
      <div className="flex items-start gap-3">
        <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-signal-cyan/12 text-signal-cyan">
          <Sparkles size={22} />
        </div>
        <div>
          <h1 className="text-4xl font-semibold tracking-normal text-white md:text-6xl">{title}</h1>
          <p className="mt-4 max-w-4xl text-base leading-7 text-slate-300">{purpose}</p>
        </div>
      </div>
    </Panel>
  );
}

export function ChapterSourceArtifacts({ chapterId }: { chapterId: string }) {
  const chapter = chapters.find((item) => item.id === chapterId);
  return <SourceList sources={chapter?.sourceArtifacts ?? []} />;
}

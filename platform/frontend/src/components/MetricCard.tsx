import { StatusBadge } from "./StatusBadge";
import type { MetricCard as MetricCardType } from "@/lib/types";

export function MetricCard({ metric }: { metric: MetricCardType }) {
  return (
    <article className="rounded-2xl border border-white/10 bg-white/[0.045] p-5">
      <div className="mb-4 flex items-center justify-between gap-3">
        <p className="text-sm text-slate-400">{metric.label}</p>
        <StatusBadge tone={metric.tone}>{metric.tone}</StatusBadge>
      </div>
      <p className="text-4xl font-semibold tracking-normal text-white">{metric.value}</p>
      <p className="mt-3 text-sm leading-6 text-slate-400">{metric.detail}</p>
    </article>
  );
}


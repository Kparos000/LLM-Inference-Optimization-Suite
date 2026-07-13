"use client";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { sloRows, verticalRows } from "@/lib/facts";

export function VerticalPressureChart() {
  return (
    <div className="h-72">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={verticalRows}>
          <CartesianGrid stroke="rgba(148,163,184,0.12)" vertical={false} />
          <XAxis dataKey="vertical" stroke="#94a3b8" tick={{ fontSize: 11 }} />
          <YAxis stroke="#94a3b8" tick={{ fontSize: 11 }} />
          <Tooltip contentStyle={{ background: "#101825", border: "1px solid rgba(255,255,255,.12)" }} />
          <Bar dataKey="pressure" fill="#47d7ff" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function SloChart() {
  const data = sloRows.filter((row) => typeof row.observed === "number");
  return (
    <div className="h-72">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <CartesianGrid stroke="rgba(148,163,184,0.12)" vertical={false} />
          <XAxis dataKey="metric" stroke="#94a3b8" tick={{ fontSize: 10 }} interval={0} angle={-20} />
          <YAxis stroke="#94a3b8" tick={{ fontSize: 11 }} />
          <Tooltip contentStyle={{ background: "#101825", border: "1px solid rgba(255,255,255,.12)" }} />
          <Bar dataKey="observed" fill="#54e6a5" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ReplayLineChart({ data }: { data: Array<Record<string, unknown>> }) {
  return (
    <div className="h-72">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid stroke="rgba(148,163,184,0.12)" vertical={false} />
          <XAxis dataKey="compressed_second" stroke="#94a3b8" tick={{ fontSize: 11 }} />
          <YAxis stroke="#94a3b8" tick={{ fontSize: 11 }} />
          <Tooltip contentStyle={{ background: "#101825", border: "1px solid rgba(255,255,255,.12)" }} />
          <Line dataKey="completed_requests" stroke="#47d7ff" strokeWidth={3} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function TelemetryAreaChart({ data }: { data: Array<Record<string, unknown>> }) {
  return (
    <div className="h-72">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data}>
          <CartesianGrid stroke="rgba(148,163,184,0.12)" vertical={false} />
          <XAxis dataKey="timestamp" hide />
          <YAxis stroke="#94a3b8" tick={{ fontSize: 11 }} />
          <Tooltip contentStyle={{ background: "#101825", border: "1px solid rgba(255,255,255,.12)" }} />
          <Area
            dataKey="utilization_gpu_percent"
            stroke="#54e6a5"
            fill="rgba(84,230,165,.18)"
            strokeWidth={2}
          />
          <Area
            dataKey="temperature_c"
            stroke="#ffbf5f"
            fill="rgba(255,191,95,.12)"
            strokeWidth={2}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}


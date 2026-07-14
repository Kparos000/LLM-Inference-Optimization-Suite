"use client";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
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
  const data = sloRows
    .filter((row) => typeof row.observed === "number")
    .map((row) => {
      let attainment = 100;
      if (row.metric === "Safety findings") {
        attainment = row.observed === 0 ? 100 : 0;
      } else if (row.metric === "Runtime" || row.metric === "Cost") {
        attainment = row.status === "PASS" ? 100 : 0;
      } else {
        const target = Number(String(row.target).replace(/[^0-9.]/g, ""));
        attainment = target > 0 ? Math.min(120, (Number(row.observed) / target) * 100) : 0;
      }
      return {
        ...row,
        attainment: Number(attainment.toFixed(1)),
        label: row.status === "PASS" ? "PASS" : "FAIL"
      };
    });
  return (
    <div className="h-72">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <CartesianGrid stroke="rgba(148,163,184,0.12)" vertical={false} />
          <XAxis dataKey="metric" stroke="#94a3b8" tick={{ fontSize: 10 }} interval={0} angle={-20} />
          <YAxis
            stroke="#94a3b8"
            tick={{ fontSize: 11 }}
            domain={[0, 120]}
            label={{ value: "% of target", angle: -90, position: "insideLeft", fill: "#94a3b8" }}
          />
          <Tooltip
            contentStyle={{ background: "#101825", border: "1px solid rgba(255,255,255,.12)" }}
            formatter={(value) => [`${value}% of target`, "SLO attainment"]}
          />
          <ReferenceLine y={100} stroke="rgba(255,255,255,.35)" strokeDasharray="4 4" />
          <Bar dataKey="attainment" minPointSize={4} radius={[6, 6, 0, 0]}>
            {data.map((row) => (
              <Cell
                key={row.metric}
                fill={row.status === "PASS" ? "#54e6a5" : "#ff6b8a"}
              />
            ))}
          </Bar>
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

export function LatencyPercentileChart({ data }: { data: Array<Record<string, unknown>> }) {
  return (
    <div className="h-72">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <CartesianGrid stroke="rgba(148,163,184,0.12)" vertical={false} />
          <XAxis dataKey="metric" stroke="#94a3b8" tick={{ fontSize: 11 }} />
          <YAxis stroke="#94a3b8" tick={{ fontSize: 11 }} />
          <Tooltip contentStyle={{ background: "#101825", border: "1px solid rgba(255,255,255,.12)" }} />
          <Legend />
          <Bar dataKey="p50" name="p50 ms" fill="#47d7ff" radius={[5, 5, 0, 0]} />
          <Bar dataKey="p95" name="p95 ms" fill="#ffbf5f" radius={[5, 5, 0, 0]} />
          <Bar dataKey="p99" name="p99 ms" fill="#ff6b8a" radius={[5, 5, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function QualityRateChart({ data }: { data: Array<Record<string, unknown>> }) {
  return (
    <div className="h-72">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <CartesianGrid stroke="rgba(148,163,184,0.12)" vertical={false} />
          <XAxis dataKey="metric" stroke="#94a3b8" tick={{ fontSize: 10 }} interval={0} angle={-15} />
          <YAxis
            stroke="#94a3b8"
            tick={{ fontSize: 11 }}
            domain={[0, 100]}
            label={{ value: "rate %", angle: -90, position: "insideLeft", fill: "#94a3b8" }}
          />
          <Tooltip contentStyle={{ background: "#101825", border: "1px solid rgba(255,255,255,.12)" }} />
          <Bar dataKey="rate" fill="#47d7ff" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function RequestSplitChart({ data }: { data: Array<Record<string, unknown>> }) {
  return (
    <div className="h-72">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <CartesianGrid stroke="rgba(148,163,184,0.12)" vertical={false} />
          <XAxis dataKey="track" stroke="#94a3b8" tick={{ fontSize: 11 }} />
          <YAxis stroke="#94a3b8" tick={{ fontSize: 11 }} />
          <Tooltip contentStyle={{ background: "#101825", border: "1px solid rgba(255,255,255,.12)" }} />
          <Bar dataKey="requests" name="requests" fill="#54e6a5" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function CostSplitChart({ data }: { data: Array<Record<string, unknown>> }) {
  return (
    <div className="h-72">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <CartesianGrid stroke="rgba(148,163,184,0.12)" vertical={false} />
          <XAxis dataKey="track" stroke="#94a3b8" tick={{ fontSize: 11 }} />
          <YAxis stroke="#94a3b8" tick={{ fontSize: 11 }} />
          <Tooltip contentStyle={{ background: "#101825", border: "1px solid rgba(255,255,255,.12)" }} />
          <Bar dataKey="cost" name="cost USD" fill="#ffbf5f" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

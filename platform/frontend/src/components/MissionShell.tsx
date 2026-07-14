"use client";

import clsx from "clsx";
import { ArrowLeft, ArrowRight, Database, Gauge, ShieldCheck, Sparkles } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { chapters } from "@/lib/facts";
import { useExperimentSession } from "@/lib/session";
import { StatusBadge } from "./StatusBadge";

export function MissionShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { session } = useExperimentSession();
  const activeIndex = Math.max(0, chapters.findIndex((chapter) => chapter.path === pathname));
  const previous = chapters[Math.max(0, activeIndex - 1)];
  const next = chapters[Math.min(chapters.length - 1, activeIndex + 1)];

  return (
    <main className="mission-grid min-h-screen">
      <div className="mx-auto flex min-h-screen max-w-[1500px] gap-5 px-4 py-4 lg:px-6">
        <aside className="hidden w-72 shrink-0 rounded-2xl border border-white/10 bg-graphite-900/80 p-4 shadow-mission backdrop-blur-xl lg:block">
          <div className="mb-7 flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-signal-cyan/15 text-signal-cyan">
              <Sparkles size={22} />
            </div>
            <div>
              <p className="text-sm font-semibold text-white">Inference Mission</p>
              <p className="text-xs text-slate-400">No GPU replay product</p>
            </div>
          </div>
          <nav className="space-y-1">
            {chapters.map((chapter, index) => {
              const active = chapter.path === pathname;
              return (
                <Link
                  key={chapter.id}
                  href={chapter.path}
                  className={clsx(
                    "group flex items-center gap-3 rounded-xl px-3 py-3 text-sm transition",
                    active
                      ? "bg-white/10 text-white"
                      : "text-slate-400 hover:bg-white/5 hover:text-white"
                  )}
                >
                  <span
                    className={clsx(
                      "flex h-7 w-7 items-center justify-center rounded-lg border text-xs",
                      active ? "border-signal-cyan/50 text-signal-cyan" : "border-white/10"
                    )}
                  >
                    {index + 1}
                  </span>
                  <span>{chapter.shortTitle}</span>
                </Link>
              );
            })}
          </nav>
        </aside>

        <section className="flex min-w-0 flex-1 flex-col">
          <header className="mb-4 rounded-2xl border border-white/10 bg-graphite-900/70 px-4 py-3 shadow-mission backdrop-blur-xl">
            <div className="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
              <div className="flex flex-wrap items-center gap-3">
                <StatusBadge tone={session.resultType === "measured" ? "pass" : "warn"}>
                  {session.resultType === "measured" ? "artifact replay" : "pre-run design"}
                </StatusBadge>
                <span className="text-sm text-slate-300">
                  Baseline: <strong className="text-white">{session.baselineRunId}</strong>
                </span>
                <span className="text-sm text-slate-300">
                  Scenario: <strong className="text-white">{session.selectedScenarioId}</strong>
                </span>
              </div>
              <div className="flex flex-wrap gap-2 text-xs text-slate-300">
                <span className="inline-flex items-center gap-1 rounded-full bg-white/5 px-3 py-1">
                  <Database size={14} /> Main_Inference_V1 artifacts
                </span>
                <span className="inline-flex items-center gap-1 rounded-full bg-white/5 px-3 py-1">
                  <Gauge size={14} /> A100-SXM4 measured
                </span>
                <span className="inline-flex items-center gap-1 rounded-full bg-white/5 px-3 py-1">
                  <ShieldCheck size={14} /> browser-only replay
                </span>
              </div>
            </div>
            <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-white/10">
              <div
                className="h-full rounded-full bg-gradient-to-r from-signal-cyan to-signal-green"
                style={{ width: `${((activeIndex + 1) / chapters.length) * 100}%` }}
              />
            </div>
          </header>

          <div className="flex-1">{children}</div>

          <footer className="mt-4 flex items-center justify-between rounded-2xl border border-white/10 bg-graphite-900/70 p-3 backdrop-blur-xl">
            <Link
              href={previous.path}
              className={clsx(
                "inline-flex items-center gap-2 rounded-xl border border-white/10 px-4 py-2 text-sm text-slate-200",
                activeIndex === 0 && "pointer-events-none opacity-40"
              )}
            >
              <ArrowLeft size={16} /> {previous.shortTitle}
            </Link>
            <Link
              href={next.path}
              className={clsx(
                "inline-flex items-center gap-2 rounded-xl bg-signal-cyan px-4 py-2 text-sm font-semibold text-graphite-950",
                activeIndex === chapters.length - 1 && "pointer-events-none opacity-40"
              )}
            >
              {next.shortTitle} <ArrowRight size={16} />
            </Link>
          </footer>
        </section>
      </div>
    </main>
  );
}

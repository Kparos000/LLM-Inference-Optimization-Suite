import clsx from "clsx";

export function StatusBadge({
  children,
  tone = "neutral"
}: {
  children: React.ReactNode;
  tone?: "neutral" | "pass" | "fail" | "warn";
}) {
  return (
    <span
      className={clsx(
        "inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em]",
        tone === "pass" && "border-signal-green/40 bg-signal-green/10 text-signal-green",
        tone === "fail" && "border-signal-red/40 bg-signal-red/10 text-signal-red",
        tone === "warn" && "border-signal-amber/40 bg-signal-amber/10 text-signal-amber",
        tone === "neutral" && "border-white/15 bg-white/5 text-slate-200"
      )}
    >
      {children}
    </span>
  );
}


import { ReactNode } from "react";

export function Panel({ title, icon, right, children, className = "" }:
  { title?: string; icon?: ReactNode; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <div className={`rounded-xl border border-slate-800/80 bg-slate-900/50 backdrop-blur shadow-xl shadow-black/30 ${className}`}>
      {(title || right) && (
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-slate-800/80">
          <div className="flex items-center gap-2 text-slate-200">
            {icon && <span className="text-indigo-400">{icon}</span>}
            <h3 className="text-[13px] font-semibold tracking-wide">{title}</h3>
          </div>
          {right}
        </div>
      )}
      <div className="p-4">{children}</div>
    </div>
  );
}

export function Stat({ label, value, tone = "default", hint }:
  { label: string; value: ReactNode; tone?: "default" | "good" | "bad" | "warn" | "accent"; hint?: string }) {
  const tones: Record<string, string> = {
    default: "text-slate-100", good: "text-emerald-400", bad: "text-rose-400",
    warn: "text-amber-400", accent: "text-indigo-300",
  };
  return (
    <div className="rounded-lg border border-slate-800/80 bg-slate-950/50 px-3.5 py-2.5">
      <div className="text-[10px] uppercase tracking-widest text-slate-500">{label}</div>
      <div className={`text-2xl font-semibold tabular-nums ${tones[tone]}`}>{value}</div>
      {hint && <div className="text-[11px] text-slate-500 mt-0.5">{hint}</div>}
    </div>
  );
}

export function Badge({ children, tone = "slate" }: { children: ReactNode; tone?: string }) {
  const tones: Record<string, string> = {
    slate: "bg-slate-800 text-slate-300",
    green: "bg-emerald-950 text-emerald-300 ring-1 ring-emerald-800/60",
    red: "bg-rose-950 text-rose-300 ring-1 ring-rose-800/60",
    amber: "bg-amber-950 text-amber-300 ring-1 ring-amber-800/60",
    blue: "bg-sky-950 text-sky-300 ring-1 ring-sky-800/60",
    indigo: "bg-indigo-950 text-indigo-300 ring-1 ring-indigo-800/60",
  };
  return <span className={`inline-flex items-center rounded-md px-2 py-0.5 text-[11px] font-medium ${tones[tone]}`}>{children}</span>;
}

export function StatusDot({ tone = "green", live = false }: { tone?: "green" | "red" | "amber" | "slate"; live?: boolean }) {
  const c: Record<string, string> = { green: "bg-emerald-400", red: "bg-rose-400", amber: "bg-amber-400", slate: "bg-slate-500" };
  return <span className={`inline-block h-2 w-2 rounded-full ${c[tone]} ${live ? "live-dot" : ""}`} />;
}

export function Table({ columns, rows }: { columns: string[]; rows: ReactNode[][] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[13px]">
        <thead>
          <tr className="text-left text-slate-500 border-b border-slate-800">
            {columns.map((c) => <th key={c} className="py-1.5 pr-4 font-medium">{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-slate-800/40 hover:bg-slate-800/20">
              {r.map((cell, j) => <td key={j} className="py-1.5 pr-4 tabular-nums">{cell}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length === 0 && <div className="text-slate-600 text-sm py-3">no data</div>}
    </div>
  );
}

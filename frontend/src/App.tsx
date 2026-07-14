import { useState } from "react";
import {
  LayoutDashboard, HeartPulse, ListChecks, GitBranch, Database,
  ShieldAlert, Boxes, Radio, Circle,
} from "lucide-react";
import { api } from "./api";
import { usePoll } from "./usePoll";
import { StatusDot } from "./components/ui";
import Overview from "./panels/Overview";
import TrainingHealth from "./panels/TrainingHealth";

type View = "overview" | "training" | "jobs" | "policies" | "checkpoints" | "trajectories" | "recovery";
const NAV: { id: View; label: string; icon: any }[] = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "training", label: "Training Health", icon: HeartPulse },
  { id: "jobs", label: "Jobs & Queue", icon: ListChecks },
  { id: "policies", label: "Policies", icon: GitBranch },
  { id: "checkpoints", label: "Checkpoints", icon: Database },
  { id: "trajectories", label: "Trajectories", icon: Boxes },
  { id: "recovery", label: "Recovery", icon: ShieldAlert },
];

export default function App() {
  const [view, setView] = useState<View>("overview");
  const [live, setLive] = useState(true);
  const health = usePoll(api.health, 5000, true);

  return (
    <div className="flex h-screen">
      {/* sidebar */}
      <aside className="w-60 shrink-0 border-r border-slate-800/80 bg-slate-950/40 flex flex-col">
        <div className="px-4 py-4 border-b border-slate-800/80">
          <div className="text-slate-100 font-semibold tracking-tight">ShopRL Fabric</div>
          <div className="text-[11px] text-slate-500">RL post-training · mission control</div>
        </div>
        <nav className="flex-1 p-2 space-y-0.5">
          {NAV.map((n) => {
            const Icon = n.icon; const active = view === n.id;
            return (
              <button key={n.id} onClick={() => setView(n.id)}
                className={`w-full flex items-center gap-2.5 rounded-lg px-3 py-2 text-[13px] transition
                  ${active ? "bg-indigo-600/20 text-indigo-200 ring-1 ring-indigo-700/50"
                           : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/40"}`}>
                <Icon size={16} /> {n.label}
              </button>
            );
          })}
        </nav>
        <div className="p-3 border-t border-slate-800/80 text-[11px] text-slate-500">
          <div className="flex items-center gap-2">
            <StatusDot tone={health.data ? "green" : "red"} live={!!health.data} />
            {health.data ? "API connected" : "API unreachable"}
          </div>
          {health.data && <div className="mt-1 truncate text-slate-600">root: {health.data.root}</div>}
        </div>
      </aside>

      {/* main */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <header className="flex items-center justify-between px-6 py-3 border-b border-slate-800/80 bg-slate-950/30">
          <div>
            <h1 className="text-slate-100 text-lg font-semibold">{NAV.find((n) => n.id === view)?.label}</h1>
            <p className="text-[11px] text-slate-500">rollout → reward → optimize · KL / policy-version / GPU telemetry</p>
          </div>
          <button onClick={() => setLive((v) => !v)}
            className={`flex items-center gap-2 rounded-lg px-3 py-1.5 text-[12px] ring-1 transition
              ${live ? "bg-emerald-950/60 text-emerald-300 ring-emerald-800/60"
                     : "bg-slate-800/60 text-slate-400 ring-slate-700"}`}>
            {live ? <Radio size={14} className="live-dot" /> : <Circle size={14} />}
            {live ? "LIVE" : "PAUSED"}
          </button>
        </header>

        <main className="flex-1 overflow-y-auto p-6">
          {view === "overview" && <Overview live={live} />}
          {view === "training" && <TrainingHealth live={live} />}
          {!["overview", "training"].includes(view) && (
            <div className="rounded-xl border border-dashed border-slate-800 bg-slate-900/30 p-10 text-center text-slate-500">
              <div className="text-slate-300 font-medium">{NAV.find((n) => n.id === view)?.label}</div>
              <div className="text-sm mt-1">Part 2 — wiring next (jobs/scheduler controls, policies, checkpoints, trajectories, recovery + dev controls).</div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

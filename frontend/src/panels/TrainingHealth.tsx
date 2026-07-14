import { useEffect, useState } from "react";
import { LineChartIcon, MemoryStick, GitCompare, AlertTriangle } from "lucide-react";
import { api } from "../api";
import { usePoll } from "../usePoll";
import { Panel, Badge, Table } from "../components/ui";
import { LinePanel, RewardBand } from "../components/charts";

const C = { reward: "#818cf8", kl: "#38bdf8", entropy: "#34d399", grad: "#f59e0b",
            alloc: "#22d3ee", reserved: "#a78bfa", max: "#f472b6",
            rloo: "#34d399", grpo: "#f59e0b", ppo: "#f43f5e" };

export default function TrainingHealth({ live }: { live: boolean }) {
  const runs = usePoll(api.metricsRuns, 5000, live);
  const [run, setRun] = useState<string>("");
  useEffect(() => { if (!run && runs.data?.length) setRun(runs.data[0]); }, [runs.data, run]);

  const metrics = usePoll(() => (run ? api.runMetrics(run) : Promise.resolve(null)), 3000, live && !!run);
  const alerts = usePoll(() => (run ? api.runAlerts(run) : Promise.resolve(null)), 5000, live && !!run);
  const comps = usePoll(api.comparisons, 10000, live);

  const rows = (metrics.data?.metrics ?? []).map((x) => ({
    ...x, band: [x.reward_mean - (x.reward_std ?? 0), x.reward_mean + (x.reward_std ?? 0)],
  }));
  const hasGpu = rows.some((r) => "gpu_mem_reserved_gb" in r);
  const compData = buildCompareKL(comps.data ?? []);

  return (
    <div className="space-y-4">
      {/* --- comparison overlays (historical) --- */}
      <Panel title="ALGORITHM COMPARISON — KL STABILITY (historical, committed)" icon={<GitCompare size={15} />}>
        {comps.data?.length ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Table columns={["algo", "final KL", "max KL", "reward gain", "crit alerts"]}
              rows={comps.data.map((c) => [
                <span className="font-semibold uppercase" style={{ color: (C as any)[c.algorithm] }}>{c.algorithm}</span>,
                c.final_kl?.toFixed(3), c.max_kl?.toFixed(3),
                (c.reward_gain ?? 0).toFixed(4),
                c.alerts?.critical
                  ? <Badge tone="red">{c.alerts.critical}</Badge>
                  : <Badge tone="green">0</Badge>,
              ])} />
            <div>
              <div className="text-[11px] text-slate-500 mb-1">KL vs reference, per training step</div>
              <LinePanel data={compData} series={[
                { key: "rloo", color: C.rloo }, { key: "grpo", color: C.grpo }, { key: "ppo", color: C.ppo },
              ]} height={200} />
            </div>
          </div>
        ) : <div className="text-slate-600 text-sm">No comparison artifacts (comparisons/*.json).</div>}
      </Panel>

      {/* --- single run (live) --- */}
      <Panel title="SINGLE RUN — LIVE METRICS" icon={<LineChartIcon size={15} />}
        right={
          <select value={run} onChange={(e) => setRun(e.target.value)}
            className="bg-slate-800 border border-slate-700 rounded-md text-[12px] px-2 py-1 text-slate-200">
            {(runs.data ?? []).map((r) => <option key={r}>{r}</option>)}
          </select>
        }>
        {rows.length ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            <div><Label>Reward mean ±std</Label><RewardBand data={rows} /></div>
            <div><Label>KL divergence</Label><LinePanel data={rows} series={[{ key: "kl", color: C.kl }]} /></div>
            <div><Label>Policy entropy</Label><LinePanel data={rows} series={[{ key: "entropy", color: C.entropy }]} /></div>
            <div><Label>Gradient norm</Label><LinePanel data={rows} series={[{ key: "grad_norm", color: C.grad }]} /></div>
            <div className="lg:col-span-2">
              <Label><span className="inline-flex items-center gap-1.5"><MemoryStick size={13} className="text-cyan-400" />
                GPU memory (GB) — REAL, measured per step (torch.cuda)</span></Label>
              {hasGpu ? (
                <LinePanel data={rows} height={200} series={[
                  { key: "gpu_mem_allocated_gb", name: "allocated", color: C.alloc },
                  { key: "gpu_mem_reserved_gb", name: "reserved", color: C.reserved },
                  { key: "gpu_mem_max_allocated_gb", name: "peak", color: C.max },
                ]} />
              ) : <div className="text-slate-600 text-[13px] py-6">Not recorded for this run (CPU/MPS — no CUDA). Real values appear only for GPU runs.</div>}
            </div>
          </div>
        ) : <div className="text-slate-600 text-sm">Select a run with a metrics.jsonl.</div>}
      </Panel>

      {/* --- alerts --- */}
      <Panel title="ACTIVE + HISTORICAL ALERTS" icon={<AlertTriangle size={15} />}>
        {alerts.data?.n_alerts ? (
          <>
            <div className="mb-2"><Badge tone={alerts.data.max_level === "critical" ? "red" : "amber"}>
              {alerts.data.n_alerts} alert(s) · max {alerts.data.max_level}</Badge></div>
            <Table columns={["level", "rule", "message", "step"]}
              rows={alerts.data.alerts.map((a: any) => [
                <Badge tone={a.level === "critical" ? "red" : "amber"}>{a.level}</Badge>,
                a.rule, <span className="text-slate-400">{a.message}</span>, a.step,
              ])} />
          </>
        ) : <div className="text-emerald-400/80 text-sm">No active alerts for this run.</div>}
      </Panel>
    </div>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return <div className="text-[11px] text-slate-400 mb-1">{children}</div>;
}

function buildCompareKL(comps: any[]) {
  const maxLen = Math.max(0, ...comps.map((c) => (c.step_metrics ?? []).length));
  const out: any[] = [];
  for (let i = 0; i < maxLen; i++) {
    const row: any = { step: i };
    for (const c of comps) row[c.algorithm] = c.step_metrics?.[i]?.kl;
    out.push(row);
  }
  return out;
}

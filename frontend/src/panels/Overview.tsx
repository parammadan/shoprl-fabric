import { Activity, Cpu, Layers, ListChecks } from "lucide-react";
import { api } from "../api";
import { usePoll } from "../usePoll";
import { Panel, Stat, StatusDot } from "../components/ui";

const STAGE_TONE: Record<string, "default" | "good" | "bad" | "warn"> = {
  succeeded: "good", failed: "bad", dead_letter: "bad", running: "warn",
};

export default function Overview({ live }: { live: boolean }) {
  const ov = usePoll(api.overview, 3000, live);
  const sch = usePoll(api.scheduler, 3000, live);
  const o = ov.data, s = sch.data;

  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
      <Panel title="JOB STATES" icon={<ListChecks size={15} />} className="xl:col-span-2"
        right={<span className="text-[11px] text-slate-500 flex items-center gap-1.5"><StatusDot live={live} /> live</span>}>
        <div className="grid grid-cols-3 sm:grid-cols-4 gap-2.5">
          {["pending", "running", "succeeded", "retrying", "failed", "dead_letter", "paused", "cancelled"].map((st) => (
            <Stat key={st} label={st.replace("_", " ")} value={o?.job_counts?.[st] ?? 0}
                  tone={(o?.job_counts?.[st] ? STAGE_TONE[st] : undefined) ?? "default"} />
          ))}
        </div>
        {o?.reward_stats?.count ? (
          <div className="text-[12px] text-slate-500 mt-3">
            {o.reward_stats.count} trajectories · reward mean{" "}
            <span className="text-slate-300">{o.reward_stats.mean?.toFixed(3)}</span>{" "}
            (min {o.reward_stats.min?.toFixed(3)} / max {o.reward_stats.max?.toFixed(3)})
          </div>
        ) : null}
      </Panel>

      <Panel title="SCHEDULER" icon={<Cpu size={15} />}>
        <div className="grid grid-cols-2 gap-2.5">
          <Stat label="GPU slots" value={`${s?.gpu.used ?? 0}/${s?.gpu.slots ?? 0}`} tone="accent"
                hint={`${s?.gpu.queued ?? 0} queued`} />
          <Stat label="CPU workers" value={`${s?.cpu.used ?? 0}/${s?.cpu.slots ?? 0}`}
                hint={`${s?.cpu.queued ?? 0} queued`} />
          <Stat label="Running" value={`${s?.running_total ?? 0}/${s?.max_concurrent ?? 0}`} />
          <Stat label="Capacity" value={s?.at_capacity ? "FULL" : "OK"} tone={s?.at_capacity ? "warn" : "good"} />
        </div>
        <div className="mt-3 text-[11px] text-slate-500 flex items-center gap-1.5">
          <Activity size={13} className="text-indigo-400" /> admission control · priority · GPU-slot accounting
        </div>
      </Panel>
    </div>
  );
}

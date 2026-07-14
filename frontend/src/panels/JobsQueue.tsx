import { ListChecks } from "lucide-react";
import { api } from "../api";
import { usePoll } from "../usePoll";
import { Panel, Badge, Table, Stat } from "../components/ui";

const STATE_TONE: Record<string, string> = {
  running: "amber", succeeded: "green", failed: "red", dead_letter: "red",
  pending: "slate", paused: "blue", retrying: "amber", cancelled: "slate",
};

export default function JobsQueue({ live }: { live: boolean }) {
  const jobs = usePoll(api.jobs, 3000, live);
  const sch = usePoll(api.scheduler, 3000, live);
  const rows = (jobs.data ?? []).map((j) => [
    <span className="font-mono text-[12px] text-slate-400">{j.id.slice(0, 12)}</span>,
    j.kind,
    <Badge tone={STATE_TONE[j.state] || "slate"}>{j.state}</Badge>,
    j.resource, j.priority, j.attempts,
  ]);
  const s = sch.data;
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
        <Stat label="GPU" value={`${s?.gpu.used ?? 0}/${s?.gpu.slots ?? 0}`} tone="accent" hint={`${s?.gpu.queued ?? 0} queued`} />
        <Stat label="CPU workers" value={`${s?.cpu.used ?? 0}/${s?.cpu.slots ?? 0}`} hint={`${s?.cpu.queued ?? 0} queued`} />
        <Stat label="Running" value={`${s?.running_total ?? 0}/${s?.max_concurrent ?? 0}`} />
        <Stat label="Capacity" value={s?.at_capacity ? "FULL" : "OK"} tone={s?.at_capacity ? "warn" : "good"} />
      </div>
      <Panel title="JOBS" icon={<ListChecks size={15} />}
        right={<span className="text-[11px] text-slate-500">{jobs.data?.length ?? 0} jobs</span>}>
        <Table columns={["id", "kind", "state", "resource", "priority", "attempts"]} rows={rows} />
      </Panel>
    </div>
  );
}

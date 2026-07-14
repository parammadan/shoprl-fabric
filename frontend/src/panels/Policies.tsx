import { GitBranch } from "lucide-react";
import { api } from "../api";
import { usePoll } from "../usePoll";
import { Panel, Table, Stat } from "../components/ui";

export default function Policies({ live }: { live: boolean }) {
  const pols = usePoll(api.policies, 4000, live);
  const stale = usePoll(api.staleness, 5000, live);
  const s = stale.data;
  return (
    <div className="space-y-4">
      {s?.n ? (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
          <Stat label="current version" value={`v${s.current_version}`} tone="accent" />
          <Stat label="max staleness" value={s.max_staleness} tone={s.max_staleness > 0 ? "warn" : "good"} />
          <Stat label="on-policy" value={`${s.on_policy_count}/${s.n}`} tone="good" />
          <Stat label="stale" value={s.stale_count} tone={s.stale_count > 0 ? "warn" : "default"} />
        </div>
      ) : null}
      <Panel title="POLICY VERSIONS" icon={<GitBranch size={15} />}
        right={<span className="text-[11px] text-slate-500">weight-sync lifecycle</span>}>
        <Table columns={["version", "fingerprint", "created"]}
          rows={(pols.data ?? []).map((p) => [
            <span className="font-semibold text-indigo-300">v{p.version}</span>,
            <span className="font-mono text-[12px] text-slate-400">{p.fingerprint}</span>,
            new Date((p.created_at || 0) * 1000).toLocaleTimeString(),
          ])} />
        <div className="text-[11px] text-slate-500 mt-2">
          Trajectory staleness = trainer version − generating policy version (on-policy ⇒ 0).
        </div>
      </Panel>
    </div>
  );
}

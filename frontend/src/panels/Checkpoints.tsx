import { Database, ShieldCheck, ShieldX } from "lucide-react";
import { api } from "../api";
import { usePoll } from "../usePoll";
import { Panel, Table, Badge } from "../components/ui";

export default function Checkpoints({ live }: { live: boolean }) {
  const cks = usePoll(api.checkpoints, 3000, live);
  const bad = (cks.data ?? []).filter((c) => c.integrity !== "OK");
  return (
    <Panel title="CHECKPOINT REGISTRY" icon={<Database size={15} />}
      right={bad.length
        ? <Badge tone="red"><ShieldX size={12} className="mr-1" />{bad.length} corrupt</Badge>
        : <Badge tone="green"><ShieldCheck size={12} className="mr-1" />all verify OK</Badge>}>
      <Table columns={["checkpoint", "step", "reward_mean", "files", "integrity"]}
        rows={(cks.data ?? []).map((c) => [
          <span className="font-mono text-[12px] text-slate-300">{c.ckpt_id}</span>,
          c.step,
          c.reward_mean == null ? "—" : c.reward_mean.toFixed(3),
          c.n_files,
          c.integrity === "OK"
            ? <Badge tone="green">OK</Badge>
            : <Badge tone="red">{c.integrity}</Badge>,
        ])} />
      <div className="text-[11px] text-slate-500 mt-2">
        Integrity re-checked live via sha256 on every poll (atomic write · corruption detection).
      </div>
    </Panel>
  );
}

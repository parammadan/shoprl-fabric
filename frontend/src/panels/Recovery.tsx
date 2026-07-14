import { useState } from "react";
import { ShieldAlert, Skull, FileWarning, Flame } from "lucide-react";
import { api } from "../api";
import { usePoll } from "../usePoll";
import { Panel, Table, Badge } from "../components/ui";

export default function Recovery({ live }: { live: boolean }) {
  const ov = usePoll(api.overview, 3000, live);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const run = (fn: () => Promise<any>, fmt: (r: any) => string) =>
    fn().then((r) => setMsg({ ok: true, text: fmt(r) }))
        .catch((e) => setMsg({ ok: false, text: String(e) }));

  const evs = ov.data?.recovery_events ?? [];
  return (
    <div className="space-y-4">
      <Panel title="RECOVERY SCENARIOS (DEV MODE)" icon={<ShieldAlert size={15} />}>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <Control icon={<Skull size={16} />} tone="amber" title="Kill worker"
            tag="REAL recovery"
            desc="Expire a claimed job's lease → the reaper requeues it."
            onRun={() => run(api.killWorker, (r) => `reaped ${r.reaped} → ${r.resulting_state}`)} />
          <Control icon={<FileWarning size={16} />} tone="red" title="Corrupt checkpoint"
            tag="REAL detection"
            desc="Flip a byte → sha256 verify() flags it CORRUPT (see Checkpoints)."
            onRun={() => run(api.corruptCheckpoint, (r) => `${r.ckpt_id} → ${r.integrity}`)} />
          <Control icon={<Flame size={16} />} tone="red" title="Trigger OOM"
            tag="SIMULATION (laptop)"
            desc="Real OOM runs on GPU (demo_gpu_oom → shrink+restore+resume). This is the labeled fallback."
            onRun={() => run(api.oom, (r) => `${r.action}, microbatch ${r.microbatch}`)} />
        </div>
        {msg && (
          <div className={`mt-3 rounded-lg px-3 py-2 text-[13px] ${msg.ok ? "bg-emerald-950/60 text-emerald-300 ring-1 ring-emerald-800/50" : "bg-rose-950/60 text-rose-300 ring-1 ring-rose-800/50"}`}>
            {msg.text}
          </div>
        )}
      </Panel>

      <Panel title="RECOVERY EVENTS" icon={<ShieldAlert size={15} />}
        right={<span className="text-[11px] text-slate-500">{evs.length} events</span>}>
        <Table columns={["class", "action", "microbatch", "restored", "result", "trigger"]}
          rows={evs.map((e: any) => [
            <Badge tone="amber">{e.failure_class}</Badge>,
            e.action,
            e.microbatch_after != null ? `${e.microbatch_before}→${e.microbatch_after}` : "—",
            e.restored_ckpt || "—",
            e.resulting_state,
            e.simulated ? <Badge tone="slate">SIMULATED</Badge> : <Badge tone="blue">real</Badge>,
          ])} />
      </Panel>
    </div>
  );
}

function Control({ icon, title, tag, desc, tone, onRun }:
  { icon: React.ReactNode; title: string; tag: string; desc: string; tone: string; onRun: () => void }) {
  const [armed, setArmed] = useState(false);
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-3 flex flex-col gap-2">
      <div className="flex items-center gap-2 text-slate-200">
        <span className="text-slate-400">{icon}</span>
        <span className="text-[13px] font-medium">{title}</span>
        <Badge tone={tone}>{tag}</Badge>
      </div>
      <p className="text-[11px] text-slate-500 leading-snug min-h-[32px]">{desc}</p>
      <label className="flex items-center gap-2 text-[11px] text-slate-400">
        <input type="checkbox" checked={armed} onChange={(e) => setArmed(e.target.checked)}
          className="accent-indigo-500" /> confirm
      </label>
      <button disabled={!armed} onClick={() => { onRun(); setArmed(false); }}
        className={`rounded-md px-3 py-1.5 text-[12px] font-medium transition
          ${armed ? "bg-indigo-600 hover:bg-indigo-500 text-white" : "bg-slate-800 text-slate-600 cursor-not-allowed"}`}>
        Trigger
      </button>
    </div>
  );
}

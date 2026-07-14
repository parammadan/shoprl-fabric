import { useEffect, useState } from "react";
import { Boxes } from "lucide-react";
import { api } from "../api";
import { usePoll } from "../usePoll";
import { Panel, Badge } from "../components/ui";
import { BarPanel } from "../components/charts";

export default function Trajectories({ live }: { live: boolean }) {
  const list = usePoll(() => api.trajectories(300), 5000, live);
  const [sel, setSel] = useState<string>("");
  useEffect(() => { if (!sel && list.data?.length) setSel(list.data[0].id); }, [list.data, sel]);
  const [detail, setDetail] = useState<any>(null);
  useEffect(() => { if (sel) api.trajectory(sel).then(setDetail).catch(() => setDetail(null)); }, [sel]);

  const comp = detail?.reward_components
    ? Object.entries(detail.reward_components)
        .filter(([, v]) => typeof v === "number")
        .map(([k, v]) => ({ name: k.replace("quality_", ""), value: v as number }))
    : [];

  return (
    <Panel title="TRAJECTORY EXPLORER" icon={<Boxes size={15} />}
      right={
        <select value={sel} onChange={(e) => setSel(e.target.value)}
          className="bg-slate-800 border border-slate-700 rounded-md text-[12px] px-2 py-1 text-slate-200 max-w-[280px]">
          {(list.data ?? []).map((t) => (
            <option key={t.id} value={t.id}>
              {t.id.slice(0, 8)} · {t.policy_id} · r={t.reward?.toFixed?.(3) ?? "—"}
            </option>
          ))}
        </select>
      }>
      {detail ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <div className="space-y-3">
            <Field label="Prompt" value={detail.prompt} mono />
            <Field label="Response" value={detail.response} mono />
          </div>
          <div className="space-y-3">
            <div className="flex gap-2 flex-wrap">
              <Badge tone="indigo">policy {detail.policy_id}</Badge>
              <Badge tone="slate">reward {detail.total_reward?.toFixed?.(3) ?? "—"}</Badge>
              <Badge tone="blue">advantage {detail.advantage == null ? "N/A" : detail.advantage.toFixed(3)}</Badge>
              <Badge tone="slate">KL N/A (per-step, not per-trajectory)</Badge>
            </div>
            {comp.length > 0 && (
              <div>
                <div className="text-[11px] text-slate-400 mb-1">Reward components (real)</div>
                <BarPanel data={comp} xKey="name" bars={[{ key: "value", color: "#818cf8" }]} height={180} />
              </div>
            )}
            <div className="text-[11px] text-slate-500">
              lineage: job {String(detail.job_id).slice(0, 8)} · prompt {detail.prompt_id} · parent {detail.parent_id ?? "—"}
            </div>
          </div>
        </div>
      ) : <div className="text-slate-600 text-sm">No trajectories yet.</div>}
    </Panel>
  );
}

function Field({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <div className="text-[11px] text-slate-400 mb-1">{label}</div>
      <div className={`rounded-lg border border-slate-800 bg-slate-950/50 px-3 py-2 text-[13px] text-slate-300 whitespace-pre-wrap max-h-40 overflow-y-auto ${mono ? "font-mono" : ""}`}>
        {value || "—"}
      </div>
    </div>
  );
}

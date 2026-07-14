// Typed client over the EXISTING FastAPI backend (via the Vite /api proxy).
// The frontend never touches the platform stores — only these HTTP endpoints.
const j = <T,>(p: string): Promise<T> =>
  fetch(`/api${p}`).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status} ${p}`))));
const jOpt = <T,>(p: string): Promise<T | null> =>
  fetch(`/api${p}`).then((r) => (r.status === 404 ? null : r.ok ? r.json() : Promise.reject(new Error(`${r.status} ${p}`))));
const post = <T,>(p: string): Promise<T> =>
  fetch(`/api${p}`, { method: "POST" }).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status} ${p}`))));

export interface Scheduler {
  gpu: { used: number; slots: number; queued: number };
  cpu: { used: number; slots: number; queued: number };
  running_total: number; max_concurrent: number; at_capacity: boolean;
}
export interface Overview {
  job_counts: Record<string, number>;
  reward_stats: { count?: number; min?: number; mean?: number; max?: number };
  reward_by_policy: [string, number, number][];
  checkpoints: { ckpt_id: string; step: number; reward_mean: number | null; n_files: number; integrity: string }[];
  recovery_events: any[];
}
export type Metric = Record<string, number>;

export const api = {
  health: () => j<{ status: string; root: string }>("/health"),
  overview: () => j<Overview>("/overview"),
  scheduler: () => j<Scheduler>("/scheduler"),
  jobs: () => j<any[]>("/jobs"),
  policies: () => j<any[]>("/policies"),
  staleness: () => jOpt<any>("/policies/staleness"),
  checkpoints: () => j<any[]>("/checkpoints"),
  metricsRuns: () => j<string[]>("/metrics-runs"),
  runMetrics: (id: string) => jOpt<{ run_id: string; n_steps: number; metrics: Metric[] }>(`/runs/${id}/metrics`),
  runAlerts: (id: string) => jOpt<any>(`/runs/${id}/alerts`),
  comparisons: () => j<any[]>("/comparisons"),
  trajectories: (limit = 200) => j<any[]>(`/trajectories?limit=${limit}`),
  trajectory: (id: string) => j<any>(`/trajectories/${id}`),
  // dev-mode recovery
  killWorker: () => post<any>("/dev/kill-worker"),
  corruptCheckpoint: () => post<any>("/dev/corrupt-checkpoint"),
  oom: () => post<any>("/dev/oom"),
};

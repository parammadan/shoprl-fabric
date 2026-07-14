import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend,
} from "recharts";

const AXIS = { stroke: "#475569", fontSize: 11 } as const;
const GRID = "#16202e";
const TIP = {
  contentStyle: { background: "#0b1220", border: "1px solid #334155", borderRadius: 8, fontSize: 12 },
  labelStyle: { color: "#94a3b8" },
} as const;

export interface Series { key: string; name?: string; color: string; }

export function LinePanel({ data, series, xKey = "step", height = 190 }:
  { data: any[]; series: Series[]; xKey?: string; height?: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 6, right: 14, left: -10, bottom: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey={xKey} {...AXIS} />
        <YAxis {...AXIS} width={46} />
        <Tooltip {...TIP} />
        {series.length > 1 && <Legend wrapperStyle={{ fontSize: 11 }} />}
        {series.map((s) => (
          <Line key={s.key} type="monotone" dataKey={s.key} name={s.name || s.key}
                stroke={s.color} strokeWidth={2} dot={false} isAnimationActive={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function RewardBand({ data, height = 210 }: { data: any[]; height?: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 6, right: 14, left: -10, bottom: 0 }}>
        <defs>
          <linearGradient id="band" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#818cf8" stopOpacity={0.28} />
            <stop offset="100%" stopColor="#818cf8" stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="step" {...AXIS} />
        <YAxis {...AXIS} width={46} />
        <Tooltip {...TIP} />
        <Area type="monotone" dataKey="band" stroke="none" fill="url(#band)" isAnimationActive={false} name="±std" />
        <Line type="monotone" dataKey="reward_mean" stroke="#818cf8" strokeWidth={2} dot={false} isAnimationActive={false} name="reward" />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function BarPanel({ data, xKey, bars, height = 210 }:
  { data: any[]; xKey: string; bars: Series[]; height?: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 6, right: 14, left: -10, bottom: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey={xKey} {...AXIS} />
        <YAxis {...AXIS} width={46} />
        <Tooltip {...TIP} />
        {bars.length > 1 && <Legend wrapperStyle={{ fontSize: 11 }} />}
        {bars.map((b) => <Bar key={b.key} dataKey={b.key} name={b.name || b.key} fill={b.color} radius={[3, 3, 0, 0]} />)}
      </BarChart>
    </ResponsiveContainer>
  );
}

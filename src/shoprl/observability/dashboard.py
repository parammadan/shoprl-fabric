"""Render a self-contained HTML dashboard from a training metrics log.

    python -m shoprl.observability.dashboard --metrics runs/<exp>/metrics.jsonl \
        --out runs/<exp>/dashboard.html

Reads the JSONL the trainer appends per step and draws six panels — reward
(mean ± std band), reward-by-component, KL, entropy, clip fraction, grad norm —
so the first AWS run is watchable. Output is one HTML file with inline SVG and
no external dependencies (open it locally, or scp it off the GPU box).

Design follows the data-viz method: line charts (change-over-time = step);
single-series panels use one hue (title names the series, no legend); the
component panel uses the fixed-order categorical palette with a legend AND
direct end-labels (the relief rule — two of those hues are sub-3:1 on the light
surface, so identity is never carried by color alone). Palette validated with
the skill's validator (worst adjacent CVD ΔE 24.2).
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

# Fixed-order categorical slots 1-5 (light / dark) from the validated palette.
_CATEGORICAL = [
    ("budget", "#2a78d6", "#3987e5"),
    ("groundedness", "#1baf7a", "#199e70"),
    ("coverage", "#eda100", "#c98500"),
    ("format", "#008300", "#008300"),
    ("comparison", "#4a3aa7", "#9085e9"),
]
_COMPONENT_KEYS = {
    "budget": "reward_budget",
    "groundedness": "reward_groundedness",
    "coverage": "reward_coverage",
    "format": "reward_quality_format",
    "comparison": "reward_quality_comparison",
}
# Single-series panels: (metric key, title).
_SINGLE = [
    ("kl", "KL vs reference"),
    ("entropy", "Policy entropy"),
    ("clip_frac", "Clip fraction"),
    ("grad_norm", "Grad norm"),
]

W, H = 380, 170
PADL, PADR, PADT, PADB = 46, 18, 10, 22
PADR_COMP = 92  # extra right margin for direct end-labels in the component panel


def load_metrics(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).open() if l.strip()]


def _scale(values: list[float]):
    lo, hi = min(values), max(values)
    if hi == lo:  # constant series -> pad so the line sits mid-panel
        pad = abs(hi) * 0.1 or 0.5
        lo, hi = lo - pad, hi + pad
    else:
        margin = (hi - lo) * 0.08
        lo, hi = lo - margin, hi + margin
    return lo, hi


def _xy(steps, i, v, lo, hi, padr):
    n = len(steps)
    x = PADL if n == 1 else PADL + (W - PADL - padr) * (i / (n - 1))
    y = PADT + (H - PADT - PADB) * (1 - (v - lo) / (hi - lo))
    return x, y


def _panel_svg(pid, steps, series, padr=PADR, band=None):
    """series: list of (label, light_color, dark_color, values). band: (lo[],hi[])."""
    all_vals = [v for _, _, _, vs in series for v in vs]
    if band:
        all_vals += band[0] + band[1]
    lo, hi = _scale(all_vals)

    parts = [f'<svg class="plot" viewBox="0 0 {W} {H}" data-pid="{pid}" '
             f'preserveAspectRatio="none" role="img">']
    # gridlines + y labels (3 rows)
    for f in (0.0, 0.5, 1.0):
        yv = lo + (hi - lo) * (1 - f)
        y = PADT + (H - PADT - PADB) * f
        parts.append(f'<line x1="{PADL}" y1="{y:.1f}" x2="{W-padr}" y2="{y:.1f}" '
                     f'class="grid"/>')
        parts.append(f'<text x="{PADL-6}" y="{y+3:.1f}" class="ylab">{yv:.2f}</text>')
    # baseline (x axis) + x labels
    yb = H - PADB
    parts.append(f'<line x1="{PADL}" y1="{yb}" x2="{W-padr}" y2="{yb}" class="axis"/>')
    parts.append(f'<text x="{PADL}" y="{H-6}" class="xlab">step {steps[0]}</text>')
    parts.append(f'<text x="{W-padr}" y="{H-6}" class="xlab" text-anchor="end">'
                 f'step {steps[-1]}</text>')

    # optional ±std band (reward panel), drawn under the line
    if band:
        top = " ".join(f"{_xy(steps,i,band[1][i],lo,hi,padr)[0]:.1f},"
                       f"{_xy(steps,i,band[1][i],lo,hi,padr)[1]:.1f}"
                       for i in range(len(steps)))
        bot = " ".join(f"{_xy(steps,i,band[0][i],lo,hi,padr)[0]:.1f},"
                       f"{_xy(steps,i,band[0][i],lo,hi,padr)[1]:.1f}"
                       for i in range(len(steps) - 1, -1, -1))
        parts.append(f'<polygon points="{top} {bot}" class="band"/>')

    for si, (label, lc, dc, vs) in enumerate(series):
        pts = [_xy(steps, i, v, lo, hi, padr) for i, v in enumerate(vs)]
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        parts.append(f'<polyline points="{d}" class="line" '
                     f'style="stroke:{lc}" data-dc="{dc}"/>')
        if len(pts) == 1:  # single point -> a dot so it's visible
            parts.append(f'<circle cx="{pts[0][0]:.1f}" cy="{pts[0][1]:.1f}" r="3.5" '
                         f'style="fill:{lc}" data-dc="{dc}"/>')
        if len(series) > 1:  # direct end-label (relief rule)
            ex, ey = pts[-1]
            parts.append(f'<text x="{ex+5:.1f}" y="{ey+3:.1f}" class="endlab" '
                         f'style="fill:{lc}" data-dc="{dc}">{html.escape(label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def render_dashboard(metrics: list[dict], out_path: str | Path,
                     title: str = "ShopRL Fabric — training run") -> Path:
    steps = [m.get("step", i) for i, m in enumerate(metrics)]

    def col(key):
        return [float(m.get(key, 0.0)) for m in metrics]

    panels = []  # (title, svg, legend_html)
    # 1. Reward (mean ± std band)
    mean, std = col("reward_mean"), col("reward_std")
    band = ([mean[i] - std[i] for i in range(len(mean))],
            [mean[i] + std[i] for i in range(len(mean))])
    panels.append((
        "Reward (mean ± std)",
        _panel_svg("reward", steps, [("reward", "#2a78d6", "#3987e5", mean)], band=band),
        "",
    ))
    # 2. Reward by component (categorical, legend + end-labels)
    comp_series = [(lbl, lc, dc, col(_COMPONENT_KEYS[lbl]))
                   for lbl, lc, dc in _CATEGORICAL]
    legend = "".join(
        f'<span class="lg"><i style="background:{lc}" data-dc="{dc}"></i>'
        f'{html.escape(lbl)}</span>' for lbl, lc, dc in _CATEGORICAL
    )
    panels.append((
        "Reward by component",
        _panel_svg("components", steps, comp_series, padr=PADR_COMP),
        legend,
    ))
    # 3-6. single-series metrics
    for key, ttl in _SINGLE:
        panels.append((
            ttl, _panel_svg(key, steps, [(key, "#2a78d6", "#3987e5", col(key))]), "",
        ))

    # embed per-panel data for the JS hover layer
    hover_data = {}
    hover_data["reward"] = {"steps": steps,
                            "series": [{"label": "reward", "values": mean}]}
    hover_data["components"] = {"steps": steps,
                                "series": [{"label": l, "values": col(_COMPONENT_KEYS[l])}
                                           for l, _, _ in _CATEGORICAL]}
    for key, _ in _SINGLE:
        hover_data[key] = {"steps": steps, "series": [{"label": key, "values": col(key)}]}

    cards = "".join(
        f'<figure class="card"><figcaption>{html.escape(t)}</figcaption>'
        f'{lg and f"<div class=legend>{lg}</div>"}{svg}</figure>'
        for t, svg, lg in panels
    )

    doc = _TEMPLATE.format(
        title=html.escape(title),
        n_steps=len(steps),
        cards=cards,
        data=json.dumps(hover_data),
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    return out


_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  --surface:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,.10);
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --surface:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink2:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,.10); }}
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--plane); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif; }}
header {{ padding:20px 24px 6px; }}
h1 {{ font-size:17px; margin:0; }}
.sub {{ color:var(--ink2); font-size:13px; margin-top:2px; }}
.grid-wrap {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
  gap:14px; padding:16px 24px 28px; }}
.card {{ background:var(--surface); border:1px solid var(--ring); border-radius:10px;
  margin:0; padding:12px 12px 6px; }}
figcaption {{ font-size:13px; color:var(--ink2); margin-bottom:6px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:10px; margin-bottom:4px; }}
.lg {{ font-size:11px; color:var(--ink2); display:flex; align-items:center; gap:4px; }}
.lg i {{ width:10px; height:10px; border-radius:2px; display:inline-block; }}
.plot {{ width:100%; height:170px; display:block; overflow:visible; }}
.grid {{ stroke:var(--grid); stroke-width:1; }}
.axis {{ stroke:var(--axis); stroke-width:1; }}
.line {{ fill:none; stroke-width:2; stroke-linejoin:round; stroke-linecap:round; }}
.band {{ fill:#2a78d6; opacity:.12; stroke:none; }}
.ylab {{ fill:var(--muted); font-size:9px; text-anchor:end; font-variant-numeric:tabular-nums; }}
.xlab {{ fill:var(--muted); font-size:9px; }}
.endlab {{ font-size:9px; font-weight:600; }}
#tip {{ position:fixed; pointer-events:none; background:var(--surface);
  border:1px solid var(--ring); border-radius:6px; padding:6px 8px; font-size:11px;
  color:var(--ink); box-shadow:0 2px 8px rgba(0,0,0,.15); opacity:0; z-index:9; }}
</style></head>
<body class="viz-root">
<header><h1>{title}</h1><div class="sub">{n_steps} steps · reward should trend up · KL small &amp; finite · entropy not collapsing</div></header>
<div class="grid-wrap">{cards}</div>
<div id="tip"></div>
<script>
// dark-mode: swap each mark's stroke/fill to its dark step
if (matchMedia('(prefers-color-scheme: dark)').matches) {{
  document.querySelectorAll('[data-dc]').forEach(function(el){{
    var dc = el.getAttribute('data-dc');
    if (el.tagName === 'text' || el.tagName === 'polyline') el.style.stroke && (el.style.stroke = dc);
    if (el.tagName === 'text') el.style.fill = dc;
    if (el.tagName === 'circle' || el.tagName === 'i' || el.tagName==='I') el.style.background ? el.style.background = dc : el.style.fill = dc;
    if (el.tagName === 'polyline') el.style.stroke = dc;
  }});
}}
var DATA = {data};
var tip = document.getElementById('tip');
document.querySelectorAll('.plot').forEach(function(svg){{
  var pid = svg.getAttribute('data-pid'); var d = DATA[pid]; if(!d) return;
  svg.addEventListener('mousemove', function(e){{
    var r = svg.getBoundingClientRect();
    var frac = (e.clientX - r.left) / r.width;
    var i = Math.round(frac * (d.steps.length - 1));
    i = Math.max(0, Math.min(d.steps.length - 1, i));
    var rows = d.series.map(function(s){{
      return s.label + ': ' + Number(s.values[i]).toFixed(3);
    }}).join('<br>');
    tip.innerHTML = '<b>step ' + d.steps[i] + '</b><br>' + rows;
    tip.style.left = (e.clientX + 12) + 'px';
    tip.style.top = (e.clientY + 12) + 'px';
    tip.style.opacity = 1;
  }});
  svg.addEventListener('mouseleave', function(){{ tip.style.opacity = 0; }});
}});
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(prog="shoprl.observability.dashboard")
    ap.add_argument("--metrics", required=True, help="path to metrics.jsonl")
    ap.add_argument("--out", default=None, help="output html (default: alongside metrics)")
    ap.add_argument("--title", default="ShopRL Fabric — training run")
    args = ap.parse_args()
    metrics = load_metrics(args.metrics)
    if not metrics:
        raise SystemExit(f"no metrics in {args.metrics}")
    out = args.out or str(Path(args.metrics).with_name("dashboard.html"))
    path = render_dashboard(metrics, out, title=args.title)
    print(f"[dashboard] {len(metrics)} steps -> {path}")


if __name__ == "__main__":
    main()

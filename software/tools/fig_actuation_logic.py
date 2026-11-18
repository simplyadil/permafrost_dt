"""Paper figure: at one decision time, measured r_max forks into the two branches.

Black is what the monitor has measured up to the decision. From there the two
lookahead branches run 60 min ahead: keep heating and stop now. The annotation is
the action the service logged at that decision.

Standalone: reads only the cached predictive run and needs matplotlib. Nothing is
written back to the cache or to InfluxDB.

Run:

    python software/tools/fig_actuation_logic.py
    python software/tools/fig_actuation_logic.py --at 4.67 --out artifacts/paper/fig_actuation_fork_467
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R_LIM_MM = 40.0
R_WARN_MM = 24.0
HISTORY_H = 1.0

MEASURED = "#000000"
KEEP = "#D55E00"
STOP = "#009E73"
INK_MUTED = "#555555"

ACTION_LABELS = {"heat": "heat at 10 °C", "reduce": "reduce to 5 °C", "stop": "stop, 0 °C"}


def plot(rows: list[dict], traces: list[dict], at_hours: float, out_stem: Path) -> None:
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    index = min(range(len(traces)), key=lambda i: abs(traces[i]["t"] - at_hours))
    trace = traces[index]
    row = next(r for r in rows if abs(float(r["time_hours"]) - trace["t"]) < 1e-3)
    t0 = float(trace["t"])
    r0 = row["r_max_measured_m"] * 1000.0

    history = [(float(r["time_hours"]), r["r_max_measured_m"] * 1000.0) for r in rows
               if r["r_max_measured_m"] is not None and t0 - HISTORY_H <= float(r["time_hours"]) <= t0]
    times = [t0, *trace["times"]]
    keep = [r0, *(v * 1000.0 for v in trace["keep"])]
    stop = [r0, *(v * 1000.0 for v in trace["stop"])]

    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    ax.axhline(R_LIM_MM, color=INK_MUTED, lw=0.9, ls="--", zorder=1)
    ax.axhline(R_WARN_MM, color=INK_MUTED, lw=0.8, ls=":", zorder=1)
    ax.axvline(t0, color=INK_MUTED, lw=0.8, zorder=1)
    ax.annotate("$r_{lim}$ 40 mm", xy=(0.01, R_LIM_MM), xycoords=("axes fraction", "data"), xytext=(0, 2),
                textcoords="offset points", fontsize=7.5, color=INK_MUTED)
    ax.annotate("warning 24 mm", xy=(0.01, R_WARN_MM), xycoords=("axes fraction", "data"), xytext=(0, 2),
                textcoords="offset points", fontsize=7.5, color=INK_MUTED)

    ax.plot(*zip(*history), color=MEASURED, lw=1.8, zorder=4, label="monitored (measured) $r_{max}$")
    ax.plot(times, keep, "-o", color=KEEP, lw=1.8, ms=3, zorder=3, label="keep heating branch")
    ax.plot(times, stop, "-s", color=STOP, lw=1.8, ms=3, zorder=3, label="stop now branch")
    ax.plot([t0], [r0], "o", color=MEASURED, ms=6, zorder=5)

    for values in (keep, stop):
        ax.annotate(f"{values[-1]:.1f} mm", xy=(times[-1], values[-1]), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=7.5)
    action = ACTION_LABELS.get(row["action"], str(row["action"]))
    ax.annotate(f"decision at t = {t0:.2f} h\naction: {action}", xy=(t0, 0.97),
                xycoords=("data", "axes fraction"), xytext=(-5, 0), textcoords="offset points",
                ha="right", va="top", fontsize=8)

    ax.set_xlim(t0 - HISTORY_H, t0 + 1.12)
    ax.set_ylim(15, 50)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Thaw radius (mm)")
    ax.legend(loc="lower right", frameon=False)

    for suffix in (".pdf", ".png"):
        fig.savefig(out_stem.with_suffix(suffix), dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--traces", default="artifacts/predictive_control_dtwin/lookahead_traces.json")
    parser.add_argument("--at", type=float, default=3.97, help="Decision time in hours.")
    parser.add_argument("--out", default="artifacts/paper/fig_actuation_logic")
    args = parser.parse_args()

    cache = json.loads(Path(args.traces).read_text(encoding="utf-8"))
    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    plot(cache["rows"], cache["traces"], args.at, out_stem)
    print(f"Saved {out_stem.with_suffix('.pdf')} and {out_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()

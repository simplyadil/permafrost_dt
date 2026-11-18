"""Paper figure: measured r_max and the two lookahead branches at three decisions.

Standalone: reads only the cached predictive run and needs numpy + matplotlib.
Also recomputes the predicate firing times from that cache and prints them
against the paper's values. Nothing is written back to the cache or to InfluxDB.

Run:

    python software/tools/fig_branch_forecasts.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R_LIM_MM = 40.0
R_WARN_MM = 24.0
R_RELEASE_MM = 20.0
CONFIRM_UPDATES = 3
DECISION_TIMES_H = (1.22, 3.97, 4.67)
PAPER_TIMES_H = {"M_warn": 3.35, "F_heat": 3.97, "F_stop": 4.67, "M_lim": 4.72}

MEASURED = "#000000"
KEEP = "#D55E00"
STOP = "#009E73"
INK_MUTED = "#555555"


def recompute_predicates(rows: list[dict], traces: list[dict]) -> dict[str, float | None]:
    """First firing times; forecast predicates need CONFIRM_UPDATES in a row."""

    def first_confirmed(branch: str) -> float | None:
        streak = 0
        for trace in traces:
            streak = streak + 1 if max(trace[branch]) * 1000.0 >= R_LIM_MM else 0
            if streak >= CONFIRM_UPDATES:
                return float(trace["t"])
        return None

    def first_measured(limit_mm: float) -> float | None:
        for row in rows:
            radius = row["r_max_measured_m"]
            if radius is not None and radius * 1000.0 >= limit_mm:
                return float(row["time_hours"])
        return None

    return {
        "M_warn": first_measured(R_WARN_MM),
        "F_heat": first_confirmed("keep"),
        "F_stop": first_confirmed("stop"),
        "M_lim": first_measured(R_LIM_MM),
    }


def plot(rows: list[dict], traces: list[dict], out_stem: Path) -> None:
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    t = np.array([float(r["time_hours"]) for r in rows])
    r_mm = np.array([np.nan if r["r_max_measured_m"] is None else r["r_max_measured_m"] * 1000.0
                     for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True, constrained_layout=True)
    for ax, wanted in zip(axes, DECISION_TIMES_H):
        trace = min(traces, key=lambda tr: abs(tr["t"] - wanted))
        t0 = float(trace["t"])
        x_min, x_max = t0 - 0.5, t0 + 1.05

        for value in (R_LIM_MM, R_WARN_MM, R_RELEASE_MM):
            ax.axhline(value, color=INK_MUTED, lw=0.8, ls="--" if value == R_LIM_MM else ":", zorder=1)
        ax.axvline(t0, color=INK_MUTED, lw=0.8, zorder=1)

        mask = (t >= x_min) & (t <= x_max)
        ax.plot(t[mask], r_mm[mask], color=MEASURED, lw=1.5, zorder=4, label="measured $r_{max}$")
        times = np.array(trace["times"], dtype=float)
        ax.plot(times, np.array(trace["keep"]) * 1000.0, "-o", color=KEEP, lw=1.5, ms=3,
                zorder=3, label="keep heating ($F_{heat}$ branch)")
        ax.plot(times, np.array(trace["stop"]) * 1000.0, "-s", color=STOP, lw=1.5, ms=3,
                zorder=3, label="stop now ($F_{stop}$ branch)")

        ax.set_title(f"t = {t0:.2f} h", fontsize=8.5)
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel("Time (h)")
        ax.set_ylim(0, 60)
    axes[0].set_ylabel("Thaw radius (mm)")

    # The y-axis is shared, so the lines are labelled once, in the clear panel.
    for value, label, x_pos, align in ((R_LIM_MM, "$r_{lim}$ 40 mm", 0.02, "left"),
                                       (R_WARN_MM, "warning 24 mm", 0.98, "right"),
                                       (R_RELEASE_MM, "release 20 mm", 0.98, "right")):
        axes[1].annotate(label, xy=(x_pos, value), xycoords=("axes fraction", "data"), xytext=(0, 2),
                         textcoords="offset points", ha=align, va="bottom", fontsize=6.5,
                         color=INK_MUTED)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    for suffix in (".pdf", ".png"):
        fig.savefig(out_stem.with_suffix(suffix), dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--traces", default="artifacts/predictive_control_dtwin/lookahead_traces.json")
    parser.add_argument("--out", default="artifacts/paper/fig_branch_forecasts")
    args = parser.parse_args()

    cache = json.loads(Path(args.traces).read_text(encoding="utf-8"))
    rows, traces = cache["rows"], cache["traces"]

    print(f"{'predicate':<10}{'recomputed (h)':>16}{'paper (h)':>12}{'diff (min)':>12}")
    for name, value in recompute_predicates(rows, traces).items():
        paper = PAPER_TIMES_H[name]
        if value is None:
            print(f"{name:<10}{'never':>16}{paper:>12.2f}{'-':>12}")
        else:
            print(f"{name:<10}{value:>16.3f}{paper:>12.2f}{(value - paper) * 60.0:>12.1f}")

    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    plot(rows, traces, out_stem)
    print(f"Saved {out_stem.with_suffix('.pdf')} and {out_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()

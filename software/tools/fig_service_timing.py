"""What one digital-twin update costs, stage by stage, in milliseconds.

Replaces the diagnostic ``04_computation_time.png``, which had two problems that
made it unreadable rather than merely plain. Its stacked area was drawn on a
linear axis spanning a 27 s cold-start transient and a ~1 s steady state, so
99% of the run was a sliver along the bottom; and its legend sat on top of the
cadence line it was supposed to be read against.

Both are fixed by choosing the axis to match the data rather than the other way
round. Stage costs span four decades -- 0.12 ms for the heater command, 812 ms
for the boundary build -- so the bars are drawn on a log axis, where that range
is legible instead of being three invisible bars next to one long one.

Two panels, two different questions:

  (a) what each stage costs, as median and 95th percentile
  (b) whether the whole update fits inside the sensor cadence, over the run

No categorical colour is used or needed: every stage is named on the axis, so
hue would be redundant encoding. The one colour split that carries meaning is
on-critical-path (the six pipeline stages) against off-path (the inversion,
which runs on its own cadence in parallel and never delays an update).

The inversion bar is read from an ``inversion_updates.csv`` when one is given,
because it is the one service whose cost does not appear in ``dt_updates.csv``
at all -- by construction, since it is not part of the per-update path.

Run:

    python -m software.tools.fig_service_timing
    python -m software.tools.fig_service_timing --updates artifacts/dt_frontres/dt_updates.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

# Stage order is the pipeline order, kept fixed so the figure reads the same way
# across runs; the bars are sorted by cost, the table below them is not.
STAGES = [
    ("ms_thaw_front", "thaw front"),
    ("ms_boundary", "boundary build"),
    ("ms_forecast", "FEM forecast"),
    ("ms_lookahead", "lookahead (2 branches)"),
    ("ms_safety", "safety monitor"),
    ("ms_heater", "heater command"),
]


def _column(rows: list[dict], key: str) -> np.ndarray:
    out = []
    for row in rows:
        value = (row.get(key) or "").strip()
        if value:
            try:
                out.append(float(value))
            except ValueError:
                pass
    return np.array(out, dtype=float)


def warmup_index(rows: list[dict]) -> int | None:
    """The one update that builds the mesh and its geometry cache.

    Identified from the data rather than assumed to be the first row -- on the
    April 8 replay it is the fifth. It is the single update whose boundary build
    runs an order of magnitude past every other one, and including it would set
    the worst case for the whole pipeline from an event paid once per service
    lifetime rather than from steady-state behaviour.
    """
    values = _column(rows, "ms_boundary")
    if values.size < 20:
        return None
    candidate = int(np.argmax(values))
    rest = np.delete(values, candidate)
    return candidate if values[candidate] > 5.0 * np.percentile(rest, 99.9) else None


def stage_stats(rows: list[dict], *, drop: int | None = None) -> list[dict]:
    """Median / p95 / max per stage, plus the duty cycle of intermittent ones.

    An intermittent stage is summarised over the updates where it actually ran.
    The lookahead fires every few updates, so its median across all rows is 0 ms
    and its mean is a blend of the two regimes -- neither is the cost of running
    it, which is the number a reader needs to size the machine.
    """
    stats = []
    for key, label in STAGES:
        values = _column(rows, key)
        if drop is not None and drop < values.size:
            values = np.delete(values, drop)
        if values.size == 0:
            continue
        active = values[values > 0.0]
        duty = float(active.size) / float(values.size) if values.size else 0.0
        used = active if active.size else values
        stats.append({
            "key": key,
            "label": label,
            "mean_ms": float(used.mean()),
            "median_ms": float(np.median(used)),
            "p95_ms": float(np.percentile(used, 95)),
            "max_ms": float(used.max()),
            "duty": duty,
            "intermittent": duty < 0.9,
        })
    return stats


def inversion_stat(path: Path | None) -> dict | None:
    """Cost of one inversion fit, from an audit CSV, in milliseconds."""
    if path is None or not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values = _column(rows, "elapsed_seconds") * 1000.0
    if values.size == 0:
        return None
    return {
        "key": "inversion",
        "label": "inversion fit",
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(values.max()),
        "duty": float("nan"),
        "intermittent": True,
        "off_path": True,
    }


def render(rows: list[dict], stats: list[dict], out_dir: Path, *,
           cadence_s: float, dpi: int, panel: str = "both", stat: str = "max_ms",
           drop: int | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from software.tools.paper_figures import DOUBLE, INK, MONITORED, MUTED, STATUS_LIMIT, grid, save

    ON_PATH = "#D55E00"
    OFF_PATH = MONITORED

    # Late import of style so the module can be imported without touching rcParams.
    from software.tools.paper_figures import style
    style()

    budget_ms = cadence_s * 1000.0
    # Panel letters only exist when there is more than one panel to tell apart.
    lettered = panel == "both"
    ax_budget = None
    if panel == "total":
        fig, ax2 = plt.subplots(1, 1, figsize=(DOUBLE, DOUBLE * 0.30))
        ax = None
    elif panel == "stages":
        # Two axes, not one: the bars need a scale that fits them and the budget
        # needs a scale that fits it, and forcing both onto one axis is what made
        # every bar a sliver. The strip below carries the budget comparison once,
        # geometrically, so it is still shown rather than reduced to a number.
        fig, (ax, ax_budget) = plt.subplots(
            2, 1, figsize=(DOUBLE, DOUBLE * 0.40),
            gridspec_kw={"height_ratios": [6.0, 1.0], "hspace": 0.55})
        ax2 = None
    else:
        ax_budget = None
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(DOUBLE, DOUBLE * 0.60),
                                      gridspec_kw={"height_ratios": [1.35, 1.0],
                                                   "hspace": 0.42})

    # --- (a) per-stage cost -------------------------------------------------
    if ax is not None:
        ordered = sorted(stats, key=lambda s: s[stat])
        y = np.arange(len(ordered))
        colours = [OFF_PATH if s.get("off_path") else ON_PATH for s in ordered]
        values = np.array([s[stat] for s in ordered])
        p95 = np.array([s["p95_ms"] for s in ordered])

        ax.barh(y, values, height=0.62, color=colours, zorder=3)
        # A linear axis running the whole budget. The earlier log axis compressed
        # the distance to the limit so hard that a 10 s bar appeared to sit close
        # to 30 s; on a linear axis the proportion is what it looks like, which is
        # the entire point of drawing the budget at all. The sub-millisecond
        # stages become hairlines as a result -- correct, and why every bar
        # carries its value.
        if stat == "max_ms":
            for yi, hi, val, colour in zip(y, p95, values, colours):
                if hi < val:
                    ax.plot([hi], [yi], "|", color="white", ms=7, mew=1.2, zorder=5)

        for yi, val in zip(y, values):
            ax.annotate(f"{val:,.1f} ms" if val < 10 else f"{val:,.0f} ms",
                        xy=(val, yi), xytext=(6, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=6.8, color=INK)

        ax.set_yticks(y)
        ax.set_yticklabels([s["label"] for s in ordered])
        span = float(values.max())
        if ax_budget is None:
            # Single-axis layouts keep the budget on the bar axis, since there is
            # no strip to carry it.
            ax.set_xlim(0, budget_ms * 1.02)
            ticks = np.arange(0, budget_ms + 1, 5000)
            ax.axvline(budget_ms, color=STATUS_LIMIT, lw=1.6, ls="--", zorder=6)
            ax.annotate(f"{cadence_s:.0f} s budget", xy=(budget_ms, -0.55),
                        xytext=(-4, 0), textcoords="offset points", fontsize=7.2,
                        color=STATUS_LIMIT, ha="right", va="center", fontweight="bold")
        else:
            ax.set_xlim(0, span * 1.32)
            step = 2000.0 if span > 6000 else 1000.0
            ticks = np.arange(0, span * 1.32, step)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{v/1000:g}" for v in ticks])
        ax.set_xlabel("worst-case time per call (s)" if stat == "max_ms"
                      else "time per call (s)")
        ax.set_ylim(-0.9, len(ordered) - 0.25)
        grid(ax, axis="x")
        if lettered:
            ax.annotate("(a)", xy=(0.0, 1.0), xycoords="axes fraction", xytext=(-46, 2),
                        textcoords="offset points", fontsize=8, fontweight="bold",
                        va="bottom", ha="left", color=INK)

    if ax_budget is not None:
        # One bar, the worst complete update, drawn against the whole budget. The
        # critical-path stages only: the inversion is off that path and adding it
        # here would claim a serialisation that does not happen.
        totals = _column(rows, "ms_total")
        # Same exclusion as the bars above. Reading the raw column here would put
        # the cold start in the strip and the steady state in the bars, so the
        # two halves of one figure would disagree about the worst case.
        if drop is not None and drop < totals.size:
            totals = np.delete(totals, drop)
        worst = float(np.nanmax(totals)) if totals.size else 0.0
        ax_budget.barh([0], [worst], height=0.55, color=ON_PATH, zorder=3)
        ax_budget.barh([0], [budget_ms - worst], left=[worst], height=0.55,
                       color=MUTED, alpha=0.13, zorder=2)
        ax_budget.set_xlim(0, budget_ms)
        ax_budget.set_ylim(-0.6, 0.6)
        ax_budget.set_yticks([])
        ax_budget.set_xticks(np.arange(0, budget_ms + 1, 5000))
        ax_budget.set_xticklabels([f"{int(v/1000)}" for v in np.arange(0, budget_ms + 1, 5000)])
        ax_budget.set_xlabel(f"one complete update against the {cadence_s:.0f} s budget (s)")
        for spine in ("top", "right", "left"):
            ax_budget.spines[spine].set_visible(False)
        ax_budget.axvline(budget_ms, color=STATUS_LIMIT, lw=1.6, ls="--", zorder=6)
        ax_budget.annotate(f"worst update {worst/1000:.1f} s = "
                           f"{worst/budget_ms*100:.0f}% of budget",
                           xy=(worst, 0), xytext=(8, 0), textcoords="offset points",
                           va="center", ha="left", fontsize=7.0, color=INK)
        ax_budget.annotate(f"{cadence_s:.0f} s", xy=(budget_ms, 0.62),
                           xytext=(-2, 0), textcoords="offset points", fontsize=7.2,
                           color=STATUS_LIMIT, ha="right", va="bottom", fontweight="bold")

    if ax2 is None:
        save(fig, out_dir, "fig_service_stages")
        return

    if ax2 is not None:
        # --- (b) total per update over time -------------------------------------
        t = _column(rows, "time_hours")
        total = _column(rows, "ms_total")
        n = min(t.size, total.size)
        t, total = t[:n], total[:n]

        ax2.plot(t, total, "-", color=ON_PATH, lw=1.0, zorder=4)
        ax2.axhline(budget_ms, color=STATUS_LIMIT, lw=1.1, ls="--", zorder=3)
        ax2.set_yscale("log")
        ax2.set_xlabel("experiment time (h)")
        ax2.set_ylabel("total per update (ms)")
        ax2.set_xlim(float(t.min()), float(t.max()))
        grid(ax2)

        med_total = float(np.median(total))
        ax2.axhline(med_total, color=MUTED, lw=0.8, ls=":", zorder=3)
        # Parked in the empty band under the budget line, not on the median itself:
        # the trace is dense enough there that a label sitting on it is unreadable.
        ax2.annotate(f"median {med_total:,.0f} ms  ({med_total / budget_ms * 100:.1f}% of budget)",
                     xy=(0.015, 0.74), xycoords="axes fraction",
                     fontsize=6.6, color=MUTED, ha="left", va="center")
        # The banding is the lookahead duty cycle, not noise, and saying so stops it
        # reading as instability in the pipeline.
        ax2.annotate("upper band = updates that also run the lookahead",
                     xy=(0.985, 0.74), xycoords="axes fraction",
                     fontsize=6.6, color=MUTED, ha="right", va="center")
        ax2.annotate(f"{cadence_s:.0f} s budget", xy=(0.985, budget_ms),
                     xycoords=("axes fraction", "data"), xytext=(0, 4),
                     textcoords="offset points", fontsize=6.6,
                     color=STATUS_LIMIT, ha="right", va="bottom")

        # The cold start is one update, and on a linear axis it flattened everything
        # else; on a log axis it is legible, so it is labelled rather than trimmed.
        peak = int(np.argmax(total))
        if total[peak] > med_total * 4:
            ax2.annotate(f"cold start {total[peak] / 1000:.1f} s",
                         xy=(t[peak], total[peak]), xytext=(8, -2),
                         textcoords="offset points", fontsize=6.6, color=INK,
                         ha="left", va="top")
        if lettered:
            ax2.annotate("(b)", xy=(0.0, 1.0), xycoords="axes fraction", xytext=(-46, 2),
                         textcoords="offset points", fontsize=8, fontweight="bold",
                         va="bottom", ha="left", color=INK)

    name = "fig_update_time" if panel == "total" else "fig_service_timing"
    save(fig, out_dir, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--updates", default="artifacts/dt_evidence/dt_updates.csv",
                        help="Per-update timing CSV written by dt_decision_evidence.")
    parser.add_argument("--inversion", default="artifacts/inversion_audit/inversion_updates.csv",
                        help="Audit CSV, for the inversion's own cost. '' to omit it.")
    parser.add_argument("--out", default="artifacts/paper/service_timing")
    parser.add_argument("--cadence-seconds", type=float, default=30.0,
                        help="Sensor cadence, drawn as the real-time budget.")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--stat", choices=("max", "median", "mean"), default="max",
                        help="Statistic the bars show. Default max: a real-time margin is "
                             "argued from the worst case, not the typical case.")
    parser.add_argument("--keep-warmup", action="store_true",
                        help="Keep the one-off mesh-build update. Excluded by default, since "
                             "it would set the worst case from an event paid once per service "
                             "lifetime.")
    parser.add_argument("--panel", choices=("both", "stages", "total"), default="both",
                        help="'total' writes just the per-update total over time as "
                             "fig_update_time; 'stages' just the per-stage bars.")
    args = parser.parse_args()

    path = Path(args.updates)
    if not path.exists():
        raise SystemExit(f"No timing data at {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No rows in {path}")

    drop = None if args.keep_warmup else warmup_index(rows)
    if drop is not None:
        print(f"excluding warm-up update #{drop} "
              f"(boundary {_column(rows, 'ms_boundary')[drop]:,.0f} ms)")
    stats = stage_stats(rows, drop=drop)
    inv = inversion_stat(Path(args.inversion) if args.inversion else None)
    if inv is not None:
        stats.append(inv)

    print(f"{len(rows)} updates from {path}")
    print(f"{'stage':<24}{'mean':>10}{'median':>10}{'p95':>10}{'max':>11}"
          f"{'% budget':>10}   duty")
    budget = args.cadence_seconds * 1000.0
    for stat in stats:
        duty = "n/a" if not np.isfinite(stat["duty"]) else f"{stat['duty'] * 100:.0f}%"
        share = "--" if stat.get("off_path") else f"{stat['max_ms'] / budget * 100:.2f}%"
        print(f"{stat['label']:<24}{stat['mean_ms']:>10.2f}{stat['median_ms']:>10.2f}"
              f"{stat['p95_ms']:>10.2f}{stat['max_ms']:>11.2f}{share:>10}   {duty:>6}")

    total = _column(rows, "ms_total")
    if drop is not None:
        total = np.delete(total, drop)
    print(f"\ntotal per update: median {np.median(total):,.0f} ms, "
          f"p95 {np.percentile(total, 95):,.0f} ms, max {total.max():,.0f} ms")
    print(f"budget {budget:,.0f} ms -> median uses {np.median(total) / budget * 100:.1f}%, "
          f"p95 uses {np.percentile(total, 95) / budget * 100:.1f}%")
    over = int(np.count_nonzero(total > budget))
    print(f"updates over budget: {over} of {total.size}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    render(rows, stats, out_dir, cadence_s=float(args.cadence_seconds),
           dpi=int(args.dpi), panel=args.panel, stat=f"{args.stat}_ms", drop=drop)


if __name__ == "__main__":
    main()

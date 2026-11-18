"""Publication figures and tables for the predictive thaw-control results.

Renders from the CSVs the analysis tools already wrote, so nothing here re-runs a
replay and the numbers in the paper are the same objects that were verified:

    artifacts/dt_evidence/dt_updates.csv    per-update record of the full replay
    artifacts/dt_frontres/dt_updates.csv    a coarser pass that also records the
                                            pre-interpolation front metric

Every figure is written as PDF (vector, for the manuscript) and PNG at 600 dpi
(for previews and slides), at either single-column (89 mm) or double-column
(183 mm) width. Tables are written twice: LaTeX with booktabs for the manuscript
and CSV for reuse.

**Colour.** Three categorical series -- the FEM keep-branch, the FEM stop-branch,
and the monitored time-to-breach readout -- taken from the Okabe-Ito set and
checked against the six colour
checks (lightness band, chroma floor, CVD separation under Machado-Oliveira-
Fernandes severity 1.0, normal-vision floor, contrast) rather than chosen by eye.
The reported worst adjacent CVD separation is 11.0 OKLab dE, above the target of
8.0, and every colour clears 3:1 against the page. Measurements are drawn in ink
black because they are the reference, not a category, and the safety limit uses a
reserved status red that is never used for a data series. Line style varies with
colour throughout, so the figures survive greyscale printing.

Run:

    python -m software.tools.paper_figures
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from software.startup.utils.config import load_startup_config
from software.tools.verify_prediction_skill import (
    LEAD_BUCKETS,
    breach_within,
    consistency,
    first_crossing,
    load_rows,
    measured_series,
    score_trigger,
)

# Validated categorical palette (see module docstring).
MONITORED = "#0072B2"
FEM_KEEP = "#D55E00"
FEM_STOP = "#009E73"
INK = "#1A1A1A"          # measurements: the reference, not a series
MUTED = "#6B6B6B"        # axes, grid, secondary annotation
STATUS_LIMIT = "#B3261E"  # reserved: the safety limit, never a data series

MM_PER_IN = 25.4
SINGLE = 89 / MM_PER_IN
DOUBLE = 183 / MM_PER_IN


def style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "grid.color": "#D8D8D8",
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.3,
        "pdf.fonttype": 42,      # embed TrueType, not Type 3: required by most journals
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def save(fig, out_dir: Path, name: str):
    for ext, kwargs in (("pdf", {}), ("png", {"dpi": 600})):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, **kwargs)
    print(f"  wrote {out_dir / name}.pdf / .png")
    import matplotlib.pyplot as plt
    plt.close(fig)


def grid(ax, axis="both"):
    ax.grid(True, axis=axis, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def _num(row: dict, key: str):
    v = row.get(key, "")
    if v in ("", "None", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def series(rows, key, scale=1.0):
    return np.array([np.nan if _num(r, key) is None else _num(r, key) * scale for r in rows])


# ---------------------------------------------------------------------------
# TABLES
# ---------------------------------------------------------------------------
def write_table(out_dir: Path, name: str, header: list[str], body: list[list], caption: str,
                align: str | None = None):
    with (out_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(body)

    align = align or ("l" + "r" * (len(header) - 1))
    lines = [
        r"\begin{table}[t]", r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{tab:{name}}}",
        f"  \\begin{{tabular}}{{{align}}}",
        r"    \toprule",
        "    " + " & ".join(str(h) for h in header) + r" \\",
        r"    \midrule",
    ]
    lines += ["    " + " & ".join(str(c) for c in row) + r" \\" for row in body]
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (out_dir / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {out_dir / name}.tex / .csv")


# ---------------------------------------------------------------------------
# FIGURES
# ---------------------------------------------------------------------------
def fig_front_resolution(rows, out_dir, spacing_m):
    """Fig 1: the front metric before and after interpolation."""
    import matplotlib.pyplot as plt

    t = series(rows, "time_hours")
    node = series(rows, "r_max_nowcast_node_m", 1000)
    interp = series(rows, "r_max_nowcast_m", 1000)
    ok = np.isfinite(node) & np.isfinite(interp)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(SINGLE, 3.4),
                                   gridspec_kw={"height_ratios": [1.5, 1]})

    ax1.step(t[ok], node[ok], where="post", color=MUTED, lw=1.1, ls="-",
             label="outermost thawed node")
    ax1.plot(t[ok], interp[ok], color=FEM_KEEP, lw=1.3, label="interpolated crossing")
    ax1.set_ylabel("forecast $r_{\\max}$ (mm)")
    ax1.legend(loc="upper left")
    grid(ax1)

    # A window where the staircase and the continuous curve are both legible.
    lo, hi = 2.4, 4.4
    win = ok & (t >= lo) & (t <= hi)
    ax2.step(t[win], node[win], where="post", color=MUTED, lw=1.1)
    ax2.plot(t[win], interp[win], color=FEM_KEEP, lw=1.3)
    ax2.set_xlabel("experiment time (h)")
    ax2.set_ylabel("(detail, mm)")
    ax2.set_xlim(lo, hi)
    grid(ax2)

    n_node = len(np.unique(np.round(node[ok], 9)))
    n_interp = len(np.unique(np.round(interp[ok], 9)))
    ax1.annotate(f"{n_node} distinct values", xy=(0.985, 0.30), xycoords="axes fraction",
                 ha="right", fontsize=7, color=MUTED)
    ax1.annotate(f"{n_interp} distinct values", xy=(0.985, 0.18), xycoords="axes fraction",
                 ha="right", fontsize=7, color=FEM_KEEP)
    save(fig, out_dir, "fig1_front_resolution")
    return n_node, n_interp


def fig_verification(rows, out_dir, actual_cross, tolerance_min):
    """Fig 2: is the predicted crossing time right, and for how long."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.7))

    specs = [("ttb_monitored_h", "monitored", MONITORED, "o", "-"),
             ("ttb_fem_h", "FEM forecast", FEM_KEEP, "s", "--")]
    results = {}
    for key, label, colour, marker, ls in specs:
        res = consistency(rows, key, actual_cross)
        results[key] = res
        if res["n"]:
            ax1.plot(res["t"], res["pred"], marker, ms=2.6, color=colour, alpha=0.7,
                     mec="none", label=label)
    ax1.axhline(actual_cross, color=INK, lw=1.2, label="observed crossing")
    ax1.set_xlabel("time prediction was made (h)")
    ax1.set_ylabel("predicted crossing time (h)")
    ax1.set_ylim(0, actual_cross * 2.0)
    ax1.set_xlim(0.5, actual_cross)
    ax1.legend(loc="upper left")
    grid(ax1)
    ax1.set_title("(a) consistency", loc="left")

    for key, label, colour, marker, ls in specs:
        buckets = results[key]["buckets"]
        if not buckets:
            continue
        centres = [(b["lo"] + b["hi"]) / 2 for b in buckets]
        ax2.plot(centres, [b["median_abs"] for b in buckets], marker=marker, ls=ls,
                 color=colour, ms=3.4, label=f"{label}, median")
        ax2.plot(centres, [b["p90_abs"] for b in buckets], marker=marker, ls=":",
                 color=colour, ms=3.0, alpha=0.55, label=f"{label}, 90th pct")
    ax2.axhline(tolerance_min, color=STATUS_LIMIT, lw=1.1, ls="-.")
    ax2.annotate(f"tolerance {tolerance_min:g} min", xy=(0.98, tolerance_min), xycoords=("axes fraction", "data"),
                 ha="right", va="bottom", fontsize=7, color=STATUS_LIMIT)
    ax2.set_yscale("log")
    ax2.set_xlabel("lead at which prediction was made (min)")
    ax2.set_ylabel("|error| in crossing time (min)")
    ax2.legend(loc="upper left", ncol=2)
    grid(ax2)
    ax2.set_title("(b) error against lead", loc="left")

    save(fig, out_dir, "fig2_prediction_verification")
    return results


def fig_skill(sweeps, out_dir):
    """Fig 3: the rule scored as a warning system."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(SINGLE, 2.4))
    scores = sweeps["filtered"]
    leads = [s["lead_h"] * 60 for s in scores]
    ax.plot(leads, [s["pod"] for s in scores], "-o", color=MONITORED, ms=3.4, label="POD (hits)")
    ax.plot(leads, [s["far"] for s in scores], "--s", color=FEM_KEEP, ms=3.4, label="FAR (false alarms)")
    ax.plot(leads, [s["csi"] for s in scores], "-.^", color=FEM_STOP, ms=3.6, label="CSI (skill)")
    best = max(scores, key=lambda s: s["csi"] if np.isfinite(s["csi"]) else -1)
    ax.axvline(best["lead_h"] * 60, color=MUTED, lw=0.9, ls=":")
    ax.annotate(f"best {best['lead_h'] * 60:.0f} min", xy=(best["lead_h"] * 60, 0.34),
                xytext=(4, 0), textcoords="offset points", fontsize=7, color=MUTED)
    ax.set_xlabel("rule lead time (min)")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.04, 1.08)
    # Legend above the axes: every corner inside the frame is occupied by a curve
    # at some lead time, and a legend over data is worse than one outside it.
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01, 1, 0.12), mode="expand",
              ncol=3, borderaxespad=0)
    grid(ax)
    save(fig, out_dir, "fig3_warning_skill")
    return best


def fig_decision(rows, out_dir, config, event_h, limit_m):
    """Fig 4: predictive vs reactive decision against the real intervention."""
    import matplotlib.pyplot as plt

    t = series(rows, "time_hours")
    meas = series(rows, "r_max_measured_m", 1000)
    keep = series(rows, "r_max_keep_m", 1000)
    stop = series(rows, "r_max_stop_m", 1000)
    valid = series(rows, "prediction_valid_at_h")

    t_react = next((float(r["time_hours"]) for r in rows
                    if _num(r, "r_max_measured_m") is not None
                    and _num(r, "r_max_measured_m") >= limit_m), None)
    t_pred = next((float(r["time_hours"]) for r in rows if r.get("predictive_stop") == "True"), None)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(DOUBLE, 3.6), sharex=True,
                                   gridspec_kw={"height_ratios": [2.4, 1]})

    ax1.plot(t, meas, color=INK, lw=1.4, label="measured")
    ok = np.isfinite(keep)
    ax1.plot(valid[ok], keep[ok], color=FEM_KEEP, lw=1.2, ls="--", label="predicted, heating continues")
    ax1.plot(valid[ok], stop[ok], color=FEM_STOP, lw=1.2, ls=":", label="predicted, heater stopped")
    ax1.axhline(limit_m * 1000, color=STATUS_LIMIT, lw=1.1, ls="-.")
    ax1.annotate("safety limit", xy=(0.995, limit_m * 1000), xycoords=("axes fraction", "data"),
                 ha="right", va="bottom", fontsize=7, color=STATUS_LIMIT)
    ax1.set_ylabel("thaw radius (mm)")
    ax1.legend(loc="upper left", ncol=3)
    grid(ax1)

    marks = [(t_pred, "predictive stop", MONITORED, "--"),
             (t_react, "reactive stop", STATUS_LIMIT, "--"),
             (event_h, "operator reduces power", INK, "-")]
    for when, label, colour, ls in marks:
        if when is None:
            continue
        for ax in (ax1, ax2):
            ax.axvline(when, color=colour, lw=1.0, ls=ls, zorder=1)
        # Along the bottom: the top of this panel carries the legend, and the
        # lower-right region is empty once the front has risen away from it.
        ax1.annotate(label, xy=(when, ax1.get_ylim()[0]), xytext=(2, 3),
                     textcoords="offset points", rotation=90, va="bottom", ha="left",
                     fontsize=6.5, color=colour)

    for key, label, colour, ls in (("ttb_monitored_h", "monitored", MONITORED, "-"),
                                   ("ttb_fem_h", "FEM forecast", FEM_KEEP, "--")):
        vals = series(rows, key, 60.0)
        if np.any(np.isfinite(vals)):
            ax2.plot(t, vals, color=colour, lw=1.1, ls=ls, label=label)
    ax2.set_ylabel("time to limit (min)")
    ax2.set_xlabel("experiment time (h)")
    ax2.set_ylim(0, 120)
    ax2.legend(loc="upper right", ncol=2)
    grid(ax2)

    save(fig, out_dir, "fig4_predictive_vs_reactive")
    return t_pred, t_react


def fig_cost(rows, out_dir, cadence_s):
    """Fig 5: cost of one update, by stage."""
    import matplotlib.pyplot as plt

    stages = [("boundary", "boundary + pipe calibration"), ("thaw_front", "front reconstruction"),
              ("lookahead", "predictive lookahead"), ("forecast", "FEM forecast"),
              ("safety", "safety monitor"), ("heater", "actuation")]
    means = [(label, float(np.nanmean(series(rows, f"ms_{k}")))) for k, label in stages]
    means.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(SINGLE, 2.2))
    # Magnitude, one hue, light-to-dark with value: a sequential job, not categorical.
    vals = np.array([m[1] for m in means])
    shades = plt.cm.Blues(0.35 + 0.5 * vals / vals.max())
    ax.barh([m[0] for m in means], vals, color=shades, height=0.68)
    for i, (_label, v) in enumerate(means):
        # Two stages are well under a millisecond; rounding them to "0 ms" reads
        # as missing data rather than as negligible.
        text = f"{v:.1f} ms" if v < 10 else f"{v:.0f} ms"
        ax.annotate(text, xy=(v, i), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=7, color=INK)
    ax.set_xlabel("mean time per update (ms)")
    ax.set_xlim(0, vals.max() * 1.35)
    grid(ax, axis="x")
    total = float(np.nanmean(series(rows, "ms_total")))
    ax.set_title(f"total {total:.0f} ms of a {cadence_s:.0f} s budget "
                 f"({100 * total / (cadence_s * 1000):.1f}%)", loc="left", color=MUTED)
    save(fig, out_dir, "fig5_computation_cost")
    return means, total


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default="artifacts/dt_evidence/dt_updates.csv")
    parser.add_argument("--frontres-csv", default="artifacts/dt_frontres/dt_updates.csv")
    parser.add_argument("--config", default="startup.conf")
    parser.add_argument("--out", default="artifacts/paper")
    parser.add_argument("--tolerance-minutes", type=float, default=10.0)
    parser.add_argument("--persistence", type=int, default=3)
    parser.add_argument("--min-rate-mm-per-h", type=float, default=1.0)
    parser.add_argument("--event-hours", type=float, default=5.133)
    parser.add_argument("--cadence-seconds", type=float, default=30.0)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    style()

    config = load_startup_config(args.config)
    limit_m = float(config["safety_monitor"]["limit_radius_m"])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Channel count actually fed to the twin: the file's H*D* columns less the
    # surface band and the stuck thermocouple, matching what the replay excluded.
    from software.tools.dt_decision_evidence import (
        DEFAULT_EXCLUDE, DEFAULT_SENSOR_PATH, surface_band_channels,
    )
    from software.tools.replay_csv import _iter_rows, _parse_header
    header = next(_iter_rows(Path(DEFAULT_SENSOR_PATH)), [])
    _idx, columns = _parse_header(header, None)
    dropped = set(DEFAULT_EXCLUDE.split(",")) | surface_band_channels(
        Path(DEFAULT_SENSOR_PATH),
        float(config["thaw_front_reconstruction"]["surface_exclusion_m"]))
    n_channels = len([c for c in columns if c["sensor_id"] not in dropped])

    rows = load_rows(Path(args.csv))
    t, r = measured_series(rows)
    actual = first_crossing(t, r, limit_m)
    print(f"limit {limit_m * 1000:.0f} mm, observed crossing t = {actual:.3f} h, {len(rows)} updates\n")

    print("figures")
    front_rows = load_rows(Path(args.frontres_csv))
    n_node, n_interp = fig_front_resolution(front_rows, out_dir, 0.010)
    results = fig_verification(rows, out_dir, actual, args.tolerance_minutes)

    lead_grid = [x / 60.0 for x in (10, 15, 20, 30, 45, 60, 90)]
    keys = ["ttb_monitored_h", "ttb_fem_h"]
    sweeps = {
        "raw": [score_trigger(rows, t, r, limit_m=limit_m, lead_h=L, keys=keys,
                              persistence=0, min_rate_m_h=0.0) for L in lead_grid],
        "filtered": [score_trigger(rows, t, r, limit_m=limit_m, lead_h=L, keys=keys,
                                   persistence=args.persistence,
                                   min_rate_m_h=args.min_rate_mm_per_h / 1000.0)
                     for L in lead_grid],
    }
    best = fig_skill(sweeps, out_dir)
    t_pred, t_react = fig_decision(rows, out_dir, config, args.event_hours, limit_m)
    means, total_ms = fig_cost(rows, out_dir, args.cadence_seconds)

    print("\ntables")
    body = []
    for key, label in (("ttb_monitored_h", "Monitored"), ("ttb_fem_h", "FEM forecast")):
        for b in results[key]["buckets"]:
            body.append([label, f"{b['lo']}--{b['hi']}", b["n"],
                         f"{b['median_abs']:.1f}", f"{b['p90_abs']:.1f}", f"{b['bias']:+.1f}",
                         f"{b['median_abs_debiased']:.1f}"])
    write_table(out_dir, "table1_prediction_error",
                ["Source", "Lead (min)", "$n$", "Median $|e|$ (min)",
                 "90th pct $|e|$ (min)", "Bias (min)", "Debiased $|e|$ (min)"],
                body,
                "Error in the predicted crossing time, by how far ahead the prediction was made. "
                "Errors are against the single observed crossing. The 90th percentile is reported "
                "alongside the median because the monitored readout, being a rate extrapolation, "
                "diverges when the front plateaus and its rate approaches zero.")

    body = []
    for s in sweeps["filtered"]:
        body.append([f"{s['lead_h'] * 60:.0f}", s["hits"], s["misses"], s["false_alarms"],
                     f"{s['pod']:.2f}", f"{s['far']:.2f}", f"{s['csi']:.2f}",
                     f"{s['median_lead_min']:.0f}"])
    write_table(out_dir, "table2_warning_skill",
                ["Rule lead (min)", "Hits", "Misses", "False alarms", "POD", "FAR", "CSI",
                 "Median warning (min)"],
                body,
                "The predictive rule scored as a warning system over all updates at which the "
                "front had not yet reached the limit. POD is the hit rate, FAR the false-alarm "
                "ratio, CSI the critical success index.")

    body = [[label, f"{v:.1f}", f"{100 * v / total_ms:.1f}"] for label, v in reversed(means)]
    body.append([r"\textbf{Total}", f"\\textbf{{{total_ms:.0f}}}", r"\textbf{100.0}"])
    write_table(out_dir, "table3_computation_cost",
                ["Stage", "Mean time (ms)", "Share (\\%)"], body,
                f"Computational cost of one digital-twin update, against a "
                f"{args.cadence_seconds:.0f}~s sensor cadence "
                f"({100 * total_ms / (args.cadence_seconds * 1000):.1f}\\% of budget).")

    kin_h = next((b["hi"] for b in reversed(results["ttb_monitored_h"]["buckets"])
                  if b["median_abs"] <= args.tolerance_minutes), None)
    body = [
        ["Experiment duration", f"{t[-1]:.2f} h"],
        ["Sensor cadence", f"{args.cadence_seconds:.0f} s"],
        ["Thermocouples used", f"{n_channels}"],
        ["Mesh", "775 nodes, 720 elements, 10 mm spacing"],
        ["Safety limit $r_{\\max}$", f"{limit_m * 1000:.0f} mm"],
        ["Front metric resolution", f"{n_node} $\\rightarrow$ {n_interp} distinct values"],
        ["Observed crossing", f"{actual:.2f} h"],
        ["Reactive trigger", f"{t_react:.2f} h" if t_react else "--"],
        ["Predictive trigger", f"{t_pred:.2f} h" if t_pred else "--"],
        ["Operator intervention", f"{args.event_hours:.2f} h"],
        ["Usable prediction horizon", f"{kin_h if kin_h else '<10'} min"],
        ["Best rule lead", f"{best['lead_h'] * 60:.0f} min"],
        ["Median warning obtained", f"{best['median_lead_min']:.0f} min"],
    ]
    write_table(out_dir, "table4_configuration",
                ["Quantity", "Value"], body,
                "Experiment and configuration summary.", align="ll")

    print(f"\nheadline numbers")
    print(f"  front metric        : {n_node} -> {n_interp} distinct values")
    print(f"  usable horizon      : {kin_h if kin_h else '<10'} min")
    print(f"  best rule lead      : {best['lead_h'] * 60:.0f} min "
          f"(POD {best['pod']:.2f}, FAR {best['far']:.2f}, CSI {best['csi']:.2f})")
    print(f"  median warning      : {best['median_lead_min']:.0f} min")
    print(f"  predictive vs react : {t_pred:.2f} h vs {t_react:.2f} h "
          f"({(t_react - t_pred) * 60:+.0f} min)")
    print(f"  cost per update     : {total_ms:.0f} ms of {args.cadence_seconds * 1000:.0f} ms")


if __name__ == "__main__":
    main()

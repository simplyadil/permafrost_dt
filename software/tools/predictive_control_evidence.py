"""Evidence figures for the predictive safety/control rules (M1/M2/F1/F2).

Replays a recorded experiment through the real thaw-front reconstruction,
boundary builder, pipe-boundary calibration, FEM forecast, predictive safety
monitor and heater actuation -- RabbitMQ and InfluxDB stubbed -- and renders the
two figures that go with the control section of the paper:

    fig_decision_timeline   what each source said, which predicates fired, and
                            what the actuator was told to do
    fig_lookahead_branches  the two counterfactual branches at four update
                            times, which is what F1 and F2 actually test

**On the pipe calibration.** The pipe-boundary service gates recalibration on a
wall clock (``recalibrate_seconds``), which never elapses in a replay running
far faster than real time -- so a naive replay would either never recalibrate,
or, if the gate is simply disabled, recalibrate on every message and do far more
fitting than the live service ever does. Neither reproduces the deployed
behaviour. This tool instead gates on *data* time: with the shipped 300 s
interval and a 30 s sensor cadence the live service refits every tenth snapshot,
and ``--pipe-recalibrate-data-seconds`` reproduces exactly that. The pipe
boundary actually in force is printed at the start and end of the run, so the
figures cannot silently be made from the wrong one.

**On the horizon.** F1 and F2 are evaluated over ``fem_forecast.lookahead_hours``
by advancing a copy of the solver along two branches that differ only in the
pipe boundary: ``keep`` holds the current setpoint, ``stop`` forces it to
``lookahead_stop_setpoint_c``. Everything else -- mesh, material, current field,
the other three boundaries -- is identical, so the gap between them is the
effect of the decision and nothing else.

Run:

    python -m software.tools.predictive_control_evidence
    python -m software.tools.predictive_control_evidence --stride 6   # quick look
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from software.digital_twin.data_access.influx_utils import InfluxConfig
from software.startup.service_runners import (
    build_influx_config,
    create_boundary_condition_builder_server,
    create_fem_forecast_server,
    create_heater_actuation_server,
    create_pipe_boundary_server,
    create_safety_monitor_server,
    create_thaw_front_reconstruction_server,
)
from software.startup.utils.config import load_startup_config
from software.tools.fem_debug_replay import _NullInflux, _Sink, _sensor_rows
from software.tools.paper_figures import (
    DOUBLE,
    FEM_KEEP,
    FEM_STOP,
    INK,
    MUTED,
    STATUS_LIMIT,
    grid,
    save,
    style,
)
from software.tools.replay_csv import _load_fluid_temps

DEFAULT_PATH = "data/CR1000_Table1_April8_replay_Final.xlsx"
DEFAULT_EXCLUDE = "H10D75"

# Rule names as they are set in the paper. Held in one place so a rename lands
# in every panel of a figure at once instead of in whichever occurrence was
# noticed; the internal keys stay "f1" / "f2" / "m1" and are untouched.
LBL_F1 = r"$F_{\mathrm{heat}}$"
LBL_F2 = r"$F_{\mathrm{stop}}$"
LBL_M1 = r"$M_{\mathrm{lim}}$"
LBL_M2 = r"$M_{\mathrm{warn}}$"

# Status shading. Deliberately a single hue ramping with severity rather than a
# categorical set: these are ordered states, and colouring them categorically
# would imply differences in kind where there are only differences in degree.
STATUS_TINT = {
    "ok": None,
    "warning": "#FDF0D5",
    "critical": "#FADFBE",
    "predictive_warning": "#FDF0D5",
    "pre_alert": "#F6CBA8",
    "committed": "#EFAF9A",
    "alert": "#E8938F",
}
STATUS_ORDER = ["ok", "warning", "critical", "predictive_warning",
                "pre_alert", "committed", "alert"]


class _Recorder:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def publish(self, message: dict) -> None:
        self.messages.append(message)

    @property
    def last(self) -> dict | None:
        return self.messages[-1] if self.messages else None


def _pipe_bc(boundary: dict, pipe_set: str) -> dict:
    for bc in boundary.get("boundaries", []) or []:
        if bc.get("boundary_set") == pipe_set:
            return bc
    return {}


def _describe_bc(boundary: dict, pipe_set: str) -> str:
    bc = _pipe_bc(boundary, pipe_set)
    profile = bc.get("temperature_profile")
    return (f"type={bc.get('bc_type')} T={bc.get('temperature')} "
            f"h={bc.get('h_coeff')} profile={'yes' if profile else 'no'}")


# ---------------------------------------------------------------------------
# REPLAY
# ---------------------------------------------------------------------------
def run_replay(config: dict, *, path: Path, fluid_cols, start_row: int,
               limit: int | None, stride: int, exclude_sensors: set[str],
               recalibrate_data_s: float) -> tuple[list[dict], dict]:
    rabbit_cfg = config.get("rabbitmq", {})
    influx_cfg: InfluxConfig = build_influx_config(config.get("influxdb", {}))

    thaw = create_thaw_front_reconstruction_server(
        influx_cfg, rabbit_cfg, config.get("thaw_front_reconstruction", {}))
    thaw.mq_out, thaw.influx = _Recorder(), _NullInflux()

    builder = create_boundary_condition_builder_server(
        rabbit_cfg, config.get("boundary_condition_builder", {}))

    fem_cfg = dict(config.get("fem_forecast", {}))
    fem_cfg["run_baseline"] = False
    forecast = create_fem_forecast_server(influx_cfg, rabbit_cfg, fem_cfg)
    forecast.mq_out, forecast.influx = _Recorder(), _NullInflux()

    pipe = create_pipe_boundary_server(influx_cfg, rabbit_cfg, config.get("pipe_boundary", {}))
    pipe.mq_boundary_out = _Sink()
    inlet_col, outlet_col, time_col = fluid_cols
    fluid = _load_fluid_temps(path, inlet_col=inlet_col, outlet_col=outlet_col, time_col=time_col)

    monitor = create_safety_monitor_server(
        influx_cfg, rabbit_cfg, dict(config.get("safety_monitor", {}), mode="predictive"))
    monitor.mq_out, monitor.influx = _Recorder(), _NullInflux()

    heater = create_heater_actuation_server(
        influx_cfg, rabbit_cfg, config.get("heater_actuation", {}))
    heater.mq_command, heater.mq_action, heater.influx = _Recorder(), _Recorder(), _NullInflux()

    pipe_set = str(config.get("pipe_boundary", {}).get("pipe_boundary_set", "left"))
    limit_m = float(config.get("safety_monitor", {}).get("limit_radius_m", 0.040))

    snapshots = _sensor_rows(path, None, start_row, limit, stride, exclude_sensors)
    if not snapshots:
        raise SystemExit(f"No usable sensor rows parsed from {path}")

    # Data-time recalibration: see the module docstring. The wall-clock gate is
    # forced open on exactly the snapshots where the live service's timer would
    # have expired, and held shut on the rest.
    if len(snapshots) > 1:
        cadence_s = (float(snapshots[1]["time_hours"])
                     - float(snapshots[0]["time_hours"])) * 3600.0
    else:
        cadence_s = 30.0
    recal_every = max(int(round(recalibrate_data_s / max(cadence_s, 1e-9))), 1)
    print(f"Replay cadence {cadence_s:.0f} s -> pipe refit every {recal_every} snapshot(s) "
          f"({recal_every * cadence_s:.0f} s of data time)")
    print(f"Loaded {len(snapshots)} snapshots (t = {snapshots[0]['time_hours']:.3f} .. "
          f"{snapshots[-1]['time_hours']:.3f} h), excluding {sorted(exclude_sensors) or 'nothing'}")
    print(f"Limit {limit_m * 1000:.0f} mm | lookahead {forecast.lookahead_hours:g} h "
          f"in {forecast.lookahead_samples} samples | confirm {monitor.confirm_updates} "
          f"| f2_action {monitor.f2_action}")

    rows: list[dict] = []
    traces: list[dict] = []
    bc_seen: dict = {}

    for index, sensor_msg in enumerate(snapshots):
        t_h = float(sensor_msg["time_hours"])

        thaw._process_message(sensor_msg)
        thaw_msg = thaw.mq_out.last or {}

        boundary = builder._build_snapshot(sensor_msg)
        t_arr, tin_arr, tout_arr = fluid
        pipe.recalibrate_seconds = 0.0 if index % recal_every == 0 else 1.0e12
        pipe._on_fluid({
            "time_hours": t_h, "timestamp": sensor_msg["timestamp"],
            "t_inlet_c": float(np.interp(t_h, t_arr, tin_arr)),
            "t_outlet_c": float(np.interp(t_h, t_arr, tout_arr)),
            "site_id": "default",
        })
        pipe._on_boundary(boundary)
        enriched = pipe.mq_boundary_out.last
        if isinstance(enriched, dict):
            boundary = enriched

        if index == 0:
            bc_seen["first"] = (t_h, _describe_bc(boundary, pipe_set))
        bc_seen["last"] = (t_h, _describe_bc(boundary, pipe_set))

        forecast._process_message(boundary)
        fmsg = forecast.mq_out.last or {}

        monitor._handle_thaw(thaw_msg)
        monitor._handle_forecast(fmsg)
        alert = monitor.mq_out.last or {}

        heater._process_message(alert)
        cmd = heater.mq_command.last or {}

        if fmsg.get("lookahead_times_h"):
            traces.append({
                # Index, not time: matching a trace back to its row by float
                # equality silently misses (the row stores a rounded time) and
                # the panel then reports no predicates at all.
                "row": len(rows),
                "t": t_h,
                "times": list(fmsg["lookahead_times_h"]),
                "keep": list(fmsg["lookahead_radius_keep_m"]),
                "stop": list(fmsg["lookahead_radius_stop_m"]),
            })

        rows.append({
            "time_hours": round(t_h, 6),
            "r_max_measured_m": thaw_msg.get("radius_max_m"),
            "r_min_measured_m": thaw_msg.get("radius_min_m"),
            "r_keep_m": fmsg.get("radius_max_keep_m"),
            "r_stop_m": fmsg.get("radius_max_stop_m"),
            "ttb_hours": alert.get("time_to_breach_hours"),
            "m1": alert.get("predicate_m1"),
            "m2": alert.get("predicate_m2"),
            "f1": alert.get("predicate_f1"),
            "f2": alert.get("predicate_f2"),
            "status": alert.get("status"),
            "action": cmd.get("action"),
            "setpoint": cmd.get("setpoint"),
            "reason": alert.get("reason"),
        })

        if index % max(len(snapshots) // 10, 1) == 0 or index == len(snapshots) - 1:
            r_m = thaw_msg.get("radius_max_m")
            print(f"  [{index + 1:4d}/{len(snapshots)}] t={t_h:5.2f}h "
                  f"r_max={'--' if r_m is None else f'{r_m * 1000:5.1f}mm'} "
                  f"{alert.get('status', '-'):<19} {cmd.get('action', '-')}")

    print(f"\n  pipe BC at t={bc_seen['first'][0]:.2f}h: {bc_seen['first'][1]}")
    print(f"  pipe BC at t={bc_seen['last'][0]:.2f}h: {bc_seen['last'][1]}")
    return rows, {"traces": traces, "limit_m": limit_m,
                  "lookahead_hours": forecast.lookahead_hours,
                  "warn": monitor.warn_fraction, "critical": monitor.critical_fraction}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def col(rows, key, scale=1.0):
    out = []
    for r in rows:
        v = r.get(key)
        out.append(np.nan if v is None else float(v) * scale)
    return np.array(out)


def first_true(rows, key):
    for r in rows:
        if r.get(key):
            return float(r["time_hours"])
    return None


def status_spans(rows):
    """Contiguous runs of one status, as (status, t_start, t_end)."""
    spans = []
    if not rows:
        return spans
    cur = rows[0].get("status")
    start = float(rows[0]["time_hours"])
    for prev, nxt in zip(rows, rows[1:]):
        if nxt.get("status") != cur:
            spans.append((cur, start, float(nxt["time_hours"])))
            cur = nxt.get("status")
            start = float(nxt["time_hours"])
    spans.append((cur, start, float(rows[-1]["time_hours"])))
    return spans


# ---------------------------------------------------------------------------
# FIGURE 1 -- the decision timeline
# ---------------------------------------------------------------------------
def _stop_story(fig, axes, rows, t, limit_mm, mm):
    """Draw the stop instant in the stop-branch colour through every panel, so
    the cut reads as caused by that branch."""
    ax_r, ax_p, ax_a = axes
    t_stop = first_true(rows, "f2")
    if t_stop is None:
        return
    for ax in axes:
        ax.axvline(t_stop, color=FEM_STOP, lw=1.6, zorder=7)
    ax_a.annotate(f"stop, triggered by {LBL_F2}  {t_stop:.2f} h",
                  xy=(t_stop, 0), xytext=(-6, 4), textcoords="offset points",
                  ha="right", va="bottom", fontsize=6.8, color=FEM_STOP,
                  bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.2), zorder=8)


def fig_decision_timeline(rows, meta, out_dir: Path, *, model_label: str = "FEM",
                          stop_story: bool = False):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    mm = 1000.0
    limit_mm = meta["limit_m"] * mm
    t = col(rows, "time_hours")
    spans = status_spans(rows)

    fig, axes = plt.subplots(
        3, 1, figsize=(DOUBLE, DOUBLE * 0.78), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.05, 1.0], "hspace": 0.14})
    ax_r, ax_p, ax_a = axes

    # Status shading behind every panel, so a vertical read at any instant lands
    # in the same regime on all three.
    for ax in axes:
        for status, t0, t1 in spans:
            tint = STATUS_TINT.get(status)
            if tint:
                ax.axvspan(t0, t1, color=tint, lw=0, zorder=0)

    # --- radii ------------------------------------------------------------
    ax_r.axhline(limit_mm, color=STATUS_LIMIT, lw=1.2, ls="--", zorder=3)
    ax_r.annotate(f"limit {limit_mm:.0f} mm", xy=(0.008, limit_mm),
                  xycoords=("axes fraction", "data"), xytext=(0, 3),
                  textcoords="offset points", fontsize=7, color=STATUS_LIMIT)
    for frac, label in ((meta["warn"], "warn"), (meta["critical"], "critical")):
        ax_r.axhline(limit_mm * frac, color=MUTED, lw=0.7, ls=":", zorder=2)
        ax_r.annotate(f"{label} {limit_mm * frac:.0f}", xy=(0.008, limit_mm * frac),
                      xycoords=("axes fraction", "data"), xytext=(0, 2),
                      textcoords="offset points", fontsize=6.3, color=MUTED)

    ax_r.plot(t, col(rows, "r_keep_m", mm), "-.", color=FEM_KEEP, lw=1.4, zorder=5,
              label=f"{model_label}, keep heating")
    ax_r.plot(t, col(rows, "r_stop_m", mm), "--", color=FEM_STOP, lw=1.4, zorder=5,
              label=f"{model_label}, stop now")
    ax_r.plot(t, col(rows, "r_max_measured_m", mm), "-", color=INK, lw=1.7, zorder=6,
              label="measured $r_{max}$")
    ax_r.set_ylabel("thaw radius (mm)")
    ax_r.set_ylim(0, max(limit_mm * 1.45, np.nanmax(col(rows, "r_keep_m", mm)) * 1.06))
    ax_r.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, 1.02, 1, 0.1),
                mode="expand", borderaxespad=0)
    grid(ax_r)

    # --- predicates -------------------------------------------------------
    # Colour follows the *source*: measured predicates in ink, forecast
    # predicates in the branch colour they are computed from.
    preds = [("m1", f"{LBL_M1}  measured $\\geq$ limit", INK),
             ("m2", f"{LBL_M2}  measured elevated", MUTED),
             ("f1", f"{LBL_F1}  breach if heating continues", FEM_KEEP),
             ("f2", f"{LBL_F2}  breach even if stopped", FEM_STOP)]
    if stop_story:
        preds = [p for p in preds if p[0] != "m1"]
    for i, (key, label, colour) in enumerate(preds):
        y = len(preds) - 1 - i
        on = np.array([bool(r.get(key)) for r in rows])
        ax_p.broken_barh(
            [(t[j], (t[j + 1] - t[j]) if j + 1 < len(t) else 0.01)
             for j in range(len(t)) if on[j]],
            (y + 0.16, 0.68), facecolors=colour, edgecolors="none", zorder=4)
        ax_p.axhline(y + 0.5, color=MUTED, lw=0.4, alpha=0.35, zorder=1)
        ax_p.annotate(label, xy=(0.008, y + 0.5), xycoords=("axes fraction", "data"),
                      xytext=(0, 0), textcoords="offset points", va="center",
                      fontsize=6.6, color=colour, zorder=6,
                      bbox=dict(fc="white", ec="none", alpha=0.72, pad=1.2))
    ax_p.set_ylim(0, len(preds))
    ax_p.set_yticks([])
    ax_p.set_ylabel("predicates")

    # --- actuation --------------------------------------------------------
    ax_a.step(t, col(rows, "setpoint"), where="post", color=INK, lw=1.5, zorder=5)
    ax_a.set_ylabel("setpoint (°C)")
    ax_a.set_xlabel("time (h)")
    ax_a.set_ylim(-1.4, 12.5)
    grid(ax_a)

    # F_stop and M_lim fire ~20 min apart at the right-hand end of a 5 h axis,
    # so their callouts overlap each other on one line and the last one runs off
    # the edge. Step each callout down a row and flip it to the left of its rule
    # once the rule is close to the axis end.
    span = float(t[-1] - t[0]) or 1.0
    for stack, (key, label) in enumerate((("f1", LBL_F1), ("f2", LBL_F2), ("m1", LBL_M1))):
        when = first_true(rows, key)
        if when is None:
            continue
        for ax in axes:
            ax.axvline(when, color=MUTED, lw=0.8, ls=":", alpha=0.85, zorder=3)
        near_end = (t[-1] - when) / span < 0.18
        # In the radius panel, where there is headroom above the curves.
        ax_r.annotate(f"{label} {when:.2f} h", xy=(when, ax_r.get_ylim()[1]),
                      xytext=(-3 if near_end else 3, -3 - 11 * stack),
                      textcoords="offset points", fontsize=6.6,
                      color=MUTED, va="top", ha="right" if near_end else "left",
                      bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))

    used = [s for s in STATUS_ORDER if any(sp[0] == s for sp in spans) and STATUS_TINT.get(s)]
    handles = [Patch(facecolor=STATUS_TINT[s], edgecolor="none", label=s.replace("_", " "))
               for s in used]
    handles.append(Line2D([], [], color=INK, lw=1.5, label="commanded setpoint"))
    # A figure-level legend below everything: an axes-level one at this position
    # lands on top of the shared x-label.
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, 0.045), frameon=False, fontsize=6.8)
    fig.subplots_adjust(bottom=0.14)

    ax_a.set_xlim(t[0], t[-1])
    if stop_story:
        _stop_story(fig, axes, rows, t, limit_mm, mm)
    save(fig, out_dir, "fig_decision_timeline")


# ---------------------------------------------------------------------------
# FIGURE 2 -- what F1 and F2 actually test
# ---------------------------------------------------------------------------
def fig_lookahead_branches(rows, meta, out_dir: Path, *, show_measured: bool = True):
    """The two branches at the four moments the rule actually turns on.

    Panels are chosen by event, not spread evenly through the run: a nominal
    update, then the first firing of F_heat, of F_stop, and of M_lim. Evenly
    spaced panels mostly show the uninteresting middle and can miss every
    transition.
    """
    import matplotlib.pyplot as plt

    mm = 1000.0
    limit_mm = meta["limit_m"] * mm
    traces = meta["traces"]
    if not traces:
        print("  no lookahead traces recorded; skipping fig_lookahead_branches")
        return

    def trace_at(when):
        if when is None:
            return None
        return min(traces, key=lambda tr: abs(tr["t"] - when))

    events = [("nominal", trace_at(traces[len(traces) // 4]["t"])),
              (f"{LBL_F1} first fires", trace_at(first_true(rows, "f1"))),
              (f"{LBL_F2} first fires", trace_at(first_true(rows, "f2"))),
              (f"{LBL_M1} measured breach", trace_at(first_true(rows, "m1")))]
    events = [(lbl, tr) for lbl, tr in events if tr is not None]
    if not events:
        print("  no events to show; skipping fig_lookahead_branches")
        return

    fig, axes = plt.subplots(1, len(events), figsize=(DOUBLE, DOUBLE * 0.32),
                             sharey=True, gridspec_kw={"wspace": 0.08})
    axes = np.atleast_1d(axes)

    top = limit_mm * 1.45
    for ax, (label, tr) in zip(axes, events):
        t0 = tr["t"]
        # Index, not float equality: see the note where traces are recorded.
        row = rows[tr["row"]] if tr.get("row") is not None and tr["row"] < len(rows) else {}
        rel = (np.array(tr["times"]) - t0) * 60.0
        keep = np.array([np.nan if v is None else v * mm for v in tr["keep"]])
        stop = np.array([np.nan if v is None else v * mm for v in tr["stop"]])
        top = max(top, np.nanmax(keep) * 1.06)

        ax.axhline(limit_mm, color=STATUS_LIMIT, lw=1.1, ls="--", zorder=3)
        # The gap between the branches is what stopping buys; shading it makes
        # the closing of that gap the visible story across the four panels.
        ax.fill_between(rel, stop, keep, color=FEM_KEEP, alpha=0.13, lw=0, zorder=2)
        ax.plot(rel, keep, "-.", color=FEM_KEEP, lw=1.4, zorder=4)
        ax.plot(rel, stop, "--", color=FEM_STOP, lw=1.4, zorder=4)

        meas = row.get("r_max_measured_m")
        if show_measured and meas is not None:
            ax.plot([0], [meas * mm], "o", ms=4.4, color=INK, mec="white", mew=0.9, zorder=6)

        fired = [n for n, k in ((LBL_F1, "f1"), (LBL_F2, "f2"), (LBL_M1, "m1")) if row.get(k)]
        ax.set_title(f"{label}\nt = {t0:.2f} h", fontsize=7.2, pad=4)
        ax.annotate("firing: " + ", ".join(fired) if fired else "nothing fires",
                    xy=(0.05, 0.95), xycoords="axes fraction", fontsize=6.6,
                    va="top", color=STATUS_LIMIT if fired else MUTED,
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.5))
        ax.set_xlim(0, rel[-1])
        grid(ax)

    axes[0].set_ylabel("predicted $r_{max}$ (mm)")
    axes[0].set_ylim(0, top)
    # Pinned, not auto: moving the key below the row shortens the axes enough
    # that the automatic locator drops to 20 mm steps, and 10 mm is what makes
    # the distance between a branch and the 40 mm limit readable by eye.
    axes[0].yaxis.set_major_locator(plt.MultipleLocator(10))
    axes[0].annotate("limit", xy=(0.04, limit_mm), xycoords=("axes fraction", "data"),
                     xytext=(0, 3), textcoords="offset points", fontsize=6.5,
                     color=STATUS_LIMIT)
    # One shared x-label rather than four copies of the same three words, and
    # the key below it rather than inside a panel: it describes all four panels
    # equally, so parking it in the first one both reads as if it belonged to
    # that panel and covers the only stretch of it the curves occupy. The extra
    # bottom margin is what keeps the stack -- ticks, label, key -- clear of the
    # axes.
    fig.subplots_adjust(bottom=0.31)
    fig.supxlabel("minutes ahead of $t$", fontsize=8, y=0.125)

    handles = [
        plt.Line2D([], [], color=FEM_KEEP, ls="-.", lw=1.4, label=f"keep heating  ({LBL_F1})"),
        plt.Line2D([], [], color=FEM_STOP, ls="--", lw=1.4, label=f"stop now  ({LBL_F2})"),
    ]
    if show_measured:
        handles.append(plt.Line2D([], [], color=INK, marker="o", ls="none", ms=4.4,
                                  label="measured at $t$"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               ncol=len(handles), columnspacing=2.4, handlelength=2.6,
               fontsize=6.8, frameon=False)
    save(fig, out_dir, "fig_lookahead_branches")


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", default=DEFAULT_PATH)
    parser.add_argument("--fluid-inlet-col", default="Inlet")
    parser.add_argument("--fluid-outlet-col", default="Outlet")
    parser.add_argument("--fluid-time-col", default="Time (h).1")
    parser.add_argument("--start-row", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--exclude", default=DEFAULT_EXCLUDE)
    parser.add_argument("--pipe-recalibrate-data-seconds", type=float, default=300.0,
                        help="Refit the pipe model every N seconds of DATA time. The "
                             "default matches pipe_boundary.recalibrate_seconds, so the "
                             "replay refits exactly as often as the live service does.")
    parser.add_argument("--stop-after-breach-hours", type=float, default=0.15,
                        help="How far past the measured breach to keep plotting.")
    parser.add_argument("--config", default="startup.conf")
    parser.add_argument("--out", default="artifacts/predictive_control")
    parser.add_argument("--replot", action="store_true",
                        help="Re-render the figures from a previous run's "
                             "lookahead_traces.json instead of replaying.")
    args = parser.parse_args()

    config = load_startup_config(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.replot:
        cached = json.loads((out_dir / "lookahead_traces.json").read_text(encoding="utf-8"))
        rows = cached["rows"]
        meta = {k: cached[k] for k in ("traces", "limit_m", "lookahead_hours", "warn", "critical")}
        print(f"Replotting {len(rows)} cached updates from {out_dir}")
        style()
        fig_decision_timeline(rows, meta, out_dir)
        fig_lookahead_branches(rows, meta, out_dir)
        return

    rows, meta = run_replay(
        config, path=Path(args.path),
        fluid_cols=(args.fluid_inlet_col, args.fluid_outlet_col, args.fluid_time_col),
        start_row=args.start_row, limit=args.limit, stride=args.stride,
        exclude_sensors={s.strip() for s in args.exclude.split(",") if s.strip()},
        recalibrate_data_s=args.pipe_recalibrate_data_seconds)

    # The experiment is treated as ending shortly after the safety cut: what the
    # controller does once it has latched is not informative.
    breach = first_true(rows, "m1")
    if breach is not None:
        cutoff = breach + args.stop_after_breach_hours
        rows = [r for r in rows if float(r["time_hours"]) <= cutoff]
        meta["traces"] = [tr for tr in meta["traces"] if tr["t"] <= cutoff]

    print("\n--- first occurrence ---")
    for key, label in (("f1", "F1  breach if heating continues"),
                       ("f2", "F2  breach even if stopped now"),
                       ("m1", "M1  measured breach")):
        when = first_true(rows, key)
        print(f"  {label:<34} {'never' if when is None else f'{when:.3f} h'}")
    for act in ("reduce", "stop"):
        hit = next((float(r["time_hours"]) for r in rows if r.get("action") == act), None)
        print(f"  first '{act}' command{'':<15} {'never' if hit is None else f'{hit:.3f} h'}")
    if breach is not None:
        for act in ("reduce", "stop"):
            hit = next((float(r["time_hours"]) for r in rows if r.get("action") == act), None)
            if hit is not None:
                print(f"  lead of '{act}' over measured breach: {(breach - hit) * 60:.0f} min")

    path = out_dir / "predictive_updates.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)")

    # The replay costs the better part of an hour, so everything the figures
    # need is persisted alongside the CSV: --replot then re-renders in seconds.
    trace_path = out_dir / "lookahead_traces.json"
    trace_path.write_text(json.dumps({
        "rows": rows,
        "traces": meta["traces"],
        "limit_m": meta["limit_m"],
        "lookahead_hours": meta["lookahead_hours"],
        "warn": meta["warn"],
        "critical": meta["critical"],
    }), encoding="utf-8")
    print(f"  wrote {trace_path}")

    style()
    fig_decision_timeline(rows, meta, out_dir)
    fig_lookahead_branches(rows, meta, out_dir)


if __name__ == "__main__":
    main()

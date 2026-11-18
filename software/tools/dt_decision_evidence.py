"""Evidence pack for what the digital twin predicted, decided, and cost to run.

Replays a recorded experiment through the real thaw-front reconstruction,
boundary builder, pipe-boundary calibration, FEM forecast, safety monitor and
heater-actuation services -- RabbitMQ and InfluxDB stubbed -- and produces four
figures:

    01_short_horizon_predictions.png  what the twin predicted, one update ahead
    02_status_and_decision.png        the safety status and the decision it drove
    03_decisions_around_event.png     those decisions against the real intervention
    04_computation_time.png           what each update cost to compute

**On the horizon.** With ``fem_forecast.track_observed_time`` on -- which is the
shipped setting -- ``_resolve_step_hours`` advances the solver by the *elapsed*
time since the previous message, not by ``horizon_hours``. The published message
still carries ``horizon_hours``, but the state in it is a nowcast at the
observation time, not a prediction ahead of it. So a genuine short-horizon
prediction has to be produced deliberately: at each update this script snapshots
the solver, advances the copy by ``horizon_hours`` on the same boundary
condition, records the predicted front and thawed volume, and restores the state
so the tracked run is untouched. Figure 1 places each prediction at the time it
is *valid for*, next to what was actually measured then, so the two can be read
against each other rather than side by side.

**On the volume.** The forecast service publishes a front but no volume, so the
predicted volume is computed from the predicted front points using the thaw-front
service's own ``_compute_volume_metric``. Both sides of the comparison therefore
come from the same estimator, and any bias in it cancels.

**On the event.** The April 8 run has a real intervention: the circulator inlet
drops from about 20 C to about 12 C shortly after t = 5.25 h. Figure 3 puts the
twin's decisions on the same axis as that drop, so the question "did the twin
call it before the operator did, and by how long" has an answer with a number.
Note this needs the full-length sensor file: the polished export stops at
5.258 h, right as the event begins.

Run:

    python -m software.tools.dt_decision_evidence
    python -m software.tools.dt_decision_evidence --stride 4      # quick look
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.stats import theilslopes

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
from software.tools.fem_debug_replay import DEFAULT_FLUID_PATH, _NullInflux, _Sink, _sensor_rows
from software.tools.replay_csv import _iter_rows, _load_fluid_temps, _parse_header
from software.tools.safety_limit_calibration import (
    STATUS_COLOUR,
    _command_transitions,
    _Recorder,
)
from software.tools.verify_front_resolution import node_based_metrics

# The full-length export: 0 - 6.575 h, so the replay runs through the
# temperature-reduction event. The polished file stops at 5.258 h.
DEFAULT_SENSOR_PATH = "data/CR1000_Table1_April8_replay.xlsx"
# Stuck thermocouple in the unpolished export: the initial-condition
# interpolation would smear its reading across the mesh before the first step.
DEFAULT_EXCLUDE = "H290D165"

def surface_band_channels(path: Path, surface_exclusion_m: float) -> set[str]:
    """Channels shallower than the thaw service's surface exclusion depth.

    ``_compute_radius_metrics`` drops points above ``surface_exclusion_m`` before
    reducing them to a radius, but falls back to the *unfiltered* set whenever
    fewer than three survive. Early in a run the only thawed points are the
    near-surface artefact -- a horizontal 0 C layer spanning the full radial
    width -- so the fallback fires and pins r_max to the domain edge from the
    first snapshot, which reads downstream as an instant safety alert.

    Removing the shallow channels at the source keeps the fallback out of the
    picture. Derived from the file header and the configured depth rather than
    hardcoded: naming a row by hand silently misses a channel, and the resulting
    alert looks like a physical finding rather than a typo.
    """
    header = next(_iter_rows(path), [])
    _time_idx, columns = _parse_header(header, None)
    return {
        str(column["sensor_id"])
        for column in columns
        if float(column["z_m"]) < float(surface_exclusion_m)
    }


MEAS = "#111111"
NOWCAST = "#1F6FA8"
PREDICT = "#C2531F"
KEEP = "#C2531F"
STOP = "#3F7D3A"


# ---------------------------------------------------------------------------
# PREDICTIVE MACHINERY
# ---------------------------------------------------------------------------
def stopped_boundary(boundary: dict, pipe_set: str, stop_setpoint_c: float) -> dict:
    """The same boundary snapshot with the heater turned off.

    Only the pipe boundary changes: everything else -- the deep frozen anchor,
    the insulated faces, the mesh, the current field -- is what it was. The
    difference between advancing this and advancing the original is therefore
    the effect of the decision and nothing else, which is the whole point of
    running the two branches.
    """
    snapshot = deepcopy(boundary)
    for bc in snapshot.get("boundaries", []):
        if bc.get("boundary_set") == pipe_set and bc.get("bc_type") in {"robin", "dirichlet"}:
            # A depth-resolved profile would otherwise keep driving heat in and
            # silently override the scalar setpoint below.
            bc.pop("temperature_profile", None)
            bc["temperature"] = float(stop_setpoint_c)
    return snapshot


def lookahead(solver, boundary: dict, lead_hours: float, samples: int, limit_m: float):
    """Advance a copy of the solver, sampling the front as it goes.

    Returns (final r_max, time-to-limit, final front points). The solver state is
    restored before returning: the tracked run must not inherit the lookahead, or
    simulated time races ahead of observed time and every later step is wrong.

    Sampling rather than advancing in one jump is what makes a time-to-limit
    available at all -- the crossing time is the quantity the decision needs, and
    it cannot be recovered from the endpoint alone.
    """
    saved_T = solver.T.copy()
    saved_time = solver.time_hours
    step = lead_hours / max(samples, 1)
    ttl = None
    r_final = None
    points: list[dict] = []
    try:
        for k in range(1, max(samples, 1) + 1):
            solver.advance(boundary, horizon_hours=step)
            r_final, _r_avg, points = solver.compute_thaw_metrics()
            if ttl is None and r_final is not None and r_final >= limit_m:
                ttl = k * step
    finally:
        solver.T = saved_T
        solver.time_hours = saved_time
    return r_final, ttl, points


def monitored_time_to_breach(history: deque, limit_m: float) -> tuple[float | None, float | None]:
    """Time to breach implied by the monitored front, as a monitoring readout.

    A robust (Theil-Sen) slope over the recent measured radii, extrapolated to
    the limit. It reports how much time is left *at the currently observed rate*
    and nothing more: it is a reduction of the monitoring stream, not a model of
    the process.

    Deliberately naive, and structurally limited in a way worth stating. Knowing
    only a rate, it cannot represent heat already stored in the ground, cannot
    anticipate the front accelerating, and above all cannot answer the question
    the control decision actually turns on -- what happens if the heater is
    switched off now. Only the forward model can evaluate that counterfactual,
    because only the forward model has a boundary condition to change. This
    quantity corroborates the forecast from the measurement side; it does not
    substitute for it.

    Returns (slope in m/h, time-to-breach in hours).
    """
    if len(history) < 4:
        return None, None
    t = np.array([h[0] for h in history], dtype=float)
    r = np.array([h[1] for h in history], dtype=float)
    good = np.isfinite(t) & np.isfinite(r)
    if int(np.count_nonzero(good)) < 4 or float(np.ptp(t[good])) <= 0.0:
        return None, None
    slope = float(theilslopes(r[good], t[good])[0])
    if slope <= 0.0:
        return slope, None
    remaining = limit_m - float(r[good][-1])
    return slope, max(remaining, 0.0) / slope
STAGES = ["thaw_front", "boundary", "forecast", "lookahead", "safety", "heater"]
STAGE_COLOUR = {
    "thaw_front": "#3F7D3A", "boundary": "#7B6FA8", "forecast": "#1F6FA8",
    "lookahead": "#C2531F", "safety": "#D8A21B", "heater": "#B3261E",
}


# ---------------------------------------------------------------------------
# REPLAY
# ---------------------------------------------------------------------------
def run_replay(config: dict, *, path: Path, fluid_path: Path | None, start_row: int,
               limit: int | None, stride: int, exclude_sensors: set[str] | None,
               predict_every: int, lead_hours: float, breach_samples: int,
               monitor_window_h: float) -> tuple[list[dict], list[dict], list[dict]]:
    rabbit_cfg = config.get("rabbitmq", {})
    influx_cfg: InfluxConfig = build_influx_config(config.get("influxdb", {}))
    horizon_hours = float(config.get("fem_forecast", {}).get("horizon_hours", 0.17))

    thaw = create_thaw_front_reconstruction_server(influx_cfg, rabbit_cfg,
                                                   config.get("thaw_front_reconstruction", {}))
    thaw.mq_out, thaw.influx = _Recorder(), _NullInflux()

    builder = create_boundary_condition_builder_server(rabbit_cfg,
                                                       config.get("boundary_condition_builder", {}))
    fem_cfg = dict(config.get("fem_forecast", {}))
    fem_cfg["run_baseline"] = False
    forecast = create_fem_forecast_server(influx_cfg, rabbit_cfg, fem_cfg)
    forecast.mq_out, forecast.influx = _Recorder(), _NullInflux()

    pipe = None
    fluid = None
    pipe_cfg = config.get("pipe_boundary", {})
    if pipe_cfg and fluid_path is not None:
        pipe = create_pipe_boundary_server(influx_cfg, rabbit_cfg, pipe_cfg)
        pipe.mq_boundary_out = _Sink()
        fluid = _load_fluid_temps(fluid_path, inlet_col="Fluid Temperature_Inlet",
                                  outlet_col="Fluid Temperatue_Outlet", time_col="Time (h)")

    safety_cfg = config.get("safety_monitor", {})
    limit_m = float(safety_cfg.get("limit_radius_m", 0.040))
    pipe_set = str(config.get("pipe_boundary", {}).get("pipe_boundary_set", "left"))
    stop_setpoint = float(config.get("heater_actuation", {}).get("setpoint_on_alert", 0.0) or 0.0)

    monitor = create_safety_monitor_server(influx_cfg, rabbit_cfg, safety_cfg)
    monitor.mq_out, monitor.influx = _Recorder(), _NullInflux()

    heater = create_heater_actuation_server(influx_cfg, rabbit_cfg, config.get("heater_actuation", {}))
    heater.mq_command, heater.mq_action, heater.influx = _Recorder(), _Recorder(), _NullInflux()

    snapshots = _sensor_rows(path, None, start_row, limit, stride, exclude_sensors)
    if not snapshots:
        raise SystemExit(f"No usable sensor rows parsed from {path}")
    print(f"Loaded {len(snapshots)} snapshots (t = {snapshots[0]['time_hours']:.3f} .. "
          f"{snapshots[-1]['time_hours']:.3f} h)")
    print(f"Lead time: {lead_hours:g} h ({lead_hours * 60:.0f} min) in {breach_samples} samples, "
          f"lookahead every {predict_every} update(s)")
    print(f"Limit: {limit_m * 1000:.0f} mm   heater-off setpoint for the stop branch: {stop_setpoint:g} C")

    rows: list[dict] = []
    alerts: list[dict] = []
    # Causal window for the monitored rate: only past measurements ever enter.
    history: deque = deque()
    for index, sensor_msg in enumerate(snapshots):
        t_h = float(sensor_msg["time_hours"])
        timing: dict[str, float] = {}

        # -- 1. measured front ------------------------------------------------
        t0 = time.perf_counter()
        thaw._process_message(sensor_msg)
        timing["thaw_front"] = time.perf_counter() - t0
        thaw_msg = thaw.mq_out.last or {}

        # -- 2. boundary conditions -------------------------------------------
        t0 = time.perf_counter()
        boundary = builder._build_snapshot(sensor_msg)
        if pipe is not None and fluid is not None:
            t_arr, tin_arr, tout_arr = fluid
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
        timing["boundary"] = time.perf_counter() - t0

        # -- 3. forecast (tracked step: a nowcast, see module docstring) -------
        t0 = time.perf_counter()
        forecast._process_message(boundary)
        timing["forecast"] = time.perf_counter() - t0
        fmsg = forecast.mq_out.last or {}

        # -- 4. two-branch lookahead: keep heating, or stop now ----------------
        pred = {"keep_r": None, "stop_r": None, "volume": None, "valid_at": None,
                "ttb_fem": None, "ttb_monitored": None, "slope": None,
                "unavoidable": False, "imminent": False}
        timing["lookahead"] = 0.0

        measured_r = thaw_msg.get("radius_max_m")
        if measured_r is not None:
            history.append((t_h, float(measured_r)))
            while history and history[0][0] < t_h - monitor_window_h:
                history.popleft()
        slope, ttb_monitored = monitored_time_to_breach(history, limit_m)
        pred["slope"], pred["ttb_monitored"] = slope, ttb_monitored

        if forecast._solver is not None and index % predict_every == 0:
            t0 = time.perf_counter()
            keep_r, ttb_fem, keep_points = lookahead(
                forecast._solver, boundary, lead_hours, breach_samples, limit_m)
            stop_r, _stop_ttl, _stop_points = lookahead(
                forecast._solver,
                stopped_boundary(boundary, pipe_set, stop_setpoint),
                lead_hours, breach_samples, limit_m)
            timing["lookahead"] = time.perf_counter() - t0

            pred.update({
                "keep_r": keep_r,
                "stop_r": stop_r,
                "volume": thaw._compute_volume_metric(keep_points) if keep_points else None,
                "valid_at": t_h + lead_hours,
                "ttb_fem": ttb_fem,
            })

        # The decision rule. "unavoidable" is the last-safe-moment test: cutting
        # heat right now still breaches, so now is the latest instant at which
        # stopping would have worked. "imminent" is the softer one: on current
        # behaviour the limit arrives inside the lead time.
        pred["unavoidable"] = bool(pred["stop_r"] is not None and pred["stop_r"] >= limit_m)
        ttls = [x for x in (pred["ttb_fem"], pred["ttb_monitored"]) if x is not None]
        pred["imminent"] = bool(ttls and min(ttls) < lead_hours)

        # -- 5. safety monitor -------------------------------------------------
        t0 = time.perf_counter()
        stamp = f"t={t_h:.4f}"
        if fmsg.get("radius_max_m") is not None:
            monitor._handle_forecast({"timestamp": stamp, "time_hours": t_h,
                                      "radius_max_m": fmsg.get("radius_max_m"),
                                      "radius_min_m": None, "radius_avg_m": fmsg.get("radius_avg_m")})
        if thaw_msg.get("radius_max_m") is not None:
            monitor._handle_thaw({"timestamp": stamp, "time_hours": t_h,
                                  "radius_max_m": thaw_msg.get("radius_max_m"),
                                  "radius_min_m": thaw_msg.get("radius_min_m"),
                                  "radius_avg_m": thaw_msg.get("radius_avg_m")})
        timing["safety"] = time.perf_counter() - t0

        # -- 6. heater actuation ------------------------------------------------
        t0 = time.perf_counter()
        status = action = reason = None
        while monitor.mq_out.messages:
            alert = monitor.mq_out.messages.pop(0)
            alert["_time_hours"] = t_h
            heater._process_message(alert)
            alerts.append(alert)
            status, action, reason = alert.get("status"), alert.get("recommended_action"), alert.get("reason")
        timing["heater"] = time.perf_counter() - t0

        rows.append({
            "time_hours": round(t_h, 6),
            "r_max_measured_m": thaw_msg.get("radius_max_m"),
            "r_min_measured_m": thaw_msg.get("radius_min_m"),
            "volume_measured_m3": thaw_msg.get("thaw_volume_m3"),
            "r_max_nowcast_m": fmsg.get("radius_max_m"),
            # The pre-interpolation metric, computed from the same field, so the
            # resolution change can be shown exactly rather than reconstructed.
            "r_max_nowcast_node_m": (node_based_metrics(forecast._solver)[0]
                                     if forecast._solver is not None else None),
            "r_max_keep_m": pred["keep_r"],
            "r_max_stop_m": pred["stop_r"],
            "volume_predicted_m3": pred["volume"],
            "prediction_valid_at_h": pred["valid_at"],
            "ttb_fem_h": pred["ttb_fem"],
            "ttb_monitored_h": pred["ttb_monitored"],
            "front_rate_m_per_h": pred["slope"],
            "predict_unavoidable": pred["unavoidable"],
            "predict_imminent": pred["imminent"],
            "predictive_stop": bool(pred["unavoidable"] or pred["imminent"]),
            "status": status, "recommended_action": action, "reason": reason,
            **{f"ms_{k}": timing.get(k, 0.0) * 1000.0 for k in STAGES},
            "ms_total": sum(timing.values()) * 1000.0,
        })

        if index % max(len(snapshots) // 12, 1) == 0 or index == len(snapshots) - 1:
            print(f"  [{index + 1:4d}/{len(snapshots)}] t={t_h:6.3f} h  "
                  f"status={status or '-':<8} {rows[-1]['ms_total']:6.0f} ms")

    return rows, alerts, _command_transitions(heater.mq_command.messages)


# ---------------------------------------------------------------------------
def detect_event(fluid, *, drop_c: float = 3.0, window_h: float = 0.15) -> float | None:
    """Onset of the temperature-reduction event, from the fluid inlet trace.

    The first moment the inlet falls by ``drop_c`` within ``window_h``. A simple
    threshold on a rolling drop rather than a change-point fit: the intervention
    is an operator turning the setpoint down, so it is abrupt and large compared
    with the noise, and anything cleverer would only add assumptions.
    """
    if fluid is None:
        return None
    t, inlet, _outlet = fluid
    for i in range(len(t)):
        later = (t > t[i]) & (t <= t[i] + window_h)
        if np.any(later) and (inlet[i] - float(np.min(inlet[later]))) >= drop_c:
            return float(t[i])
    return None


def _status_spans(rows):
    """Contiguous runs of identical status, as (start_h, end_h, status)."""
    spans, start, current = [], None, None
    for row in rows:
        s = row.get("status")
        if s != current:
            if current is not None and start is not None:
                spans.append((start, row["time_hours"], current))
            current, start = s, row["time_hours"]
    if current is not None and start is not None:
        spans.append((start, rows[-1]["time_hours"], current))
    return [s for s in spans if s[2] is not None]


# ---------------------------------------------------------------------------
# FIGURES
# ---------------------------------------------------------------------------
def fig_predictions(rows, out_dir, horizon_hours):
    import matplotlib.pyplot as plt

    t = np.array([r["time_hours"] for r in rows], dtype=float)
    meas_r = np.array([np.nan if r["r_max_measured_m"] is None else r["r_max_measured_m"] * 1000 for r in rows])
    meas_v = np.array([np.nan if r["volume_measured_m3"] is None else r["volume_measured_m3"] * 1e6 for r in rows])

    pred = [r for r in rows if r["r_max_keep_m"] is not None]
    pt = np.array([r["prediction_valid_at_h"] for r in pred], dtype=float)
    pr = np.array([r["r_max_keep_m"] * 1000 for r in pred], dtype=float)
    ps = np.array([np.nan if r["r_max_stop_m"] is None else r["r_max_stop_m"] * 1000 for r in pred])
    pv = np.array([np.nan if r["volume_predicted_m3"] is None else r["volume_predicted_m3"] * 1e6 for r in pred])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 9.0), sharex=True,
                                        gridspec_kw={"height_ratios": [2, 2, 1.2]})

    ax1.plot(t, meas_r, color=MEAS, lw=1.9, label="measured (actual outcome)")
    ax1.plot(t, [np.nan if r["r_max_nowcast_m"] is None else r["r_max_nowcast_m"] * 1000 for r in rows],
             color=NOWCAST, lw=1.5, alpha=0.85, label="FEM nowcast (tracked step)")
    ax1.plot(pt, pr, color=KEEP, lw=1.7, ls="--", label="predicted if heating continues")
    ax1.plot(pt, ps, color=STOP, lw=1.7, ls=":", label="predicted if stopped now")
    ax1.set_ylabel("front radius r_max (mm)")
    ax1.set_title(f"Predictions at each update, both branches "
                  f"(lead time = {horizon_hours * 60:.0f} min)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(alpha=0.22)

    ax2.plot(t, meas_v, color=MEAS, lw=1.9, label="measured thawed volume")
    ax2.plot(pt, pv, color=PREDICT, lw=1.6, ls="--", label="predicted thawed volume")
    ax2.set_ylabel("thawed volume (cm$^3$)")
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.grid(alpha=0.22)

    # Prediction error: each prediction against what was actually measured at the
    # time it claimed to describe.
    finite = np.isfinite(meas_r)
    if np.any(finite) and pt.size:
        actual = np.interp(pt, t[finite], meas_r[finite], left=np.nan, right=np.nan)
        ax3.axhline(0.0, color="0.4", lw=1.0)
        ax3.plot(pt, pr - actual, color=PREDICT, lw=1.5)
        ax3.set_ylabel("prediction error\n(mm)")
    ax3.set_xlabel("Experiment time (h)")
    ax3.grid(alpha=0.22)

    fig.tight_layout()
    fig.savefig(out_dir / "01_short_horizon_predictions.png", dpi=150)
    print(f"  wrote {out_dir / '01_short_horizon_predictions.png'}")
    plt.close(fig)


def _rows_from_csv(path: Path) -> list[dict]:
    """Rebuild the update rows from a previous run's ``dt_updates.csv``.

    Only what the status figure reads is typed back: the times and radii as
    floats, the status as a string, blanks as None. The heater commands are not
    in the CSV, which is why ``--replot`` can rebuild the two-panel status
    figure and not the setpoint panel or the event figure.
    """
    numeric = {"time_hours", "r_max_measured_m", "r_min_measured_m"}
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict = {}
            for key, value in raw.items():
                value = (value or "").strip()
                if key in numeric:
                    row[key] = float(value) if value else None
                elif key == "status":
                    row[key] = value or None
                else:
                    row[key] = value or None
            if row.get("time_hours") is not None:
                rows.append(row)
    return rows


def fig_status_decision(rows, commands, out_dir, config, *, include_decision=True):
    """Front radii against the safety thresholds, the status band, and the command.

    ``include_decision=False`` drops the heater-setpoint panel and writes
    ``02_status.png`` instead. The two remaining panels answer "where was the
    front and what did the monitor call it", which is a self-contained claim;
    the setpoint panel answers "and what was done about it", and carrying it
    along forces the figure to be a third taller than that claim needs. The
    title changes with the panel set rather than promising a decision the
    figure no longer shows.
    """
    import matplotlib.pyplot as plt

    safety = config.get("safety_monitor", {})
    limit = float(safety.get("limit_radius_m", 0.040))
    warn_f, crit_f = float(safety.get("warn_fraction", 0.6)), float(safety.get("critical_fraction", 0.9))
    t = np.array([r["time_hours"] for r in rows], dtype=float)

    include_decision = bool(include_decision and commands)
    if include_decision:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 8.0), sharex=True,
                                            gridspec_kw={"height_ratios": [2.2, 0.7, 1.2]})
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True,
                                       gridspec_kw={"height_ratios": [2.2, 0.7]})
        ax3 = None

    ax1.plot(t, [np.nan if r["r_max_measured_m"] is None else r["r_max_measured_m"] * 1000 for r in rows],
             color=MEAS, lw=1.9, label="r_max measured")
    ax1.plot(t, [np.nan if r["r_min_measured_m"] is None else r["r_min_measured_m"] * 1000 for r in rows],
             color="#4A6FA5", lw=1.4, label="r_min measured")
    for frac, name in ((warn_f, "warning"), (crit_f, "critical"), (1.0, "alert")):
        ax1.axhline(limit * frac * 1000, color=STATUS_COLOUR[name], lw=1.3, ls="--")
        ax1.text(t[-1], limit * frac * 1000, f" {name}", color=STATUS_COLOUR[name],
                 fontsize=8.5, va="center")
    ax1.set_ylabel("radius (mm)")
    ax1.set_title("Safety status and the decision it drove" if include_decision
                  else "Measured thaw front against the safety thresholds", fontsize=11)
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(alpha=0.22)

    for a, b, status in _status_spans(rows):
        ax2.axvspan(a, b, color=STATUS_COLOUR.get(status, "#999"), alpha=0.9)
        if b - a > (t[-1] - t[0]) * 0.04:
            ax2.text((a + b) / 2, 0.5, status, ha="center", va="center",
                     color="white", fontsize=9, fontweight="bold")
    ax2.set_yticks([])
    ax2.set_ylabel("status")

    if ax3 is None:
        # The status band is the bottom axis now, so it carries the x-label the
        # setpoint panel used to.
        ax2.set_xlabel("Experiment time (h)")
        fig.tight_layout()
        # Vector PDF for the manuscript, raster PNG for quick viewing. Nothing
        # here is an image, so the PDF has no resolution to speak of -- it
        # scales to any figure width. fonttype 42 embeds TrueType instead of
        # matplotlib's default Type 3, which most journals reject and which
        # leaves the text unsearchable and badly hinted on screen.
        with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
            fig.savefig(out_dir / "02_status.pdf")
        fig.savefig(out_dir / "02_status.png", dpi=300)
        print(f"  wrote {out_dir / '02_status'}.pdf / .png")
        plt.close(fig)
        return

    if commands:
        ct = [c.get("time_hours", np.nan) for c in commands]
        sp = [0.0 if c.get("setpoint") is None else float(c["setpoint"]) for c in commands]
        ax3.step(list(ct) + [t[-1]], sp + [sp[-1]], where="post", color="#B3261E", lw=2.2)
        for x, s, c in zip(ct, sp, commands):
            ax3.axvline(x, color="0.6", lw=0.9, ls=":")
            ax3.annotate(f"{c.get('action')} @ {s:g} C", xy=(x, s), xytext=(5, 7),
                         textcoords="offset points", fontsize=8, color="#B3261E")
        ax3.set_ylim(-1.5, max(sp) + 4)
    ax3.set_ylabel("heater\nsetpoint (C)")
    ax3.set_xlabel("Experiment time (h)")
    ax3.grid(alpha=0.22)

    fig.tight_layout()
    fig.savefig(out_dir / "02_status_and_decision.png", dpi=150)
    print(f"  wrote {out_dir / '02_status_and_decision.png'}")
    plt.close(fig)


def fig_event(rows, commands, fluid, event_h, out_dir, config):
    import matplotlib.pyplot as plt

    if fluid is None:
        print("  skipped 03_decisions_around_event.png (no fluid file)")
        return
    ft, inlet, outlet = fluid
    t = np.array([r["time_hours"] for r in rows], dtype=float)
    limit = float(config.get("safety_monitor", {}).get("limit_radius_m", 0.040))

    lo = max(0.0, (event_h - 1.5) if event_h else t[0])
    hi = max(t[-1], float(ft[-1]))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 8.4), sharex=True,
                                        gridspec_kw={"height_ratios": [1.5, 1.8, 0.8]})

    ax1.plot(ft, inlet, color="#B3261E", lw=1.8, label="circulator inlet")
    ax1.plot(ft, outlet, color="#1F6FA8", lw=1.6, label="outlet")
    ax1.set_ylabel("fluid temperature (C)")
    ax1.set_title("Twin decisions against the real temperature-reduction event", fontsize=11)
    ax1.legend(loc="lower left", fontsize=8.5)
    ax1.grid(alpha=0.22)

    ax2.plot(t, [np.nan if r["r_max_measured_m"] is None else r["r_max_measured_m"] * 1000 for r in rows],
             color=MEAS, lw=1.9, label="r_max measured")
    pred = [r for r in rows if r["r_max_keep_m"] is not None]
    if pred:
        ax2.plot([r["prediction_valid_at_h"] for r in pred],
                 [r["r_max_keep_m"] * 1000 for r in pred],
                 color=KEEP, lw=1.5, ls="--", label="predicted (keep heating)")
        ax2.plot([r["prediction_valid_at_h"] for r in pred],
                 [np.nan if r["r_max_stop_m"] is None else r["r_max_stop_m"] * 1000 for r in pred],
                 color=STOP, lw=1.5, ls=":", label="predicted (stop now)")
    ax2.axhline(limit * 1000, color=STATUS_COLOUR["alert"], lw=1.3, ls="--")
    ax2.text(hi, limit * 1000, " alert limit", color=STATUS_COLOUR["alert"], fontsize=8.5, va="center")
    ax2.set_ylabel("radius (mm)")
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.grid(alpha=0.22)

    for a, b, status in _status_spans(rows):
        ax3.axvspan(a, b, color=STATUS_COLOUR.get(status, "#999"), alpha=0.9)
    ax3.set_yticks([])
    ax3.set_ylabel("status")
    ax3.set_xlabel("Experiment time (h)")

    first_stop = next((c for c in commands if c.get("action") == "stop"), None)
    for ax in (ax1, ax2, ax3):
        if event_h is not None:
            ax.axvline(event_h, color="#333", lw=2.0)
        if first_stop is not None:
            ax.axvline(first_stop["time_hours"], color="#B3261E", lw=2.0, ls="--")
        ax.set_xlim(lo, hi)

    if event_h is not None:
        ax1.annotate(f"operator reduces\ntemperature\nt = {event_h:.2f} h",
                     xy=(event_h, float(np.interp(event_h, ft, inlet))),
                     xytext=(-104, -46), textcoords="offset points", fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="#333"))
    if first_stop is not None:
        ax2.annotate(f"twin says STOP\nt = {first_stop['time_hours']:.2f} h",
                     xy=(first_stop["time_hours"], limit * 1000), xytext=(-118, 22),
                     textcoords="offset points", fontsize=9, color="#B3261E",
                     arrowprops=dict(arrowstyle="->", color="#B3261E"))
        if event_h is not None:
            lead = (event_h - first_stop["time_hours"]) * 60.0
            ax2.set_title(f"twin decision leads the intervention by {lead:.0f} min"
                          if lead > 0 else
                          f"twin decision lags the intervention by {abs(lead):.0f} min",
                          fontsize=10)

    fig.tight_layout()
    fig.savefig(out_dir / "03_decisions_around_event.png", dpi=150)
    print(f"  wrote {out_dir / '03_decisions_around_event.png'}")
    plt.close(fig)


def fig_compute_time(rows, out_dir, cadence_s):
    import matplotlib.pyplot as plt

    t = np.array([r["time_hours"] for r in rows], dtype=float)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.0),
                                   gridspec_kw={"height_ratios": [2, 1.3]})

    series = [np.array([r[f"ms_{s}"] for r in rows], dtype=float) for s in STAGES]
    ax1.stackplot(t, *series, labels=STAGES, colors=[STAGE_COLOUR[s] for s in STAGES], alpha=0.9)
    ax1.axhline(cadence_s * 1000, color="black", lw=1.6, ls="--")
    ax1.text(t[-1], cadence_s * 1000, f"  {cadence_s:g} s data cadence",
             fontsize=8.5, va="center")
    ax1.set_ylabel("computation time per update (ms)")
    ax1.set_xlabel("Experiment time (h)")
    ax1.set_title("Cost of one digital-twin update, by stage", fontsize=11)
    ax1.legend(loc="upper left", fontsize=8.5, ncol=3)
    ax1.grid(alpha=0.22)

    totals = np.array([r["ms_total"] for r in rows], dtype=float)
    means = [float(np.mean([r[f"ms_{s}"] for r in rows])) for s in STAGES]
    order = np.argsort(means)[::-1]
    ax2.barh([STAGES[i] for i in order], [means[i] for i in order],
             color=[STAGE_COLOUR[STAGES[i]] for i in order])
    for y, i in enumerate(order):
        ax2.text(means[i], y, f"  {means[i]:.1f} ms ({100 * means[i] / sum(means):.0f}%)",
                 va="center", fontsize=9)
    ax2.set_xlabel("mean time per update (ms)")
    ax2.set_xlim(0, max(means) * 1.35)
    ax2.grid(alpha=0.22, axis="x")
    ax2.set_title(f"total: mean {np.mean(totals):.0f} ms, p95 {np.percentile(totals, 95):.0f} ms, "
                  f"max {np.max(totals):.0f} ms  —  budget {cadence_s * 1000:.0f} ms "
                  f"({100 * np.mean(totals) / (cadence_s * 1000):.1f}% used)", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_dir / "04_computation_time.png", dpi=150)
    print(f"  wrote {out_dir / '04_computation_time.png'}")
    plt.close(fig)


def first_true(rows, key):
    """Time of the first row where `key` is set, or None."""
    for row in rows:
        if row.get(key):
            return float(row["time_hours"])
    return None


def first_measured_breach(rows, limit_m):
    for row in rows:
        r = row.get("r_max_measured_m")
        if r is not None and float(r) >= limit_m:
            return float(row["time_hours"])
    return None


def fig_lead_time(rows, out_dir, config, event_h, lead_hours):
    """The claim, stated as a number: does prediction buy time over reacting?"""
    import matplotlib.pyplot as plt

    limit_m = float(config.get("safety_monitor", {}).get("limit_radius_m", 0.040))
    t = np.array([r["time_hours"] for r in rows], dtype=float)
    meas = np.array([np.nan if r["r_max_measured_m"] is None else r["r_max_measured_m"] * 1000
                     for r in rows])

    t_react = first_measured_breach(rows, limit_m)
    t_pred = first_true(rows, "predictive_stop")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.4), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1.4]})

    ax1.plot(t, meas, color=MEAS, lw=1.9, label="r_max measured")
    keep = [r for r in rows if r["r_max_keep_m"] is not None]
    if keep:
        ax1.plot([r["prediction_valid_at_h"] for r in keep],
                 [r["r_max_keep_m"] * 1000 for r in keep],
                 color=KEEP, lw=1.5, ls="--", label="if heating continues")
        ax1.plot([r["prediction_valid_at_h"] for r in keep],
                 [np.nan if r["r_max_stop_m"] is None else r["r_max_stop_m"] * 1000 for r in keep],
                 color=STOP, lw=1.5, ls=":", label="if stopped now")
    ax1.axhline(limit_m * 1000, color="#B3261E", lw=1.4, ls="--")
    ax1.text(t[-1], limit_m * 1000, " limit", color="#B3261E", fontsize=8.5, va="center")

    for when, colour, label in ((t_pred, "#C2531F", "predictive stop"),
                                (t_react, "#B3261E", "reactive stop (measured)"),
                                (event_h, "#333333", "operator intervention")):
        if when is not None:
            ax1.axvline(when, color=colour, lw=2.0, ls="--" if label != "operator intervention" else "-")
            ax2.axvline(when, color=colour, lw=2.0, ls="--" if label != "operator intervention" else "-")
            ax1.text(when, ax1.get_ylim()[1], f" {label}\n t={when:.2f} h", color=colour,
                     fontsize=8.5, va="top", rotation=0)
    ax1.set_ylabel("radius (mm)")
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(alpha=0.22)

    if t_pred is not None and t_react is not None:
        lead = (t_react - t_pred) * 60.0
        ax1.set_title(f"Prediction fires {lead:+.0f} min before the measurement crosses the limit",
                      fontsize=11)
    else:
        ax1.set_title("Predictive vs reactive trigger", fontsize=11)

    # Time-to-limit from both predictors: the quantity the decision reads.
    for key, colour, label in (("ttb_fem_h", KEEP, "time to limit (FEM)"),
                               ("ttb_monitored_h", "#1F6FA8", "monitored time-to-breach")):
        vals = np.array([np.nan if r.get(key) in (None, "") else float(r[key]) * 60.0 for r in rows])
        if np.any(np.isfinite(vals)):
            ax2.plot(t, vals, color=colour, lw=1.6, label=label)
    ax2.axhline(lead_hours * 60.0, color="#B3261E", lw=1.3, ls="--")
    ax2.text(t[-1], lead_hours * 60.0, "  lead time required", color="#B3261E",
             fontsize=8.5, va="center")
    ax2.set_ylabel("time to limit (min)")
    ax2.set_xlabel("Experiment time (h)")
    ax2.set_ylim(0, lead_hours * 60.0 * 4)
    if ax2.get_legend_handles_labels()[0]:
        ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(alpha=0.22)

    fig.tight_layout()
    fig.savefig(out_dir / "05_predictive_lead_time.png", dpi=150)
    print(f"  wrote {out_dir / '05_predictive_lead_time.png'}")
    plt.close(fig)
    return t_pred, t_react


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", default=DEFAULT_SENSOR_PATH)
    parser.add_argument("--fluid-path", default=DEFAULT_FLUID_PATH)
    parser.add_argument("--config", default="startup.conf")
    parser.add_argument("--out", default="artifacts/dt_evidence")
    parser.add_argument("--start-row", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--predict-every", type=int, default=5,
                        help="Run the two-branch lookahead every N updates.")
    parser.add_argument("--lead-time-hours", type=float, default=0.75,
                        help="Lead time the decision must respect. Default 0.75 h (45 min), "
                             "from the measured thermal lag after the recorded intervention.")
    parser.add_argument("--breach-samples", type=int, default=10,
                        help="Samples along each lookahead; sets time-to-limit resolution.")
    parser.add_argument("--monitor-window-minutes", type=float, default=15.0,
                        help="Window of monitored radii used for the time-to-breach readout.")
    parser.add_argument("--exclude-sensors", default=DEFAULT_EXCLUDE)
    parser.add_argument("--keep-surface-band", action="store_true",
                        help="Keep channels inside the surface-exclusion depth (shows the "
                             "artefact the thaw service falls back onto).")
    parser.add_argument("--cadence-seconds", type=float, default=30.0,
                        help="Sensor cadence, drawn as the real-time compute budget.")
    parser.add_argument("--replot-status", action="store_true",
                        help="Re-render 02_status.png (front radii + status band, no heater "
                             "panel) from a previous run's dt_updates.csv, without replaying. "
                             "The heater commands are not in the CSV, so this rebuilds only "
                             "that figure.")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    config = load_startup_config(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.replot_status:
        csv_path = out_dir / "dt_updates.csv"
        if not csv_path.exists():
            raise SystemExit(f"No cached updates at {csv_path}; run the replay first.")
        rows = _rows_from_csv(csv_path)
        print(f"Replotting {len(rows)} cached updates from {csv_path}")
        fig_status_decision(rows, [], out_dir, config, include_decision=False)
        return
    horizon_hours = float(config.get("fem_forecast", {}).get("horizon_hours", 0.17))

    fluid_path = Path(args.fluid_path) if args.fluid_path else None
    sensor_path = Path(args.path)

    excluded = set(args.exclude_sensors.split(",")) if args.exclude_sensors else set()
    surface_exclusion_m = float(
        config.get("thaw_front_reconstruction", {}).get("surface_exclusion_m", 0.03)
    )
    if not args.keep_surface_band:
        shallow = surface_band_channels(sensor_path, surface_exclusion_m)
        if shallow:
            print(f"Excluding {len(shallow)} channel(s) shallower than "
                  f"{surface_exclusion_m * 1000:.0f} mm: {','.join(sorted(shallow))}")
        excluded |= shallow

    rows, alerts, commands = run_replay(
        config,
        path=sensor_path, fluid_path=fluid_path,
        start_row=args.start_row, limit=args.limit, stride=max(int(args.stride), 1),
        exclude_sensors=excluded or None,
        predict_every=max(int(args.predict_every), 1),
        lead_hours=float(args.lead_time_hours),
        breach_samples=max(int(args.breach_samples), 1),
        monitor_window_h=float(args.monitor_window_minutes) / 60.0,
    )

    fluid = None
    if fluid_path is not None:
        fluid = _load_fluid_temps(fluid_path, inlet_col="Fluid Temperature_Inlet",
                                  outlet_col="Fluid Temperatue_Outlet", time_col="Time (h)")
    event_h = detect_event(fluid)

    print("\ndecision timeline")
    previous = None
    for a in alerts:
        key = (a.get("status"), a.get("recommended_action"))
        if key != previous:
            print(f"  t={a['_time_hours']:6.3f} h  status={a.get('status'):<9} "
                  f"action={a.get('recommended_action'):<5} reason={a.get('reason')}")
            previous = key
    print("\nheater command transitions")
    for c in commands:
        print(f"  t={c.get('time_hours', float('nan')):6.3f} h  action={c.get('action'):<5} "
              f"setpoint={c.get('setpoint')}")

    first_stop = next((c for c in commands if c.get("action") == "stop"), None)
    if event_h is not None:
        print(f"\nreal temperature-reduction event detected at t = {event_h:.3f} h")
        if first_stop is not None:
            lead = (event_h - first_stop["time_hours"]) * 60.0
            verb = "before" if lead > 0 else "after"
            print(f"  twin issued STOP at t = {first_stop['time_hours']:.3f} h "
                  f"-- {abs(lead):.0f} min {verb} the operator acted")

    totals = np.array([r["ms_total"] for r in rows], dtype=float)
    print(f"\ncomputation per update: mean {totals.mean():.0f} ms, "
          f"p95 {np.percentile(totals, 95):.0f} ms, max {totals.max():.0f} ms "
          f"(budget {args.cadence_seconds * 1000:.0f} ms)")
    for stage in STAGES:
        vals = np.array([r[f"ms_{stage}"] for r in rows], dtype=float)
        print(f"  {stage:<11} mean {vals.mean():7.1f} ms   max {vals.max():7.1f} ms")

    print("\nwriting outputs")
    with (out_dir / "dt_updates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {out_dir / 'dt_updates.csv'}")

    fig_predictions(rows, out_dir, float(args.lead_time_hours))
    fig_status_decision(rows, commands, out_dir, config)
    fig_event(rows, commands, fluid, event_h, out_dir, config)
    fig_compute_time(rows, out_dir, args.cadence_seconds)
    t_pred, t_react = fig_lead_time(rows, out_dir, config, event_h, float(args.lead_time_hours))

    print("\npredictive vs reactive")
    limit_m = float(config.get("safety_monitor", {}).get("limit_radius_m", 0.040))
    print(f"  reactive trigger (measured >= {limit_m * 1000:.0f} mm) : "
          f"{'never' if t_react is None else f'{t_react:.3f} h'}")
    print(f"  predictive trigger                        : "
          f"{'never' if t_pred is None else f'{t_pred:.3f} h'}")
    if t_pred is not None and t_react is not None:
        print(f"  lead                                      : {(t_react - t_pred) * 60:+.0f} min")
    if t_pred is not None and event_h is not None:
        print(f"  ahead of the operator                     : {(event_h - t_pred) * 60:+.0f} min")
    fired = [r for r in rows if r.get("predictive_stop")]
    if fired:
        first = fired[0]
        why = "stopping now no longer avoids the breach" if first["predict_unavoidable"] \
            else "limit arrives within the lead time"
        print(f"  first predictive reason                   : {why}")


if __name__ == "__main__":
    main()

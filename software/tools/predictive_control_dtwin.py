"""Predictive control figures (M1/M2/F1/F2) with the coupled BHE-soil twin.

Same decision chain and the same two figures as ``predictive_control_evidence``
-- the real predictive safety monitor and heater actuation servers, RabbitMQ and
InfluxDB stubbed -- but the forecaster is the coupled probe-soil model of
``DT_final_python`` (``dtwin``) instead of the FEM forecast service.

**What the twin is fed.** At every decision time ``t``:

* parameters are those the gated sequential replay (``results/replay_gate.pkl``)
  had adopted by ``t`` -- prior means before the first adopted update;
* the soil field is synchronised to the measured field at ``t``, as in the
  twin's own ``forecast_sync``; the probe state (fluid and wall temperatures) is
  carried forward from the previous decision by a run on the measured inlet;
* the two lookahead branches differ only in the inlet temperature after ``t``:
  ``keep`` holds the measured inlet at ``t``, ``stop`` forces it to
  ``fem_forecast.lookahead_stop_setpoint_c``.

The measured r_max is the twin's own front reconstruction (per sensor row, from
the probe wall) applied to the measured field, so measurement and forecast use
one definition of the front.

Run:

    python -m software.tools.predictive_control_dtwin
    python -m software.tools.predictive_control_dtwin --replot
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from software.startup.service_runners import (
    build_influx_config,
    create_heater_actuation_server,
    create_safety_monitor_server,
)
from software.startup.utils.config import load_startup_config
from software.tools.fem_debug_replay import _NullInflux
from software.tools.paper_figures import style
from software.tools.predictive_control_evidence import (
    _Recorder,
    fig_decision_timeline,
    fig_lookahead_branches,
    first_true,
)

DEFAULT_DTWIN = Path.home() / "Downloads" / "DT_final_python_1014" / "DT_final_python"
K0 = 273.15


def _r_max(front_mm: np.ndarray) -> float | None:
    """Largest per-row front [m]; None when no row has thawed yet."""
    if np.all(np.isnan(front_mm)):
        return None
    return float(np.nanmax(front_mm)) / 1000.0


def _r_min(front_mm: np.ndarray) -> float | None:
    """Smallest per-row front [m]; a row with no crossing counts as 0."""
    if np.all(np.isnan(front_mm)):
        return None
    return float(np.nanmin(np.where(np.isnan(front_mm), 0.0, front_mm))) / 1000.0


def _inlet_branch(S, t_s: float, tin_k: float, horizon_s: float):
    """Copy of S whose inlet is measured up to t and held at tin_k after it."""
    Sb = copy.copy(S)
    past = S.t_data <= t_s
    Sb.t_data = np.r_[S.t_data[past], t_s + 1.0, t_s + horizon_s + 3600.0]
    Sb.Tin = np.r_[S.Tin[past], tin_k, tin_k]
    return Sb


def _params_schedule(replay_path: Path, P0):
    """[(t_s, P)] in force from t_s on, from the gated replay."""
    with replay_path.open("rb") as fh:
        RP = pickle.load(fh)
    sched = [(0.0, P0)]
    for ev in RP["events"]:
        if ev["accepted"]:
            sched.append((float(ev["t"]), ev["P"]))
    return sched


def _params_at(sched, t_s: float):
    P = sched[0][1]
    for t0, Pk in sched:
        if t_s >= t0 - 1e-6:
            P = Pk
    return P


def _hold_warning(rows: list[dict]) -> list[dict]:
    """Plot-only: once M_warn first fires, keep it (and the warning status) on.

    The measured front hovers at the warn threshold for a while, and the raw
    predicate flickers on single-sample dips; the figure shows the regime, not
    the noise. The CSV keeps the monitor's raw output.
    """
    out, held = [], False
    for r in rows:
        held = held or bool(r.get("m2"))
        if held:
            r = dict(r, m2=True)
            if r.get("status") == "ok":
                r["status"] = "warning"
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# REPLAY
# ---------------------------------------------------------------------------
def run_replay(config: dict, *, dtwin_root: Path, stride: int,
               verbose_every: int = 20) -> tuple[list[dict], dict]:
    sys.path.insert(0, str(dtwin_root))
    from dtwin.front import thaw_front
    from dtwin.model_setup import dt_setup
    from dtwin.params import dt_params
    from dtwin.simulate import dt_initial_state, dt_simulate

    S = dt_setup(dtwin_root / "data")
    P0 = dt_params()
    sched = _params_schedule(dtwin_root / "results" / "replay_gate.pkl", P0)
    print("parameter sets in force: " + ", ".join(
        f"{t / 3600:.2f} h (h_BHE {P.h_BHE:.1f}, d_slab {P.d_slab * 1e3:.1f} mm)"
        for t, P in sched))

    fem_cfg = config.get("fem_forecast", {})
    horizon_h = float(fem_cfg.get("lookahead_hours", 1.0))
    n_samples = int(fem_cfg.get("lookahead_samples", 12))
    stop_c = float(fem_cfg.get("lookahead_stop_setpoint_c", 0.0))
    horizon_s = horizon_h * 3600.0
    dt = P0.dt
    n_steps = int(round(horizon_s / dt))
    sample_idx = np.rint(np.linspace(0, n_steps, n_samples + 1)[1:]).astype(int)

    influx_cfg = build_influx_config(config.get("influxdb", {}))
    rabbit_cfg = config.get("rabbitmq", {})
    monitor = create_safety_monitor_server(
        influx_cfg, rabbit_cfg, dict(config.get("safety_monitor", {}), mode="predictive"))
    monitor.mq_out, monitor.influx = _Recorder(), _NullInflux()
    heater = create_heater_actuation_server(
        influx_cfg, rabbit_cfg, config.get("heater_actuation", {}))
    heater.mq_command, heater.mq_action, heater.influx = _Recorder(), _Recorder(), _NullInflux()
    limit_m = float(config.get("safety_monitor", {}).get("limit_radius_m", 0.040))

    front_meas = thaw_front(S, S.T_meas)[0]                       # nz x n_meas, mm
    k_meas = np.arange(0, S.T_meas.shape[1], stride)
    print(f"{k_meas.size} decisions every {stride * dt:.0f} s over 0 .. "
          f"{S.t_meas[k_meas[-1]] / 3600:.2f} h | lookahead {horizon_h:g} h in "
          f"{n_samples} samples | stop inlet {stop_c:g} C | limit {limit_m * 1e3:.0f} mm "
          f"| confirm {monitor.confirm_updates} | f2_action {monitor.f2_action}")

    X = dt_initial_state(S, P0)
    rows: list[dict] = []
    traces: list[dict] = []
    for i, k in enumerate(k_meas):
        t_s = float(S.t_meas[k])
        t_h = t_s / 3600.0
        P = _params_at(sched, t_s)

        # Probe state carried forward on the measured inlet, soil synchronised.
        if t_s > X.t:
            X = dt_simulate(S, P, X, t_s).X
        Xs = copy.copy(X)
        Xs.T = S.T_meas[:, k] + K0

        tin_now = float(np.interp(t_s, S.t_data, S.Tin))
        branches = {}
        for name, tin in (("keep", tin_now), ("stop", stop_c + K0)):
            R = dt_simulate(_inlet_branch(S, t_s, tin, horizon_s), P, Xs, t_s + horizon_s)
            fr = thaw_front(S, R.T[:, sample_idx] - K0)[0]
            branches[name] = [(_r_max(fr[:, j]) or 0.0) for j in range(sample_idx.size)]
        times = list(t_h + sample_idx * dt / 3600.0)

        stamp = f"{t_h:.6f}"
        r_max = _r_max(front_meas[:, k])
        thaw_msg = {"time_hours": t_h, "timestamp": stamp,
                    "radius_max_m": r_max, "radius_min_m": _r_min(front_meas[:, k])}
        fmsg = {"time_hours": t_h, "timestamp": stamp,
                "lookahead_hours": horizon_h, "lookahead_times_h": times,
                "lookahead_radius_keep_m": branches["keep"],
                "lookahead_radius_stop_m": branches["stop"],
                "radius_max_keep_m": branches["keep"][-1],
                "radius_max_stop_m": branches["stop"][-1]}

        monitor._handle_thaw(thaw_msg)
        monitor._handle_forecast(fmsg)
        alert = monitor.mq_out.last or {}
        heater._process_message(alert)
        cmd = heater.mq_command.last or {}

        traces.append({"row": len(rows), "t": t_h, "times": times,
                       "keep": branches["keep"], "stop": branches["stop"]})
        rows.append({
            "time_hours": round(t_h, 6),
            "r_max_measured_m": r_max,
            "r_min_measured_m": thaw_msg["radius_min_m"],
            "r_keep_m": fmsg["radius_max_keep_m"],
            "r_stop_m": fmsg["radius_max_stop_m"],
            "ttb_hours": alert.get("time_to_breach_hours"),
            "m1": alert.get("predicate_m1"),
            "m2": alert.get("predicate_m2"),
            "f1": alert.get("predicate_f1"),
            "f2": alert.get("predicate_f2"),
            "status": alert.get("status"),
            "action": cmd.get("action"),
            "setpoint": cmd.get("setpoint"),
            "reason": alert.get("reason"),
            "inlet_c": tin_now - K0,
        })

        if i % verbose_every == 0 or i == k_meas.size - 1:
            print(f"  [{i + 1:4d}/{k_meas.size}] t={t_h:5.2f}h "
                  f"r_max={'--' if r_max is None else f'{r_max * 1e3:5.1f}mm'} "
                  f"keep={branches['keep'][-1] * 1e3:5.1f} stop={branches['stop'][-1] * 1e3:5.1f} "
                  f"{alert.get('status', '-'):<19} {cmd.get('action', '-')}", flush=True)

    return rows, {"traces": traces, "limit_m": limit_m, "lookahead_hours": horizon_h,
                  "warn": monitor.warn_fraction, "critical": monitor.critical_fraction}


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dtwin-root", default=str(DEFAULT_DTWIN))
    parser.add_argument("--stride", type=int, default=2,
                        help="Decide every N measurement samples (30 s each).")
    parser.add_argument("--stop-after-breach-hours", type=float, default=0.15)
    parser.add_argument("--after-stop-hours", type=float, default=0.04,
                        help="How far past the stop command to plot.")
    parser.add_argument("--config", default="startup.conf")
    parser.add_argument("--out", default="artifacts/predictive_control_dtwin")
    parser.add_argument("--replot", action="store_true")
    args = parser.parse_args()

    config = load_startup_config(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "lookahead_traces.json"

    if args.replot:
        cached = json.loads(trace_path.read_text(encoding="utf-8"))
        rows = cached["rows"]
        meta = {k: cached[k] for k in ("traces", "limit_m", "lookahead_hours", "warn", "critical")}
    else:
        rows, meta = run_replay(config, dtwin_root=Path(args.dtwin_root), stride=args.stride)
        breach = first_true(rows, "m1")
        if breach is not None:
            cutoff = breach + args.stop_after_breach_hours
            rows = [r for r in rows if float(r["time_hours"]) <= cutoff]
            meta["traces"] = [tr for tr in meta["traces"] if tr["t"] <= cutoff]

        with (out_dir / "predictive_updates.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        trace_path.write_text(json.dumps({"rows": rows, **meta}), encoding="utf-8")

    breach = first_true(rows, "m1")
    stop = next((float(r["time_hours"]) for r in rows if r.get("action") == "stop"), None)
    plot_rows, plot_meta = rows, meta
    if stop is not None:
        # The heater is cut here, so the recorded run past this point is not
        # what the controlled experiment would have measured.
        cutoff = stop + args.after_stop_hours
        plot_rows = [r for r in rows if float(r["time_hours"]) <= cutoff]
        plot_meta = dict(meta, traces=[tr for tr in meta["traces"] if tr["t"] <= cutoff])
    plot_rows = _hold_warning(plot_rows)

    print("\n--- first occurrence ---")
    for key, label in (("f1", "F1  breach if heating continues"),
                       ("f2", "F2  breach even if stopped now"),
                       ("m1", "M1  measured breach")):
        when = first_true(rows, key)
        print(f"  {label:<34} {'never' if when is None else f'{when:.3f} h'}")
    for act in ("reduce", "stop"):
        hit = next((float(r["time_hours"]) for r in rows if r.get("action") == act), None)
        print(f"  first '{act}' command{'':<15} {'never' if hit is None else f'{hit:.3f} h'}")
        if hit is not None and breach is not None:
            print(f"    lead over measured breach: {(breach - hit) * 60:.0f} min")

    style()
    fig_decision_timeline(plot_rows, plot_meta, out_dir, model_label="DT", stop_story=True)
    fig_lookahead_branches(plot_rows, plot_meta, out_dir, show_measured=False)


if __name__ == "__main__":
    main()

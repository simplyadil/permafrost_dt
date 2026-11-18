"""Pick safety limits that the recorded experiment actually exercises.

With ``limit_radius_m = 0.2`` the thaw front never gets within a factor of four
of the limit, so the safety monitor reports ``ok`` for the entire run and the
heater-actuation service issues one ``heat`` command at the start and nothing
afterwards. Nothing downstream is ever demonstrated: not the graded statuses,
not the safety cut, not the command that stops the circulator.

This script finds limits that do exercise them, from the data rather than by
guesswork, and then proves it by running the real services at the chosen values.

It works in three stages:

1. **Collect** -- replay the sensor file through the real thaw-front
   reconstruction service to get ``r_max``/``r_min`` exactly as production
   computes them, and (unless ``--no-forecast``) through the boundary builder,
   pipe-boundary calibration and FEM forecast to get the forecast ``r_max``. Both
   matter: the safety cut fires on measured *or* forecast radius.
2. **Sweep** -- for a range of candidate ``limit_radius_m``, work out when each
   status would first appear and how much of the run it would occupy. A limit is
   only useful for a demonstration if the run passes through the whole ladder,
   ``ok`` -> ``warning`` -> ``critical`` -> ``alert``, with each stage lasting
   long enough to see.
3. **Verify** -- drive the real ``SafetyMonitorServer`` and
   ``HeaterActuationServer`` at the recommended limits with their clients
   stubbed, and record every alert and every heater command they emit.

A note on what the two limits do, because they are not symmetric and the
difference decides how they should be chosen:

* ``limit_radius_m`` (r_max) is the safety cut. Reaching it stops the heater,
  and it is also the denominator for the warning and critical fractions.
* ``limit_min_radius_m`` (r_min) is the least-thawed radius floor. It is
  **informational only** -- ``_evaluate_and_publish_dual`` records
  ``rmin_below_floor`` and puts it in the alert reason, but it never stops the
  heater and never changes the recommended action. Choose it to be visible, not
  to be safe: it cannot cut anything.

Outputs, all under ``--out`` (default ``artifacts/safety_limits``):

    01_radius_traces.png    r_max, r_min, forecast r_max, with the chosen limits
    02_status_timeline.png  the status the monitor emits and the heater commands
    03_limit_sweep.png      status coverage against candidate limit_radius_m
    radius_traces.csv       the measured and forecast traces
    safety_events.csv       every alert and heater command from stage 3

Run:

    python -m software.tools.safety_limit_calibration
    python -m software.tools.safety_limit_calibration --stride 4 --no-forecast
"""

from __future__ import annotations

import argparse
import csv
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
from software.tools.fem_debug_replay import (
    DEFAULT_FLUID_PATH,
    DEFAULT_SENSOR_PATH,
    _NullInflux,
    _Sink,
    _sensor_rows,
)
from software.tools.replay_csv import _load_fluid_temps

STATUS_ORDER = ["ok", "warning", "critical", "alert"]
STATUS_COLOUR = {"ok": "#3F7D3A", "warning": "#D8A21B", "critical": "#D2691E", "alert": "#B3261E"}


def _command_transitions(commands: list[dict]) -> list[dict]:
    """Drop consecutive repeats, keeping only where the command actually changed."""
    out: list[dict] = []
    previous = None
    for command in commands:
        key = (command.get("action"), command.get("setpoint"))
        if key != previous:
            out.append(command)
            previous = key
    return out


class _Recorder:
    """Captures every published message instead of sending it to RabbitMQ."""

    def __init__(self) -> None:
        self.messages: list[dict] = []

    def publish(self, message: dict) -> None:
        self.messages.append(dict(message))

    @property
    def last(self) -> dict | None:
        return self.messages[-1] if self.messages else None


# ---------------------------------------------------------------------------
# STAGE 1: COLLECT
# ---------------------------------------------------------------------------
def collect_traces(config: dict, *, path: Path, fluid_path: Path | None, start_row: int,
                   limit: int | None, stride: int, with_forecast: bool,
                   exclude_sensors: set[str] | None) -> list[dict]:
    rabbit_cfg = config.get("rabbitmq", {})
    influx_cfg: InfluxConfig = build_influx_config(config.get("influxdb", {}))

    thaw = create_thaw_front_reconstruction_server(influx_cfg, rabbit_cfg,
                                                   config.get("thaw_front_reconstruction", {}))
    thaw.mq_out = _Recorder()
    thaw.influx = _NullInflux()

    builder = pipe = forecast = None
    fluid = None
    if with_forecast:
        fem_cfg = dict(config.get("fem_forecast", {}))
        fem_cfg["run_baseline"] = False
        builder = create_boundary_condition_builder_server(rabbit_cfg,
                                                           config.get("boundary_condition_builder", {}))
        forecast = create_fem_forecast_server(influx_cfg, rabbit_cfg, fem_cfg)
        forecast.mq_out = _Recorder()
        forecast.influx = _NullInflux()
        pipe_cfg = config.get("pipe_boundary", {})
        if pipe_cfg and fluid_path is not None:
            pipe = create_pipe_boundary_server(influx_cfg, rabbit_cfg, pipe_cfg)
            pipe.mq_boundary_out = _Sink()
            fluid = _load_fluid_temps(fluid_path, inlet_col="Fluid Temperature_Inlet",
                                      outlet_col="Fluid Temperatue_Outlet", time_col="Time (h)")

    snapshots = _sensor_rows(path, None, start_row, limit, stride, exclude_sensors)
    if not snapshots:
        raise SystemExit(f"No usable sensor rows parsed from {path}")
    print(f"Loaded {len(snapshots)} snapshots (t = {snapshots[0]['time_hours']:.3f} .. "
          f"{snapshots[-1]['time_hours']:.3f} h)")
    print(f"Forecast arm: {'on' if with_forecast else 'off (measured radius only)'}")

    rows: list[dict] = []
    for index, sensor_msg in enumerate(snapshots):
        time_hours = float(sensor_msg["time_hours"])

        thaw._process_message(sensor_msg)
        thaw_msg = thaw.mq_out.last

        forecast_rmax = None
        if forecast is not None:
            boundary_snapshot = builder._build_snapshot(sensor_msg)
            if pipe is not None and fluid is not None:
                t_arr, tin_arr, tout_arr = fluid
                pipe._on_fluid({
                    "time_hours": time_hours,
                    "timestamp": sensor_msg["timestamp"],
                    "t_inlet_c": float(np.interp(time_hours, t_arr, tin_arr)),
                    "t_outlet_c": float(np.interp(time_hours, t_arr, tout_arr)),
                    "site_id": "default",
                })
                pipe._on_boundary(boundary_snapshot)
                enriched = pipe.mq_boundary_out.last
                if isinstance(enriched, dict):
                    boundary_snapshot = enriched
            forecast._process_message(boundary_snapshot)
            fmsg = forecast.mq_out.last
            if isinstance(fmsg, dict):
                forecast_rmax = fmsg.get("radius_max_m")

        rows.append({
            "time_hours": round(time_hours, 6),
            "r_max_m": (thaw_msg or {}).get("radius_max_m"),
            "r_min_m": (thaw_msg or {}).get("radius_min_m"),
            "r_avg_m": (thaw_msg or {}).get("radius_avg_m"),
            "forecast_r_max_m": forecast_rmax,
        })

        if index % max(len(snapshots) // 12, 1) == 0 or index == len(snapshots) - 1:
            r = rows[-1]["r_max_m"]
            print(f"  [{index + 1:4d}/{len(snapshots)}] t={time_hours:6.3f} h  "
                  f"r_max={'n/a' if r is None else f'{r * 1000:5.1f} mm'}")
    return rows


# ---------------------------------------------------------------------------
# STAGE 2: SWEEP
# ---------------------------------------------------------------------------
def _status_for(r_max, forecast_r_max, limit_max, warn_f, crit_f) -> str:
    """Reproduces the ladder in ``_evaluate_and_publish_dual`` exactly."""
    over = (r_max is not None and r_max >= limit_max) or \
           (forecast_r_max is not None and forecast_r_max >= limit_max)
    if over:
        return "alert"
    ratio = (r_max / limit_max) if (r_max is not None and limit_max > 0) else None
    if ratio is None:
        return "ok"
    if ratio >= crit_f:
        return "critical"
    if ratio >= warn_f:
        return "warning"
    return "ok"


def sweep_limits(rows: list[dict], warn_f: float, crit_f: float,
                 candidates: np.ndarray) -> list[dict]:
    times = np.array([r["time_hours"] for r in rows], dtype=float)
    total = times[-1] - times[0] if len(times) > 1 else 1.0

    out: list[dict] = []
    for limit_max in candidates:
        statuses = [_status_for(r["r_max_m"], r["forecast_r_max_m"], limit_max, warn_f, crit_f)
                    for r in rows]
        entry = {"limit_radius_m": float(limit_max)}
        for name in STATUS_ORDER:
            mask = np.array([s == name for s in statuses])
            entry[f"frac_{name}"] = float(np.mean(mask))
            entry[f"first_{name}_h"] = float(times[mask][0]) if np.any(mask) else float("nan")
        entry["stages_reached"] = sum(1 for n in STATUS_ORDER if entry[f"frac_{n}"] > 0)
        entry["duration_alert_h"] = entry[f"frac_alert"] * total
        out.append(entry)
    return out


def recommend(sweep: list[dict], *, min_alert_h: float, min_stage_frac: float) -> dict | None:
    """The largest limit that still walks the whole ladder with visible stages.

    Largest rather than smallest on purpose: a low limit trips into ``alert``
    almost immediately and the interesting part -- the graded approach -- is over
    before it can be seen. Among the candidates that do show every stage, the
    biggest one keeps the front climbing for as long as possible first.
    """
    viable = [
        e for e in sweep
        if e["stages_reached"] == 4
        and e["duration_alert_h"] >= min_alert_h
        and min(e[f"frac_{n}"] for n in STATUS_ORDER) >= min_stage_frac
    ]
    if not viable:
        # Relax the per-stage floor before giving up: reaching all four stages at
        # all is the property that matters most.
        viable = [e for e in sweep if e["stages_reached"] == 4 and e["duration_alert_h"] >= min_alert_h]
    if not viable:
        return None
    return max(viable, key=lambda e: e["limit_radius_m"])


def recommend_min_limit(rows: list[dict]) -> float | None:
    """A floor r_min crosses during the run, so the flag is seen to change.

    ``limit_min_radius_m`` never stops anything, so the only sensible criterion
    for a demonstration is that it is crossed: the midpoint of the range r_min
    actually visits, rounded to a millimetre.
    """
    values = np.array([r["r_min_m"] for r in rows if r["r_min_m"] is not None], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0 or float(np.ptp(values)) == 0.0:
        return None
    return float(np.round((values.min() + values.max()) / 2.0, 3))


# ---------------------------------------------------------------------------
# STAGE 3: VERIFY WITH THE REAL SERVICES
# ---------------------------------------------------------------------------
def run_services(config: dict, rows: list[dict], *, limit_max: float, limit_min: float) -> tuple[list, list]:
    """Drive the real monitor and actuation services at the chosen limits."""
    rabbit_cfg = config.get("rabbitmq", {})
    influx_cfg: InfluxConfig = build_influx_config(config.get("influxdb", {}))

    safety_cfg = dict(config.get("safety_monitor", {}))
    safety_cfg["limit_radius_m"] = float(limit_max)
    safety_cfg["limit_min_radius_m"] = float(limit_min)
    monitor = create_safety_monitor_server(influx_cfg, rabbit_cfg, safety_cfg)
    monitor.mq_out = _Recorder()
    monitor.influx = _NullInflux()

    heater = create_heater_actuation_server(influx_cfg, rabbit_cfg, config.get("heater_actuation", {}))
    heater.mq_command = _Recorder()
    heater.mq_action = _Recorder()
    heater.influx = _NullInflux()

    monitor_alerts: list[dict] = []
    for row in rows:
        stamp = f"t={row['time_hours']:.4f}"
        if row["forecast_r_max_m"] is not None:
            monitor._handle_forecast({
                "timestamp": stamp,
                "time_hours": row["time_hours"],
                "radius_max_m": row["forecast_r_max_m"],
                "radius_min_m": None,
                "radius_avg_m": None,
            })
        if row["r_max_m"] is not None:
            monitor._handle_thaw({
                "timestamp": stamp,
                "time_hours": row["time_hours"],
                "radius_max_m": row["r_max_m"],
                "radius_min_m": row["r_min_m"],
                "radius_avg_m": row["r_avg_m"],
            })
        # Feed whatever the monitor just emitted straight into actuation, so the
        # commands recorded here are the ones the real chain would produce.
        while monitor.mq_out.messages:
            alert = monitor.mq_out.messages.pop(0)
            alert["_time_hours"] = row["time_hours"]
            alert["_status"] = alert.get("status")
            heater._process_message(alert)
            monitor_alerts.append(alert)
    return monitor_alerts, heater.mq_command.messages


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
def plot_traces(rows, out_dir, *, limit_max, limit_min, warn_f, crit_f):
    import matplotlib.pyplot as plt

    t = [r["time_hours"] for r in rows]

    def mm(key):
        return [np.nan if r[key] is None else r[key] * 1000.0 for r in rows]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.axhspan(limit_max * 1000, ax.get_ylim()[1] if False else limit_max * 1000 * 1.6,
               color=STATUS_COLOUR["alert"], alpha=0.06)
    for frac, name in ((warn_f, "warning"), (crit_f, "critical"), (1.0, "alert")):
        y = limit_max * frac * 1000
        ax.axhline(y, color=STATUS_COLOUR[name], lw=1.4, ls="--")
        ax.text(t[-1], y, f" {name} ({frac:g}x)", color=STATUS_COLOUR[name],
                va="center", ha="left", fontsize=8.5)
    ax.axhline(limit_min * 1000, color="#4A6FA5", lw=1.4, ls=":")
    ax.text(t[-1], limit_min * 1000, "  r_min floor", color="#4A6FA5",
            va="center", ha="left", fontsize=8.5)

    ax.plot(t, mm("r_max_m"), color="#111111", lw=1.9, label="r_max (measured)")
    ax.plot(t, mm("r_min_m"), color="#4A6FA5", lw=1.6, label="r_min (measured)")
    if any(r["forecast_r_max_m"] is not None for r in rows):
        ax.plot(t, mm("forecast_r_max_m"), color="#C2531F", lw=1.6, alpha=0.9,
                label="r_max (forecast)")

    ax.set_xlabel("Experiment time (h)")
    ax.set_ylabel("radius (mm)")
    ax.set_title(f"Thaw-front radii against the proposed limits "
                 f"(limit_radius_m={limit_max:g}, limit_min_radius_m={limit_min:g})", fontsize=11)
    ax.grid(alpha=0.22)
    ax.legend(loc="upper left", fontsize=9)
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / "01_radius_traces.png", dpi=150)
    print(f"  wrote {out_dir / '01_radius_traces.png'}")
    plt.close(fig)


def plot_timeline(alerts, commands, out_dir):
    import matplotlib.pyplot as plt

    if not alerts:
        return
    t = np.array([a["_time_hours"] for a in alerts], dtype=float)
    statuses = [a.get("status", "ok") for a in alerts]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 5.0), sharex=True,
                                  gridspec_kw={"height_ratios": [1, 1.3]})

    # Status ribbon: one coloured span per contiguous run of the same status.
    start = 0
    for i in range(1, len(statuses) + 1):
        if i == len(statuses) or statuses[i] != statuses[start]:
            ax.axvspan(t[start], t[min(i, len(t) - 1)],
                       color=STATUS_COLOUR.get(statuses[start], "#999999"), alpha=0.85)
            mid = (t[start] + t[min(i, len(t) - 1)]) / 2
            if t[min(i, len(t) - 1)] - t[start] > (t[-1] - t[0]) * 0.05:
                ax.text(mid, 0.5, statuses[start], ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
            start = i
    ax.set_yticks([])
    ax.set_ylabel("safety\nmonitor")
    ax.set_title("What the services actually emit at these limits", fontsize=11)

    # Only the transitions: with publish_on_change off the service re-issues the
    # same command every message, and plotting all of them buries the two moments
    # that matter under a thousand identical marks.
    transitions = _command_transitions(commands)
    if transitions:
        ct = [c.get("time_hours", np.nan) for c in transitions]
        setpoints = [0.0 if c.get("setpoint") is None else float(c["setpoint"]) for c in transitions]
        actions = [c.get("action") for c in transitions]
        ax2.step(list(ct) + [t[-1]], setpoints + [setpoints[-1]],
                 where="post", color="#B3261E", lw=2.2)
        for x, s, a in zip(ct, setpoints, actions):
            ax2.axvline(x, color="0.6", lw=0.9, ls=":")
            ax2.annotate(f"{a} @ {s:g} C\nt={x:.2f} h", xy=(x, s), xytext=(6, 8),
                         textcoords="offset points", fontsize=8.5, color="#B3261E")
        ax2.set_ylim(-1.5, max(setpoints) + 3.5)
    ax2.set_ylabel("heater setpoint (C)")
    ax2.set_xlabel("Experiment time (h)")
    ax2.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(out_dir / "02_status_timeline.png", dpi=150)
    print(f"  wrote {out_dir / '02_status_timeline.png'}")
    plt.close(fig)


def plot_sweep(sweep, out_dir, chosen):
    import matplotlib.pyplot as plt

    limits = np.array([e["limit_radius_m"] for e in sweep]) * 1000
    fig, ax = plt.subplots(figsize=(11, 4.6))
    bottom = np.zeros_like(limits)
    for name in STATUS_ORDER:
        frac = np.array([e[f"frac_{name}"] for e in sweep]) * 100
        ax.bar(limits, frac, bottom=bottom, width=(limits[1] - limits[0]) * 0.9,
               color=STATUS_COLOUR[name], label=name)
        bottom += frac
    if chosen is not None:
        ax.axvline(chosen["limit_radius_m"] * 1000, color="black", lw=2.0, ls="--")
        ax.text(chosen["limit_radius_m"] * 1000, 103, " recommended",
                fontsize=9, va="bottom")
    ax.set_xlabel("candidate limit_radius_m (mm)")
    ax.set_ylabel("% of run in each status")
    ax.set_title("Status coverage against the safety limit", fontsize=11)
    ax.legend(loc="lower left", fontsize=9, ncol=4)
    ax.set_ylim(0, 108)
    fig.tight_layout()
    fig.savefig(out_dir / "03_limit_sweep.png", dpi=150)
    print(f"  wrote {out_dir / '03_limit_sweep.png'}")
    plt.close(fig)


# ---------------------------------------------------------------------------
def _write_csv(rows, path):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", default=DEFAULT_SENSOR_PATH)
    parser.add_argument("--fluid-path", default=DEFAULT_FLUID_PATH)
    parser.add_argument("--config", default="startup.conf")
    parser.add_argument("--out", default="artifacts/safety_limits")
    parser.add_argument("--start-row", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--no-forecast", action="store_true",
                        help="Skip the FEM arm (much faster; measured radius only).")
    parser.add_argument("--exclude-sensors", default=None)
    parser.add_argument("--min-alert-hours", type=float, default=0.3,
                        help="Alert must last at least this long to count as demonstrable.")
    parser.add_argument("--min-stage-fraction", type=float, default=0.03,
                        help="Each status must occupy at least this fraction of the run.")
    parser.add_argument("--limit-radius-m", type=float, default=None,
                        help="Skip the recommendation and verify this value instead.")
    parser.add_argument("--limit-min-radius-m", type=float, default=None)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    config = load_startup_config(args.config)
    safety_cfg = config.get("safety_monitor", {})
    warn_f = float(safety_cfg.get("warn_fraction", 0.6))
    crit_f = float(safety_cfg.get("critical_fraction", 0.9))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Current config: limit_radius_m={safety_cfg.get('limit_radius_m')}, "
          f"limit_min_radius_m={safety_cfg.get('limit_min_radius_m')}, "
          f"warn={warn_f:g}, critical={crit_f:g}\n")

    rows = collect_traces(
        config,
        path=Path(args.path),
        fluid_path=Path(args.fluid_path) if args.fluid_path else None,
        start_row=args.start_row,
        limit=args.limit,
        stride=max(int(args.stride), 1),
        with_forecast=not args.no_forecast,
        exclude_sensors=set(args.exclude_sensors.split(",")) if args.exclude_sensors else None,
    )

    rmax = np.array([r["r_max_m"] for r in rows if r["r_max_m"] is not None], dtype=float)
    rmin = np.array([r["r_min_m"] for r in rows if r["r_min_m"] is not None], dtype=float)
    print("\nobserved radii (mm)")
    if rmax.size:
        print(f"  r_max : {rmax.min() * 1000:6.1f} .. {rmax.max() * 1000:6.1f}")
    if rmin.size:
        print(f"  r_min : {rmin.min() * 1000:6.1f} .. {rmin.max() * 1000:6.1f}")
    if not rmax.size:
        raise SystemExit("No thaw front was ever reconstructed; nothing to calibrate against.")

    # Candidates span the observed r_max range: below the minimum everything is
    # instantly in alert, above the maximum nothing ever fires.
    candidates = np.round(np.linspace(rmax.min() + 0.002, rmax.max(), 40), 4)
    sweep = sweep_limits(rows, warn_f, crit_f, candidates)
    chosen = recommend(sweep, min_alert_h=args.min_alert_hours, min_stage_frac=args.min_stage_fraction)

    limit_max = args.limit_radius_m if args.limit_radius_m is not None else (
        chosen["limit_radius_m"] if chosen else None)
    if limit_max is None:
        raise SystemExit("No candidate limit walks the full status ladder on this run.")
    limit_min = args.limit_min_radius_m if args.limit_min_radius_m is not None else \
        (recommend_min_limit(rows) or float(np.round(rmax.max() * 0.25, 3)))

    print(f"\nrecommended  limit_radius_m     = {limit_max:.4g}   ({limit_max * 1000:.0f} mm)")
    print(f"             limit_min_radius_m = {limit_min:.4g}   ({limit_min * 1000:.0f} mm)")
    print(f"             warning at {limit_max * warn_f * 1000:.0f} mm, "
          f"critical at {limit_max * crit_f * 1000:.0f} mm, alert at {limit_max * 1000:.0f} mm")

    alerts, commands = run_services(config, rows, limit_max=limit_max, limit_min=limit_min)

    print("\nwhat the services emit at these limits")
    seen: list[tuple[str, float]] = []
    for a in alerts:
        key = (a.get("status"), a.get("recommended_action"))
        if not seen or seen[-1][0] != key:
            print(f"  t={a['_time_hours']:6.3f} h  status={a.get('status'):<9} "
                  f"action={a.get('recommended_action'):<5} reason={a.get('reason')}")
            seen.append((key, a["_time_hours"]))
    # heater_actuation.publish_on_change is false in this config, so it re-issues
    # the same command on every alert. Only the transitions are informative.
    transitions = _command_transitions(commands)
    print(f"\nheater commands issued: {len(commands)}  ({len(transitions)} distinct transitions)")
    for c in transitions:
        print(f"  t={c.get('time_hours', float('nan')):6.3f} h  "
              f"action={c.get('action'):<5} setpoint={c.get('setpoint')}")

    print("\nwriting outputs")
    _write_csv(rows, out_dir / "radius_traces.csv")
    _write_csv([{k: v for k, v in a.items() if not k.startswith("_")} | {"time_hours": a["_time_hours"]}
                for a in alerts], out_dir / "safety_events.csv")
    plot_traces(rows, out_dir, limit_max=limit_max, limit_min=limit_min, warn_f=warn_f, crit_f=crit_f)
    plot_timeline(alerts, commands, out_dir)
    plot_sweep(sweep, out_dir, chosen)


if __name__ == "__main__":
    main()

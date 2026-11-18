"""Thaw-front evolution up to the safety limit: measured r_max / r_min vs FEM forecast.

Replays a recorded experiment through the real thaw-front reconstruction,
boundary builder, pipe-boundary calibration and FEM forecast services (RabbitMQ
and InfluxDB stubbed) and plots three curves on one radius axis:

    r_max (measured)     outer edge of the thawed region, from the sensors
    r_min (measured)     least-thawed radius, from the sensors
    r_max (FEM forecast) the model's prediction of the outer edge

**The run is cut at the limit.** Every series stops at the moment the measured
r_max first reaches ``safety_monitor.limit_radius_m`` -- the experiment is treated
as having ended there, which is what the safety rule would have enforced. The
crossing time is interpolated between the two bracketing samples so the measured
curve terminates exactly on the limit rather than one sample past it.

**On the forecast curve.** With ``fem_forecast.track_observed_time`` on -- the
shipped setting -- the forecast service advances the solver by the elapsed time
since the previous message, so its published radius is a nowcast at the
observation time, not a prediction ahead of it. Plotting that as "forecast" would
be mislabelling it. So at each update this script snapshots the solver, advances
the copy by ``horizon_hours`` on the same boundary condition, records the
predicted radius, and restores the state so the tracked run is untouched. Each
prediction is drawn at the time it is *valid for* (t + horizon), which is what
makes it readable against the measurement it is predicting.

Run:

    python -m software.tools.front_evolution
    python -m software.tools.front_evolution --stride 4        # quick look
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
    create_pipe_boundary_server,
    create_thaw_front_reconstruction_server,
)
from software.startup.utils.config import load_startup_config
from software.tools.fem_debug_replay import _NullInflux, _Sink, _sensor_rows
from software.tools.paper_figures import DOUBLE, FEM_KEEP, INK, MUTED, STATUS_LIMIT, grid, save, style
from software.tools.replay_csv import _load_fluid_temps

DEFAULT_PATH = "data/CR1000_Table1_April8_replay_Final.xlsx"
# The broken thermistor. It reads a flat, zero-clipped signal, and the solver
# seeds its initial field by interpolating one snapshot across the mesh, so a
# single stuck channel is smeared over the domain before the first time step.
DEFAULT_EXCLUDE = "H10D75"


class _Recorder:
    """Captures published messages instead of sending them to RabbitMQ."""

    def __init__(self) -> None:
        self.last: dict | None = None

    def publish(self, message: dict) -> None:
        self.last = message


# ---------------------------------------------------------------------------
# REPLAY
# ---------------------------------------------------------------------------
def forecast_radius(solver, boundary: dict, horizon_hours: float) -> float | None:
    """Advance a copy of the solver one horizon ahead and read its front.

    The state is restored before returning: the tracked run must not inherit the
    lookahead, or simulated time races ahead of observed time and every later
    step is wrong.
    """
    saved_T = solver.T.copy()
    saved_time = solver.time_hours
    try:
        solver.advance(boundary, horizon_hours=horizon_hours)
        r_max, _r_avg, _points = solver.compute_thaw_metrics()
    finally:
        solver.T = saved_T
        solver.time_hours = saved_time
    return r_max


def run_replay(config: dict, *, path: Path, fluid_path: Path | None,
               fluid_cols: tuple[str, str, str], start_row: int, limit: int | None,
               stride: int, exclude_sensors: set[str] | None) -> tuple[list[dict], float]:
    rabbit_cfg = config.get("rabbitmq", {})
    influx_cfg: InfluxConfig = build_influx_config(config.get("influxdb", {}))
    horizon_hours = float(config.get("fem_forecast", {}).get("horizon_hours", 0.17))

    thaw = create_thaw_front_reconstruction_server(
        influx_cfg, rabbit_cfg, config.get("thaw_front_reconstruction", {}))
    thaw.mq_out, thaw.influx = _Recorder(), _NullInflux()

    builder = create_boundary_condition_builder_server(
        rabbit_cfg, config.get("boundary_condition_builder", {}))

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
        inlet_col, outlet_col, time_col = fluid_cols
        fluid = _load_fluid_temps(fluid_path, inlet_col=inlet_col,
                                  outlet_col=outlet_col, time_col=time_col)

    snapshots = _sensor_rows(path, None, start_row, limit, stride, exclude_sensors)
    if not snapshots:
        raise SystemExit(f"No usable sensor rows parsed from {path}")
    print(f"Loaded {len(snapshots)} snapshots (t = {snapshots[0]['time_hours']:.3f} .. "
          f"{snapshots[-1]['time_hours']:.3f} h)")
    if exclude_sensors:
        print(f"Excluded channels: {', '.join(sorted(exclude_sensors))}")
    print(f"Forecast horizon: {horizon_hours:g} h ({horizon_hours * 60:.0f} min)")

    rows: list[dict] = []
    for index, sensor_msg in enumerate(snapshots):
        t_h = float(sensor_msg["time_hours"])

        thaw._process_message(sensor_msg)
        thaw_msg = thaw.mq_out.last or {}

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

        forecast._process_message(boundary)

        r_pred = None
        if forecast._solver is not None:
            r_pred = forecast_radius(forecast._solver, boundary, horizon_hours)

        rows.append({
            "time_hours": round(t_h, 6),
            "r_max_measured_m": thaw_msg.get("radius_max_m"),
            "r_min_measured_m": thaw_msg.get("radius_min_m"),
            "r_max_forecast_m": r_pred,
            "forecast_valid_at_h": round(t_h + horizon_hours, 6),
        })

        if index % max(len(snapshots) // 10, 1) == 0 or index == len(snapshots) - 1:
            r_m = thaw_msg.get("radius_max_m")
            print(f"  [{index + 1:4d}/{len(snapshots)}] t={t_h:6.3f} h  "
                  f"r_max={'--' if r_m is None else f'{r_m * 1000:5.1f} mm'}")

    return rows, horizon_hours


# ---------------------------------------------------------------------------
# TRUNCATION AT THE LIMIT
# ---------------------------------------------------------------------------
def _clean(rows, t_key: str, v_key: str) -> tuple[np.ndarray, np.ndarray]:
    t, v = [], []
    for r in rows:
        val = r.get(v_key)
        if val is None:
            continue
        t.append(float(r[t_key]))
        v.append(float(val))
    return np.array(t), np.array(v)


def crossing_time(t: np.ndarray, r: np.ndarray, limit_m: float) -> float | None:
    """First time the series reaches the limit, interpolated between samples."""
    hit = np.where(r >= limit_m)[0]
    if hit.size == 0:
        return None
    i = int(hit[0])
    if i == 0:
        return float(t[0])
    r0, r1 = float(r[i - 1]), float(r[i])
    if r1 == r0:
        return float(t[i])
    frac = (limit_m - r0) / (r1 - r0)
    return float(t[i - 1] + (t[i] - t[i - 1]) * frac)


def truncate(t: np.ndarray, v: np.ndarray, t_end: float) -> tuple[np.ndarray, np.ndarray]:
    """Cut a series at t_end, interpolating a final point exactly on the cut.

    Without the interpolated endpoint every curve would stop at the last sample
    *before* the limit, which reads as the experiment ending early and leaves the
    measured curve visibly short of the line it is supposed to have reached.
    """
    keep = t <= t_end
    t_cut, v_cut = t[keep], v[keep]
    if t_cut.size and t_cut[-1] < t_end and np.any(t > t_end):
        v_end = float(np.interp(t_end, t, v))
        t_cut = np.append(t_cut, t_end)
        v_cut = np.append(v_cut, v_end)
    return t_cut, v_cut


# ---------------------------------------------------------------------------
# FIGURE
# ---------------------------------------------------------------------------
def figure(rows, out_dir: Path, limit_m: float, horizon_hours: float, t_end: float):
    import matplotlib.pyplot as plt

    mm = 1000.0
    t_max, r_max = _clean(rows, "time_hours", "r_max_measured_m")
    t_min, r_min = _clean(rows, "time_hours", "r_min_measured_m")
    t_fc, r_fc = _clean(rows, "forecast_valid_at_h", "r_max_forecast_m")

    t_max, r_max = truncate(t_max, r_max, t_end)
    t_min, r_min = truncate(t_min, r_min, t_end)
    t_fc, r_fc = truncate(t_fc, r_fc, t_end)

    fig, ax = plt.subplots(figsize=(DOUBLE, DOUBLE * 0.46))

    ax.axhline(limit_m * mm, color=STATUS_LIMIT, lw=1.1, ls="--", zorder=2)
    ax.annotate(f"safety limit {limit_m * mm:.0f} mm", xy=(0.012, limit_m * mm),
                xycoords=("axes fraction", "data"), xytext=(0, -4),
                textcoords="offset points", va="top", fontsize=7, color=STATUS_LIMIT)

    # Measurements in ink, the model in colour: the sensors are the reference,
    # not one category among three. Line style tracks colour so identity never
    # rests on hue alone and the figure survives greyscale printing.
    ax.plot(t_fc, r_fc * mm, "-.", color=FEM_KEEP, lw=1.4, zorder=4,
            label=f"$r_{{max}}$ — FEM forecast (+{horizon_hours * 60:.0f} min)")
    ax.plot(t_max, r_max * mm, "-", color=INK, lw=1.6, zorder=5,
            label="$r_{max}$ — measured")
    ax.plot(t_min, r_min * mm, "--", color=MUTED, lw=1.4, zorder=3,
            label="$r_{min}$ — measured")

    # The moment the run is cut, and the point that defines it.
    ax.axvline(t_end, color=STATUS_LIMIT, lw=0.8, ls=":", alpha=0.7, zorder=2)
    ax.plot([t_end], [limit_m * mm], "o", ms=5, color=STATUS_LIMIT,
            mec="white", mew=1.0, zorder=6)
    # Right of the crossing: r_max and the forecast converge onto the limit
    # there, so anything placed left of it lands on a curve.
    ax.annotate(f"limit reached\nt = {t_end:.2f} h", xy=(t_end, limit_m * mm),
                xytext=(7, -3), textcoords="offset points", ha="left", va="top",
                fontsize=7, color=STATUS_LIMIT, linespacing=1.35)

    ax.set_xlabel("time (h)")
    ax.set_ylabel("thaw-front radius (mm)")
    ax.set_xlim(0, t_end * 1.13)
    ax.set_ylim(0, limit_m * mm * 1.16)
    # Legend above the axes: every region inside the frame is either crossed by a
    # curve or occupied by the limit line and its annotations.
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01, 1, 0.12), mode="expand",
              ncol=3, borderaxespad=0)
    grid(ax)

    save(fig, out_dir, "front_evolution_to_limit")


def write_csv(rows, out_dir: Path, t_end: float):
    path = out_dir / "front_evolution.csv"
    kept = [r for r in rows if float(r["time_hours"]) <= t_end]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(kept)
    print(f"  wrote {path}  ({len(kept)} rows)")


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", default=DEFAULT_PATH, help="Sensor replay file.")
    parser.add_argument("--fluid-path", default=None,
                        help="Fluid inlet/outlet file (defaults to --path, which carries both).")
    parser.add_argument("--fluid-inlet-col", default="Inlet")
    parser.add_argument("--fluid-outlet-col", default="Outlet")
    parser.add_argument("--fluid-time-col", default="Time (h).1",
                        help="Time column in the fluid sheet. pandas renames a duplicated "
                             "'Time (h)' header to 'Time (h).1'.")
    parser.add_argument("--start-row", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Max rows to replay.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--exclude", default=DEFAULT_EXCLUDE,
                        help="Comma-separated channels to drop. Empty string keeps all.")
    parser.add_argument("--limit-mm", type=float, default=None,
                        help="Override the safety limit (default: from startup.conf).")
    parser.add_argument("--config", default="startup.conf")
    parser.add_argument("--out", default="artifacts/front_evolution")
    args = parser.parse_args()

    config = load_startup_config(args.config)
    limit_m = (args.limit_mm / 1000.0 if args.limit_mm is not None
               else float(config.get("safety_monitor", {}).get("limit_radius_m", 0.040)))

    path = Path(args.path)
    fluid_path = Path(args.fluid_path) if args.fluid_path else path
    exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, horizon_hours = run_replay(
        config, path=path, fluid_path=fluid_path,
        fluid_cols=(args.fluid_inlet_col, args.fluid_outlet_col, args.fluid_time_col),
        start_row=args.start_row, limit=args.limit, stride=args.stride,
        exclude_sensors=exclude)

    t_meas, r_meas = _clean(rows, "time_hours", "r_max_measured_m")
    t_end = crossing_time(t_meas, r_meas, limit_m)
    if t_end is None:
        t_end = float(t_meas[-1])
        print(f"\nMeasured r_max never reached {limit_m * 1000:.0f} mm "
              f"(peak {r_meas.max() * 1000:.1f} mm) -- plotting the full run.")
    else:
        print(f"\nMeasured r_max reaches {limit_m * 1000:.0f} mm at t = {t_end:.3f} h "
              f"({t_end * 60:.0f} min); every series is cut there.")

    style()
    figure(rows, out_dir, limit_m, horizon_hours, t_end)
    write_csv(rows, out_dir, t_end)


if __name__ == "__main__":
    main()

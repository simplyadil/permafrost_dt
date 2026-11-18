"""Why a parameter pins: how much signal each one actually has in the window.

``inversion_audit`` shows *that* ``cap_s`` sits on whatever bound it is given.
It cannot show *why*, because a trajectory that never leaves a bound carries no
information about the shape of the objective it was walking. This tool measures
that shape directly: it takes a real fitting window and sweeps one parameter at
a time across its allowed range, holding the others at base, recording the loss
at each point.

What the numbers mean:

* **signal** -- ``(max loss - min loss) / min loss`` over the sweep, in percent.
  A parameter the window cannot see has a flat curve and a signal near zero, and
  L-BFGS-B will slide it to whichever bound the last decimal of noise favours.
  That is the pinning mechanism, and it is a property of the *window*, not of the
  optimiser.
* **argmin** -- the scale the sweep prefers. If it sits at an end of the range,
  widening the bound moves the answer and does not fix it.

The sweep is run at several window lengths because the suspected cause is
physical rather than numerical. ``cap_s`` is a storage term: it sets how much
heat the solid takes to change temperature, and its effect on a sensor history
accumulates with time. Over the shipped five-snapshot window -- two and a half
minutes at a 30 s cadence, started from a *measured* initial field -- there is
very little integrated storage to see, while conduction near the boundary
dominates immediately. If that is the explanation, the signal in ``cap_s`` will
grow with window length while the conduction terms stay identifiable throughout.
If instead ``cap_s`` is flat at every length, the window is not the problem and
the parameter is unidentifiable from this sensor layout however long you watch.

Run:

    python -m software.tools.inversion_identifiability
    python -m software.tools.inversion_identifiability --at-hours 3.0 --windows 5,20,60
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from software.digital_twin.data_access.influx_utils import InfluxConfig
from software.startup.service_runners import (
    build_influx_config,
    create_boundary_condition_builder_server,
    create_pipe_boundary_server,
)
from software.startup.utils.config import load_startup_config
from software.tools.fem_debug_replay import (
    DEFAULT_FLUID_PATH,
    DEFAULT_SENSOR_PATH,
    _Sink,
    _sensor_rows,
)
from software.tools.inversion_accuracy import _build_inversion
from software.tools.replay_csv import _load_fluid_temps

PRETTY = {"cap_s": "c_s", "k_parameter": "k", "lambda_s": "lambda_s"}


def live_boundary_at(config: dict, snapshots: list[dict], end: int, fluid_path: Path,
                     recalibrate_data_s: float = 300.0) -> dict | None:
    """The pipe-calibrated boundary snapshot in force at ``snapshots[end]``.

    Replays the boundary builder and the real pipe-boundary service up to that
    snapshot, refitting on the same data-time cadence the live service uses, and
    returns the enriched boundary the forecast would be marching. This is the
    boundary the inversion is *not* currently fitting against.
    """
    rabbit_cfg = config.get("rabbitmq", {})
    influx_cfg: InfluxConfig = build_influx_config(config.get("influxdb", {}))
    builder = create_boundary_condition_builder_server(
        rabbit_cfg, config.get("boundary_condition_builder", {}))
    pipe_cfg = config.get("pipe_boundary", {})
    if not pipe_cfg:
        return None
    pipe = create_pipe_boundary_server(influx_cfg, rabbit_cfg, pipe_cfg)
    pipe.mq_boundary_out = _Sink()

    t_arr, tin_arr, tout_arr = _load_fluid_temps(
        fluid_path,
        inlet_col="Fluid Temperature_Inlet",
        outlet_col="Fluid Temperatue_Outlet",
        time_col="Time (h)",
    )
    cadence_s = ((float(snapshots[1]["time_hours"]) - float(snapshots[0]["time_hours"])) * 3600.0
                 if len(snapshots) > 1 else 30.0)
    recal_every = max(int(round(recalibrate_data_s / max(cadence_s, 1e-9))), 1)

    boundary = None
    for index in range(end + 1):
        snap = snapshots[index]
        t_h = float(snap["time_hours"])
        boundary = builder._build_snapshot(snap)
        pipe.recalibrate_seconds = 0.0 if index % recal_every == 0 else 1.0e12
        pipe._on_fluid({"time_hours": t_h, "timestamp": snap["timestamp"],
                        "t_inlet_c": float(np.interp(t_h, t_arr, tin_arr)),
                        "t_outlet_c": float(np.interp(t_h, t_arr, tout_arr)),
                        "site_id": "default"})
        pipe._on_boundary(boundary)
        if isinstance(pipe.mq_boundary_out.last, dict):
            boundary = pipe.mq_boundary_out.last
    return boundary


def sweep_window(inversion, snapshots: list[dict], *, window: int, points: int) -> list[dict]:
    """Loss curve for each parameter over its bounds, on one window of length N."""
    # The server trims its buffer to ``window_size``, so both gates move together
    # or the window silently stays at the configured five.
    inversion.window_size = window
    inversion.min_snapshots = window
    inversion._snapshots = []
    for snap in snapshots[-window:]:
        inversion._upsert_snapshot(snap)

    built = inversion._build_observation_window()
    if built is None:
        return []
    times_hours, positions, obs_series, ic_sensors = built
    span_min = (times_hours[-1] - times_hours[0]) * 60.0

    base = {name: float(value) for name, value in inversion.base_params.items()}
    rows: list[dict] = []
    for name in inversion.invert_params:
        lo, hi = inversion.relative_bounds[name]
        scales = np.linspace(float(lo), float(hi), points)
        losses = []
        for scale in scales:
            params = dict(base)
            params[name] = base[name] * float(scale)
            series = inversion._generate_series(params, positions, times_hours, ic_sensors)
            losses.append(float(inversion._compute_loss(series, obs_series)))
        losses = np.array(losses, dtype=float)
        finite = losses[np.isfinite(losses)]
        if finite.size < 2:
            continue
        lo_loss, hi_loss = float(finite.min()), float(finite.max())
        signal = (hi_loss - lo_loss) / lo_loss * 100.0 if lo_loss > 0 else float("nan")
        best = float(scales[int(np.nanargmin(losses))])
        rows.append({
            "window": window,
            "span_minutes": round(span_min, 2),
            "parameter": name,
            "signal_pct": signal,
            "argmin_scale": best,
            "argmin_value": base[name] * best,
            "at_edge": bool(best <= float(lo) + 1e-9 or best >= float(hi) - 1e-9),
            "loss_min": lo_loss,
            "loss_max": hi_loss,
            "scales": ";".join(f"{s:.4g}" for s in scales),
            "losses": ";".join(f"{v:.6g}" for v in losses),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", default=DEFAULT_SENSOR_PATH)
    parser.add_argument("--config", default="startup.conf")
    parser.add_argument("--out", default="artifacts/inversion_identifiability")
    parser.add_argument("--at-hours", type=float, default=3.0,
                        help="Experiment time the window ends at. Default 3.0 h, where the "
                             "front is moving and the parameters should be at their most visible.")
    parser.add_argument("--windows", default="5,20,60",
                        help="Window lengths in snapshots. The shipped service uses 5.")
    parser.add_argument("--points", type=int, default=9, help="Sweep points per parameter.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fluid-path", default=DEFAULT_FLUID_PATH)
    parser.add_argument("--matched-bc", action="store_true",
                        help="Sweep against the pipe-calibrated boundary the forecast actually "
                             "runs on, instead of the configured default the inversion fits "
                             "under today. The decisive comparison: if the optima move inside "
                             "the box, the pinning is a boundary error wearing a parameter mask.")
    args = parser.parse_args()

    config = load_startup_config(args.config)
    snapshots = _sensor_rows(Path(args.path), None, 1, None, max(int(args.stride), 1), None)
    if not snapshots:
        raise SystemExit(f"No usable sensor rows parsed from {args.path}")

    times = np.array([float(s["time_hours"]) for s in snapshots])
    end = int(np.argmin(np.abs(times - float(args.at_hours))))
    windows = [int(w) for w in args.windows.split(",") if w.strip()]
    print(f"Loaded {len(snapshots)} snapshots; window ends at index {end} (t = {times[end]:.3f} h)")
    print(f"Sweeping {args.points} points per parameter at window lengths {windows}\n")

    inversion = _build_inversion(config, maxiter=None, maxfun=None, warmup_hours=0.0)
    if args.matched_bc:
        from software.digital_twin.simulator.fem_inversion.window_forward_model import (
            WindowForwardModel,
        )
        boundary = live_boundary_at(config, snapshots, end, Path(args.fluid_path))
        if boundary is None:
            raise SystemExit("No pipe_boundary configured; --matched-bc needs one.")
        inversion._window_model = WindowForwardModel(args.config)
        inversion._window_model._snapshot = boundary
        pipe_set = str(config.get("pipe_boundary", {}).get("pipe_boundary_set", "left"))
        for bc in boundary.get("boundaries", []) or []:
            if bc.get("boundary_set") == pipe_set:
                print(f"matched boundary: type={bc.get('bc_type')} T={bc.get('temperature')} "
                      f"h={bc.get('h_coeff')}")
        print()
    for name in inversion.invert_params:
        lo, hi = inversion.relative_bounds[name]
        base = float(inversion.base_params[name])
        print(f"  {name:<12} base {base:g}  scale [{lo:g}, {hi:g}]  -> {base * lo:g}..{base * hi:g}")
    print()

    rows: list[dict] = []
    for window in windows:
        if end + 1 < window:
            print(f"window {window}: not enough history before t={times[end]:.3f} h; skipped")
            continue
        start = time.perf_counter()
        got = sweep_window(inversion, snapshots[: end + 1], window=window, points=int(args.points))
        rows.extend(got)
        if not got:
            continue
        span = got[0]["span_minutes"]
        print(f"window {window:>3} snapshots ({span:.1f} min of data)"
              f"   [{time.perf_counter() - start:.0f}s]")
        print(f"    {'parameter':<12} {'signal':>9}  {'argmin':>9}  {'value':>10}  verdict")
        for row in got:
            verdict = ("flat: unidentifiable" if row["signal_pct"] < 1.0
                       else "pinned at edge" if row["at_edge"]
                       else "interior optimum")
            print(f"    {PRETTY.get(row['parameter'], row['parameter']):<12} "
                  f"{row['signal_pct']:>8.2f}% {row['argmin_scale']:>9.3f} "
                  f"{row['argmin_value']:>10.4g}  {verdict}")
        print()

    if not rows:
        raise SystemExit("No windows could be built; nothing to report.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "identifiability.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")

    print("\nsignal (%) by window length")
    names = list(dict.fromkeys(r["parameter"] for r in rows))
    widths = sorted({r["window"] for r in rows})
    print(f"    {'parameter':<12}" + "".join(f"{w:>12}" for w in widths))
    for name in names:
        cells = []
        for w in widths:
            hit = next((r for r in rows if r["parameter"] == name and r["window"] == w), None)
            cells.append(f"{hit['signal_pct']:>11.2f}%" if hit else f"{'-':>12}")
        print(f"    {PRETTY.get(name, name):<12}" + "".join(cells))


if __name__ == "__main__":
    main()

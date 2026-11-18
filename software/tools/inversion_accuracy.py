"""Does the inversion actually make the FEM more accurate?

Offline analysis tool, not a service. It replays a recorded sensor file and runs
*both* simulator services on it -- ``fem_inversion`` and ``fem_forecast`` -- with
RabbitMQ and InfluxDB stubbed out, then measures the forecast error against the
thermocouples with and without the inverted parameters.

The two arms come from one forecast server, so they are exact:

    with inversion     the solver receiving the ``material_override`` the
                       inversion service just published
    without inversion  the forecast server's own ``run_baseline`` twin, built
                       from the configured material block, seeded from the same
                       initial field and advanced on the same boundary snapshot

Only the material parameters differ between them. Boundary conditions, initial
condition, mesh and time stepping are shared, so the gap between the curves is
the inversion's contribution and nothing else.

Error metric is RMSE over the thermocouples: the FEM field is sampled at the
sensor positions by linear interpolation and differenced against the measured
values, at every replayed step. Note the well-known limitation, spelled out in
``fem_debug_replay``: a forecast can sit within a couple of percent of every
thermocouple and still put the thaw front in the wrong place, because the
sensors are sparse near the front. Read this plot together with the thaw-front
and volume ones, not instead of them.

Runtime: the inversion is the expensive part -- one update runs the forward FEM
``window_size`` times per objective evaluation, up to ``maxfun`` evaluations, so
a few hundred solves. Inversions therefore fire every ``--invert-every``
replayed steps, and ``--maxfun`` / ``--maxiter`` / ``--stride`` bound the rest.
Expect minutes per inversion update; the tool prints the elapsed time of each.

One caveat worth knowing when reading the result: the inversion's own forward
model calls ``run_fem_case.FEM`` with ``boundary_snapshot=None``, so it fits
against the *configured default* boundaries rather than the pipe-calibrated ones
the forecast actually runs on. Parameters fitted under one boundary condition and
applied under another are part of what this plot measures.

By default the whole recording is replayed, end to end; ``--limit`` is opt-in
truncation for a quick wiring check and is off unless asked for. All three
outputs are rewritten after every inversion update, so a run stopped part way
through still leaves the results it reached.

Outputs, under ``--out`` (default ``artifacts/inversion_accuracy``):

* ``inversion_accuracy.png``  -- the two panels
* ``inversion_accuracy.csv``  -- per-step RMSE for both arms
* ``inversion_updates.csv``   -- one row per inversion update (params, loss, elapsed)

    python -m software.tools.inversion_accuracy                          # whole run, config budget
    python -m software.tools.inversion_accuracy --invert-every 8 --maxfun 20
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from software.digital_twin.data_access.influx_utils import InfluxConfig
from software.startup.service_runners import build_influx_config, create_fem_inversion_server
from software.startup.utils.config import load_startup_config
from software.tools.fem_debug_replay import (
    DEFAULT_FLUID_PATH,
    DEFAULT_SENSOR_PATH,
    _NullInflux,
    _Sink,
    _build_servers,
    _interp_to_grid,
    _sensor_rows,
    _solver_field_c,
)
from software.tools.replay_csv import _load_fluid_temps

# With inversion is the subject, without is the foil: grey reads as recessive
# next to it, and the improvement panel is coloured by sign, not by series.
CALIBRATED_COLOR = "#0369a1"
BASELINE_COLOR = "#94a3b8"
BETTER_COLOR = "#15803d"
WORSE_COLOR = "#b91c1c"
EVENT_COLOR = "#c2410c"
GRID_COLOR = "#e2e8f0"
INK = "#1e293b"
INK_MUTED = "#64748b"


# ---------------------------------------------------------------------------
# ERROR
# ---------------------------------------------------------------------------
def _sensor_rmse(solver, obs_x: np.ndarray, obs_z: np.ndarray, obs_t: np.ndarray) -> tuple[float, float]:
    """RMSE and max |error| of a solver field against the measured sensors.

    The field is interpolated to the sensor positions rather than the sensors
    being snapped to the nearest node: nearest-node sampling quantises the error
    onto the mesh and would credit or blame a parameter set for the mesh.
    """
    node_xy, temps_c = _solver_field_c(solver)
    predicted = _interp_to_grid(node_xy[:, 0], node_xy[:, 1], temps_c, np.abs(obs_x), obs_z)
    residual = predicted - obs_t
    finite = np.isfinite(residual)
    if not np.any(finite):
        return float("nan"), float("nan")
    return (
        float(np.sqrt(np.mean(np.square(residual[finite])))),
        float(np.max(np.abs(residual[finite]))),
    )


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------
def _style_axes(ax, *, xlabel: str, title: str) -> None:
    ax.set_title(title, fontsize=10, color=INK, pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color=INK_MUTED)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(labelsize=8, colors=INK_MUTED, length=3)


def _plot_accuracy(
    times: np.ndarray,
    rmse_cal: np.ndarray,
    rmse_base: np.ndarray,
    update_times: list[float],
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.errstate(divide="ignore", invalid="ignore"):
        improvement = np.where(rmse_base > 1e-12, (rmse_base - rmse_cal) / rmse_base * 100.0, np.nan)

    # On a partial render the newest update can sit past the last replayed step.
    update_times = [t for t in update_times if float(times[0]) <= t <= float(times[-1])]

    fig, (ax_rmse, ax_gain) = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=160)

    # --- absolute error ----------------------------------------------------
    # Before the first update both arms are the same model; say so rather than
    # letting a flat zero in the right-hand panel read as "inversion did nothing".
    if update_times:
        ax_rmse.axvspan(
            float(times[0]), float(update_times[0]),
            color=GRID_COLOR, alpha=0.55, linewidth=0, zorder=0,
            label="before first update",
        )
    for t_update in update_times:
        ax_rmse.axvline(t_update, color=EVENT_COLOR, linewidth=0.9, alpha=0.35, zorder=1)
    if update_times:
        # Label the rule once; one entry, not one per update.
        ax_rmse.axvline(update_times[0], color=EVENT_COLOR, linewidth=0.9, alpha=0.35,
                        label="inversion update", zorder=1)
    # Dashed baseline under the solid calibrated line: the two coincide until the
    # parameters have had time to bite, and a solid line hidden under a solid line
    # looks like a missing series.
    ax_rmse.plot(times, rmse_base, color=BASELINE_COLOR, linewidth=2.6, linestyle="--",
                 label="without inversion", zorder=2)
    ax_rmse.plot(times, rmse_cal, color=CALIBRATED_COLOR, linewidth=2.0, label="with inversion", zorder=3)
    _style_axes(ax_rmse, xlabel="Time (h)", title="Forecast error at the thermocouples")
    ax_rmse.set_ylabel("RMSE (C)", fontsize=9, color=INK_MUTED)
    ax_rmse.set_ylim(bottom=0.0)
    ax_rmse.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK_MUTED)

    # --- relative change ---------------------------------------------------
    finite = np.isfinite(improvement)
    ax_gain.axhline(0.0, color=INK_MUTED, linewidth=1.0)
    ax_gain.fill_between(
        times, improvement, 0.0, where=finite & (improvement >= 0),
        color=BETTER_COLOR, alpha=0.16, linewidth=0, interpolate=True,
    )
    ax_gain.fill_between(
        times, improvement, 0.0, where=finite & (improvement < 0),
        color=WORSE_COLOR, alpha=0.16, linewidth=0, interpolate=True,
    )
    ax_gain.plot(times, improvement, color=INK, linewidth=1.8, zorder=3)
    _style_axes(ax_gain, xlabel="Time (h)", title="Change in RMSE from inversion")
    ax_gain.set_ylabel("RMSE reduction (%)", fontsize=9, color=INK_MUTED)
    if np.any(finite):
        span = max(float(np.max(np.abs(improvement[finite]))), 1.0)
        ax_gain.set_ylim(-span * 1.2, span * 1.2)
        ax_gain.annotate(
            "inversion helps",
            xy=(0.02, 0.96), xycoords="axes fraction", va="top",
            fontsize=8, color=BETTER_COLOR,
        )
        ax_gain.annotate(
            "inversion hurts",
            xy=(0.02, 0.04), xycoords="axes fraction", va="bottom",
            fontsize=8, color=WORSE_COLOR,
        )

    for ax in (ax_rmse, ax_gain):
        ax.set_xlim(float(times[0]), float(times[-1]))

    mean_gain = float(np.nanmean(improvement)) if np.any(finite) else float("nan")
    fig.suptitle(
        f"Effect of inversion on FEM accuracy   |   t = {times[0]:.2f}-{times[-1]:.2f} h   |   "
        f"{len(update_times)} inversion update(s)   |   mean RMSE change {mean_gain:+.1f}%",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# REPLAY
# ---------------------------------------------------------------------------
def _build_inversion(config: dict, *, maxiter: int | None, maxfun: int | None,
                     warmup_hours: float | None = None):
    """The real inversion service with its clients stubbed out.

    ``min_update_hours`` / ``min_update_seconds`` are zeroed: those gates pace the
    live service against wall-clock and experiment time, which in a replay would
    make the update cadence depend on how fast this machine solves. Cadence is
    controlled explicitly by ``--invert-every`` instead.

    ``warmup_hours`` is deliberately *not* zeroed. It is measured on the sensor
    clock, so it covers the same stretch of experiment however fast the replay
    runs, and it changes which fits exist at all -- a replay that skipped it
    would not be showing the service's behaviour. Pass a value to override it.
    """
    rabbit_cfg = config.get("rabbitmq", {})
    influx_cfg: InfluxConfig = build_influx_config(config.get("influxdb", {}))
    inversion_cfg = dict(config.get("fem_inversion", {}))
    inversion_cfg["min_update_hours"] = 0.0
    inversion_cfg["min_update_seconds"] = 0.0
    if warmup_hours is not None:
        inversion_cfg["warmup_hours"] = float(warmup_hours)
    if maxiter is not None:
        inversion_cfg["maxiter"] = int(maxiter)
    if maxfun is not None:
        inversion_cfg["maxfun"] = int(maxfun)

    inversion = create_fem_inversion_server(
        influx_cfg,
        rabbit_cfg,
        inversion_cfg,
        config.get("boundary_condition_builder", {}),
        config.get("fem_forecast", {}),
    )
    inversion.mq_out = _Sink()
    inversion.mq_fem_in = _Sink()
    inversion.influx = _NullInflux()
    return inversion


def _run_inversion_now(inversion) -> tuple[dict[str, float] | None, dict | None, float]:
    """Run one inversion update synchronously; returns (params, payload, seconds).

    The service normally hands the job to a worker thread. Here it is run inline:
    a replay must not advance the forecast past the snapshot the parameters were
    fitted on, or the comparison silently measures a stale override.
    """
    job = inversion._prepare_inversion_job(datetime.utcnow())
    if job is None:
        return None, None, 0.0
    (trigger_time, trigger_snapshot, times_hours, positions, obs_series, ic_sensors,
     boundary_schedule) = job

    start = time.monotonic()
    inversion._run_inversion_job(
        trigger_time=trigger_time,
        trigger_snapshot=trigger_snapshot,
        times_hours=times_hours,
        positions=positions,
        obs_series=obs_series,
        ic_sensors=ic_sensors,
        boundary_schedule=boundary_schedule,
    )
    elapsed = time.monotonic() - start

    payload = inversion.mq_out.last
    if not isinstance(payload, dict):
        return None, None, elapsed
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        return None, payload, elapsed
    return {str(k): float(v) for k, v in parameters.items()}, payload, elapsed


def run_inversion_accuracy(
    *,
    path: Path,
    config_path: str | Path,
    fluid_path: Path | None,
    fluid_inlet_col: str,
    fluid_outlet_col: str,
    fluid_time_col: str,
    out_dir: Path,
    time_column: str | None,
    start_row: int,
    limit: int | None,
    stride: int,
    invert_every: int,
    maxiter: int | None,
    maxfun: int | None,
    horizon_hours: float | None,
    use_pipe_boundary: bool,
    exclude_sensors: set[str] | None,
) -> None:
    config = load_startup_config(config_path)

    snapshots = _sensor_rows(path, time_column, start_row, limit, stride, exclude_sensors)
    if not snapshots:
        raise ValueError(f"No usable sensor rows parsed from {path}")
    if exclude_sensors:
        print(f"Excluded sensor channels: {sorted(exclude_sensors)}")
    print(
        f"Loaded {len(snapshots)} sensor snapshots from {path.name} "
        f"(t = {snapshots[0]['time_hours']:.3f} .. {snapshots[-1]['time_hours']:.3f} h)"
    )

    fluid = None
    if fluid_path is not None and use_pipe_boundary:
        fluid = _load_fluid_temps(
            fluid_path,
            inlet_col=fluid_inlet_col,
            outlet_col=fluid_outlet_col,
            time_col=fluid_time_col,
        )
        print(f"Loaded {len(fluid[0])} fluid samples from {fluid_path.name}")

    # run_baseline is forced on: the uncalibrated twin *is* the control arm here.
    builder, pipe, forecast = _build_servers(config, run_baseline=True, horizon_hours=horizon_hours)
    if not use_pipe_boundary:
        pipe = None
    inversion = _build_inversion(config, maxiter=maxiter, maxfun=maxfun)
    print(
        "Pipe boundary stage: "
        + ("enabled" if pipe is not None and fluid is not None else "disabled (configured boundaries used as-is)")
    )
    print(
        f"Inversion: params={inversion.invert_params}, window={inversion.window_size}, "
        f"maxiter={inversion.maxiter}, maxfun={inversion.maxfun}, every {invert_every} step(s)"
    )

    planned_updates = len([i for i in range(1, len(snapshots)) if i % invert_every == 0])
    print(
        f"Plan: {len(snapshots)} steps covering the full "
        f"{snapshots[-1]['time_hours'] - snapshots[0]['time_hours']:.2f} h of data, "
        f"{planned_updates} inversion update(s). Partial results are written after each update, "
        "so an interrupted run still leaves a plot."
    )

    # The forecast advances by the elapsed time between messages, capped at
    # max_step_hours. A stride that opens a gap wider than the cap makes simulated
    # time fall behind observed time, and both arms then lag the sensors -- which
    # this plot would report as model error.
    step_gap_hours = float(np.median(np.diff([s["time_hours"] for s in snapshots]))) if len(snapshots) > 1 else 0.0
    max_step_hours = float(config.get("fem_forecast", {}).get("max_step_hours", 0.5) or 0.0)
    if max_step_hours > 0.0 and step_gap_hours > max_step_hours + 1e-9:
        print(
            f"WARNING: --stride {stride} leaves {step_gap_hours:.3f} h between steps but "
            f"fem_forecast.max_step_hours is {max_step_hours:.3f} h. Simulated time will fall "
            "behind observed time. Lower --stride or raise max_step_hours."
        )

    rows: list[dict] = []
    update_rows: list[dict] = []
    material_override: dict[str, float] | None = None

    for index, sensor_msg in enumerate(snapshots):
        time_hours = float(sensor_msg["time_hours"])

        # 1. Feed the inversion service its snapshot, and let it fit on cadence.
        parsed = inversion._parse_snapshot(sensor_msg)
        if parsed is not None:
            inversion._upsert_snapshot(parsed)
        if index > 0 and index % invert_every == 0:
            params, payload, elapsed = _run_inversion_now(inversion)
            if params is not None:
                material_override = params
                update_rows.append(
                    {
                        "time_hours": round(time_hours, 6),
                        "step": index,
                        "elapsed_seconds": round(elapsed, 3),
                        "final_loss": float(payload.get("final_loss", float("nan"))) if payload else float("nan"),
                        "success": bool(payload.get("success", False)) if payload else False,
                        "eval_count": int(payload.get("eval_count", 0)) if payload else 0,
                        **{name: params[name] for name in sorted(params) if name in inversion.invert_params},
                    }
                )
                pretty = ", ".join(f"{k}={params[k]:.4g}" for k in inversion.invert_params if k in params)
                print(f"  * inversion @ t={time_hours:6.3f} h  ({elapsed:.1f}s)  {pretty}")

                # Cost per update is dominated by window_size x maxfun forward
                # solves and barely varies, so the first one predicts the rest.
                remaining = planned_updates - len(update_rows)
                if remaining > 0:
                    mean_cost = float(np.mean([row["elapsed_seconds"] for row in update_rows]))
                    print(
                        f"    {remaining} update(s) left, ~{mean_cost / 60.0:.0f} min each "
                        f"-> ~{remaining * mean_cost / 3600.0:.1f} h remaining"
                    )
                # Flush results now: this run may have hours left in it.
                _write_outputs(rows, update_rows, out_dir, quiet=True)

        # 2. Boundary condition builder, then the pipe calibration stage.
        boundary_snapshot = builder._build_snapshot(sensor_msg)
        if pipe is not None and fluid is not None:
            t_arr, tin_arr, tout_arr = fluid
            pipe._on_fluid(
                {
                    "time_hours": time_hours,
                    "timestamp": sensor_msg["timestamp"],
                    "t_inlet_c": float(np.interp(time_hours, t_arr, tin_arr)),
                    "t_outlet_c": float(np.interp(time_hours, t_arr, tout_arr)),
                    "site_id": "default",
                }
            )
            pipe._on_boundary(boundary_snapshot)
            enriched = pipe.mq_boundary_out.last
            if isinstance(enriched, dict):
                boundary_snapshot = enriched

        # 3. Attach the inverted parameters exactly as the inversion service does
        #    when publish_material_override_boundary is on.
        if material_override:
            boundary_snapshot["material_override"] = dict(material_override)

        # 4. One forecast call advances both arms: the override solver and the
        #    configured baseline twin.
        forecast._process_message(boundary_snapshot)
        if forecast._solver is None:
            raise RuntimeError("FEM solver was never built; check the forecast config.")
        if forecast._baseline_solver is None:
            raise RuntimeError("Baseline twin was never built; fem_forecast.run_baseline must be on.")

        obs_x = np.array([s["x_m"] for s in sensor_msg["sensors"]], dtype=float)
        obs_z = np.array([s["z_m"] for s in sensor_msg["sensors"]], dtype=float)
        obs_t = np.array([s["temperature"] for s in sensor_msg["sensors"]], dtype=float)

        rmse_cal, max_cal = _sensor_rmse(forecast._solver, obs_x, obs_z, obs_t)
        rmse_base, max_base = _sensor_rmse(forecast._baseline_solver, obs_x, obs_z, obs_t)

        rows.append(
            {
                "time_hours": round(time_hours, 6),
                "rmse_with_inversion_c": rmse_cal,
                "rmse_without_inversion_c": rmse_base,
                "max_abs_with_inversion_c": max_cal,
                "max_abs_without_inversion_c": max_base,
                "inverted": bool(material_override),
            }
        )
        print(
            f"  [{index + 1}/{len(snapshots)}] t={time_hours:7.3f} h  "
            f"RMSE with {rmse_cal:6.3f} C   without {rmse_base:6.3f} C"
            + ("" if material_override else "   (no override yet)")
        )

    _write_outputs(rows, update_rows, out_dir)
    _print_summary(rows, update_rows)


def _write_outputs(rows: list[dict], update_rows: list[dict], out_dir: Path, *, quiet: bool = False) -> None:
    """Write the CSVs and the plot from whatever has been replayed so far.

    Called after every inversion update as well as at the end: a full run takes
    hours, and a run that is interrupted at hour three should still leave the
    results it has rather than nothing.
    """
    if not rows:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "inversion_accuracy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if update_rows:
        updates_path = out_dir / "inversion_updates.csv"
        fieldnames: list[str] = []
        for row in update_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with updates_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(update_rows)

    png_path = out_dir / "inversion_accuracy.png"
    _plot_accuracy(
        np.array([row["time_hours"] for row in rows], dtype=float),
        np.array([row["rmse_with_inversion_c"] for row in rows], dtype=float),
        np.array([row["rmse_without_inversion_c"] for row in rows], dtype=float),
        [float(row["time_hours"]) for row in update_rows],
        png_path,
    )
    if quiet:
        print(f"    (partial results written to {out_dir})")
    else:
        print(f"\nWrote per-step accuracy to {csv_path}")
        if update_rows:
            print(f"Wrote inversion updates to {out_dir / 'inversion_updates.csv'}")
        print(f"Wrote plot to {png_path}")


def _print_summary(rows: list[dict], update_rows: list[dict]) -> None:
    cal = np.array([row["rmse_with_inversion_c"] for row in rows], dtype=float)
    base = np.array([row["rmse_without_inversion_c"] for row in rows], dtype=float)
    inverted = np.array([bool(row["inverted"]) for row in rows])

    print("\n--- summary ---")
    print(f"steps                  : {len(rows)}")
    print(f"inversion updates      : {len(update_rows)}")
    if update_rows:
        elapsed = np.array([row["elapsed_seconds"] for row in update_rows], dtype=float)
        print(f"inversion cost         : mean {elapsed.mean():.1f}s  total {elapsed.sum():.1f}s")
    print(f"RMSE without inversion : mean {np.nanmean(base):.3f} C  final {base[-1]:.3f} C")
    print(f"RMSE with inversion    : mean {np.nanmean(cal):.3f} C  final {cal[-1]:.3f} C")

    if not np.any(inverted):
        print("No override was ever applied -- the two arms are identical by construction.")
        return

    # Only the steps that actually ran with an override say anything about the
    # inversion; before the first update the arms are the same model.
    cal_i, base_i = cal[inverted], base[inverted]
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(base_i > 1e-12, (base_i - cal_i) / base_i * 100.0, np.nan)
    print(
        f"after first override   : {int(inverted.sum())} steps, "
        f"RMSE {np.nanmean(base_i):.3f} -> {np.nanmean(cal_i):.3f} C "
        f"(mean change {np.nanmean(gain):+.1f}%)"
    )
    better = int(np.sum(cal_i < base_i))
    print(f"steps improved         : {better}/{int(inverted.sum())}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay recorded sensor data through the inversion and forecast services "
        "and plot the effect of inversion on FEM accuracy.",
    )
    parser.add_argument(
        "--path",
        default=DEFAULT_SENSOR_PATH,
        help=f"Recorded sensor CSV/XLSX with H*D* columns (default: {DEFAULT_SENSOR_PATH}).",
    )
    parser.add_argument("--config", default="startup.conf", help="Startup config to take the service setup from.")
    parser.add_argument("--out", default="artifacts/inversion_accuracy", help="Output directory.")
    parser.add_argument("--time-column", default=None, help="Override the time column header.")
    parser.add_argument("--start-row", type=int, default=1, help="First data row (1-based).")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Truncate to the first N rows. Default is no limit: the whole recording.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=25,
        help="Use every Nth row (default 25). This thins the sampling but still spans the whole "
        "recording; it must stay below fem_forecast.max_step_hours worth of rows.",
    )
    parser.add_argument(
        "--invert-every",
        type=int,
        default=5,
        help="Run an inversion update every Nth replayed step (default 5). Each update costs "
        "roughly window_size x maxfun forward FEM solves.",
    )
    parser.add_argument("--maxiter", type=int, default=None, help="Override fem_inversion.maxiter.")
    parser.add_argument("--maxfun", type=int, default=None, help="Override fem_inversion.maxfun.")
    parser.add_argument("--horizon-hours", type=float, default=None, help="Override fem_forecast.horizon_hours.")
    parser.add_argument(
        "--no-pipe-boundary",
        action="store_true",
        help="Skip the pipe calibration stage and use the configured boundaries as-is.",
    )
    parser.add_argument(
        "--exclude-sensors",
        default=None,
        help="Comma-separated channels to drop, e.g. 'H290D165'.",
    )

    fluid_group = parser.add_argument_group("fluid temperatures (drive the pipe boundary)")
    fluid_group.add_argument(
        "--fluid-path",
        default=DEFAULT_FLUID_PATH,
        help=f"Fluid inlet/outlet XLSX (default: {DEFAULT_FLUID_PATH}). "
        "Pass '' to run without it; the pipe boundary then stays at its configured values.",
    )
    fluid_group.add_argument("--fluid-inlet-col", default="Fluid Temperature_Inlet")
    fluid_group.add_argument("--fluid-outlet-col", default="Fluid Temperatue_Outlet")
    fluid_group.add_argument("--fluid-time-col", default="Time (h)")

    args = parser.parse_args()

    run_inversion_accuracy(
        path=Path(args.path),
        config_path=args.config,
        fluid_path=Path(args.fluid_path) if args.fluid_path else None,
        fluid_inlet_col=args.fluid_inlet_col,
        fluid_outlet_col=args.fluid_outlet_col,
        fluid_time_col=args.fluid_time_col,
        out_dir=Path(args.out),
        time_column=args.time_column,
        start_row=args.start_row,
        limit=args.limit,
        stride=max(int(args.stride), 1),
        invert_every=max(int(args.invert_every), 1),
        maxiter=args.maxiter,
        maxfun=args.maxfun,
        horizon_hours=args.horizon_hours,
        use_pipe_boundary=not args.no_pipe_boundary,
        exclude_sensors=set(args.exclude_sensors.split(",")) if args.exclude_sensors else None,
    )


if __name__ == "__main__":
    main()

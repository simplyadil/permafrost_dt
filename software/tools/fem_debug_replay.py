"""Offline replay: measured temperature field vs FEM forecast, side by side.

Debugging tool, not a service. It replays a recorded sensor CSV through the
*same* code the live pipeline runs -- boundary condition builder, pipe boundary
calibration, FEM forecast server -- but with RabbitMQ and InfluxDB stubbed out,
and draws the measured field next to the forecast field at each sampled step,
each with its 0 C thaw front.

Why the servers are driven directly instead of reimplemented: a debug rig that
reimplements the boundary handling would debug the rig, not the pipeline. Here
the only substitutions are the message-bus and database clients.

The point of the plots is *shape*, not magnitude: a forecast can sit within a
couple of percent of every thermocouple and still put the thaw front in the
wrong place, because the sensors are sparse near the front and percent error on
absolute temperature is a very forgiving metric. Each panel therefore carries
the measured front as a dashed overlay, and ``front_metrics.csv`` records the
front radii from both fields on the same grid and the same per-depth rule.

Outputs, all under ``--out`` (default ``artifacts/fem_debug``):

* ``fem_debug_replay.gif`` -- the animation, written by default
* ``frames/frame_<step>_t<hours>h.png`` -- the individual frames it is built from
* ``front_metrics.csv`` -- per-step sensor RMSE and measured/FEM front radii

The sensor and fluid files both default to the polished April 8 dataset, so the
usual invocation is just:

    python -m software.tools.fem_debug_replay --plot-every 10

Point it elsewhere with ``--path`` / ``--fluid-path``.

Add ``--material-override cap_s=1200,lambda_s=5.0`` to inject inverted
parameters the way the inversion service does, and ``--baseline`` for a third
panel that runs the configured (uncalibrated) parameters alongside.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.interpolate import griddata

from software.digital_twin.data_access.influx_utils import InfluxConfig
from software.startup.service_runners import (
    build_influx_config,
    create_boundary_condition_builder_server,
    create_fem_forecast_server,
    create_pipe_boundary_server,
)
from software.startup.utils.config import load_startup_config
from software.tools.replay_csv import _iter_rows, _load_fluid_temps, _parse_header

# Default dataset: the polished April 8 export. Clean 80-channel H*D* grid over
# 0-5.26 h with no gaps, and the stuck H290D165 channel of the raw export
# already corrected. The fluid inlet/outlet stream lives in its own file and
# covers a wider window (0-6.58 h), so it is interpolated onto the sensor times.
DEFAULT_SENSOR_PATH = "data/CR1000_Table1_April8_replay (3).xlsx"
DEFAULT_FLUID_PATH = "data/exports/CR1000_Table1_April8_fluid.xlsx"


# ---------------------------------------------------------------------------
# STUBS
# Stand in for the message bus and the database so the real servers can run
# without any infrastructure. Nothing else about them is changed.
# ---------------------------------------------------------------------------
class _Sink:
    """Captures published messages instead of sending them to RabbitMQ."""

    def __init__(self) -> None:
        self.last: dict | None = None
        self.count = 0

    def publish(self, message: dict) -> None:
        self.last = message
        self.count += 1


class _NullInflux:
    """Swallows every Influx write the servers attempt."""

    def __getattr__(self, _name: str):
        def _noop(*_args: Any, **_kwargs: Any) -> None:
            return None

        return _noop


# ---------------------------------------------------------------------------
# FIELD / FRONT HELPERS
# ---------------------------------------------------------------------------
def _grid_from_mesh(node_xy: np.ndarray, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Plot grid spanning the mesh footprint (radius x depth)."""
    r_max = float(np.max(np.abs(node_xy[:, 0])))
    z_max = float(np.max(node_xy[:, 1]))
    z_min = float(np.min(node_xy[:, 1]))
    r_axis = np.linspace(0.0, r_max, resolution)
    z_axis = np.linspace(z_min, z_max, resolution)
    return np.meshgrid(r_axis, z_axis)


def _interp_to_grid(
    x_m: np.ndarray,
    z_m: np.ndarray,
    values: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """Interpolation onto the plot grid; NaN outside the convex hull.

    Linear by default for both fields even though the FEM nodes could support
    something fancier -- the two panels must be produced identically or the
    front shapes are not comparable, and ``method`` exists so a caller can pick
    a smoother reconstruction for *both* rather than for one. Points outside
    the hull stay NaN and plot as blank, which is honest about where the
    thermocouples stop constraining anything.
    """
    finite = np.isfinite(x_m) & np.isfinite(z_m) & np.isfinite(values)
    if int(np.count_nonzero(finite)) < 3:
        return np.full(R.shape, np.nan)
    points = np.column_stack([np.abs(x_m[finite]), z_m[finite]])
    return griddata(points, values[finite], (R, Z), method=method)


def _front_from_grid(
    R: np.ndarray,
    Z: np.ndarray,
    T: np.ndarray,
    threshold_c: float = 0.0,
    surface_exclusion_m: float = 0.0,
) -> tuple[float | None, float | None, np.ndarray, np.ndarray]:
    """Per-depth outermost thawed radius, mirroring compute_thaw_metrics.

    The solver takes the outermost thawed *node* per depth level; here the same
    rule is applied to the interpolated grid so the measured and forecast fronts
    are extracted by an identical procedure.

    ``surface_exclusion_m`` drops depths shallower than the given value, the same
    guard ``thaw_front_reconstruction`` applies: the near-surface horizontal 0 C
    layer is an experimental artefact, and because the rule takes the *outermost*
    thawed cell per depth, one warm shallow row otherwise pins r_max to the edge
    of the domain in both fields and hides the very mismatch being debugged.

    Returns (r_max, r_avg, radii, depths).
    """
    thawed = np.isfinite(T) & (T >= threshold_c)
    radii: list[float] = []
    depths: list[float] = []
    for row in range(T.shape[0]):
        depth = float(Z[row][0])
        if depth < surface_exclusion_m:
            continue
        if not np.any(thawed[row]):
            continue
        radii.append(float(np.max(R[row][thawed[row]])))
        depths.append(depth)
    if not radii:
        return None, None, np.array([]), np.array([])
    radii_arr = np.asarray(radii)
    return float(np.max(radii_arr)), float(np.mean(radii_arr)), radii_arr, np.asarray(depths)


def _solver_field_c(solver) -> tuple[np.ndarray, np.ndarray]:
    """Nodal coordinates and temperatures in Celsius from a live solver."""
    zero_c = float(solver.solver.temperature_offset_k)
    return solver.node.copy(), solver.T.copy() - zero_c


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------
def _draw_panel(
    ax,
    R: np.ndarray,
    Z: np.ndarray,
    T: np.ndarray,
    *,
    title: str,
    temp_range: tuple[float, float],
    cmap,
    reference_front: tuple[np.ndarray, np.ndarray] | None = None,
    markers: tuple[np.ndarray, np.ndarray] | None = None,
):
    masked = np.ma.masked_invalid(T)
    mesh = ax.pcolormesh(
        R * 1000.0,
        Z * 1000.0,
        masked,
        cmap=cmap,
        vmin=temp_range[0],
        vmax=temp_range[1],
        shading="auto",
    )
    # The thaw front: this field's own 0 C isotherm.
    if np.any(np.isfinite(T)):
        ax.contour(
            R * 1000.0,
            Z * 1000.0,
            np.nan_to_num(T, nan=-999.0),
            levels=[0.0],
            colors="black",
            linewidths=2.0,
        )
    # The measured front repeated on every panel, so a shape mismatch is visible
    # in place rather than by eye across panels.
    if reference_front is not None and reference_front[0].size:
        ax.plot(
            reference_front[0] * 1000.0,
            reference_front[1] * 1000.0,
            "--",
            color="white",
            linewidth=1.6,
            label="measured front",
        )
        ax.legend(loc="lower right", fontsize=7, framealpha=0.6)
    if markers is not None and markers[0].size:
        ax.scatter(
            np.abs(markers[0]) * 1000.0,
            markers[1] * 1000.0,
            s=6,
            c="black",
            alpha=0.35,
            linewidths=0,
        )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Radius (mm)")
    return mesh


def _render_frame(
    fig,
    panels: list[dict],
    *,
    time_hours: float,
    temp_range: tuple[float, float],
    summary: str,
):
    import matplotlib.pyplot as plt

    fig.clear()
    axes = fig.subplots(1, len(panels), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad(color=(0.85, 0.85, 0.85, 1.0))

    mesh = None
    for ax, panel in zip(axes, panels):
        mesh = _draw_panel(
            ax,
            panel["R"],
            panel["Z"],
            panel["T"],
            title=panel["title"],
            temp_range=temp_range,
            cmap=cmap,
            reference_front=panel.get("reference_front"),
            markers=panel.get("markers"),
        )
    axes[0].set_ylabel("Depth (mm)")
    # Once only: the axes share y, so inverting per panel would cancel out on an
    # even number of panels and leave the depth axis upside-down.
    axes[0].invert_yaxis()
    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes, fraction=0.035, pad=0.02)
        cbar.set_label("Temperature (C)")
    fig.suptitle(f"t = {time_hours:.3f} h   |   {summary}", fontsize=10)
    return fig


def _write_animation(
    frame_paths: list[Path],
    gif_path: Path,
    *,
    fps: float,
    max_width: int,
) -> None:
    """Assemble the saved frames into a looping GIF.

    Built by re-reading the PNGs rather than buffering rendered frames in
    memory: a three-panel frame is several megabytes of raw RGB, so a long
    replay would otherwise hold hundreds of megabytes just to write one file.
    """
    try:
        from PIL import Image
    except ImportError:
        print("Animation skipped: Pillow is not installed (pip install pillow).")
        return

    duration_ms = max(int(round(1000.0 / max(float(fps), 0.1))), 20)
    frames = []
    try:
        for path in frame_paths:
            with Image.open(path) as handle:
                frame = handle.convert("RGB")
            if max_width > 0 and frame.width > max_width:
                scale = max_width / float(frame.width)
                frame = frame.resize(
                    (max_width, max(int(round(frame.height * scale)), 1)),
                    Image.LANCZOS,
                )
            frames.append(frame)

        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
    except Exception as exc:  # noqa: BLE001 - the frames are still on disk
        print(f"Animation export failed ({exc}); the PNG frames are still in {frame_paths[0].parent}")
        return

    size_mb = gif_path.stat().st_size / 1e6
    print(
        f"Wrote animation: {gif_path}  ({len(frames)} frames, {fps:g} fps, "
        f"{frames[0].width}x{frames[0].height}, {size_mb:.1f} MB)"
    )


# ---------------------------------------------------------------------------
# REPLAY
# ---------------------------------------------------------------------------
def _parse_material_override(raw: str | None) -> dict[str, float] | None:
    if not raw:
        return None
    override: dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Bad --material-override entry '{item}'; expected name=value.")
        name, value = item.split("=", 1)
        override[name.strip()] = float(value)
    return override or None


def _sensor_rows(
    path: Path,
    time_column: str | None,
    start_row: int,
    limit: int | None,
    stride: int,
    exclude_sensors: set[str] | None = None,
) -> list[dict]:
    """Parse the replay file into sensor snapshots (same layout as the queue).

    ``exclude_sensors`` drops named channels entirely. A single stuck
    thermocouple matters far more here than its share of the RMSE suggests: the
    solver seeds its initial field by interpolating one snapshot across the
    mesh, so one bad reading is smeared over the domain and can plant a thawed
    region before the first time step.
    """
    excluded = {name.strip() for name in (exclude_sensors or set()) if name.strip()}
    row_iter = _iter_rows(path)
    header = next(row_iter, [])
    time_idx, sensor_columns = _parse_header(header, time_column)

    snapshots: list[dict] = []
    for row_index, row in enumerate(row_iter, start=1):
        if row_index < start_row:
            continue
        if limit is not None and (row_index - start_row) >= limit:
            break
        if (row_index - start_row) % stride != 0:
            continue
        if time_idx >= len(row):
            continue
        raw_time = row[time_idx].strip()
        if not raw_time:
            continue
        try:
            time_hours = float(raw_time)
        except ValueError:
            continue

        sensors: list[dict] = []
        for column in sensor_columns:
            idx = column["index"]
            if idx >= len(row):
                continue
            if column["sensor_id"] in excluded:
                continue
            value = row[idx].strip()
            if value == "":
                continue
            try:
                temperature = float(value)
            except ValueError:
                continue
            sensors.append(
                {
                    "sensor_id": column["sensor_id"],
                    "x_m": column["x_m"],
                    "z_m": column["z_m"],
                    "temperature": temperature,
                }
            )
        if not sensors:
            continue
        snapshots.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "time_hours": time_hours,
                "sensors": sensors,
            }
        )
    return snapshots


def _build_servers(config: dict, *, run_baseline: bool, horizon_hours: float | None):
    """Instantiate the real services with their clients stubbed out."""
    rabbit_cfg = config.get("rabbitmq", {})
    influx_cfg: InfluxConfig = build_influx_config(config.get("influxdb", {}))
    bc_cfg = config.get("boundary_condition_builder", {})
    fem_cfg = dict(config.get("fem_forecast", {}))
    if horizon_hours is not None:
        fem_cfg["horizon_hours"] = float(horizon_hours)
    fem_cfg["run_baseline"] = bool(run_baseline)

    builder = create_boundary_condition_builder_server(rabbit_cfg, bc_cfg)
    forecast = create_fem_forecast_server(influx_cfg, rabbit_cfg, fem_cfg)
    forecast.mq_out = _Sink()
    forecast.influx = _NullInflux()

    pipe = None
    pipe_cfg = config.get("pipe_boundary", {})
    if pipe_cfg:
        pipe = create_pipe_boundary_server(influx_cfg, rabbit_cfg, pipe_cfg)
        pipe.mq_boundary_out = _Sink()
        # influx / mq_calibration stay None: the server already guards on them.
    return builder, pipe, forecast


def run_debug_replay(
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
    plot_every: int,
    grid_resolution: int,
    temp_range: tuple[float, float] | None,
    material_override: dict[str, float] | None,
    run_baseline: bool,
    use_pipe_boundary: bool,
    horizon_hours: float | None,
    surface_exclusion_m: float | None,
    exclude_sensors: set[str] | None,
    animate: bool = True,
    fps: float = 4.0,
    gif_max_width: int = 1600,
    clean: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = load_startup_config(config_path)
    if surface_exclusion_m is None:
        thaw_cfg = config.get("thaw_front_reconstruction", {})
        surface_exclusion_m = float(thaw_cfg.get("surface_exclusion_m", 0.0)) if isinstance(thaw_cfg, dict) else 0.0
    snapshots = _sensor_rows(path, time_column, start_row, limit, stride, exclude_sensors)
    if not snapshots:
        raise ValueError(f"No usable sensor rows parsed from {path}")
    if exclude_sensors:
        print(f"Excluded sensor channels: {sorted(exclude_sensors)}")
    print(f"Loaded {len(snapshots)} sensor snapshots from {path.name} "
          f"(t = {snapshots[0]['time_hours']:.3f} .. {snapshots[-1]['time_hours']:.3f} h)")

    fluid = None
    if fluid_path is not None and use_pipe_boundary:
        fluid = _load_fluid_temps(
            fluid_path,
            inlet_col=fluid_inlet_col,
            outlet_col=fluid_outlet_col,
            time_col=fluid_time_col,
        )
        print(f"Loaded {len(fluid[0])} fluid samples from {fluid_path.name}")

    builder, pipe, forecast = _build_servers(
        config, run_baseline=run_baseline, horizon_hours=horizon_hours
    )
    if not use_pipe_boundary:
        pipe = None
    print(
        "Pipe boundary stage: "
        + ("enabled" if pipe is not None and fluid is not None else "disabled (configured boundaries used as-is)")
    )

    if temp_range is None:
        all_temps = np.array(
            [s["temperature"] for snap in snapshots for s in snap["sensors"]], dtype=float
        )
        temp_range = (float(np.floor(np.nanmin(all_temps))), float(np.ceil(np.nanmax(all_temps))))
    print(f"Colour scale: {temp_range[0]:.1f} .. {temp_range[1]:.1f} C")
    print(f"Front metrics ignore depths shallower than {surface_exclusion_m * 1000:.0f} mm")

    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / "fem_debug_replay.gif"

    # Frames are named by step index, so a re-run with a different --stride or
    # --plot-every leaves the previous run's frames interleaved in the same
    # directory. The animation is built from this run's list only and is never
    # affected, but the directory misleads; say so, or clear it on request.
    stale = sorted(frames_dir.glob("frame_*.png"))
    if stale:
        if clean:
            for path in stale:
                path.unlink()
            print(f"Removed {len(stale)} frame(s) from a previous run in {frames_dir}")
        else:
            print(
                f"Note: {len(stale)} frame(s) from a previous run are already in {frames_dir}; "
                "the animation uses only this run's frames. Pass --clean to delete them."
            )
    print(f"Frames -> {frames_dir}")
    if animate:
        print(f"Animation -> {gif_path}")

    metrics_rows: list[dict] = []
    frame_paths: list[Path] = []
    grid: tuple[np.ndarray, np.ndarray] | None = None
    fig = plt.figure(figsize=(5.6 * (3 if run_baseline else 2), 4.4), dpi=140)

    for index, sensor_msg in enumerate(snapshots):
        time_hours = float(sensor_msg["time_hours"])

        # 1. Boundary condition builder (real service method).
        boundary_snapshot = builder._build_snapshot(sensor_msg)

        # 2. Pipe boundary enrichment: feed the fluid sample through the real
        #    calibration path, then let the server rewrite the pipe boundary.
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

        # 3. Material override, injected exactly as the inversion service does.
        if material_override:
            boundary_snapshot["material_override"] = dict(material_override)

        # 4. FEM forecast: the real message handler, clients stubbed.
        forecast._process_message(boundary_snapshot)
        if forecast._solver is None:
            raise RuntimeError("FEM solver was never built; check the forecast config.")

        if grid is None:
            grid = _grid_from_mesh(forecast._solver.node, grid_resolution)
        R, Z = grid

        # Measured field.
        obs_x = np.array([s["x_m"] for s in sensor_msg["sensors"]], dtype=float)
        obs_z = np.array([s["z_m"] for s in sensor_msg["sensors"]], dtype=float)
        obs_t = np.array([s["temperature"] for s in sensor_msg["sensors"]], dtype=float)
        T_obs = _interp_to_grid(obs_x, obs_z, obs_t, R, Z)

        # Forecast field, sampled onto the same grid by the same interpolation.
        node_xy, temps_c = _solver_field_c(forecast._solver)
        T_fem = _interp_to_grid(node_xy[:, 0], node_xy[:, 1], temps_c, R, Z)

        obs_rmax, obs_ravg, obs_radii, obs_depths = _front_from_grid(
            R, Z, T_obs, surface_exclusion_m=surface_exclusion_m
        )
        fem_rmax, fem_ravg, _, _ = _front_from_grid(
            R, Z, T_fem, surface_exclusion_m=surface_exclusion_m
        )
        solver_rmax, solver_ravg, _ = forecast._solver.compute_thaw_metrics()

        # Error at the sensor positions only (what the dashboard's % error sees).
        fem_at_sensors = _interp_to_grid(node_xy[:, 0], node_xy[:, 1], temps_c, np.abs(obs_x), obs_z)
        residual = fem_at_sensors - obs_t
        finite = np.isfinite(residual)
        rmse_c = float(np.sqrt(np.mean(np.square(residual[finite])))) if np.any(finite) else float("nan")
        max_abs_c = float(np.max(np.abs(residual[finite]))) if np.any(finite) else float("nan")

        metrics_rows.append(
            {
                "time_hours": round(time_hours, 6),
                "rmse_c": rmse_c,
                "max_abs_error_c": max_abs_c,
                "r_max_measured_m": obs_rmax,
                "r_avg_measured_m": obs_ravg,
                "r_max_fem_m": fem_rmax,
                "r_avg_fem_m": fem_ravg,
                "r_max_fem_mesh_m": solver_rmax,
                "r_avg_fem_mesh_m": solver_ravg,
            }
        )

        is_last = index == len(snapshots) - 1
        if index % plot_every != 0 and not is_last:
            continue

        summary = (
            f"RMSE {rmse_c:.2f} C   |   r_max measured "
            f"{'n/a' if obs_rmax is None else f'{obs_rmax * 1000:.0f} mm'} vs FEM "
            f"{'n/a' if fem_rmax is None else f'{fem_rmax * 1000:.0f} mm'}"
        )
        reference_front = (obs_radii, obs_depths) if obs_radii.size else None
        panels = [
            {
                "R": R,
                "Z": Z,
                "T": T_obs,
                "title": "Measured field (thermocouples) + thaw front",
                "markers": (obs_x, obs_z),
            },
            {
                "R": R,
                "Z": Z,
                "T": T_fem,
                "title": "FEM forecast field + thaw front",
                "reference_front": reference_front,
                "markers": (obs_x, obs_z),
            },
        ]
        if run_baseline and forecast._baseline_solver is not None:
            base_xy, base_c = _solver_field_c(forecast._baseline_solver)
            panels.append(
                {
                    "R": R,
                    "Z": Z,
                    "T": _interp_to_grid(base_xy[:, 0], base_xy[:, 1], base_c, R, Z),
                    "title": "FEM without inversion (fixed parameters)",
                    "reference_front": reference_front,
                }
            )

        _render_frame(fig, panels, time_hours=time_hours, temp_range=temp_range, summary=summary)
        frame_path = frames_dir / f"frame_{index:05d}_t{time_hours:07.3f}h.png"
        # bbox_inches is deliberately not "tight" here: a tight box is sized per
        # frame, so panels jitter between frames and the animation is unreadable.
        fig.savefig(frame_path)
        frame_paths.append(frame_path)
        print(f"  [{index + 1}/{len(snapshots)}] t={time_hours:7.3f} h  {summary}  -> {frame_path.name}")

    plt.close(fig)

    metrics_path = out_dir / "front_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)
    print(f"\nWrote per-step front metrics to {metrics_path}")

    if animate and frame_paths:
        _write_animation(frame_paths, gif_path, fps=fps, max_width=gif_max_width)

    _print_summary(metrics_rows)


def _print_summary(rows: Iterable[dict]) -> None:
    rows = list(rows)
    paired = [
        row for row in rows
        if row["r_max_measured_m"] is not None and row["r_max_fem_m"] is not None
    ]
    print("\n--- summary ---")
    rmse = np.array([row["rmse_c"] for row in rows], dtype=float)
    print(f"steps                : {len(rows)}")
    print(f"RMSE at sensors (C)  : mean {np.nanmean(rmse):.3f}  max {np.nanmax(rmse):.3f}")
    if not paired:
        print("thaw front           : never present in both fields (nothing above 0 C)")
        return
    obs = np.array([row["r_max_measured_m"] for row in paired], dtype=float)
    fem = np.array([row["r_max_fem_m"] for row in paired], dtype=float)
    print(f"steps with a front   : {len(paired)}")
    print(
        f"r_max measured (mm)  : mean {obs.mean() * 1000:.1f}  final {obs[-1] * 1000:.1f}\n"
        f"r_max FEM (mm)       : mean {fem.mean() * 1000:.1f}  final {fem[-1] * 1000:.1f}\n"
        f"front radius gap (mm): mean {np.mean(fem - obs) * 1000:+.1f}  "
        f"max |gap| {np.max(np.abs(fem - obs)) * 1000:.1f}"
    )
    print(
        "A small sensor RMSE next to a large front-radius gap is the signature of a model that "
        "matches where it is measured and drifts where it is not."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay recorded sensor data and compare the measured and FEM temperature fields.",
    )
    parser.add_argument(
        "--path",
        default=DEFAULT_SENSOR_PATH,
        help=f"Recorded sensor CSV/XLSX with H*D* columns (default: {DEFAULT_SENSOR_PATH}).",
    )
    parser.add_argument("--config", default="startup.conf", help="Startup config to take the FEM/BC setup from.")
    parser.add_argument("--out", default="artifacts/fem_debug", help="Output directory.")
    parser.add_argument("--time-column", default=None, help="Override the time column header.")
    parser.add_argument("--start-row", type=int, default=1, help="First data row (1-based).")
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows to read.")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth row (thins the replay).")
    parser.add_argument("--plot-every", type=int, default=25, help="Render a frame every Nth replayed step.")
    parser.add_argument("--grid-resolution", type=int, default=160, help="Plot grid resolution.")
    parser.add_argument("--tmin", type=float, default=None, help="Colour scale minimum (C).")
    parser.add_argument("--tmax", type=float, default=None, help="Colour scale maximum (C).")
    parser.add_argument("--horizon-hours", type=float, default=None, help="Override fem_forecast.horizon_hours.")
    parser.add_argument(
        "--surface-exclusion-m",
        type=float,
        default=None,
        help="Ignore depths shallower than this when extracting the front "
        "(default: thaw_front_reconstruction.surface_exclusion_m; 0 disables).",
    )
    parser.add_argument(
        "--material-override",
        default=None,
        help="Inverted parameters to inject, e.g. 'cap_s=1200,k_parameter=8,lambda_s=5'.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also run the uncalibrated twin and add it as a third panel.",
    )
    parser.add_argument(
        "--no-pipe-boundary",
        action="store_true",
        help="Skip the pipe calibration stage and use the configured boundaries as-is.",
    )
    parser.add_argument(
        "--exclude-sensors",
        default=None,
        help="Comma-separated channels to drop, e.g. 'H290D165'. Useful for stuck "
        "thermocouples, which the initial-condition interpolation smears across the mesh.",
    )
    animation_group = parser.add_argument_group("animation (written by default)")
    animation_group.add_argument(
        "--no-animation",
        action="store_true",
        help="Only write the PNG frames, skip the GIF.",
    )
    animation_group.add_argument("--fps", type=float, default=4.0, help="Animation frame rate (default 4).")
    animation_group.add_argument(
        "--clean",
        action="store_true",
        help="Delete frames left in the output directory by a previous run before starting.",
    )
    animation_group.add_argument(
        "--gif-max-width",
        type=int,
        default=1600,
        help="Downscale animation frames wider than this, in pixels (0 disables).",
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
    temp_range = None
    if args.tmin is not None and args.tmax is not None:
        temp_range = (float(args.tmin), float(args.tmax))

    run_debug_replay(
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
        plot_every=max(int(args.plot_every), 1),
        grid_resolution=max(int(args.grid_resolution), 20),
        temp_range=temp_range,
        material_override=_parse_material_override(args.material_override),
        run_baseline=bool(args.baseline),
        use_pipe_boundary=not args.no_pipe_boundary,
        horizon_hours=args.horizon_hours,
        surface_exclusion_m=args.surface_exclusion_m,
        exclude_sensors=set(args.exclude_sensors.split(",")) if args.exclude_sensors else None,
        animate=not args.no_animation,
        fps=float(args.fps),
        gif_max_width=max(int(args.gif_max_width), 0),
        clean=bool(args.clean),
    )


if __name__ == "__main__":
    main()

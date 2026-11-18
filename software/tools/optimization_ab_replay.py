"""Side-by-side replay of the current and optimised pipelines.

The optimisation work on the FEM solver and the inversion forward model is
supposed to change how long the twin takes, not what it predicts. This tool is
the evidence for that claim: it replays one recorded sensor file through two
complete pipelines -- the shipping one, and one running the pre-optimisation
code from ``software/tools/golden_reference.py`` -- and draws them next to each
other with a difference panel, so a divergence shows up as colour rather than as
a number in a log nobody reads.

Both pipelines run the real services (boundary condition builder, pipe boundary
calibration, FEM forecast, and the inversion driven inline at a fixed interval)
with only RabbitMQ and InfluxDB stubbed out, exactly as ``fem_debug_replay``
does. They are stepped in lockstep on the same snapshots, so the difference
panel compares like with like at every frame.

Outputs, all under ``--out`` (default ``artifacts/optimization_ab``):

* ``fem_ab_fields.gif``      -- measured / current / optimised / difference
* ``fem_ab_thaw_front.gif``  -- front envelopes and the r_max traces
* ``frames_fields/``, ``frames_front/`` -- the PNGs the GIFs are built from
* ``frames_current/``        -- the FEM forecast - CURRENT panel on its own
* ``ab_metrics.csv``         -- per-step field difference, front radii, timings

Typical use:

    python -m software.tools.optimization_ab_replay --limit 120 --stride 4

The default window is deliberately short: the current pipeline is the slow one,
and it has to run every frame for the comparison to mean anything.
"""

from __future__ import annotations

import argparse
import csv
import time
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from software.digital_twin.simulator.fem_inversion.fem_inversion_server import FEMInversionServer
from software.startup.service_runners import (
    build_influx_config,
    create_boundary_condition_builder_server,
    create_fem_forecast_server,
    create_pipe_boundary_server,
)
from software.startup.utils.config import load_startup_config
from software.tools.golden_reference import reference_assembly, reference_forward_model
from software.tools.fem_debug_replay import (
    DEFAULT_FLUID_PATH,
    DEFAULT_SENSOR_PATH,
    _front_from_grid,
    _grid_from_mesh,
    _interp_to_grid,
    _NullInflux,
    _Sink,
    _sensor_rows,
    _solver_field_c,
    _write_animation,
)
from software.tools.replay_csv import _load_fluid_temps


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------
class _Pipeline:
    """One complete twin, either the shipping code or the pre-optimisation code."""

    def __init__(
        self,
        label: str,
        config: dict,
        config_path: Path,
        *,
        use_pipe_boundary: bool,
        run_inversion: bool,
        use_reference: bool,
    ) -> None:
        self.label = label
        self.config = config
        self.config_path = config_path
        # The reference pipeline runs the original assembly and the original
        # forward model, patched in for the duration of each call. Patching is
        # scoped rather than global because both pipelines live in one process
        # and are stepped alternately on the same snapshot.
        self.use_reference = use_reference

        rabbit_cfg = config.get("rabbitmq", {})
        influx_cfg = build_influx_config(config.get("influxdb", {}))
        fem_cfg = dict(config.get("fem_forecast", {}))
        # The uncalibrated baseline twin is a third solve per message and plays
        # no part in this comparison, so it is off in both pipelines.
        fem_cfg["run_baseline"] = False

        self.builder = create_boundary_condition_builder_server(rabbit_cfg, config.get("boundary_condition_builder", {}))
        self.forecast = create_fem_forecast_server(influx_cfg, rabbit_cfg, fem_cfg)
        self.forecast.mq_out = _Sink()
        self.forecast.influx = _NullInflux()

        self.pipe = None
        pipe_cfg = config.get("pipe_boundary", {})
        if use_pipe_boundary and pipe_cfg:
            self.pipe = create_pipe_boundary_server(influx_cfg, rabbit_cfg, pipe_cfg)
            self.pipe.mq_boundary_out = _Sink()

        self.inversion = _make_inversion_server(config, config_path) if run_inversion else None
        self.material_override: dict[str, float] | None = None

        self.forecast_seconds = 0.0
        self.inversion_seconds = 0.0
        self.inversion_count = 0
        self.inversion_evals = 0

    # ------------------------------------------------------------------
    def _implementation(self) -> ExitStack:
        """Context selecting which implementation this pipeline runs under."""
        stack = ExitStack()
        if self.use_reference:
            stack.enter_context(reference_assembly())
            if self.inversion is not None:
                stack.enter_context(reference_forward_model(self.inversion))
        return stack

    def feed_inversion(self, sensor_msg: dict) -> None:
        if self.inversion is not None:
            self.inversion._upsert_snapshot(deepcopy(sensor_msg))

    def run_inversion(self) -> dict[str, float] | None:
        """Fit the material parameters inline, the way the service would."""
        if self.inversion is None:
            return None
        window = self.inversion._build_observation_window()
        if window is None:
            return None
        times_hours, positions, obs_series, ic_sensors = window

        start = time.perf_counter()
        with self._implementation():
            best_params, info = self.inversion._run_inversion(times_hours, positions, obs_series, ic_sensors)
        self.inversion_seconds += time.perf_counter() - start
        self.inversion_count += 1
        self.inversion_evals += int(info.get("eval_count", 0))

        self.material_override = {name: float(best_params[name]) for name in self.inversion.invert_params}
        return self.material_override

    def step(self, sensor_msg: dict, fluid: tuple | None) -> dict:
        """Push one snapshot through builder -> pipe -> forecast."""
        boundary_snapshot = self.builder._build_snapshot(sensor_msg)

        if self.pipe is not None and fluid is not None:
            t_arr, tin_arr, tout_arr = fluid
            time_hours = float(sensor_msg["time_hours"])
            self.pipe._on_fluid(
                {
                    "time_hours": time_hours,
                    "timestamp": sensor_msg["timestamp"],
                    "t_inlet_c": float(np.interp(time_hours, t_arr, tin_arr)),
                    "t_outlet_c": float(np.interp(time_hours, t_arr, tout_arr)),
                    "site_id": "default",
                }
            )
            self.pipe._on_boundary(boundary_snapshot)
            enriched = self.pipe.mq_boundary_out.last
            if isinstance(enriched, dict):
                boundary_snapshot = enriched

        if self.material_override:
            boundary_snapshot["material_override"] = dict(self.material_override)

        start = time.perf_counter()
        with self._implementation():
            self.forecast._process_message(boundary_snapshot)
        self.forecast_seconds += time.perf_counter() - start

        if self.forecast._solver is None:
            raise RuntimeError(f"{self.label}: FEM solver was never built; check the forecast config.")
        return boundary_snapshot


def _make_inversion_server(config: dict, config_path: Path) -> FEMInversionServer:
    """Inversion service instance with no message bus attached."""
    inv_cfg = config.get("fem_inversion", {})
    fem_cfg = config.get("fem_forecast", {})
    return FEMInversionServer(
        site_id=str(inv_cfg.get("site_id", "default")),
        invert_params=list(inv_cfg.get("invert_params", [])) or None,
        base_params=fem_cfg.get("material", {}),
        relative_bounds=inv_cfg.get("relative_bounds", {}),
        loss=str(inv_cfg.get("loss", "mse")),
        window_size=int(inv_cfg.get("window_size", 5)),
        min_snapshots=int(inv_cfg.get("min_snapshots", 5)),
        min_sensors=int(inv_cfg.get("min_sensors", 3)),
        relative_time_window=bool(inv_cfg.get("relative_time_window", True)),
        min_horizon_hours=float(inv_cfg.get("min_horizon_hours", 1e-3)),
        maxiter=int(inv_cfg.get("maxiter", 40)),
        maxfun=int(inv_cfg.get("maxfun", 80)),
        seed=inv_cfg.get("seed", 42),
        config_path=str(config_path),
    )


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------
def _draw_field(ax, R, Z, T, *, title, vmin, vmax, cmap, front=None, markers=None):
    mesh = ax.pcolormesh(R * 1000.0, Z * 1000.0, np.ma.masked_invalid(T),
                         cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    if np.any(np.isfinite(T)):
        ax.contour(R * 1000.0, Z * 1000.0, np.nan_to_num(T, nan=-999.0),
                   levels=[0.0], colors="black", linewidths=1.8)
    if front is not None and front[0].size:
        # Magenta is the one hue jet never produces, so the measured front stays
        # legible over both the warm core and the cyan/blue far field.
        ax.plot(front[0] * 1000.0, front[1] * 1000.0, "--", color="magenta", linewidth=1.8,
                label="measured front")
    if markers is not None and markers[0].size:
        ax.scatter(np.abs(markers[0]) * 1000.0, markers[1] * 1000.0, s=4, c="black", alpha=0.3, linewidths=0)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Radius (mm)")
    return mesh


def _render_field_frame(fig, panels, diff, *, time_hours, vmin, vmax, diff_scale, summary):
    import matplotlib.pyplot as plt

    fig.clear()
    axes = fig.subplots(1, len(panels) + 1, sharey=True)
    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad(color=(0.85, 0.85, 0.85, 1.0))

    mesh = None
    for ax, panel in zip(axes[:-1], panels):
        mesh = _draw_field(ax, panel["R"], panel["Z"], panel["T"], title=panel["title"],
                           vmin=vmin, vmax=vmax, cmap=cmap,
                           front=panel.get("front"), markers=panel.get("markers"))

    # RdBu_r puts pure white at zero. coolwarm centres on a light grey that is
    # indistinguishable from the "outside the sensor hull" mask, which on a
    # panel whose whole job is to show nothing is precisely the wrong ambiguity.
    diff_cmap = plt.get_cmap("RdBu_r").copy()
    diff_cmap.set_bad(color=(0.55, 0.55, 0.55, 1.0))
    ax_diff = axes[-1]
    diff_mesh = ax_diff.pcolormesh(diff["R"] * 1000.0, diff["Z"] * 1000.0, np.ma.masked_invalid(diff["T"]),
                                   cmap=diff_cmap, vmin=-diff_scale, vmax=diff_scale, shading="auto")
    ax_diff.set_title(diff["title"], fontsize=9)
    ax_diff.set_xlabel("Radius (mm)")
    # State the scale in the panel: a blank panel only means something once the
    # reader knows how sensitive it is.
    ax_diff.text(
        0.5, 0.5,
        "blank = identical"
        f"\nscale +-{diff_scale * 1000:g} mK"
        "\n(thermocouple resolution ~100 mK)",
        transform=ax_diff.transAxes, ha="center", va="center",
        fontsize=8, color="0.35",
    )

    axes[0].set_ylabel("Depth (mm)")
    axes[0].invert_yaxis()

    cbar = fig.colorbar(mesh, ax=axes[:-1].tolist(), fraction=0.030, pad=0.02)
    cbar.set_label("Temperature (C)")
    dbar = fig.colorbar(diff_mesh, ax=ax_diff, fraction=0.055, pad=0.04)
    dbar.set_label("optimised - current (K)")

    fig.suptitle(f"t = {time_hours:.3f} h   |   {summary}", fontsize=10)
    return fig


def _render_single_panel(fig, panel, *, time_hours, vmin, vmax):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig.clear()
    ax = fig.subplots()
    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad(color=(0.85, 0.85, 0.85, 1.0))
    mesh = _draw_field(ax, panel["R"], panel["Z"], panel["T"], title=f"{panel['title']}  (t = {time_hours:.3f} h)",
                       vmin=vmin, vmax=vmax, cmap=cmap,
                       front=panel.get("front"), markers=panel.get("markers"))
    ax.set_ylabel("Depth (mm)")
    ax.invert_yaxis()
    fig.colorbar(mesh, ax=ax, pad=0.03).set_label("Temperature (C)")
    handles = [Line2D([], [], color="black", linewidth=1.8, label="FEM front (0 C)")]
    if panel.get("front") is not None:
        handles.append(Line2D([], [], color="magenta", linestyle="--", linewidth=1.8, label="measured front"))
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.85)
    fig.tight_layout()
    return fig


def _render_front_frame(fig, *, time_hours, fronts, history, r_limit_m, summary):
    fig.clear()
    ax_env, ax_trace = fig.subplots(1, 2)

    styles = {
        "measured": dict(color="black", linestyle="--", linewidth=1.6, zorder=3),
        "current": dict(color="tab:blue", linestyle="-", linewidth=2.4, alpha=0.9, zorder=1),
        "optimised": dict(color="tab:orange", linestyle=":", linewidth=2.0, zorder=2),
    }
    for name, (radii, depths) in fronts.items():
        if radii.size:
            ax_env.plot(radii * 1000.0, depths * 1000.0, label=name, **styles[name])
    ax_env.set_xlabel("Front radius (mm)")
    ax_env.set_ylabel("Depth (mm)")
    ax_env.set_title("Thaw-front envelope", fontsize=9)
    ax_env.invert_yaxis()
    ax_env.grid(alpha=0.25)
    # Before the front appears there is nothing plotted, and an empty legend warns.
    if ax_env.get_legend_handles_labels()[0]:
        ax_env.legend(loc="lower right", fontsize=8)
    ax_env.set_xlim(0.0, r_limit_m * 1000.0)

    for name in ("measured", "current", "optimised"):
        times = history["time_hours"]
        values = history[name]
        if times:
            ax_trace.plot(times, [v * 1000.0 if v is not None else np.nan for v in values],
                          label=name, **styles[name])
    ax_trace.set_xlabel("Time (h)")
    ax_trace.set_ylabel("r_max (mm)")
    ax_trace.set_title("Maximum thaw radius", fontsize=9)
    ax_trace.grid(alpha=0.25)
    if ax_trace.get_legend_handles_labels()[0]:
        ax_trace.legend(loc="upper left", fontsize=8)

    fig.suptitle(f"t = {time_hours:.3f} h   |   {summary}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# ---------------------------------------------------------------------------
# REPLAY
# ---------------------------------------------------------------------------
def run_ab_replay(
    *,
    path: Path,
    fluid_path: Path | None,
    config_path: str | Path,
    out_dir: Path,
    start_row: int,
    limit: int | None,
    stride: int,
    plot_every: int,
    grid_resolution: int,
    inversion_every: int,
    run_inversion: bool,
    use_pipe_boundary: bool,
    exclude_sensors: set[str] | None,
    diff_scale: float,
    fps: float,
    gif_max_width: int,
    animate: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_startup_config(config_path)

    snapshots = _sensor_rows(path, None, start_row, limit, stride, exclude_sensors)
    if not snapshots:
        raise ValueError(f"No usable sensor rows parsed from {path}")
    print(f"Loaded {len(snapshots)} sensor snapshots from {path.name} "
          f"(t = {snapshots[0]['time_hours']:.3f} .. {snapshots[-1]['time_hours']:.3f} h)")

    fluid = None
    if fluid_path is not None and use_pipe_boundary:
        fluid = _load_fluid_temps(fluid_path, inlet_col="Fluid Temperature_Inlet",
                                  outlet_col="Fluid Temperatue_Outlet", time_col="Time (h)")
        print(f"Loaded {len(fluid[0])} fluid samples from {fluid_path.name}")

    pipelines = {
        "current": _Pipeline("current", base_config, Path(config_path),
                             use_pipe_boundary=use_pipe_boundary, run_inversion=run_inversion,
                             use_reference=True),
        "optimised": _Pipeline("optimised", base_config, Path(config_path),
                               use_pipe_boundary=use_pipe_boundary, run_inversion=run_inversion,
                               use_reference=False),
    }
    print("Pipelines: current (pre-optimisation code from golden_reference) vs "
          "optimised (shipping code)")
    print(f"Inversion: {'inline every ' + str(inversion_every) + ' steps' if run_inversion else 'disabled'}")

    thaw_cfg = base_config.get("thaw_front_reconstruction", {})
    surface_exclusion_m = float(thaw_cfg.get("surface_exclusion_m", 0.0)) if isinstance(thaw_cfg, dict) else 0.0
    r_limit_m = float(base_config.get("safety_monitor", {}).get("limit_radius_m", 0.2))

    all_temps = np.array([s["temperature"] for snap in snapshots for s in snap["sensors"]], dtype=float)
    vmin, vmax = float(np.floor(np.nanmin(all_temps))), float(np.ceil(np.nanmax(all_temps)))
    print(f"Colour scale: {vmin:.1f} .. {vmax:.1f} C     difference scale: +-{diff_scale:g} K")

    frames_fields = out_dir / "frames_fields"
    frames_front = out_dir / "frames_front"
    frames_current = out_dir / "frames_current"
    for directory in (frames_fields, frames_front, frames_current):
        directory.mkdir(parents=True, exist_ok=True)
        for stale in directory.glob("frame_*.png"):
            stale.unlink()

    metrics_rows: list[dict] = []
    field_frames: list[Path] = []
    front_frames: list[Path] = []
    history: dict[str, list] = {"time_hours": [], "measured": [], "current": [], "optimised": []}
    grid: tuple[np.ndarray, np.ndarray] | None = None
    worst_field_diff = 0.0
    worst_front_diff = 0.0

    fig_fields = plt.figure(figsize=(19.0, 4.4), dpi=110)
    fig_front = plt.figure(figsize=(11.0, 4.6), dpi=120)
    fig_single = plt.figure(figsize=(5.6, 5.2), dpi=150)

    wall_start = time.perf_counter()
    for index, sensor_msg in enumerate(snapshots):
        time_hours = float(sensor_msg["time_hours"])

        for pipeline in pipelines.values():
            pipeline.feed_inversion(sensor_msg)
        if run_inversion and index > 0 and index % inversion_every == 0:
            for name, pipeline in pipelines.items():
                override = pipeline.run_inversion()
                if override is not None:
                    pretty = ", ".join(f"{k}={v:.4g}" for k, v in override.items())
                    print(f"  [{index + 1}/{len(snapshots)}] inversion ({name}): {pretty}")

        for pipeline in pipelines.values():
            pipeline.step(sensor_msg, fluid)

        if grid is None:
            grid = _grid_from_mesh(pipelines["current"].forecast._solver.node, grid_resolution)
        R, Z = grid

        obs_x = np.array([s["x_m"] for s in sensor_msg["sensors"]], dtype=float)
        obs_z = np.array([s["z_m"] for s in sensor_msg["sensors"]], dtype=float)
        obs_t = np.array([s["temperature"] for s in sensor_msg["sensors"]], dtype=float)
        T_obs = _interp_to_grid(obs_x, obs_z, obs_t, R, Z)

        fields: dict[str, np.ndarray] = {}
        nodal: dict[str, np.ndarray] = {}
        for name, pipeline in pipelines.items():
            node_xy, temps_c = _solver_field_c(pipeline.forecast._solver)
            nodal[name] = temps_c
            fields[name] = _interp_to_grid(node_xy[:, 0], node_xy[:, 1], temps_c, R, Z)

        # The nodal comparison is the honest one: the gridded fields go through
        # an interpolation that could mask or invent a difference.
        nodal_diff = float(np.max(np.abs(nodal["optimised"] - nodal["current"])))
        worst_field_diff = max(worst_field_diff, nodal_diff)
        grid_diff = fields["optimised"] - fields["current"]

        obs_rmax, _, obs_radii, obs_depths = _front_from_grid(R, Z, T_obs, surface_exclusion_m=surface_exclusion_m)
        fronts: dict[str, tuple[np.ndarray, np.ndarray]] = {"measured": (obs_radii, obs_depths)}
        rmax: dict[str, float | None] = {"measured": obs_rmax}
        for name in ("current", "optimised"):
            value, _, radii, depths = _front_from_grid(R, Z, fields[name], surface_exclusion_m=surface_exclusion_m)
            fronts[name] = (radii, depths)
            rmax[name] = value
        if rmax["current"] is not None and rmax["optimised"] is not None:
            worst_front_diff = max(worst_front_diff, abs(rmax["optimised"] - rmax["current"]))

        history["time_hours"].append(time_hours)
        for name in ("measured", "current", "optimised"):
            history[name].append(rmax[name])

        metrics_rows.append(
            {
                "time_hours": round(time_hours, 6),
                "max_abs_nodal_diff_k": nodal_diff,
                "r_max_measured_m": obs_rmax,
                "r_max_current_m": rmax["current"],
                "r_max_optimised_m": rmax["optimised"],
                "forecast_seconds_current": pipelines["current"].forecast_seconds,
                "forecast_seconds_optimised": pipelines["optimised"].forecast_seconds,
                "inversion_seconds_current": pipelines["current"].inversion_seconds,
                "inversion_seconds_optimised": pipelines["optimised"].inversion_seconds,
            }
        )

        is_last = index == len(snapshots) - 1
        if index % plot_every != 0 and not is_last:
            continue

        rmax_current = rmax["current"]
        rmax_optimised = rmax["optimised"]
        summary = (
            f"max |optimised - current| = {nodal_diff:.2e} K   |   r_max: "
            f"measured {'n/a' if obs_rmax is None else f'{obs_rmax * 1000:.0f} mm'}, "
            f"current {'n/a' if rmax_current is None else f'{rmax_current * 1000:.1f} mm'}, "
            f"optimised {'n/a' if rmax_optimised is None else f'{rmax_optimised * 1000:.1f} mm'}"
        )

        reference_front = (obs_radii, obs_depths) if obs_radii.size else None
        panels = [
            {"R": R, "Z": Z, "T": T_obs, "title": "Measured (thermocouples)", "markers": (obs_x, obs_z)},
            {"R": R, "Z": Z, "T": fields["current"], "title": "FEM forecast - CURRENT",
             "front": reference_front, "markers": (obs_x, obs_z)},
            {"R": R, "Z": Z, "T": fields["optimised"], "title": "FEM forecast - OPTIMISED",
             "front": reference_front, "markers": (obs_x, obs_z)},
        ]
        diff_panel = {"R": R, "Z": Z, "T": grid_diff,
                      "title": f"Difference (max {nodal_diff:.1e} K)"}

        _render_field_frame(fig_fields, panels, diff_panel, time_hours=time_hours,
                            vmin=vmin, vmax=vmax, diff_scale=diff_scale, summary=summary)
        field_path = frames_fields / f"frame_{index:05d}_t{time_hours:07.3f}h.png"
        fig_fields.savefig(field_path)
        field_frames.append(field_path)

        _render_single_panel(fig_single, panels[1], time_hours=time_hours, vmin=vmin, vmax=vmax)
        fig_single.savefig(frames_current / f"frame_{index:05d}_t{time_hours:07.3f}h.png")

        _render_front_frame(fig_front, time_hours=time_hours, fronts=fronts,
                            history=history, r_limit_m=r_limit_m, summary=summary)
        front_path = frames_front / f"frame_{index:05d}_t{time_hours:07.3f}h.png"
        fig_front.savefig(front_path)
        front_frames.append(front_path)

        print(f"  [{index + 1}/{len(snapshots)}] t={time_hours:7.3f} h  {summary}")

    plt.close(fig_fields)
    plt.close(fig_front)
    plt.close(fig_single)
    wall_total = time.perf_counter() - wall_start

    metrics_path = out_dir / "ab_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)
    print(f"\nWrote per-step metrics to {metrics_path}")

    if animate:
        _write_animation(field_frames, out_dir / "fem_ab_fields.gif", fps=fps, max_width=gif_max_width)
        _write_animation(front_frames, out_dir / "fem_ab_thaw_front.gif", fps=fps, max_width=gif_max_width)

    _print_summary(pipelines, worst_field_diff, worst_front_diff, len(snapshots), wall_total)


def _print_summary(pipelines: dict, worst_field_diff: float, worst_front_diff: float,
                   n_steps: int, wall_total: float) -> None:
    current, optimised = pipelines["current"], pipelines["optimised"]

    print("\n--- agreement ---")
    print(f"worst nodal temperature difference : {worst_field_diff:.3e} K")
    print(f"worst r_max difference             : {worst_front_diff * 1000:.3e} mm")
    verdict = "IDENTICAL (round-off only)" if worst_field_diff < 1e-9 else "DIVERGED"
    print(f"verdict                            : {verdict}")

    print("\n--- speed ---")
    print(f"{'stage':<22} {'current':>12} {'optimised':>12} {'speedup':>9}")
    print("-" * 58)
    f_cur, f_opt = current.forecast_seconds, optimised.forecast_seconds
    print(f"{'FEM forecast':<22} {f_cur:11.1f}s {f_opt:11.1f}s {f_cur / max(f_opt, 1e-9):8.1f}x")
    if current.inversion_count:
        i_cur, i_opt = current.inversion_seconds, optimised.inversion_seconds
        print(f"{'FEM inversion':<22} {i_cur:11.1f}s {i_opt:11.1f}s {i_cur / max(i_opt, 1e-9):8.1f}x")
        print(f"{'  inversions run':<22} {current.inversion_count:11d}  {optimised.inversion_count:11d}")
        print(f"{'  objective evals':<22} {current.inversion_evals:11d}  {optimised.inversion_evals:11d}")
        t_cur, t_opt = f_cur + i_cur, f_opt + i_opt
        print("-" * 58)
        print(f"{'total compute':<22} {t_cur:11.1f}s {t_opt:11.1f}s {t_cur / max(t_opt, 1e-9):8.1f}x")
        print(f"\nper sensor snapshot: current {t_cur / n_steps:.2f} s, optimised {t_opt / n_steps:.2f} s "
              f"(data arrives every 30 s)")
    print(f"\nwall clock for this replay (both pipelines + plotting): {wall_total / 60.0:.1f} min")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", default=DEFAULT_SENSOR_PATH, help="Recorded sensor CSV/XLSX.")
    parser.add_argument("--fluid-path", default=DEFAULT_FLUID_PATH, help="Fluid inlet/outlet XLSX ('' to skip).")
    parser.add_argument("--config", default="startup.conf")
    parser.add_argument("--out", default="artifacts/optimization_ab")
    parser.add_argument("--start-row", type=int, default=1)
    parser.add_argument("--limit", type=int, default=120, help="Rows to read before striding.")
    parser.add_argument("--stride", type=int, default=4, help="Use every Nth row.")
    parser.add_argument("--plot-every", type=int, default=1, help="Render a frame every Nth replayed step.")
    parser.add_argument("--grid-resolution", type=int, default=160)
    parser.add_argument("--inversion-every", type=int, default=10, help="Run the inversion every N steps.")
    parser.add_argument("--no-inversion", action="store_true", help="Compare the forecast only.")
    parser.add_argument("--no-pipe-boundary", action="store_true")
    parser.add_argument("--exclude-sensors", default=None)
    parser.add_argument("--diff-scale", type=float, default=1e-3,
                        help="Full scale of the difference panel in K (default 1e-3, i.e. 1 mK).")
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--gif-max-width", type=int, default=1600)
    parser.add_argument("--no-animation", action="store_true")
    args = parser.parse_args()

    run_ab_replay(
        path=Path(args.path),
        fluid_path=Path(args.fluid_path) if args.fluid_path else None,
        config_path=args.config,
        out_dir=Path(args.out),
        start_row=args.start_row,
        limit=args.limit,
        stride=max(int(args.stride), 1),
        plot_every=max(int(args.plot_every), 1),
        grid_resolution=max(int(args.grid_resolution), 20),
        inversion_every=max(int(args.inversion_every), 1),
        run_inversion=not args.no_inversion,
        use_pipe_boundary=not args.no_pipe_boundary,
        exclude_sensors=set(args.exclude_sensors.split(",")) if args.exclude_sensors else None,
        diff_scale=float(args.diff_scale),
        fps=float(args.fps),
        gif_max_width=max(int(args.gif_max_width), 0),
        animate=not args.no_animation,
    )


if __name__ == "__main__":
    main()

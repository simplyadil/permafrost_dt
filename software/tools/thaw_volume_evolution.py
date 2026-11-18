"""Thawed volume evolution: observed next to simulated.

Offline analysis tool, not a service. It replays a recorded sensor file through
the *same* code the live pipeline runs -- boundary condition builder, pipe
boundary calibration, FEM forecast server -- with RabbitMQ and InfluxDB stubbed
out, and tracks the thawed volume of both fields at every step. The result is
two side-by-side panels on a shared vertical scale:

    left   observed  V_thaw(t)  from the interpolated thermocouple field
    right  simulated V_thaw(t)  from the FEM field, with the observed curve
                                repeated as a dashed reference

Why a shared scale and a repeated reference: the interesting failure is not that
one curve is noisier, it is that the twin melts too much or too fast. Two panels
with independent y-axes hide exactly that, and reading a magnitude gap by eye
across panels is unreliable, so the observed curve is drawn on both.

Both volumes come from one procedure applied to one grid: per depth level take
the outermost thawed radius R(z) (0 where that depth is entirely frozen) and
revolve it, V = int pi*R(z)^2 dz. This is the discrete form of the frustum
integration ``thaw_front_reconstruction`` publishes, and it is the same
axisymmetry assumption -- the fields are only comparable if extracted
identically, so neither side gets a smarter estimator than the other.

Depths shallower than ``--surface-exclusion-m`` are dropped from the integral
(default: ``thaw_front_reconstruction.surface_exclusion_m``). The near-surface
horizontal 0 C layer is an experimental artefact, and because the rule takes the
outermost thawed cell per depth, one warm shallow row otherwise contributes a
full-radius disc to both volumes and swamps the difference being measured.

Outputs, under ``--out`` (default ``artifacts/thaw_volume``):

* ``thaw_volume_evolution.png`` -- the two panels
* ``thaw_volume_evolution.csv`` -- per-step observed/simulated volume and gap

Usual invocation, using the same April 8 defaults as ``fem_debug_replay``:

    python -m software.tools.thaw_volume_evolution

Add ``--material-override cap_s=1200,lambda_s=5.0`` to inject inverted
parameters the way the inversion service does, and ``--baseline`` to draw the
uncalibrated twin on the simulated panel as well.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from software.startup.utils.config import load_startup_config
from software.tools.fem_debug_replay import (
    DEFAULT_FLUID_PATH,
    DEFAULT_SENSOR_PATH,
    _build_servers,
    _grid_from_mesh,
    _interp_to_grid,
    _parse_material_override,
    _sensor_rows,
    _solver_field_c,
)
from software.tools.replay_csv import _load_fluid_temps

# Two hues, fixed roles: observed is always blue, simulated always orange, and
# the uncalibrated baseline is a muted grey so it never competes with either.
# Deep steps rather than the dashboard's bright ones -- these are read on a white
# figure, not a dark panel.
OBSERVED_COLOR = "#0369a1"
SIMULATED_COLOR = "#c2410c"
BASELINE_COLOR = "#64748b"
REFERENCE_COLOR = "#94a3b8"
GRID_COLOR = "#e2e8f0"
INK = "#1e293b"
INK_MUTED = "#64748b"


# ---------------------------------------------------------------------------
# VOLUME
# ---------------------------------------------------------------------------
def _volume_from_grid(
    R: np.ndarray,
    Z: np.ndarray,
    T: np.ndarray,
    *,
    threshold_c: float = 0.0,
    surface_exclusion_m: float = 0.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Thawed volume by revolving the per-depth outermost thawed radius.

    Depth rows with nothing above ``threshold_c`` contribute R = 0 rather than
    being skipped: dropping them would shorten the integration interval and make
    a small thawed bulb look like a short fat cylinder. Cells outside the
    interpolation hull are NaN and count as frozen, which is honest for the
    measured field -- the thermocouples stop constraining anything out there.

    Returns (volume_m3, radii, depths).
    """
    thawed = np.isfinite(T) & (T >= threshold_c)
    radii: list[float] = []
    depths: list[float] = []
    for row in range(T.shape[0]):
        depth = float(Z[row][0])
        if depth < surface_exclusion_m:
            continue
        depths.append(depth)
        radii.append(float(np.max(R[row][thawed[row]])) if np.any(thawed[row]) else 0.0)

    if len(depths) < 2:
        return 0.0, np.asarray(radii), np.asarray(depths)

    radii_arr = np.asarray(radii, dtype=float)
    depths_arr = np.asarray(depths, dtype=float)
    volume = float(trapezoid(np.pi * radii_arr**2, depths_arr))
    return volume, radii_arr, depths_arr


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


def _label_final(ax, t: np.ndarray, v_litres: np.ndarray, color: str) -> None:
    """Direct-label the end of a curve; no value on every point."""
    if not t.size:
        return
    ax.annotate(
        f"{v_litres[-1]:.2f} L",
        xy=(t[-1], v_litres[-1]),
        xytext=(-4, 8),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=INK,
    )


def _plot_volumes(
    times: np.ndarray,
    v_obs: np.ndarray,
    v_fem: np.ndarray,
    v_base: np.ndarray | None,
    out_path: Path,
    *,
    surface_exclusion_m: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obs_l = v_obs * 1000.0
    fem_l = v_fem * 1000.0

    fig, (ax_obs, ax_fem) = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=160, sharey=True)

    # --- observed -----------------------------------------------------------
    ax_obs.fill_between(times, obs_l, color=OBSERVED_COLOR, alpha=0.12, linewidth=0)
    ax_obs.plot(times, obs_l, color=OBSERVED_COLOR, linewidth=2.0)
    _label_final(ax_obs, times, obs_l, OBSERVED_COLOR)
    _style_axes(ax_obs, xlabel="Time (h)", title="Observed (thermocouple field)")
    ax_obs.set_ylabel("Thawed volume (L)", fontsize=9, color=INK_MUTED)

    # --- simulated ----------------------------------------------------------
    ax_fem.fill_between(times, fem_l, color=SIMULATED_COLOR, alpha=0.12, linewidth=0)
    ax_fem.plot(times, obs_l, color=REFERENCE_COLOR, linewidth=1.4, linestyle="--", label="observed")
    ax_fem.plot(times, fem_l, color=SIMULATED_COLOR, linewidth=2.0, label="simulated")
    if v_base is not None:
        ax_fem.plot(
            times,
            v_base * 1000.0,
            color=BASELINE_COLOR,
            linewidth=1.4,
            linestyle=":",
            label="simulated, no inversion",
        )
    _label_final(ax_fem, times, fem_l, SIMULATED_COLOR)
    _style_axes(ax_fem, xlabel="Time (h)", title="Simulated (FEM forecast)")
    ax_fem.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK_MUTED)

    ax_obs.set_ylim(bottom=0.0)
    ax_obs.set_xlim(float(times[0]), float(times[-1]))

    fig.suptitle(
        f"Thawed volume evolution   |   t = {times[0]:.2f}-{times[-1]:.2f} h   |   "
        f"depths >= {surface_exclusion_m * 1000:.0f} mm",
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
def run_volume_evolution(
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
    grid_resolution: int,
    threshold_c: float,
    surface_exclusion_m: float | None,
    material_override: dict[str, float] | None,
    run_baseline: bool,
    use_pipe_boundary: bool,
    horizon_hours: float | None,
    exclude_sensors: set[str] | None,
    progress_every: int,
) -> None:
    config = load_startup_config(config_path)
    if surface_exclusion_m is None:
        thaw_cfg = config.get("thaw_front_reconstruction", {})
        surface_exclusion_m = float(thaw_cfg.get("surface_exclusion_m", 0.0)) if isinstance(thaw_cfg, dict) else 0.0

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

    builder, pipe, forecast = _build_servers(
        config, run_baseline=run_baseline, horizon_hours=horizon_hours
    )
    if not use_pipe_boundary:
        pipe = None
    print(
        "Pipe boundary stage: "
        + ("enabled" if pipe is not None and fluid is not None else "disabled (configured boundaries used as-is)")
    )
    print(f"Volume integral ignores depths shallower than {surface_exclusion_m * 1000:.0f} mm")

    times: list[float] = []
    vol_obs: list[float] = []
    vol_fem: list[float] = []
    vol_base: list[float] = []
    grid: tuple[np.ndarray, np.ndarray] | None = None

    for index, sensor_msg in enumerate(snapshots):
        time_hours = float(sensor_msg["time_hours"])

        # 1. Boundary condition builder (real service method).
        boundary_snapshot = builder._build_snapshot(sensor_msg)

        # 2. Pipe boundary enrichment through the real calibration path.
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

        obs_x = np.array([s["x_m"] for s in sensor_msg["sensors"]], dtype=float)
        obs_z = np.array([s["z_m"] for s in sensor_msg["sensors"]], dtype=float)
        obs_t = np.array([s["temperature"] for s in sensor_msg["sensors"]], dtype=float)
        T_obs = _interp_to_grid(obs_x, obs_z, obs_t, R, Z)

        node_xy, temps_c = _solver_field_c(forecast._solver)
        T_fem = _interp_to_grid(node_xy[:, 0], node_xy[:, 1], temps_c, R, Z)

        v_obs, _, _ = _volume_from_grid(
            R, Z, T_obs, threshold_c=threshold_c, surface_exclusion_m=surface_exclusion_m
        )
        v_fem, _, _ = _volume_from_grid(
            R, Z, T_fem, threshold_c=threshold_c, surface_exclusion_m=surface_exclusion_m
        )

        times.append(time_hours)
        vol_obs.append(v_obs)
        vol_fem.append(v_fem)

        if run_baseline:
            baseline_solver = forecast._baseline_solver
            if baseline_solver is None:
                vol_base.append(float("nan"))
            else:
                base_xy, base_c = _solver_field_c(baseline_solver)
                T_base = _interp_to_grid(base_xy[:, 0], base_xy[:, 1], base_c, R, Z)
                v_base, _, _ = _volume_from_grid(
                    R, Z, T_base, threshold_c=threshold_c, surface_exclusion_m=surface_exclusion_m
                )
                vol_base.append(v_base)

        is_last = index == len(snapshots) - 1
        if index % progress_every == 0 or is_last:
            print(
                f"  [{index + 1}/{len(snapshots)}] t={time_hours:7.3f} h  "
                f"V observed {v_obs * 1000:6.3f} L   V simulated {v_fem * 1000:6.3f} L"
            )

    times_arr = np.asarray(times, dtype=float)
    obs_arr = np.asarray(vol_obs, dtype=float)
    fem_arr = np.asarray(vol_fem, dtype=float)
    base_arr = np.asarray(vol_base, dtype=float) if run_baseline else None

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "thaw_volume_evolution.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["time_hours", "volume_observed_m3", "volume_simulated_m3", "volume_gap_m3"]
        if base_arr is not None:
            fieldnames.append("volume_baseline_m3")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, t_h in enumerate(times_arr):
            row = {
                "time_hours": round(float(t_h), 6),
                "volume_observed_m3": obs_arr[idx],
                "volume_simulated_m3": fem_arr[idx],
                "volume_gap_m3": fem_arr[idx] - obs_arr[idx],
            }
            if base_arr is not None:
                row["volume_baseline_m3"] = base_arr[idx]
            writer.writerow(row)
    print(f"\nWrote per-step volumes to {csv_path}")

    png_path = out_dir / "thaw_volume_evolution.png"
    _plot_volumes(
        times_arr,
        obs_arr,
        fem_arr,
        base_arr,
        png_path,
        surface_exclusion_m=surface_exclusion_m,
    )
    print(f"Wrote plot to {png_path}")

    _print_summary(times_arr, obs_arr, fem_arr)


def _print_summary(times: np.ndarray, obs: np.ndarray, fem: np.ndarray) -> None:
    gap = fem - obs
    print("\n--- summary ---")
    print(f"steps                  : {times.size}")
    print(f"V observed (L)         : final {obs[-1] * 1000:.3f}  max {obs.max() * 1000:.3f}")
    print(f"V simulated (L)        : final {fem[-1] * 1000:.3f}  max {fem.max() * 1000:.3f}")
    print(
        f"gap simulated-observed : mean {np.mean(gap) * 1000:+.3f} L  "
        f"max |gap| {np.max(np.abs(gap)) * 1000:.3f} L"
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(obs > 1e-9, fem / obs, np.nan)
    if np.any(np.isfinite(ratio)):
        print(f"simulated / observed   : mean {np.nanmean(ratio):.2f}x  final {ratio[-1]:.2f}x")
    print(
        "A twin that tracks the volume but not the front shape, or the reverse, is a different "
        "problem than one that is simply too warm -- check fem_debug_replay for the field itself."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay recorded sensor data and plot observed vs simulated thawed volume.",
    )
    parser.add_argument(
        "--path",
        default=DEFAULT_SENSOR_PATH,
        help=f"Recorded sensor CSV/XLSX with H*D* columns (default: {DEFAULT_SENSOR_PATH}).",
    )
    parser.add_argument("--config", default="startup.conf", help="Startup config to take the FEM/BC setup from.")
    parser.add_argument("--out", default="artifacts/thaw_volume", help="Output directory.")
    parser.add_argument("--time-column", default=None, help="Override the time column header.")
    parser.add_argument("--start-row", type=int, default=1, help="First data row (1-based).")
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows to read.")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth row (thins the replay).")
    parser.add_argument("--grid-resolution", type=int, default=160, help="Integration grid resolution.")
    parser.add_argument("--threshold-c", type=float, default=0.0, help="Thaw threshold (C).")
    parser.add_argument("--horizon-hours", type=float, default=None, help="Override fem_forecast.horizon_hours.")
    parser.add_argument(
        "--surface-exclusion-m",
        type=float,
        default=None,
        help="Ignore depths shallower than this in the volume integral "
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
        help="Also run the uncalibrated twin and draw its volume on the simulated panel.",
    )
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
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print a progress line every Nth replayed step.",
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

    run_volume_evolution(
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
        grid_resolution=max(int(args.grid_resolution), 20),
        threshold_c=float(args.threshold_c),
        surface_exclusion_m=args.surface_exclusion_m,
        material_override=_parse_material_override(args.material_override),
        run_baseline=bool(args.baseline),
        use_pipe_boundary=not args.no_pipe_boundary,
        horizon_hours=args.horizon_hours,
        exclude_sensors=set(args.exclude_sensors.split(",")) if args.exclude_sensors else None,
        progress_every=max(int(args.progress_every), 1),
    )


if __name__ == "__main__":
    main()

"""Run the inversion service over a whole recorded experiment and plot what it did.

One command, no infrastructure: replays a sensor file through the real boundary
builder, pipe-boundary calibration, inversion and forecast services with RabbitMQ
and InfluxDB stubbed out, driving the inversion inline at a fixed cadence, and
writes a set of diagnostic figures plus the raw numbers behind them.

It exists to answer one question -- is the inversion doing anything useful? --
which decomposes into four that the figures separate deliberately, because a
service can pass any one of them while failing the others:

1. **Are the parameters moving sensibly, or pinned?**  A fit that sits on a
   bound is not an estimate, it is the optimiser running out of room. Figures 1
   and 2.
2. **Does the fit improve its own objective?**  The loss at the fitted
   parameters against the loss at the configured literature values, on the same
   window. If these are close, the fit is not finding anything the base values
   did not already have. Figure 3.
3. **Does that translate into a better forecast?**  Sensor RMSE with the
   inverted parameters against the forecast service's own uncalibrated baseline
   twin, which shares the mesh, the initial field and the boundary conditions,
   so the gap is the parameters and nothing else. Figure 4.
4. **Does it put the thaw front in the right place?**  RMSE can look fine while
   the front is badly wrong, because the thermocouples are sparse near it.
   Figure 5.

Figure 6 plots the fitted parameters against each other rather than against
time: two parameters that trade off along a line are jointly unidentifiable, and
the fit is choosing an arbitrary point on that line.

Cadence note: ``min_update_hours`` and ``min_update_seconds`` are zeroed, since
in a replay those gates would pace the service against how fast this machine
solves. ``--invert-every`` controls it explicitly instead. Warm starting is left
on, as the live service has it, so each fit begins from the previous one -- which
means the trajectories are path-dependent by design, not noise.

Outputs, all under ``--out`` (default ``artifacts/inversion_audit``):

    01_parameter_evolution.png   fitted values, with base value and bounds
    02_bound_occupancy.png       position within the allowed range, all three
    03_fit_quality.png           fitted loss vs loss at the base parameters
    04_forecast_error.png        sensor RMSE with and without inversion
    05_thaw_front.png            r_max measured, calibrated, uncalibrated
    06_parameter_tradeoff.png    fitted parameters against each other
    inversion_updates.csv        one row per fit
    forecast_steps.csv           one row per replayed snapshot

Runtime is dominated by the forecast, which solves twice per snapshot (calibrated
and baseline). The whole April 8 file is roughly 6-8 minutes; ``--stride 4`` cuts
that to under two if you only want the shape.

Run:

    python -m software.tools.inversion_audit
    python -m software.tools.inversion_audit --stride 4 --invert-every 3
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
    create_fem_forecast_server,
    create_pipe_boundary_server,
)
from software.startup.utils.config import load_startup_config
from software.tools.fem_debug_replay import (
    DEFAULT_FLUID_PATH,
    DEFAULT_SENSOR_PATH,
    _front_from_grid,
    _grid_from_mesh,
    _interp_to_grid,
    _NullInflux,
    _Sink,
    _sensor_rows,
)
from software.tools.inversion_accuracy import _build_inversion, _run_inversion_now, _sensor_rmse
from software.tools.replay_csv import _load_fluid_temps

CAL = "#C2531F"     # with inversion
BASE = "#1F6FA8"    # without inversion (configured parameters)
MEAS = "#111111"    # measured
BOUND = "#C2531F"


# ---------------------------------------------------------------------------
# REPLAY
# ---------------------------------------------------------------------------
def _build_services(config: dict, *, use_pipe: bool, horizon_hours: float | None):
    rabbit_cfg = config.get("rabbitmq", {})
    influx_cfg: InfluxConfig = build_influx_config(config.get("influxdb", {}))

    fem_cfg = dict(config.get("fem_forecast", {}))
    # The baseline twin is what "without inversion" means here, so it is required
    # rather than optional: it shares mesh, initial field and boundaries with the
    # calibrated solver, leaving the material parameters as the only difference.
    fem_cfg["run_baseline"] = True
    if horizon_hours is not None:
        fem_cfg["horizon_hours"] = float(horizon_hours)

    builder = create_boundary_condition_builder_server(rabbit_cfg, config.get("boundary_condition_builder", {}))
    forecast = create_fem_forecast_server(influx_cfg, rabbit_cfg, fem_cfg)
    forecast.mq_out = _Sink()
    forecast.influx = _NullInflux()

    pipe = None
    pipe_cfg = config.get("pipe_boundary", {})
    if use_pipe and pipe_cfg:
        pipe = create_pipe_boundary_server(influx_cfg, rabbit_cfg, pipe_cfg)
        pipe.mq_boundary_out = _Sink()

    return builder, pipe, forecast


def _loss_at_base(inversion) -> tuple[float, dict | None]:
    """Objective value at the configured parameters, on the current window.

    The reference the fitted loss has to beat. Without it a small final loss is
    uninterpretable: it might mean the fit found something, or it might mean the
    window is easy and the literature values already fit it.
    """
    window = inversion._build_observation_window()
    if window is None:
        return float("nan"), None
    times_hours, positions, obs_series, ic_sensors = window
    base_params = {name: float(value) for name, value in inversion.base_params.items()}
    series = inversion._generate_series(base_params, positions, times_hours, ic_sensors)
    return inversion._compute_loss(series, obs_series), window


def run_audit(
    *,
    path: Path,
    fluid_path: Path | None,
    config_path: str | Path,
    out_dir: Path,
    start_row: int,
    limit: int | None,
    stride: int,
    invert_every: int,
    use_pipe: bool,
    horizon_hours: float | None,
    maxiter: int | None,
    maxfun: int | None,
    warmup_hours: float | None = None,
    exclude_sensors: set[str] | None,
    grid_resolution: int,
) -> tuple[list[dict], list[dict], dict]:
    config = load_startup_config(config_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshots = _sensor_rows(path, None, start_row, limit, stride, exclude_sensors)
    if not snapshots:
        raise SystemExit(f"No usable sensor rows parsed from {path}")
    print(f"Loaded {len(snapshots)} snapshots from {path.name} "
          f"(t = {snapshots[0]['time_hours']:.3f} .. {snapshots[-1]['time_hours']:.3f} h)")

    fluid = None
    if fluid_path is not None and use_pipe:
        fluid = _load_fluid_temps(
            fluid_path,
            inlet_col="Fluid Temperature_Inlet",
            outlet_col="Fluid Temperatue_Outlet",
            time_col="Time (h)",
        )
        print(f"Loaded {len(fluid[0])} fluid samples from {fluid_path.name}")

    builder, pipe, forecast = _build_services(config, use_pipe=use_pipe, horizon_hours=horizon_hours)
    inversion = _build_inversion(config, maxiter=maxiter, maxfun=maxfun,
                                 warmup_hours=warmup_hours)
    print(f"Inverting: {', '.join(inversion.invert_params)}   every {invert_every} snapshots")
    print(f"Pipe boundary: {'enabled' if pipe is not None and fluid is not None else 'disabled'}")

    thaw_cfg = config.get("thaw_front_reconstruction", {})
    surface_exclusion_m = float(thaw_cfg.get("surface_exclusion_m", 0.0)) if isinstance(thaw_cfg, dict) else 0.0

    steps: list[dict] = []
    updates: list[dict] = []
    material_override: dict[str, float] | None = None
    grid: tuple[np.ndarray, np.ndarray] | None = None
    wall_start = time.perf_counter()

    for index, sensor_msg in enumerate(snapshots):
        time_hours = float(sensor_msg["time_hours"])

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

        parsed = inversion._parse_snapshot(sensor_msg)
        if parsed is not None:
            inversion._upsert_snapshot(parsed)

        # Fit under the boundary the forecast is actually marching, not the
        # configured fallback. Without this the optimiser has to absorb the
        # boundary difference into the material, and the parameters it produces
        # carry that compensation into a forecast running the other boundary.
        inversion.set_boundary_snapshot(boundary_snapshot)

        if index > 0 and index % invert_every == 0:
            base_loss, _window = _loss_at_base(inversion)
            params, payload, elapsed = _run_inversion_now(inversion)
            if params is not None and payload is not None:
                material_override = params
                row = {
                    "time_hours": round(time_hours, 6),
                    "step": index,
                    "final_loss": float(payload.get("final_loss", float("nan"))),
                    "loss_at_base": float(base_loss),
                    "eval_count": int(payload.get("eval_count", 0)),
                    "success": bool(payload.get("success", False)),
                    "elapsed_seconds": round(elapsed, 3),
                }
                scales = payload.get("best_scales", {}) or {}
                for name in inversion.invert_params:
                    lo, hi = inversion.relative_bounds[name]
                    scale = float(scales.get(name, float(params[name]) / float(inversion.base_params[name])))
                    row[name] = float(params[name])
                    row[f"{name}_scale"] = scale
                    # Fraction of the way across the allowed range: 0 and 1 are
                    # the bounds themselves, and a trajectory that lives there is
                    # the signature of a parameter the data cannot pin down.
                    row[f"{name}_position"] = (scale - lo) / (hi - lo) if hi > lo else float("nan")
                    row[f"{name}_at_bound"] = bool(abs(scale - lo) < 1e-6 or abs(scale - hi) < 1e-6)
                updates.append(row)

        if material_override:
            boundary_snapshot["material_override"] = dict(material_override)

        forecast._process_message(boundary_snapshot)
        if forecast._solver is None:
            raise SystemExit("FEM solver was never built; check the forecast config.")

        obs_x = np.array([s["x_m"] for s in sensor_msg["sensors"]], dtype=float)
        obs_z = np.array([s["z_m"] for s in sensor_msg["sensors"]], dtype=float)
        obs_t = np.array([s["temperature"] for s in sensor_msg["sensors"]], dtype=float)

        rmse_cal, max_cal = _sensor_rmse(forecast._solver, obs_x, obs_z, obs_t)
        if forecast._baseline_solver is not None:
            rmse_base, max_base = _sensor_rmse(forecast._baseline_solver, obs_x, obs_z, obs_t)
        else:
            rmse_base, max_base = float("nan"), float("nan")

        if grid is None:
            grid = _grid_from_mesh(forecast._solver.node, grid_resolution)
        R, Z = grid
        T_obs = _interp_to_grid(obs_x, obs_z, obs_t, R, Z)
        r_meas, _, _, _ = _front_from_grid(R, Z, T_obs, surface_exclusion_m=surface_exclusion_m)
        r_cal, _, _ = forecast._solver.compute_thaw_metrics()
        r_base = None
        if forecast._baseline_solver is not None:
            r_base, _, _ = forecast._baseline_solver.compute_thaw_metrics()

        steps.append({
            "time_hours": round(time_hours, 6),
            "rmse_with_inversion_c": rmse_cal,
            "rmse_without_inversion_c": rmse_base,
            "max_abs_with_inversion_c": max_cal,
            "max_abs_without_inversion_c": max_base,
            "r_max_measured_m": r_meas,
            "r_max_with_inversion_m": r_cal,
            "r_max_without_inversion_m": r_base,
            "inverted": bool(material_override),
        })

        if index % max(len(snapshots) // 20, 1) == 0 or index == len(snapshots) - 1:
            done = time.perf_counter() - wall_start
            print(f"  [{index + 1:4d}/{len(snapshots)}] t={time_hours:6.3f} h  "
                  f"RMSE {rmse_cal:5.2f} / {rmse_base:5.2f} C  "
                  f"({len(updates)} fits, {done / 60:.1f} min)")

    meta = {
        "invert_params": list(inversion.invert_params),
        "base_params": {n: float(inversion.base_params[n]) for n in inversion.invert_params},
        "bounds": {n: tuple(inversion.relative_bounds[n]) for n in inversion.invert_params},
        "wall_minutes": (time.perf_counter() - wall_start) / 60.0,
    }
    return steps, updates, meta


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
def _finish(ax, *, xlabel="Experiment time (h)", ylabel="", title=""):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.22, linewidth=0.7)
    ax.margins(x=0.01)


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  wrote {path}")


def plot_parameter_evolution(updates, meta, out_dir):
    import matplotlib.pyplot as plt

    names = meta["invert_params"]
    t = [r["time_hours"] for r in updates]
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 2.6 * len(names)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, name in zip(axes, names):
        base = meta["base_params"][name]
        lo, hi = meta["bounds"][name]
        ax.axhspan(base * lo, base * hi, color=BOUND, alpha=0.07, zorder=0)
        ax.axhline(base * lo, color=BOUND, lw=1.0, ls=":", zorder=1)
        ax.axhline(base * hi, color=BOUND, lw=1.0, ls=":", zorder=1)
        ax.axhline(base, color="0.35", lw=1.2, ls="--", zorder=2,
                   label=f"configured = {base:g}")
        ax.plot(t, [r[name] for r in updates], color=CAL, lw=1.8, zorder=3, label="fitted")
        ax.legend(loc="best", fontsize=8, framealpha=0.85)
        _finish(ax, xlabel="", ylabel=name, title=f"{name}   (shaded = allowed range)")
    axes[-1].set_xlabel("Experiment time (h)")
    fig.suptitle("Inverted parameters over the experiment", fontsize=13, y=0.998)
    _save(fig, out_dir / "01_parameter_evolution.png")
    plt.close(fig)


def fig_parameter_evolution(updates, meta, out_dir):
    """Paper figure: the three fitted parameters over the experiment.

    The diagnostic version of this (``01_parameter_evolution.png``) answers "is
    it moving"; this one has to answer "is it an estimate", which is a different
    question and needs one thing the diagnostic does not draw: where the fit is
    sitting *on* a bound. A trajectory pinned to a bound is the optimiser
    running out of room, not a measurement of the soil, and on a plain line plot
    it is indistinguishable from a parameter that genuinely converged to a
    constant. Pinned stretches are therefore drawn as a thick pale underlay
    along the bound, so the reader sees the distinction without the caption.

    One panel per parameter, absolute units, because the three span three orders
    of magnitude; normalising them onto one axis would hide that ``k`` is the
    only one with room above its base.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from software.tools.inversion_margin import PRETTY
    from software.tools.paper_figures import (
        DOUBLE,
        FEM_KEEP,
        INK,
        MUTED,
        STATUS_LIMIT,
        grid,
        save,
        style,
    )

    style()
    names = list(meta["invert_params"])
    t = np.array([float(r["time_hours"]) for r in updates], dtype=float)

    fig, axes = plt.subplots(len(names), 1, figsize=(DOUBLE, DOUBLE * 0.62),
                             sharex=True, gridspec_kw={"hspace": 0.30})
    axes = np.atleast_1d(axes)

    for letter, ax, name in zip("abcdefghij", axes, names):
        base = float(meta["base_params"][name])
        lo, hi = (float(v) for v in meta["bounds"][name])
        values = np.array([float(r[name]) for r in updates], dtype=float)
        pinned = np.array([bool(r.get(f"{name}_at_bound")) for r in updates], dtype=bool)

        ax.axhspan(base * lo, base * hi, color=MUTED, alpha=0.07, lw=0, zorder=0)
        for edge in (base * lo, base * hi):
            ax.axhline(edge, color=MUTED, lw=0.8, ls=":", zorder=1)
        ax.axhline(base, color=INK, lw=1.1, ls="--", zorder=2)

        # Pinned stretches first, as a wide pale underlay, so the thin fitted
        # line still reads on top of it rather than being replaced by it.
        if np.any(pinned):
            ax.plot(np.where(pinned, t, np.nan), np.where(pinned, values, np.nan),
                    "-", color=STATUS_LIMIT, lw=5.0, alpha=0.30,
                    solid_capstyle="butt", zorder=3)
        ax.plot(t, values, "-", color=FEM_KEEP, lw=1.5, zorder=4)

        ax.set_ylabel(PRETTY.get(name, name), fontsize=7.2)
        pad = 0.04 * base * (hi - lo)
        ax.set_ylim(base * lo - pad, base * hi + pad)
        # The two bounds and the configured value carry the only numbers worth
        # printing, and they are already at fixed heights -- so they go on the
        # y-ticks rather than into floating annotations, which in the earlier
        # draft landed on the bound lines and on the trajectory itself. In panel
        # (b) the base and the lower bound coincide (k cannot go below its base
        # under the shipped [1, 5]), and one tick is the honest way to show that.
        ticks = sorted({round(base * lo, 6), round(base, 6), round(base * hi, 6)})
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{v:g}" for v in ticks])
        # Both callouts ride at 0.90 rather than the top edge: in panel (a) the
        # trajectory is glued to the upper bound, so the top corners are taken
        # and only a band just below them is free in all three panels.
        ax.annotate(f"({letter})", xy=(0.006, 0.90), xycoords="axes fraction",
                    fontsize=8, fontweight="bold", va="top", ha="left", color=INK)
        pct = 100.0 * float(np.count_nonzero(pinned)) / max(len(pinned), 1)
        ax.annotate(f"at a bound for {pct:.0f}% of fits", xy=(0.997, 0.90),
                    xycoords="axes fraction", fontsize=6.4,
                    color=STATUS_LIMIT if pct > 50 else MUTED,
                    va="top", ha="right")
        grid(ax)

    axes[-1].set_xlabel("experiment time (h)")
    axes[-1].set_xlim(0.0, float(t[-1]))

    handles = [
        Line2D([], [], color=FEM_KEEP, lw=1.5, label="fitted"),
        Line2D([], [], color=STATUS_LIMIT, lw=5.0, alpha=0.30, label="pinned at a bound"),
        Line2D([], [], color=INK, lw=1.1, ls="--", label="configured value"),
        Patch(facecolor=MUTED, alpha=0.07, edgecolor=MUTED, ls=":", label="allowed range"),
    ]
    fig.subplots_adjust(bottom=0.16)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               ncol=len(handles), frameon=False, fontsize=7.0, columnspacing=2.2)
    save(fig, out_dir, "fig_parameter_evolution")


def plot_bound_occupancy(updates, meta, out_dir):
    """All parameters on one axis as position within their own allowed range."""
    import matplotlib.pyplot as plt

    names = meta["invert_params"]
    t = [r["time_hours"] for r in updates]
    fig, ax = plt.subplots(figsize=(10, 4.4))

    ax.axhspan(-0.02, 0.02, color=BOUND, alpha=0.12, zorder=0)
    ax.axhspan(0.98, 1.02, color=BOUND, alpha=0.12, zorder=0)
    ax.axhline(0.0, color=BOUND, lw=1.0, ls=":")
    ax.axhline(1.0, color=BOUND, lw=1.0, ls=":")
    for name, colour in zip(names, ("#C2531F", "#1F6FA8", "#3F7D3A")):
        ax.plot(t, [r[f"{name}_position"] for r in updates], lw=1.8, label=name, color=colour)

    ax.set_ylim(-0.08, 1.08)
    ax.legend(loc="best", fontsize=9, ncol=len(names))
    _finish(ax, ylabel="position within allowed range",
            title="0 = lower bound, 1 = upper bound. A flat line on either is a pinned fit, not an estimate.")
    _save(fig, out_dir / "02_bound_occupancy.png")
    plt.close(fig)


def plot_fit_quality(updates, out_dir):
    import matplotlib.pyplot as plt

    t = [r["time_hours"] for r in updates]
    fitted = np.array([r["final_loss"] for r in updates], dtype=float)
    base = np.array([r["loss_at_base"] for r in updates], dtype=float)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})
    ax.plot(t, base, color=BASE, lw=1.7, label="loss at configured parameters")
    ax.plot(t, fitted, color=CAL, lw=1.9, label="loss after fitting")
    ax.fill_between(t, fitted, base, where=base >= fitted, color=CAL, alpha=0.16,
                    label="improvement won by fitting")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=9)
    _finish(ax, xlabel="", ylabel="objective (MSE, C$^2$)",
            title="Does the fit improve its own objective?")

    with np.errstate(divide="ignore", invalid="ignore"):
        pct = 100.0 * (base - fitted) / base
    ax2.axhline(0.0, color="0.4", lw=1.0)
    ax2.plot(t, pct, color=CAL, lw=1.6)
    _finish(ax2, ylabel="loss reduction (%)", title="")
    _save(fig, out_dir / "03_fit_quality.png")
    plt.close(fig)


def plot_forecast_error(steps, updates, out_dir):
    import matplotlib.pyplot as plt

    t = np.array([r["time_hours"] for r in steps], dtype=float)
    cal = np.array([r["rmse_with_inversion_c"] for r in steps], dtype=float)
    base = np.array([r["rmse_without_inversion_c"] for r in steps], dtype=float)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})
    ax.plot(t, base, color=BASE, lw=1.7, label="without inversion (configured parameters)")
    ax.plot(t, cal, color=CAL, lw=1.9, label="with inversion")
    ax.fill_between(t, cal, base, where=base >= cal, color=CAL, alpha=0.16, label="inversion helps")
    ax.fill_between(t, cal, base, where=base < cal, color=BASE, alpha=0.16, label="inversion hurts")
    for r in updates:
        ax.axvline(r["time_hours"], color="0.75", lw=0.4, zorder=0)
    ax.legend(loc="best", fontsize=9)
    _finish(ax, xlabel="", ylabel="sensor RMSE (C)",
            title="Forecast error against the thermocouples (grey lines = a fit ran)")

    ax2.axhline(0.0, color="0.4", lw=1.0)
    ax2.plot(t, base - cal, color=CAL, lw=1.6)
    _finish(ax2, ylabel="RMSE reduction (C)", title="")
    _save(fig, out_dir / "04_forecast_error.png")
    plt.close(fig)


def plot_thaw_front(steps, out_dir):
    import matplotlib.pyplot as plt

    t = [r["time_hours"] for r in steps]

    def mm(key):
        return [np.nan if r[key] is None else r[key] * 1000.0 for r in steps]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(t, mm("r_max_measured_m"), color=MEAS, lw=1.8, ls="--", label="measured")
    ax.plot(t, mm("r_max_without_inversion_m"), color=BASE, lw=1.7, label="without inversion")
    ax.plot(t, mm("r_max_with_inversion_m"), color=CAL, lw=1.9, label="with inversion")
    ax.legend(loc="best", fontsize=9)
    _finish(ax, ylabel="thaw radius r_max (mm)",
            title="Thaw front: the quantity the safety monitor acts on")
    _save(fig, out_dir / "05_thaw_front.png")
    plt.close(fig)


def plot_tradeoff(updates, meta, out_dir):
    """Fitted parameters against each other; a line means they are trading off."""
    import matplotlib.pyplot as plt

    names = meta["invert_params"]
    if len(names) < 2:
        return
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    t = np.array([r["time_hours"] for r in updates], dtype=float)

    fig, axes = plt.subplots(1, len(pairs), figsize=(4.6 * len(pairs), 4.3))
    axes = np.atleast_1d(axes)
    scatter = None
    for ax, (a, b) in zip(axes, pairs):
        scatter = ax.scatter([r[a] for r in updates], [r[b] for r in updates],
                             c=t, cmap="viridis", s=22, linewidths=0)
        ax.axvline(meta["base_params"][a], color="0.6", lw=1.0, ls="--")
        ax.axhline(meta["base_params"][b], color="0.6", lw=1.0, ls="--")
        # A pinned parameter is constant, so its standard deviation is zero and
        # the correlation is undefined rather than merely small. Say so: "never
        # moved" is the finding, and a NaN label would hide it.
        va = np.asarray([r[a] for r in updates], dtype=float)
        vb = np.asarray([r[b] for r in updates], dtype=float)
        flat = [n for n, v in ((a, va), (b, vb)) if float(np.std(v)) == 0.0]
        if len(updates) < 3:
            label = "too few fits to correlate"
        elif flat:
            label = f"{' and '.join(flat)} never moved"
        else:
            label = f"r = {float(np.corrcoef(va, vb)[0, 1]):+.2f}"
        _finish(ax, xlabel=a, ylabel=b, title=label)
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes.tolist(), fraction=0.03, pad=0.02)
        cbar.set_label("experiment time (h)")
    fig.suptitle("Fitted parameters against each other (dashed = configured values)", fontsize=12)
    fig.savefig(out_dir / "06_parameter_tradeoff.png", dpi=150, bbox_inches="tight")
    print(f"  wrote {out_dir / '06_parameter_tradeoff.png'}")
    plt.close(fig)


# ---------------------------------------------------------------------------
def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path}")


def _summary(steps, updates, meta) -> None:
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    if not updates:
        print("No inversion ever ran. Check --invert-every and min_snapshots.")
        return

    print(f"fits run            : {len(updates)} over {len(steps)} snapshots")
    print(f"replay wall clock   : {meta['wall_minutes']:.1f} min")

    print("\nparameter behaviour")
    print(f"  {'parameter':<14} {'first':>10} {'last':>10} {'range':>18} {'at a bound':>12}")
    for name in meta["invert_params"]:
        values = np.array([r[name] for r in updates], dtype=float)
        pinned = 100.0 * np.mean([r[f"{name}_at_bound"] for r in updates])
        print(f"  {name:<14} {values[0]:10.4g} {values[-1]:10.4g} "
              f"{values.min():8.4g}..{values.max():<8.4g} {pinned:11.0f}%")

    fitted = np.array([r["final_loss"] for r in updates], dtype=float)
    base = np.array([r["loss_at_base"] for r in updates], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        reduction = 100.0 * (base - fitted) / base
    print("\nobjective")
    print(f"  loss at configured parameters : median {np.nanmedian(base):.5f}")
    print(f"  loss after fitting            : median {np.nanmedian(fitted):.5f}")
    print(f"  reduction                     : median {np.nanmedian(reduction):.1f}%")

    cal = np.array([r["rmse_with_inversion_c"] for r in steps], dtype=float)
    bas = np.array([r["rmse_without_inversion_c"] for r in steps], dtype=float)
    inverted = np.array([r["inverted"] for r in steps], dtype=bool)
    print("\nforecast error against the thermocouples (once an override exists)")
    if np.any(inverted):
        c, b = cal[inverted], bas[inverted]
        print(f"  with inversion    : mean {np.nanmean(c):.3f} C   max {np.nanmax(c):.3f} C")
        print(f"  without inversion : mean {np.nanmean(b):.3f} C   max {np.nanmax(b):.3f} C")
        delta = np.nanmean(b) - np.nanmean(c)
        verdict = "better" if delta > 0 else "WORSE"
        print(f"  inversion makes the forecast {verdict} by {abs(delta):.3f} C on average")
        print(f"  steps where it helped : {100.0 * np.mean(c < b):.0f}%")
    else:
        print("  no override was ever applied")

    meas = np.array([np.nan if r["r_max_measured_m"] is None else r["r_max_measured_m"] for r in steps])
    rcal = np.array([np.nan if r["r_max_with_inversion_m"] is None else r["r_max_with_inversion_m"] for r in steps])
    rbas = np.array([np.nan if r["r_max_without_inversion_m"] is None else r["r_max_without_inversion_m"] for r in steps])
    ok = np.isfinite(meas) & np.isfinite(rcal) & np.isfinite(rbas)
    if np.any(ok):
        print("\nthaw radius error vs measured (mm)")
        print(f"  with inversion    : mean |error| {np.nanmean(np.abs(rcal[ok] - meas[ok])) * 1000:.1f}")
        print(f"  without inversion : mean |error| {np.nanmean(np.abs(rbas[ok] - meas[ok])) * 1000:.1f}")


def _updates_from_csv(path: Path, config_path: str | Path) -> tuple[list[dict], dict]:
    """Rebuild the fit records from ``inversion_updates.csv`` for a replot.

    Only the per-fit rows come from the CSV. ``meta`` -- the base values and the
    relative bounds -- is read back off the *config*, through the same service
    constructor the run used, rather than inferred from the recorded scales: a
    run whose fits never left a bound pins nothing, and guessing the range from
    the trajectory would silently redraw the band the figure is there to show.
    A CSV written under a different material set will therefore disagree with
    the config, which is why the resolved base values are printed.
    """
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict = {}
            for key, value in raw.items():
                value = (value or "").strip()
                if value in ("", "None"):
                    row[key] = None
                elif value in ("True", "False"):
                    row[key] = value == "True"
                else:
                    try:
                        row[key] = float(value)
                    except ValueError:
                        row[key] = value
            rows.append(row)
    if not rows:
        raise SystemExit(f"No fit records in {path}")

    config = load_startup_config(config_path)
    inversion = _build_inversion(config, maxiter=None, maxfun=None)
    meta = {
        "invert_params": list(inversion.invert_params),
        "base_params": {n: float(inversion.base_params[n]) for n in inversion.invert_params},
        "bounds": {n: tuple(inversion.relative_bounds[n]) for n in inversion.invert_params},
    }
    for name in meta["invert_params"]:
        lo, hi = meta["bounds"][name]
        base = meta["base_params"][name]
        print(f"  {name:<12} base {base:g}  bounds [{base * lo:g}, {base * hi:g}]")
    return rows, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", default=DEFAULT_SENSOR_PATH)
    parser.add_argument("--fluid-path", default=DEFAULT_FLUID_PATH, help="'' to skip the pipe boundary.")
    parser.add_argument("--config", default="startup.conf")
    parser.add_argument("--out", default="artifacts/inversion_audit")
    parser.add_argument("--start-row", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Rows to read before striding (default: all).")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth row.")
    parser.add_argument("--invert-every", type=int, default=5, help="Run a fit every N snapshots.")
    parser.add_argument("--no-pipe-boundary", action="store_true")
    parser.add_argument("--horizon-hours", type=float, default=None)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--maxfun", type=int, default=None)
    parser.add_argument("--exclude-sensors", default=None)
    parser.add_argument("--grid-resolution", type=int, default=160)
    parser.add_argument("--warmup-hours", type=float, default=None,
                        help="Override fem_inversion.warmup_hours: data time the service "
                             "accumulates before its first fit (config default 0.5 h). "
                             "0 disables the warm-up.")
    parser.add_argument("--replot", action="store_true",
                        help="Re-render the paper figure from a previous run's "
                             "inversion_updates.csv instead of replaying. The diagnostic "
                             "figures that need the per-snapshot forecast are skipped.")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    out_dir = Path(args.out)

    if args.replot:
        csv_path = out_dir / "inversion_updates.csv"
        if not csv_path.exists():
            raise SystemExit(f"No cached fits at {csv_path}; run the audit first.")
        updates, meta = _updates_from_csv(csv_path, args.config)
        print(f"Replotting {len(updates)} fits from {csv_path}")
        fig_parameter_evolution(updates, meta, out_dir)
        plot_parameter_evolution(updates, meta, out_dir)
        plot_bound_occupancy(updates, meta, out_dir)
        plot_fit_quality(updates, out_dir)
        plot_tradeoff(updates, meta, out_dir)
        return
    steps, updates, meta = run_audit(
        path=Path(args.path),
        fluid_path=Path(args.fluid_path) if args.fluid_path else None,
        config_path=args.config,
        out_dir=out_dir,
        start_row=args.start_row,
        limit=args.limit,
        stride=max(int(args.stride), 1),
        invert_every=max(int(args.invert_every), 1),
        use_pipe=not args.no_pipe_boundary,
        horizon_hours=args.horizon_hours,
        maxiter=args.maxiter,
        maxfun=args.maxfun,
        warmup_hours=args.warmup_hours,
        exclude_sensors=set(args.exclude_sensors.split(",")) if args.exclude_sensors else None,
        grid_resolution=max(int(args.grid_resolution), 20),
    )

    print("\nwriting outputs")
    _write_csv(updates, out_dir / "inversion_updates.csv")
    _write_csv(steps, out_dir / "forecast_steps.csv")
    if updates:
        fig_parameter_evolution(updates, meta, out_dir)
        plot_parameter_evolution(updates, meta, out_dir)
        plot_bound_occupancy(updates, meta, out_dir)
        plot_fit_quality(updates, out_dir)
        plot_tradeoff(updates, meta, out_dir)
    plot_forecast_error(steps, updates, out_dir)
    plot_thaw_front(steps, out_dir)

    _summary(steps, updates, meta)


if __name__ == "__main__":
    main()

"""Can the inversion pull deliberately wrong soil parameters back to the truth?

The inversion normally starts from parameters that are already close to right,
which leaves it nothing visible to do: it removes an error that was small to
begin with, so its contribution is real but invisible on any plot. This tool
gives it a margin to work in. The FEM is started from *deliberately degraded*
soil parameters and the inversion is asked to recover from them.

Two experiments, answering two different questions, because one of them has a
ground truth and the other does not.

**A -- recovery, against a known truth (synthetic twin).**
Real geometry, real thermocouple positions and a real measured initial field are
borrowed from the recording, but the temperatures the inversion is asked to fit
are manufactured by the forward model from a known parameter set. The global
minimum of the objective then sits exactly at those parameters, so "did it
recover?" has an unambiguous answer. The inversion's *base* parameters are then
multiplied by the degrade factors, which is what makes the starting point wrong.

Because the synthetic data comes from the same forward model the inversion fits
with, this measures the fitting machinery -- optimiser, bounds, window, and the
parameterisation -- and nothing about whether the physics is right. Model error
against reality cancels exactly here. That is the point: it isolates recovery.

**B -- does recovery make the FEM better (real measurements).**
Three forecast solvers march the same real boundary stream from the same seeded
initial field, differing only in material parameters:

    reference   the configured values, frozen              -- the error floor
    degraded    the degraded values, frozen                -- the error ceiling
    recovered   degraded, plus the live ``material_override``
                the inversion publishes                    -- what it recovers

The gap between the first two is the margin. How much of it the third closes is
the inversion's contribution, measured as RMSE against the thermocouples.

**Bounds are relative, and that is a trap worth naming.** The service bounds each
parameter as ``base * scale`` with ``scale`` in ``relative_bounds``, so degrading
the base drags the reachable range with it. Degrading ``cap_s`` by 1.6x under the
shipped ``[0.5, 2.0]`` puts the truth at scale 0.625, near the edge; degrading
``k_parameter`` at all puts the truth *outside* its shipped ``[1.0, 5.0]``, which
cannot fall below the base. The experiment would then measure the bounds rather
than the inversion. ``--bound-lo/--bound-hi`` widen them so every truth sits
strictly interior, and the margin between the truth scale and the nearest bound
is printed so the setup cannot silently rig the result.

**The structural confound is measured, not merely disclosed.** The inversion's
forward model fits against the *configured default* boundary (``h = 5``), while
the forecast runs on the pipe-calibrated one (``h = 45``). Parameters fitted
under one boundary and applied under another need not transfer at all. Rather
than note this and move on, experiment B carries a fourth arm:

    recovered      degraded + inversion, fitted as the service does today
    recovered+bc   degraded + inversion, fitted against the *same* boundary the
                   forecast is running on

Both use the identical optimiser, window and bounds, so the difference between
them is attributable to the boundary mismatch and to nothing else. That makes
the confound a measured quantity instead of a caveat. The fix is applied here by
pointing the window model at the live boundary snapshot; it is a change to the
experiment, not to the shipped service.

Run:

    python -m software.tools.inversion_margin
    python -m software.tools.inversion_margin --windows 3 --skip-forecast   # A only
"""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from software.digital_twin.data_access.influx_utils import InfluxConfig
from software.digital_twin.simulator.fem_inversion.fem_inversion_server import FEMInversionServer
from software.startup.service_runners import (
    build_influx_config,
    create_boundary_condition_builder_server,
    create_fem_forecast_server,
    create_pipe_boundary_server,
)
from software.startup.utils.config import load_startup_config
from software.tools.fem_debug_replay import _NullInflux, _Sink, _sensor_rows
from software.tools.inversion_accuracy import _interp_to_grid, _solver_field_c
from software.tools.paper_figures import (
    DOUBLE,
    FEM_KEEP,
    FEM_STOP,
    INK,
    MONITORED,
    MUTED,
    STATUS_LIMIT,
    grid,
    save,
    style,
)
from software.tools.replay_csv import _load_fluid_temps

DEFAULT_PATH = "data/CR1000_Table1_April8_replay_Final.xlsx"
DEFAULT_EXCLUDE = "H10D75"

# How wrong the FEM is started. Chosen to be obviously wrong rather than subtly
# so -- roughly a halving or a near-doubling of each -- while staying inside the
# range a careless literature pick could plausibly land on.
DEFAULT_DEGRADE = {"cap_s": 1.6, "k_parameter": 0.4, "lambda_s": 0.5}

PRETTY = {
    "cap_s": r"$c_s$  (J kg$^{-1}$ K$^{-1}$)",
    "k_parameter": r"$k$  freezing-curve sharpness",
    "lambda_s": r"$\lambda_s$  (W m$^{-1}$ K$^{-1}$)",
}


class _Recorder:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def publish(self, message: dict) -> None:
        self.messages.append(message)

    @property
    def last(self) -> dict | None:
        return self.messages[-1] if self.messages else None


# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------
def make_inversion_server(config: dict, config_path: str, *, base_params: dict,
                          bounds: dict) -> FEMInversionServer:
    inv_cfg = config.get("fem_inversion", {})
    return FEMInversionServer(
        site_id=str(inv_cfg.get("site_id", "default")),
        invert_params=list(inv_cfg.get("invert_params", [])) or None,
        base_params=base_params,
        relative_bounds=bounds,
        loss=str(inv_cfg.get("loss", "mse")),
        window_size=int(inv_cfg.get("window_size", 5)),
        min_snapshots=int(inv_cfg.get("min_snapshots", 5)),
        min_sensors=int(inv_cfg.get("min_sensors", 3)),
        relative_time_window=bool(inv_cfg.get("relative_time_window", True)),
        min_horizon_hours=float(inv_cfg.get("min_horizon_hours", 1e-3)),
        maxiter=int(inv_cfg.get("maxiter", 40)),
        maxfun=int(inv_cfg.get("maxfun", 80)),
        seed=inv_cfg.get("seed", 42),
        config_path=config_path,
    )


def degrade(material: dict, factors: dict[str, float]) -> dict:
    out = deepcopy(material)
    for name, factor in factors.items():
        if name in out:
            out[name] = float(out[name]) * float(factor)
    return out


def check_reachable(factors: dict[str, float], bounds: dict, names) -> list[str]:
    """Is the truth strictly inside the reachable range from the degraded base?

    Returns one report line per parameter. A truth sitting on or outside a bound
    means the experiment is measuring the bound, not the inversion, so this is
    printed rather than merely asserted.
    """
    lines = []
    for name in names:
        factor = float(factors.get(name, 1.0))
        truth_scale = 1.0 / factor
        lo, hi = bounds[name]
        margin = min(truth_scale - lo, hi - truth_scale)
        ok = "ok" if margin > 1e-6 else "UNREACHABLE"
        lines.append(f"    {name:<12} degrade x{factor:<5.2f} -> truth at scale "
                     f"{truth_scale:5.3f}  bounds [{lo:g}, {hi:g}]  margin {margin:+.3f}  {ok}")
    return lines


# ---------------------------------------------------------------------------
# EXPERIMENT A -- recovery against a known truth
# ---------------------------------------------------------------------------
def experiment_recovery(config: dict, config_path: str, *, path: Path, factors: dict,
                        bounds: dict, windows: int, first_row: int, row_step: int,
                        window_stride: int, noise_c: float, seed: int) -> list[dict]:
    truth_material = config.get("fem_forecast", {}).get("material", {})
    bad_material = degrade(truth_material, factors)

    # One server holds the truth and manufactures the observations; the other is
    # the one under test and never sees the true base at all.
    truth_server = make_inversion_server(config, config_path,
                                         base_params=truth_material, bounds=bounds)
    names = list(truth_server.invert_params)

    print("\n=== A. Recovery against a known truth (synthetic twin) ===")
    print("  reachability of the truth from the degraded base:")
    for line in check_reachable(factors, bounds, names):
        print(line)

    results = []
    for w in range(windows):
        start_row = first_row + w * row_step
        # A fresh server per window: a warm start would carry the previous
        # window's answer forward and make recovery look easier than it is.
        server = make_inversion_server(config, config_path,
                                       base_params=bad_material, bounds=bounds)
        try:
            snapshots = _sensor_rows(path, None, start_row,
                                     window_stride * (server.window_size - 1) + 1,
                                     window_stride, {DEFAULT_EXCLUDE})
            if len(snapshots) < server.window_size:
                print(f"  window {w + 1}: not enough rows at start {start_row}; stopping")
                break
            for snap in snapshots:
                server._upsert_snapshot(snap)
            window = server._build_observation_window()
            if window is None:
                print(f"  window {w + 1}: could not assemble a window; skipping")
                continue
            times_hours, positions, _obs, ic_sensors = window
        except SystemExit as exc:
            print(f"  window {w + 1}: {exc}")
            break

        # Manufacture what the TRUE parameters would have produced.
        truth = {name: float(truth_material[name]) for name in names}
        full_truth = deepcopy(truth_server.base_params)
        full_truth.update(truth)
        obs = truth_server._generate_series(full_truth, positions, times_hours, ic_sensors)
        if noise_c > 0.0:
            rng = np.random.default_rng(seed + w)
            obs = {sid: [v + float(rng.normal(0.0, noise_c)) for v in vals]
                   for sid, vals in obs.items()}

        server._last_best_scales = None
        fitted, info = server._run_inversion(times_hours, positions, obs, ic_sensors)

        # times_hours is relative (relative_time_window), so it starts at 0 for
        # every window and cannot distinguish them; report where the window
        # actually sits in the experiment.
        abs_start = float(snapshots[0]["time_hours"])
        abs_end = float(snapshots[-1]["time_hours"])
        row = {"window": w + 1, "start_row": start_row,
               "t_start_h": abs_start, "t_end_h": abs_end,
               "final_loss": float(info.get("final_loss", float("nan"))),
               "eval_count": int(info.get("eval_count", 0))}
        print(f"  window {w + 1}  t = {abs_start:.2f}..{abs_end:.2f} h  "
              f"loss {row['final_loss']:.3e}  {row['eval_count']} evals")
        for name in names:
            t_val = float(truth[name])
            start_val = float(bad_material[name])
            got = float(fitted[name])
            err0 = abs(start_val - t_val) / abs(t_val) * 100.0
            err1 = abs(got - t_val) / abs(t_val) * 100.0
            lo, hi = bounds[name]
            scale = got / start_val
            at_bound = abs(scale - lo) < 1e-6 or abs(scale - hi) < 1e-6
            row.update({f"{name}_truth": t_val, f"{name}_start": start_val,
                        f"{name}_fitted": got, f"{name}_err_start_pct": err0,
                        f"{name}_err_end_pct": err1, f"{name}_at_bound": at_bound})
            print(f"      {name:<12} truth {t_val:9.3f}  start {start_val:9.3f} "
                  f"({err0:6.1f}% off)  ->  {got:9.3f} ({err1:5.1f}% off)"
                  f"{'  AT BOUND' if at_bound else ''}")
        results.append(row)
    return results


# ---------------------------------------------------------------------------
# EXPERIMENT B -- does it make the FEM better on real measurements
# ---------------------------------------------------------------------------
def sensor_rmse(solver, obs_x, obs_z, obs_t) -> float:
    node_xy, temps_c = _solver_field_c(solver)
    predicted = _interp_to_grid(node_xy[:, 0], node_xy[:, 1], temps_c,
                                np.abs(obs_x), obs_z)
    residual = predicted - obs_t
    finite = np.isfinite(residual)
    if not np.any(finite):
        return float("nan")
    return float(np.sqrt(np.mean(np.square(residual[finite]))))


def experiment_forecast(config: dict, config_path: str, *, path: Path, fluid_cols,
                        factors: dict, bounds: dict, stride: int, limit: int | None,
                        invert_every: int, recalibrate_data_s: float) -> list[dict]:
    rabbit_cfg = config.get("rabbitmq", {})
    influx_cfg: InfluxConfig = build_influx_config(config.get("influxdb", {}))
    truth_material = config.get("fem_forecast", {}).get("material", {})
    bad_material = degrade(truth_material, factors)

    def forecast_arm(material: dict):
        cfg = dict(config.get("fem_forecast", {}))
        cfg["material"] = material
        cfg["run_baseline"] = False
        # The predictive lookahead is a control-path feature and costs ~440 ms a
        # step; nothing here reads it.
        cfg["enable_lookahead"] = False
        server = create_fem_forecast_server(influx_cfg, rabbit_cfg, cfg)
        server.mq_out, server.influx = _Recorder(), _NullInflux()
        return server

    arms = {"reference": forecast_arm(truth_material),
            "degraded": forecast_arm(bad_material),
            "recovered": forecast_arm(bad_material),
            "recovered_bc": forecast_arm(bad_material)}

    builder = create_boundary_condition_builder_server(
        rabbit_cfg, config.get("boundary_condition_builder", {}))
    pipe = create_pipe_boundary_server(influx_cfg, rabbit_cfg, config.get("pipe_boundary", {}))
    pipe.mq_boundary_out = _Sink()
    inlet_col, outlet_col, time_col = fluid_cols
    fluid = _load_fluid_temps(path, inlet_col=inlet_col, outlet_col=outlet_col, time_col=time_col)

    # Two inversion services, identical but for the boundary their forward model
    # fits under. Separate instances because each carries its own warm start.
    inversion = make_inversion_server(config, config_path,
                                      base_params=bad_material, bounds=bounds)
    inversion_bc = make_inversion_server(config, config_path,
                                         base_params=bad_material, bounds=bounds)

    snapshots = _sensor_rows(path, None, 1, limit, stride, {DEFAULT_EXCLUDE})
    if not snapshots:
        raise SystemExit(f"No usable sensor rows parsed from {path}")
    cadence_s = ((float(snapshots[1]["time_hours"]) - float(snapshots[0]["time_hours"])) * 3600.0
                 if len(snapshots) > 1 else 30.0)
    recal_every = max(int(round(recalibrate_data_s / max(cadence_s, 1e-9))), 1)

    print(f"\n=== B. Forecast error on real measurements ===")
    print(f"  {len(snapshots)} snapshots at {cadence_s:.0f} s, pipe refit every "
          f"{recal_every}, inversion every {invert_every}")
    for name in inversion.invert_params:
        print(f"    {name:<12} reference {float(truth_material[name]):9.3f}   "
              f"degraded {float(bad_material[name]):9.3f}")

    rows: list[dict] = []
    override: dict | None = None
    override_bc: dict | None = None
    for index, snap in enumerate(snapshots):
        t_h = float(snap["time_hours"])
        inversion._upsert_snapshot(snap)
        inversion_bc._upsert_snapshot(snap)

        boundary = builder._build_snapshot(snap)
        t_arr, tin_arr, tout_arr = fluid
        pipe.recalibrate_seconds = 0.0 if index % recal_every == 0 else 1.0e12
        pipe._on_fluid({"time_hours": t_h, "timestamp": snap["timestamp"],
                        "t_inlet_c": float(np.interp(t_h, t_arr, tin_arr)),
                        "t_outlet_c": float(np.interp(t_h, t_arr, tout_arr)),
                        "site_id": "default"})
        pipe._on_boundary(boundary)
        if isinstance(pipe.mq_boundary_out.last, dict):
            boundary = pipe.mq_boundary_out.last

        # Fit on the window that has just closed, then hand the result to the
        # recovered arm -- the same path the live service uses.
        if index > 0 and index % invert_every == 0:
            window = inversion._build_observation_window()
            if window is not None:
                times_hours, positions, obs, ic_sensors = window
                fitted, info = inversion._run_inversion(times_hours, positions, obs, ic_sensors)
                override = {n: float(fitted[n]) for n in inversion.invert_params}

                # Same fit, but the forward model marches the boundary the
                # forecast is actually on. The window model builds lazily, so
                # force it into existence before pointing it anywhere.
                if inversion_bc._window_model is None:
                    from software.digital_twin.simulator.fem_inversion.window_forward_model import (
                        WindowForwardModel,
                    )
                    inversion_bc._window_model = WindowForwardModel(config_path)
                inversion_bc._window_model._snapshot = deepcopy(boundary)
                window_bc = inversion_bc._build_observation_window()
                if window_bc is not None:
                    tb, pb, ob, icb = window_bc
                    fitted_bc, info_bc = inversion_bc._run_inversion(tb, pb, ob, icb)
                    override_bc = {n: float(fitted_bc[n]) for n in inversion_bc.invert_params}
                else:
                    info_bc = {}
                print(f"  [{index:4d}/{len(snapshots)}] t={t_h:5.2f}h")
                print("        as-is    " + "  ".join(f"{n}={override[n]:.3f}" for n in override)
                      + f"  loss={float(info.get('final_loss', float('nan'))):.4f}")
                if override_bc is not None:
                    print("        bc-fixed " + "  ".join(f"{n}={override_bc[n]:.3f}"
                                                          for n in override_bc)
                          + f"  loss={float(info_bc.get('final_loss', float('nan'))):.4f}")

        obs_x = np.array([float(s["x_m"]) for s in snap["sensors"]], dtype=float)
        obs_z = np.array([float(s["z_m"]) for s in snap["sensors"]], dtype=float)
        obs_t = np.array([float(s["temperature"]) for s in snap["sensors"]], dtype=float)

        row = {"time_hours": round(t_h, 6)}
        for label, server in arms.items():
            msg = deepcopy(boundary)
            if label == "recovered" and override is not None:
                msg["material_override"] = dict(override)
            if label == "recovered_bc" and override_bc is not None:
                msg["material_override"] = dict(override_bc)
            server._process_message(msg)
            row[f"rmse_{label}_c"] = (sensor_rmse(server._solver, obs_x, obs_z, obs_t)
                                      if server._solver is not None else float("nan"))
        for name in inversion.invert_params:
            row[name] = None if override is None else float(override[name])
            row[f"{name}_bc"] = None if override_bc is None else float(override_bc[name])
        rows.append(row)

        if index % max(len(snapshots) // 10, 1) == 0:
            print(f"  [{index:4d}/{len(snapshots)}] t={t_h:5.2f}h  RMSE  "
                  f"ref {row['rmse_reference_c']:.3f}  deg {row['rmse_degraded_c']:.3f}  "
                  f"rec {row['rmse_recovered_c']:.3f}  rec+bc {row['rmse_recovered_bc_c']:.3f}")
    return rows


# ---------------------------------------------------------------------------
# FIGURES
# ---------------------------------------------------------------------------
def fig_recovery(results: list[dict], names: list[str], out_dir: Path):
    """Where each parameter started, where it landed, and where the truth is."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not results:
        print("  no recovery results; skipping fig_parameter_recovery")
        return

    fig, axes = plt.subplots(1, len(names), figsize=(DOUBLE, DOUBLE * 0.36))
    axes = np.atleast_1d(axes)

    for ax, name in zip(axes, names):
        truth = results[0][f"{name}_truth"]
        start = results[0][f"{name}_start"]
        x = np.arange(1, len(results) + 1)
        fitted = np.array([r[f"{name}_fitted"] for r in results], dtype=float)

        # Normalised to the truth so all three panels share one readable scale
        # and 1.0 means exactly right in every one of them.
        ax.axhline(1.0, color=INK, lw=1.4, zorder=4)
        ax.axhline(start / truth, color=STATUS_LIMIT, lw=1.3, ls="--", zorder=3)

        # An arrow per window, from the wrong start to where the fit landed: the
        # length of the arrow is the correction the inversion made.
        for xi, val in zip(x, fitted):
            ax.annotate("", xy=(xi, val / truth), xytext=(xi, start / truth),
                        arrowprops=dict(arrowstyle="-|>", color=FEM_KEEP,
                                        lw=1.2, shrinkA=0, shrinkB=0), zorder=5)
        ax.plot(x, fitted / truth, "o", ms=5, color=FEM_KEEP, mec="white", mew=0.9,
                zorder=6)

        # The residual error, stated rather than left to be read off the axis.
        mean_err = float(np.mean([r[f"{name}_err_end_pct"] for r in results]))
        ax.set_title(f"{PRETTY.get(name, name)}\nresidual {mean_err:.1f}%",
                     fontsize=7.4, pad=4)
        ax.set_xlabel("fitting window")
        ax.set_xticks(x)
        ax.set_xlim(0.4, len(results) + 0.6)
        lo = min(0.92, float(np.min(fitted / truth)) * 0.97, start / truth * 0.97)
        hi = max(1.08, float(np.max(fitted / truth)) * 1.03, start / truth * 1.03)
        ax.set_ylim(lo, hi)
        grid(ax)

    axes[0].set_ylabel("value / truth")

    # One explicit legend for the whole figure. Collecting handles from the axes
    # instead would repeat every entry once per panel, and in-panel text labels
    # for the two reference lines land on the markers at either end.
    handles = [
        Line2D([], [], color=STATUS_LIMIT, ls="--", lw=1.3, label="degraded start"),
        Line2D([], [], color=FEM_KEEP, marker="o", ls="-", lw=1.2, ms=5,
               label="after inversion"),
        Line2D([], [], color=INK, lw=1.4, label="truth"),
    ]
    fig.subplots_adjust(bottom=0.26)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.115),
               ncol=3, frameon=False, fontsize=7.2)
    save(fig, out_dir, "fig_parameter_recovery")


def fig_forecast(rows: list[dict], out_dir: Path):
    """Did closing the parameter error close the forecast error?"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if not rows:
        print("  no forecast rows; skipping fig_forecast_margin")
        return

    t = np.array([float(r["time_hours"]) for r in rows])

    def series(key):
        return np.array([float(r[key]) if r.get(key) is not None else np.nan for r in rows])

    ref = series("rmse_reference_c")
    deg = series("rmse_degraded_c")
    rec = series("rmse_recovered_c")
    recbc = series("rmse_recovered_bc_c")

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(DOUBLE, DOUBLE * 0.56), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.14})

    # The band between the two frozen arms is the room the inversion has to work
    # in. It only opens once the parameters start to matter: both arms are
    # seeded from the same measured field, so early on they agree by
    # construction and the band is meaningless there.
    ax.fill_between(t, ref, deg, color=STATUS_LIMIT, alpha=0.10, lw=0, zorder=1,
                    label="margin available")
    ax.plot(t, deg, "--", color=STATUS_LIMIT, lw=1.4, zorder=3, label="degraded, frozen")
    ax.plot(t, rec, "-", color=FEM_KEEP, lw=1.6, zorder=5, label="+ inversion (as shipped)")
    ax.plot(t, recbc, "-", color=MONITORED, lw=1.6, zorder=6, label="+ inversion (matched BC)")
    ax.plot(t, ref, "-.", color=FEM_STOP, lw=1.4, zorder=4, label="reference, frozen")
    ax.set_ylabel("RMSE at thermocouples (°C)")
    # Not anchored at zero. The four arms differ by ~0.1-0.4 °C on curves that
    # reach 3 °C, so a zero-anchored axis spends four fifths of its height on
    # agreement and compresses the entire subject of the figure into a ribbon.
    # The axis is labelled, so the absolute level is not lost by starting high.
    finite = np.concatenate([a[np.isfinite(a)] for a in (ref, deg, rec, recbc)])
    lo, hi = float(finite.min()), float(finite.max())
    ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.06 * (hi - lo))
    grid(ax)

    # Fraction of the available margin closed. Undefined wherever the margin has
    # not opened (the frozen arms within noise of each other, or the degraded one
    # accidentally ahead), and plotting a ratio through those points would
    # manufacture spikes out of a near-zero denominator -- so they are masked and
    # the masked stretch is shaded rather than silently left blank.
    span = deg - ref
    usable = span > 0.05
    for arm, colour, label in ((rec, FEM_KEEP, "as shipped"),
                               (recbc, MONITORED, "matched BC")):
        closed = np.full_like(span, np.nan)
        closed[usable] = (deg[usable] - arm[usable]) / span[usable] * 100.0
        ax2.plot(t, closed, "-", color=colour, lw=1.5, zorder=4, label=label)
    lo2, hi2 = -40.0, 120.0
    if np.any(~usable):
        ax2.fill_between(t, lo2, hi2, where=~usable, color=MUTED, alpha=0.10,
                         lw=0, zorder=0)
        first_gap = int(np.argmax(~usable))
        if not usable[first_gap]:
            ax2.annotate("margin < 0.05 °C", xy=(t[first_gap], lo2 + 6),
                         xytext=(2, 0), textcoords="offset points",
                         fontsize=6.3, color=MUTED, ha="left", va="bottom")

    ax2.axhline(0, color=MUTED, lw=0.8, zorder=2)
    ax2.axhline(100, color=FEM_STOP, lw=0.9, ls="-.", zorder=2)
    ax2.annotate("fully recovered", xy=(0.99, 100), xycoords=("axes fraction", "data"),
                 xytext=(0, 3), textcoords="offset points", fontsize=6.6,
                 color=FEM_STOP, ha="right")
    ax2.set_ylabel("margin closed (%)")
    ax2.set_xlabel("time (h)")
    ax2.set_xlim(t[0], t[-1])
    # Was (-100, 130) for data spanning roughly -20 to +70: a third of the panel
    # was empty, which flattened the very separation between the two arms that
    # the panel exists to show.
    ax2.set_ylim(lo2, hi2)
    grid(ax2)

    # (a) / (b), so the caption can name a panel instead of describing it.
    for letter, axis in (("a", ax), ("b", ax2)):
        axis.annotate(f"({letter})", xy=(0.006, 0.96), xycoords="axes fraction",
                      fontsize=8, fontweight="bold", va="top", ha="left", color=INK)

    # One key for the whole figure, below it. The previous version carried a
    # five-entry legend stretched across the top of panel (a) plus a second,
    # redundant two-entry legend inside panel (b) -- the same two colours keyed
    # twice, and the widest block of ink in the figure sitting where the curves
    # are furthest apart.
    handles = [
        Line2D([], [], color=STATUS_LIMIT, ls="--", lw=1.4, label="degraded, frozen"),
        Line2D([], [], color=FEM_KEEP, lw=1.6, label="+ inversion (as shipped)"),
        Line2D([], [], color=MONITORED, lw=1.6, label="+ inversion (matched BC)"),
        Line2D([], [], color=FEM_STOP, ls="-.", lw=1.4, label="reference, frozen"),
        Patch(facecolor=STATUS_LIMIT, alpha=0.10, edgecolor="none", label="margin available"),
    ]
    fig.subplots_adjust(bottom=0.17)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               ncol=len(handles), frameon=False, fontsize=6.8, columnspacing=1.9)

    save(fig, out_dir, "fig_forecast_margin")


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", default=DEFAULT_PATH)
    parser.add_argument("--config", default="startup.conf")
    parser.add_argument("--out", default="artifacts/inversion_margin")
    parser.add_argument("--degrade", default=None,
                        help='JSON of per-parameter factors, e.g. \'{"cap_s":1.6}\'. '
                             "Defaults to a near-doubling / halving of each.")
    parser.add_argument("--bound-lo", type=float, default=0.2,
                        help="Relative lower bound applied to every inverted parameter. "
                             "Must be below 1/degrade for the truth to be reachable.")
    parser.add_argument("--bound-hi", type=float, default=5.0)
    parser.add_argument("--windows", type=int, default=5, help="Recovery windows (experiment A).")
    parser.add_argument("--first-row", type=int, default=60)
    parser.add_argument("--row-step", type=int, default=90)
    parser.add_argument("--window-stride", type=int, default=20)
    parser.add_argument("--noise-c", type=float, default=0.1,
                        help="Gaussian noise on the synthetic observations (C). The default "
                             "is roughly thermocouple resolution; 0 is the clean correctness "
                             "gate, where recovery must be exact and is.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stride", type=int, default=4, help="Replay stride (experiment B).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--invert-every", type=int, default=10)
    parser.add_argument("--pipe-recalibrate-data-seconds", type=float, default=300.0)
    parser.add_argument("--skip-recovery", action="store_true")
    parser.add_argument("--replot", action="store_true",
                        help="Re-render both figures from the CSVs a previous run "
                             "wrote, instead of repeating the experiments.")
    parser.add_argument("--skip-forecast", action="store_true")
    parser.add_argument("--fluid-inlet-col", default="Inlet")
    parser.add_argument("--fluid-outlet-col", default="Outlet")
    parser.add_argument("--fluid-time-col", default="Time (h).1")
    args = parser.parse_args()

    config = load_startup_config(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = Path(args.path)

    factors = dict(DEFAULT_DEGRADE)
    if args.degrade:
        factors.update(json.loads(args.degrade))

    names = list(config.get("fem_inversion", {}).get("invert_params", []))
    bounds = {n: [args.bound_lo, args.bound_hi] for n in names}

    style()

    if args.replot:
        # Both experiments are expensive and both write everything the figures
        # need, so a layout change should not cost a re-run.
        def _load(name):
            path = out_dir / name
            if not path.exists():
                return []
            with path.open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            out = []
            for row in rows:
                clean = {}
                for k, v in row.items():
                    if v in ("", "None"):
                        clean[k] = None
                    elif v in ("True", "False"):
                        clean[k] = v == "True"
                    else:
                        try:
                            clean[k] = float(v)
                        except ValueError:
                            clean[k] = v
                out.append(clean)
            return out

        rec_rows = _load("recovery.csv")
        if rec_rows:
            print(f"  replotting {len(rec_rows)} recovery windows")
            fig_recovery(rec_rows, names, out_dir)
        fc_rows = _load("forecast_margin.csv")
        if fc_rows:
            print(f"  replotting {len(fc_rows)} forecast updates")
            fig_forecast(fc_rows, out_dir)
        return

    recovery: list[dict] = []
    if not args.skip_recovery:
        recovery = experiment_recovery(
            config, args.config, path=path, factors=factors, bounds=bounds,
            windows=args.windows, first_row=args.first_row, row_step=args.row_step,
            window_stride=args.window_stride, noise_c=args.noise_c, seed=args.seed)
        if recovery:
            with (out_dir / "recovery.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(recovery[0].keys()))
                writer.writeheader()
                writer.writerows(recovery)
            print(f"  wrote {out_dir / 'recovery.csv'}")
            for name in names:
                e0 = np.mean([r[f"{name}_err_start_pct"] for r in recovery])
                e1 = np.mean([r[f"{name}_err_end_pct"] for r in recovery])
                print(f"  {name:<12} mean error {e0:6.1f}%  ->  {e1:5.1f}%")
            fig_recovery(recovery, names, out_dir)

    if not args.skip_forecast:
        rows = experiment_forecast(
            config, args.config, path=path,
            fluid_cols=(args.fluid_inlet_col, args.fluid_outlet_col, args.fluid_time_col),
            factors=factors, bounds=bounds, stride=args.stride, limit=args.limit,
            invert_every=args.invert_every,
            recalibrate_data_s=args.pipe_recalibrate_data_seconds)
        with (out_dir / "forecast_margin.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  wrote {out_dir / 'forecast_margin.csv'}")

        # Summarise over the second half only: the arms share a seeded initial
        # field, so early on they agree by construction and averaging from t=0
        # would dilute the effect with a stretch where no effect is possible.
        half = len(rows) // 2
        ref = np.nanmean([r["rmse_reference_c"] for r in rows[half:]])
        deg = np.nanmean([r["rmse_degraded_c"] for r in rows[half:]])
        rec = np.nanmean([r["rmse_recovered_c"] for r in rows[half:]])
        recbc = np.nanmean([r["rmse_recovered_bc_c"] for r in rows[half:]])
        span = deg - ref
        print(f"\n  second-half mean RMSE (°C):")
        print(f"    reference, frozen          {ref:.3f}")
        print(f"    degraded, frozen           {deg:.3f}")
        print(f"    + inversion (as shipped)   {rec:.3f}")
        print(f"    + inversion (matched BC)   {recbc:.3f}")
        if span > 1e-9:
            print(f"    margin available           {span:.3f}")
            print(f"    closed, as shipped         {(deg - rec) / span * 100.0:.0f}%")
            print(f"    closed, matched BC         {(deg - recbc) / span * 100.0:.0f}%")
        else:
            print("    margin did not open over this window; no ratio reported.")
        fig_forecast(rows, out_dir)


if __name__ == "__main__":
    main()

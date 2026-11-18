"""Inversion recovery on the April 8 data: two figures, one honest setup.

The question the figures answer: started from wrong soil parameters, does the
running inversion service pull them back, and does that make the FEM better?

Why earlier attempts showed nothing. The "good" parameters they degraded from
were the literature values in ``fem_forecast.material``. Those are not what the
April 8 measurements support -- every real-data fit leaves them in the same
direction (``cap_s`` up, ``k_parameter`` and ``lambda_s`` down) -- and the chosen
degradation (``cap_s`` x1.6, ``k`` x0.4, ``lambda_s`` x0.5) happened to point the
same way. The "degraded" FEM was therefore *closer* to the data than the
reference (sensor RMSE 0.40 vs 0.54 C at t=0), leaving the inversion no gap to
close. Recovery toward a reference the data does not prefer cannot be shown.

What this tool does instead, in three stages:

1. **calibrate** -- one offline fit over long windows spread across the whole
   recording, under the pipe-calibrated boundary as it evolves in time. The
   result ``p*`` is the parameter set the data actually supports, and it is the
   reference the rest of the experiment is measured against.
2. **replay** -- the real inversion service (its ``_run_inversion``, warm start
   and all) runs over the recording, starting from ``p*`` deliberately degraded
   *against* the direction the data pulls, with bounds wide enough that ``p*`` is
   interior. Its trajectory is figure 1.
3. **skill** -- at regular times through the run, the FEM is seeded from the
   measured field and marched ``--horizon`` hours ahead under the measured
   boundary, with three parameter sets: ``p*``, the degraded start, and the
   inversion's latest estimate *as of that time* (no look-ahead). RMSE against
   the thermocouples is figure 2. A short rolling horizon keeps the comparison
   about material: a single run-long march accumulates boundary/model drift
   that no material set can remove, which buries the inversion's effect.

Two structural errors had to be removed before any parameter could converge,
both found by this tool's first run (kept in ``run1_deep_bc_minus5/``):

* **Deep edge.** The configured ``deep_frozen`` Dirichlet edge is -5 C; the
  sensors beside it read -3.5 C warming to -0.5 C. The model pulled that row to
  -5 C within a few steps (1-step RMSE already 0.6 C, worse than persistence over
  a full hour) and ``cap_s`` ran to whatever upper bound it was given, because
  "never change temperature" was the best available answer. ``--deep-sensor``
  drives the edge from a measured sensor (the builder's ``sensor_overrides``
  mechanism); with it the 1 h error early in the run drops from 0.98 to 0.33 C.
* **Pipe wall.** With the deep edge fixed, the 59 sensors beyond 2 cm from the
  axis forecast to ~0.1-0.3 C, while the 12 at the pipe wall (D0/D15) run 1.5-3 C
  too warm under the pipe Robin boundary. That is a boundary error no soil
  parameter can absorb, and it dominated the loss. ``--min-radius`` leaves them
  out of the fit and the score; they still seed the initial field.

The forward model here marches the boundary snapshot of every step, rather
than holding one snapshot for a whole window as the service's
``WindowForwardModel`` does; the replay stage drives the service's own window
model so figure 1 shows exactly what the service would publish.

Outputs under ``--out`` (default ``artifacts/inversion_recovery``):

    calibration.json             p* and how it was obtained
    inversion_updates.csv        one row per service fit (figure 1 data)
    forecast_skill.csv           one row per skill time (figure 2 data)
    fig_parameter_evolution.png/.pdf
    fig_fem_impact.png/.pdf

Run:

    python -m software.tools.inversion_recovery                 # all stages
    python -m software.tools.inversion_recovery --stage plot    # replot from CSVs
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from software.digital_twin.data_access.influx_utils import InfluxConfig
from software.digital_twin.simulator.fem_inversion.window_forward_model import WindowForwardModel
from software.startup.service_runners import (
    build_influx_config,
    create_boundary_condition_builder_server,
    create_pipe_boundary_server,
)
from software.startup.utils.config import load_startup_config
from software.tools.fem_debug_replay import _Sink, _sensor_rows
from software.tools.replay_csv import _load_fluid_temps

DEFAULT_PATH = "data/CR1000_Table1_April8_replay_Final.xlsx"
DEFAULT_EXCLUDE = "H10D75"
NAMES = ("cap_s", "k_parameter", "lambda_s")


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
def load_snapshots(path: Path) -> list[dict]:
    snapshots = _sensor_rows(path, None, 1, None, 1, {DEFAULT_EXCLUDE})
    if len(snapshots) < 2:
        raise SystemExit(f"No usable sensor rows parsed from {path}")
    return snapshots


def boundary_track(config: dict, snapshots: list[dict], path: Path, *,
                   fluid_cols=("Inlet", "Outlet", "Time (h).1"),
                   recalibrate_data_s: float = 300.0) -> list[dict]:
    """The pipe-calibrated boundary in force at every snapshot, in one pass.

    Same builder + pipe-boundary services, same refit cadence, as the live chain.
    """
    rabbit_cfg = config.get("rabbitmq", {})
    influx_cfg: InfluxConfig = build_influx_config(config.get("influxdb", {}))
    builder = create_boundary_condition_builder_server(
        rabbit_cfg, config.get("boundary_condition_builder", {}))
    pipe = create_pipe_boundary_server(influx_cfg, rabbit_cfg, config.get("pipe_boundary", {}))
    pipe.mq_boundary_out = _Sink()
    inlet_col, outlet_col, time_col = fluid_cols
    t_arr, tin_arr, tout_arr = _load_fluid_temps(path, inlet_col=inlet_col,
                                                 outlet_col=outlet_col, time_col=time_col)
    cadence_s = (float(snapshots[1]["time_hours"]) - float(snapshots[0]["time_hours"])) * 3600.0
    recal_every = max(int(round(recalibrate_data_s / max(cadence_s, 1e-9))), 1)

    track = []
    for index, snap in enumerate(snapshots):
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
        track.append(deepcopy(boundary))
    return track


def cached_boundary_track(config: dict, snapshots: list[dict], path: Path,
                          cache: Path) -> list[dict]:
    """``boundary_track``, computed once per data file.

    The pipe-boundary refit is the slow part of the whole chain (~15 s each, and
    a pipe simulation per fluid message), and it depends only on the data, not
    on anything this tool varies, so it is pickled next to the outputs.
    """
    stamp = (str(path.resolve()), path.stat().st_mtime, len(snapshots))
    if cache.exists():
        with cache.open("rb") as fh:
            saved = pickle.load(fh)
        if saved.get("stamp") == stamp:
            return saved["track"]
    start = time.perf_counter()
    print(f"building pipe-calibrated boundary track for {len(snapshots)} snapshots "
          f"(one-off, cached to {cache}) ...", flush=True)
    track = boundary_track(config, snapshots, path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as fh:
        pickle.dump({"stamp": stamp, "track": track}, fh)
    print(f"  done in {time.perf_counter() - start:.0f}s", flush=True)
    return track


def drive_deep_edge(boundaries: list[dict], snapshots: list[dict], *, boundary_set: str,
                    sensor_id: str) -> list[dict]:
    """Set the deep Dirichlet edge from a measured sensor, as ``sensor_overrides`` does.

    The configured ``deep_frozen`` edge is held at -5 C, but on April 8 the row
    beside it (z = 0.29-0.30 m) starts at -3.5 C and warms to about -0.5 C. The
    model drags that row to -5 C within a few steps; the only parameter that
    can resist is ``cap_s``, which the fit then drives to whatever bound it is
    given. Driving the edge from the measurement removes that compensation.
    """
    out = []
    for snap, boundary in zip(snapshots, boundaries):
        temps = {str(x["sensor_id"]): float(x["temperature"]) for x in snap["sensors"]}
        boundary = deepcopy(boundary)
        if sensor_id in temps:
            for bc in boundary.get("boundaries", []):
                if bc.get("boundary_set") == boundary_set and bc.get("bc_type") == "dirichlet":
                    bc["temperature"] = temps[sensor_id]
        out.append(boundary)
    return out


# ---------------------------------------------------------------------------
# FORWARD MODEL WITH A TIME-VARYING BOUNDARY
# ---------------------------------------------------------------------------
class Marcher:
    """Seed from snapshot ``i0``, march snapshot by snapshot, sample the sensors.

    One solver step per snapshot interval (the data cadence equals the solver's
    30 s step), each under that interval's own boundary. Reuses the window
    model's solver/mesh/nearest-node caching so a candidate costs only the march.
    """

    def __init__(self, config_path: str, snapshots: list[dict], boundaries: list[dict],
                 base_material: dict, min_radius_m: float = 0.0) -> None:
        self.wm = WindowForwardModel(config_path)
        self.snapshots = snapshots
        self.boundaries = boundaries
        self.base = dict(base_material)
        ids = None
        for s in snapshots:
            got = {str(x["sensor_id"]) for x in s["sensors"]}
            ids = got if ids is None else ids & got
        self.ids = sorted(ids or [])
        first = {str(x["sensor_id"]): x for x in snapshots[0]["sensors"]}
        self.positions = [{"sensor_id": sid, "x_m": float(first[sid]["x_m"]),
                           "z_m": float(first[sid]["z_m"])} for sid in self.ids]
        self.obs = np.array([[{str(x["sensor_id"]): float(x["temperature"])
                               for x in s["sensors"]}[sid] for sid in self.ids]
                             for s in snapshots], dtype=float)
        self.times = np.array([float(s["time_hours"]) for s in snapshots])
        # Scored sensors: the same pipe-wall exclusion the service applies.
        self.score = np.array([abs(p["x_m"]) >= min_radius_m for p in self.positions])
        self.n_evals = 0

    def predict(self, params: dict, i0: int, i1: int, every: int = 1):
        """Predicted sensor temps at snapshots i0+every, i0+2*every, ... <= i1."""
        merged = dict(self.base)
        merged.update(params)
        wm = self.wm
        solver = wm._ensure_solver(merged)
        # Force a re-seed each call: the IC cache keys on sensors, which is fine,
        # but i0 changes between calls so the signature changes with it.
        solver.T = wm._ensure_initial_field(solver, self.snapshots[i0]["sensors"]).copy()
        nearest = wm._ensure_nearest(solver, self.positions)
        offset = float(solver.solver.temperature_offset_k)
        dt_h = wm.dt_seconds / 3600.0
        rows, idx = [], []
        for i in range(i0 + 1, i1 + 1):
            steps = max(1, int(round((self.times[i] - self.times[i - 1]) / dt_h)))
            for _ in range(steps):
                solver._solve_step(self.boundaries[i - 1])
            if (i - i0) % every == 0:
                rows.append(solver.T[nearest] - offset)
                idx.append(i)
        self.n_evals += 1
        return np.array(idx, dtype=int), np.array(rows, dtype=float)

    def mse(self, params: dict, i0: int, i1: int, every: int = 1) -> float:
        idx, pred = self.predict(params, i0, i1, every)
        return float(np.mean((pred - self.obs[idx])[:, self.score] ** 2))

    def rmse(self, params: dict, i0: int, i1: int, every: int = 1) -> float:
        return float(np.sqrt(self.mse(params, i0, i1, every)))

    def trajectory(self, schedule: list[tuple[int, dict]]) -> np.ndarray:
        """Scored-sensor RMSE at every snapshot of ONE continuous run.

        Seeded once from snapshot 0, never re-seeded -- how the forecast twin
        actually runs. ``schedule`` is ``(index, params)`` sorted by index: from
        that snapshot on, the run continues on those parameters, the field
        carried over unchanged, as when a ``material_override`` arrives.
        """
        wm = self.wm
        merged = dict(self.base)
        merged.update(schedule[0][1])
        solver = wm._ensure_solver(merged)
        solver.T = wm._ensure_initial_field(solver, self.snapshots[0]["sensors"]).copy()
        nearest = wm._ensure_nearest(solver, self.positions)
        offset = float(solver.solver.temperature_offset_k)
        dt_h = wm.dt_seconds / 3600.0
        switch = {int(i): p for i, p in schedule[1:]}
        out = np.full(len(self.snapshots), np.nan)
        err0 = (solver.T[nearest] - offset - self.obs[0])[self.score]
        out[0] = float(np.sqrt(np.mean(err0 ** 2)))
        for i in range(1, len(self.snapshots)):
            if i - 1 in switch:
                field = solver.T.copy()
                merged = dict(self.base)
                merged.update(switch[i - 1])
                solver = wm._ensure_solver(merged)
                solver.T = field
            steps = max(1, int(round((self.times[i] - self.times[i - 1]) / dt_h)))
            for _ in range(steps):
                solver._solve_step(self.boundaries[i - 1])
            err = (solver.T[nearest] - offset - self.obs[i])[self.score]
            out[i] = float(np.sqrt(np.mean(err ** 2)))
        return out


# ---------------------------------------------------------------------------
# STAGE 1 -- CALIBRATE p*
# ---------------------------------------------------------------------------
def window_starts(marcher: Marcher, *, first_h: float, length: int, count: int) -> list[int]:
    """``count`` window starts spread evenly from ``first_h`` to the last full window."""
    i_first = int(np.searchsorted(marcher.times, first_h))
    i_last = len(marcher.times) - 1 - length
    if i_last < i_first:
        raise SystemExit("Recording too short for the requested calibration windows.")
    return [int(round(v)) for v in np.linspace(i_first, i_last, count)]


def calibrate(marcher: Marcher, literature: dict, *, starts: list[int], length: int,
              every: int, bounds: dict, maxfun: int, prior_weight: float = 0.0,
              prior_names: tuple[str, ...] = ()) -> dict:
    """Joint fit over several long windows; log-scale L-BFGS-B from the literature set.

    With ``prior_weight`` > 0 this is a MAP estimate: the data misfit plus
    ``prior_weight * sum(ln(p / literature)^2)`` over ``prior_names`` -- the same
    penalty the service applies, so ``p*`` is the point the service should reach.
    """
    prior_mask = np.array([n in prior_names for n in NAMES], dtype=float)
    names = list(NAMES)
    lo = np.log([bounds[n][0] for n in names])
    hi = np.log([bounds[n][1] for n in names])
    evals = [0]

    def params_of(x):
        return {n: literature[n] * float(np.exp(v)) for n, v in zip(names, x)}

    def objective(x):
        p = params_of(x)
        loss = float(np.mean([marcher.mse(p, i0, i0 + length, every) for i0 in starts]))
        loss += prior_weight * float(np.sum(prior_mask * np.asarray(x) ** 2))
        evals[0] += 1
        print(f"    eval {evals[0]:3d}  loss {loss:.5f}  scales "
              + " ".join(f"{n}={np.exp(v):.3f}" for n, v in zip(names, x)), flush=True)
        return loss

    start = time.perf_counter()
    loss_lit = objective(np.zeros(len(names)))
    # eps on log-scale: a 0.1 % relative step. The default 1e-8 is fine for a
    # smooth model but buys nothing here and is below the solver's own noise.
    result = minimize(objective, x0=np.zeros(len(names)), method="L-BFGS-B",
                      bounds=list(zip(lo, hi)),
                      options={"maxfun": maxfun, "maxiter": maxfun, "eps": 1e-3})
    at_bound = {n: bool(abs(v - lo[i]) < 1e-3 or abs(v - hi[i]) < 1e-3)
                for i, (n, v) in enumerate(zip(names, result.x))}
    return {
        "p_star": params_of(result.x),
        "literature": {n: float(literature[n]) for n in names},
        "scales_vs_literature": {n: float(np.exp(v)) for n, v in zip(names, result.x)},
        "at_bound": at_bound,
        "loss_literature": loss_lit,
        "loss_p_star": float(result.fun),
        "rmse_literature_c": float(np.sqrt(loss_lit)),
        "rmse_p_star_c": float(np.sqrt(result.fun)),
        "window_starts_h": [float(marcher.times[i]) for i in starts],
        "window_hours": float(marcher.times[starts[0] + length] - marcher.times[starts[0]]),
        "sample_every": every,
        "bounds_vs_literature": {n: list(bounds[n]) for n in names},
        "prior_weight": prior_weight,
        "prior_names": list(prior_names),
        "evals": evals[0],
        "message": str(result.message),
        "elapsed_s": time.perf_counter() - start,
    }


def degrade_away(p_star: dict, literature: dict, magnitude: float) -> dict[str, float]:
    """Degrade factors that push each parameter *against* the data's pull.

    The data pulls each parameter from its literature value toward ``p*``. A
    degradation along that pull lands near the data optimum and leaves nothing
    to recover -- which is what sank the earlier attempt. So each parameter is
    moved by ``magnitude`` the other way; where the pull is negligible the
    direction is arbitrary and the parameter is raised.
    """
    factors = {}
    for n in NAMES:
        pull = float(np.log(p_star[n] / literature[n]))
        factors[n] = 1.0 / magnitude if pull > 0.05 else magnitude
    return factors


# ---------------------------------------------------------------------------
# STAGE 2 -- REPLAY THE INVERSION SERVICE FROM THE DEGRADED START
# ---------------------------------------------------------------------------
def make_service(config: dict, config_path: str, *, base: dict, bound: float,
                 window_size: int, window_stride: int, maxfun: int, min_radius_m: float,
                 prior: dict | None = None, prior_weight: float = 0.0):
    """The real inversion service, clients stubbed, configured for this replay."""
    from software.digital_twin.simulator.fem_inversion.fem_inversion_server import (
        FEMInversionServer,
    )
    inv_cfg = config.get("fem_inversion", {})
    material = dict(config.get("fem_forecast", {}).get("material", {}))
    material.update(base)
    return FEMInversionServer(
        invert_params=list(NAMES),
        base_params=material,
        relative_bounds={n: [1.0 / bound, bound] for n in NAMES},
        loss=str(inv_cfg.get("loss", "mse")),
        window_size=window_size,
        window_stride=window_stride,
        min_snapshots=window_size,
        min_sensors=int(inv_cfg.get("min_sensors", 3)),
        min_sensor_radius_m=min_radius_m,
        warmup_hours=0.0,
        relative_time_window=True,
        maxiter=maxfun,
        maxfun=maxfun,
        scale_space="log",
        prior_params=prior,
        prior_weight=prior_weight,
        config_path=config_path,
    )


def replay(config: dict, config_path: str, snapshots: list[dict], boundaries: list[dict], *,
           degraded: dict, p_star: dict, bound: float, window_size: int, window_stride: int,
           invert_every: int, maxfun: int, min_radius_m: float, out_csv: Path,
           prior: dict | None = None, prior_weight: float = 0.0) -> list[dict]:
    import logging

    service = make_service(config, config_path, base=degraded, bound=bound,
                           window_size=window_size, window_stride=window_stride, maxfun=maxfun,
                           min_radius_m=min_radius_m, prior=prior, prior_weight=prior_weight)
    service.logger.setLevel(logging.WARNING)
    fields = (["time_hours", "index", "final_loss", "eval_count", "elapsed_s"]
              + list(NAMES) + [f"{n}_err_pct" for n in NAMES])
    rows: list[dict] = []
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        # The degraded start as a row at t=0, so the trajectory begins there.
        row0 = {"time_hours": float(snapshots[0]["time_hours"]), "index": 0,
                "final_loss": float("nan"), "eval_count": 0, "elapsed_s": 0.0}
        for n in NAMES:
            row0[n] = degraded[n]
            row0[f"{n}_err_pct"] = (degraded[n] / p_star[n] - 1.0) * 100.0
        writer.writerow(row0)
        rows.append(row0)

        for index, snap in enumerate(snapshots):
            service._upsert_snapshot(snap)
            # Every calibrated boundary, as the live stream would deliver it, so
            # the service can fit each window under the boundary history.
            service.set_boundary_snapshot(boundaries[index])
            if index == 0 or index % invert_every:
                continue
            window = service._build_observation_window()
            if window is None:
                continue
            schedule = service._boundary_schedule(
                float(service._window_snapshots()[0]["time_hours"]))
            start = time.perf_counter()
            fitted, info = service._run_inversion(*window, boundary_schedule=schedule)
            row = {"time_hours": float(snap["time_hours"]), "index": index,
                   "final_loss": float(info["final_loss"]), "eval_count": int(info["eval_count"]),
                   "elapsed_s": round(time.perf_counter() - start, 2)}
            for n in NAMES:
                row[n] = float(fitted[n])
                row[f"{n}_err_pct"] = (float(fitted[n]) / p_star[n] - 1.0) * 100.0
            writer.writerow(row)
            fh.flush()
            rows.append(row)
            print(f"  t={row['time_hours']:5.2f}h  loss {row['final_loss']:.4f}  "
                  f"{row['eval_count']:3d} evals {row['elapsed_s']:6.1f}s  "
                  + "  ".join(f"{n}={row[n]:.4g} ({row[n + '_err_pct']:+.0f}%)" for n in NAMES),
                  flush=True)
    return rows


# ---------------------------------------------------------------------------
# STAGE 2t -- SYNTHETIC TWIN (known truth)
# ---------------------------------------------------------------------------
def synthesize(marcher: Marcher, params: dict, *, noise_c: float, seed: int) -> list[dict]:
    """Snapshots the forward model would have recorded with ``params``.

    One continuous march from the first measured snapshot under the measured
    boundary history, sampled at every sensor, plus Gaussian noise. Geometry,
    sensor layout, initial field and forcing are all real; only the soil is
    known. The service then re-seeds each window from these (noisy, sparse)
    readings, exactly as it does from the real ones.
    """
    idx, pred = marcher.predict(params, 0, len(marcher.snapshots) - 1, 1)
    rng = np.random.default_rng(seed)
    pred = pred + rng.normal(0.0, noise_c, size=pred.shape) if noise_c > 0 else pred
    col = {sid: j for j, sid in enumerate(marcher.ids)}
    out = [deepcopy(marcher.snapshots[0])]
    for row, i in zip(pred, idx):
        snap = deepcopy(marcher.snapshots[int(i)])
        snap["sensors"] = [dict(x, temperature=float(row[col[str(x["sensor_id"])]]))
                           for x in snap["sensors"] if str(x["sensor_id"]) in col]
        out.append(snap)
    return out


# ---------------------------------------------------------------------------
# STAGE 2b -- RECURSIVE ESTIMATE (all evidence so far)
# ---------------------------------------------------------------------------
# Why this exists. On April 8 each 1 h window has its own optimum: the fit on
# the window ending at 1.0 h prefers lambda_s high and k low, the one ending at
# 4.0 h prefers both low, and on that 4 h window the *degraded* set beats p*.
# The residual model error is not stationary, so a single-window fit tracks
# that hour's error rather than the soil, and never settles -- which is what
# every single-window replay showed. The recursive estimate fits every 1 h
# window seen so far jointly, warm-started from the previous update, so it
# accumulates evidence and converges; with all the data in, it is the
# full-record calibration.
_WORKER: dict = {}


def _worker_init(config_path: str, path: str, track_pkl: str, deep_edge: str,
                 deep_sensor: str, min_radius: float) -> None:
    import logging
    logging.disable(logging.INFO)
    config = load_startup_config(config_path)
    snapshots = load_snapshots(Path(path))
    with open(track_pkl, "rb") as fh:
        boundaries = pickle.load(fh)["track"]
    if deep_sensor:
        boundaries = drive_deep_edge(boundaries, snapshots, boundary_set=deep_edge,
                                     sensor_id=deep_sensor)
    _WORKER["m"] = Marcher(config_path, snapshots, boundaries,
                           config["fem_forecast"]["material"], min_radius_m=min_radius)


def _worker_mse(job: tuple) -> float:
    params, i0, i1, every = job
    return _WORKER["m"].mse(params, i0, i1, every)


def recursive(pool, times: np.ndarray, *, degraded: dict, p_star: dict, prior: dict,
              prior_weight: float, bound: float, length: int, window_step: int, every: int,
              first_index: int, min_start: int, update_every: int, maxfun: int,
              out_csv: Path) -> list[dict]:
    names = list(NAMES)
    lo, hi = np.log(1.0 / bound), np.log(bound)
    fields = (["time_hours", "index", "final_loss", "eval_count", "elapsed_s", "n_windows"]
              + names + [f"{n}_err_pct" for n in names])
    rows: list[dict] = []
    x = np.zeros(len(names))
    last = len(times) - 1
    update_at = list(range(first_index, last + 1, update_every))
    if update_at[-1] != last:
        update_at.append(last)

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        row0 = {"time_hours": float(times[0]), "index": 0, "final_loss": float("nan"),
                "eval_count": 0, "elapsed_s": 0.0, "n_windows": 0}
        for n in names:
            row0[n] = degraded[n]
            row0[f"{n}_err_pct"] = (degraded[n] / p_star[n] - 1.0) * 100.0
        writer.writerow(row0)
        rows.append(row0)

        for idx in update_at:
            # Windows ending at the newest snapshot and every window_step before
            # it, never starting in the first quarter hour (uncalibrated pipe
            # boundary, seeded field still relaxing).
            starts = [s for s in range(idx - length, -1, -window_step) if s >= min_start]
            if not starts:
                continue
            evals = [0]

            def params_of(v):
                return {n: degraded[n] * float(np.exp(c)) for n, c in zip(names, v)}

            def objective(v):
                p = params_of(v)
                data = float(np.mean(list(pool.map(_worker_mse,
                                              [(p, s, s + length, every) for s in starts]))))
                penalty = prior_weight * sum(np.log(p[n] / prior[n]) ** 2 for n in prior)
                evals[0] += 1
                return data + penalty

            start = time.perf_counter()
            result = minimize(objective, x0=x, method="L-BFGS-B", bounds=[(lo, hi)] * len(names),
                              options={"maxfun": maxfun, "maxiter": maxfun, "eps": 1e-3})
            x = np.asarray(result.x, dtype=float)
            fitted = params_of(x)
            row = {"time_hours": float(times[idx]), "index": idx, "final_loss": float(result.fun),
                   "eval_count": evals[0], "elapsed_s": round(time.perf_counter() - start, 1),
                   "n_windows": len(starts)}
            for n in names:
                row[n] = fitted[n]
                row[f"{n}_err_pct"] = (fitted[n] / p_star[n] - 1.0) * 100.0
            writer.writerow(row)
            fh.flush()
            rows.append(row)
            print(f"  t={row['time_hours']:5.2f}h  {len(starts):2d} windows  loss {row['final_loss']:.4f}"
                  f"  {row['eval_count']:3d} evals {row['elapsed_s']:6.1f}s  "
                  + "  ".join(f"{n}={row[n]:.4g} ({row[n + '_err_pct']:+.0f}%)" for n in names),
                  flush=True)
    return rows


# ---------------------------------------------------------------------------
# STAGE 2h -- RECURSIVE HISTORY ESTIMATE (seeded once, never re-seeded)
# ---------------------------------------------------------------------------
# Why. Every windowed fit re-seeds its initial field by interpolating the
# sparse thermocouples. Once a thaw zone forms that interpolation misplaces the
# front, and latent heat makes the error last for hours. On the synthetic twin
# -- no model error at all -- the windowed objective is then NOT minimised at the
# truth (loss at truth 0.019-0.064 C^2 vs a 0.0025 noise floor; a 60 min spin-up
# fixes it at 2.5 h but not at 5 h), so the fit is pulled to the bounds. Seeding
# once at t = 0, while the field is frozen and smooth, and fitting the whole
# history in one continuous march removes that error entirely.
def _history_worker_init(config_path: str, path: str, track_pkl: str, deep_edge: str,
                         deep_sensor: str, min_radius: float, twin_params: dict | None,
                         twin_noise: float) -> None:
    _worker_init(config_path, path, track_pkl, deep_edge, deep_sensor, min_radius)
    if twin_params is not None:
        real = _WORKER["m"]
        synthetic = synthesize(real, twin_params, noise_c=twin_noise, seed=7)
        _WORKER["m"] = Marcher(config_path, synthetic, real.boundaries, real.base,
                               min_radius_m=min_radius)


def _history_mse(job: tuple) -> float:
    params, i1, every = job
    return _WORKER["m"].mse(params, 0, i1, every)


def history_estimate(pool, times: np.ndarray, *, degraded: dict, p_ref: dict, bound: float,
                     every: int, first_index: int, update_every: int, maxiter: int,
                     out_csv: Path, eps: float = 1e-3) -> list[dict]:
    names = list(NAMES)
    lo, hi = np.log(1.0 / bound), np.log(bound)
    fields = (["time_hours", "index", "final_loss", "eval_count", "elapsed_s"]
              + names + [f"{n}_err_pct" for n in names])
    last = len(times) - 1
    update_at = list(range(first_index, last + 1, update_every))
    if update_at[-1] != last:
        update_at.append(last)
    x = np.zeros(len(names))
    rows: list[dict] = []

    def params_of(v):
        return {n: degraded[n] * float(np.exp(c)) for n, c in zip(names, v)}

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        row0 = {"time_hours": float(times[0]), "index": 0, "final_loss": float("nan"),
                "eval_count": 0, "elapsed_s": 0.0}
        for n in names:
            row0[n] = degraded[n]
            row0[f"{n}_err_pct"] = (degraded[n] / p_ref[n] - 1.0) * 100.0
        writer.writerow(row0)
        rows.append(row0)

        for idx in update_at:
            evals = [0]

            def fun_and_grad(v):
                # f and the forward-difference gradient in one parallel round.
                # Steps go inward at an upper bound so no probe leaves the box.
                points = [np.array(v, dtype=float)]
                steps = []
                for j in range(len(names)):
                    h = eps if v[j] + eps <= hi else -eps
                    probe = np.array(v, dtype=float)
                    probe[j] += h
                    points.append(probe)
                    steps.append(h)
                values = list(pool.map(_history_mse,
                                       [(params_of(pt), idx, every) for pt in points]))
                evals[0] += len(points)
                f0 = float(values[0])
                grad = np.array([(float(values[j + 1]) - f0) / steps[j]
                                 for j in range(len(names))])
                return f0, grad

            start = time.perf_counter()
            result = minimize(fun_and_grad, x0=x, jac=True, method="L-BFGS-B",
                              bounds=[(lo, hi)] * len(names),
                              options={"maxiter": maxiter, "maxfun": maxiter * 2})
            x = np.asarray(result.x, dtype=float)
            fitted = params_of(x)
            row = {"time_hours": float(times[idx]), "index": idx, "final_loss": float(result.fun),
                   "eval_count": evals[0], "elapsed_s": round(time.perf_counter() - start, 1)}
            for n in names:
                row[n] = fitted[n]
                row[f"{n}_err_pct"] = (fitted[n] / p_ref[n] - 1.0) * 100.0
            writer.writerow(row)
            fh.flush()
            rows.append(row)
            print(f"  t={row['time_hours']:5.2f}h  loss {row['final_loss']:.5f}  "
                  f"{row['eval_count']:3d} evals {row['elapsed_s']:6.1f}s  "
                  + "  ".join(f"{n}={row[n]:.4g} ({row[n + '_err_pct']:+.0f}%)" for n in names),
                  flush=True)
    return rows


# ---------------------------------------------------------------------------
# STAGE 3 -- FORECAST SKILL OF EACH PARAMETER SET
# ---------------------------------------------------------------------------
def forecast_skill(marcher: Marcher, *, updates: list[dict], arms: dict[str, dict],
                   horizon: int, eval_every: int, first_h: float, every: int,
                   out_csv: Path) -> list[dict]:
    """RMSE of a ``horizon``-step forecast from each evaluation time.

    ``inverted`` uses the latest fit published at or before the evaluation time
    -- the parameters the forecast would actually hold then -- so nothing is
    taken from the future.
    """
    fit_times = np.array([float(u["time_hours"]) for u in updates])
    i_first = int(np.searchsorted(marcher.times, first_h))
    rows: list[dict] = []
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = None
        for i0 in range(i_first, len(marcher.times) - horizon, eval_every):
            t0 = float(marcher.times[i0])
            k = int(np.searchsorted(fit_times, t0 + 1e-9, side="right")) - 1
            inverted = {n: float(updates[k][n]) for n in NAMES}
            row = {"time_hours": t0, "fit_time_hours": float(fit_times[k])}
            for label, params in list(arms.items()) + [("inverted", inverted)]:
                row[f"rmse_{label}_c"] = marcher.rmse(params, i0, i0 + horizon, every)
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            fh.flush()
            rows.append(row)
            print(f"  t0={t0:5.2f}h  " + "  ".join(
                f"{key[5:-2]} {val:.3f}" for key, val in row.items() if key.startswith("rmse_")),
                flush=True)
    return rows


# ---------------------------------------------------------------------------
# FIGURES
# ---------------------------------------------------------------------------
PRETTY = {
    "cap_s": r"$c_s$ (J kg$^{-1}$ K$^{-1}$)",
    "k_parameter": r"$k$ (freezing curve)",
    "lambda_s": r"$\lambda_s$ (W m$^{-1}$ K$^{-1}$)",
}


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fig_parameter_evolution(updates: list[dict], calib: dict, degraded: dict,
                            out_dir: Path, *, name: str = "fig_parameter_evolution",
                            truth_label: str = "data-calibrated $p^*$",
                            err_label: str = "$p^*$", title: str | None = None,
                            prior_note: bool = True, show_literature: bool = True) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from software.tools.paper_figures import (DOUBLE, FEM_KEEP, FEM_STOP, MUTED,
                                              STATUS_LIMIT, grid, save, style)
    style()
    p_star, lit = calib["p_star"], calib["literature"]
    t = np.array([float(u["time_hours"]) for u in updates])

    fig, axes = plt.subplots(3, 1, figsize=(DOUBLE * 0.62, DOUBLE * 0.78), sharex=True,
                             gridspec_kw={"hspace": 0.18})
    for ax, n in zip(axes, NAMES):
        v = np.array([float(u[n]) for u in updates])
        ax.axhspan(p_star[n] * 0.9, p_star[n] * 1.1, color=FEM_STOP, alpha=0.10, lw=0, zorder=1)
        ax.axhline(p_star[n], color=FEM_STOP, lw=1.4, ls="-.", zorder=3)
        ax.axhline(degraded[n], color=STATUS_LIMIT, lw=1.2, ls="--", zorder=3)
        if show_literature:
            ax.axhline(lit[n], color=MUTED, lw=0.9, ls=":", zorder=2)
        # Step: the forecast holds each estimate until the next fit replaces it.
        ax.step(t, v, where="post", color=FEM_KEEP, lw=1.5, zorder=5)
        ax.plot(t[1:], v[1:], "o", ms=3.2, color=FEM_KEEP, mec="white", mew=0.6, zorder=6)
        ax.set_ylabel(PRETTY[n], fontsize=7.2)
        values = np.concatenate([v, [p_star[n] * 0.9, p_star[n] * 1.1, degraded[n]]
                                 + ([lit[n]] if show_literature else [])])
        lo, hi = float(values.min()), float(values.max())
        pad = 0.08 * (hi - lo)
        # Headroom so the error annotation never sits on a reference line.
        ax.set_ylim(lo - pad, hi + 2.6 * pad)
        start_err = (degraded[n] / p_star[n] - 1.0) * 100.0
        final_err = (v[-1] / p_star[n] - 1.0) * 100.0
        ax.annotate(f"error vs {err_label}: {start_err:+.0f}% → {final_err:+.0f}%",
                    xy=(0.99, 0.94), xycoords="axes fraction", ha="right", va="top",
                    fontsize=6.8, color=FEM_KEEP)
        # Said on the panel, not only in a caption: a regularised parameter's
        # convergence is partly the prior's doing, not the data's alone.
        if (prior_note and n in calib.get("prior_names", [])
                and float(calib.get("prior_weight", 0.0)) > 0.0):
            ax.annotate("weakly identifiable: regularised toward literature",
                        xy=(0.99, 0.80), xycoords="axes fraction", ha="right", va="top",
                        fontsize=6.3, color=MUTED, style="italic")
        grid(ax)
    axes[-1].set_xlabel("experiment time (h)")
    axes[-1].set_xlim(0, t[-1] * 1.01)
    for letter, ax in zip("abc", axes):
        ax.annotate(f"({letter})", xy=(0.006, 0.94), xycoords="axes fraction",
                    fontsize=8, fontweight="bold", va="top", ha="left")
    handles = [
        Line2D([], [], color=FEM_KEEP, marker="o", ms=3.2, lw=1.5, label="inverted estimate"),
        Line2D([], [], color=FEM_STOP, ls="-.", lw=1.4, label=f"{truth_label} (±10 %)"),
        Line2D([], [], color=STATUS_LIMIT, ls="--", lw=1.2, label="degraded start"),
    ]
    if show_literature:
        handles.append(Line2D([], [], color=MUTED, ls=":", lw=0.9, label="literature"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.045), ncol=2,
               frameon=False, fontsize=6.8)
    if title:
        axes[0].set_title(title, fontsize=7.6, loc="left", pad=6)
    fig.subplots_adjust(bottom=0.13)
    save(fig, out_dir, name)


def fig_fem_trajectory(rows: list[dict], updates: list[dict], out_dir: Path, *,
                       name: str, ref_label: str, title: str | None = None) -> None:
    """Sensor RMSE of the continuous twin run: frozen arms vs live inversion."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from software.tools.paper_figures import (DOUBLE, FEM_KEEP, FEM_STOP, MUTED,
                                              STATUS_LIMIT, grid, save, style)
    style()
    t = np.array([float(r["time_hours"]) for r in rows])

    def col(key):
        return np.array([float(r[key]) for r in rows])

    deg, inv, ref = col("rmse_degraded_c"), col("rmse_inverted_c"), col("rmse_reference_c")
    lit = col("rmse_literature_c") if "rmse_literature_c" in rows[0] else None
    final = col("rmse_final_c") if "rmse_final_c" in rows[0] else None
    fit_t = [float(u["time_hours"]) for u in updates[1:]]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(DOUBLE, DOUBLE * 0.55), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.12})
    if final is not None:
        ax.plot(t, final, ":", color=MUTED, lw=1.1, zorder=2)
    ax.fill_between(t, ref, deg, where=deg >= ref, color=STATUS_LIMIT, alpha=0.09, lw=0, zorder=1)
    ax.plot(t, deg, "--", color=STATUS_LIMIT, lw=1.4, zorder=3)
    if lit is not None:
        ax.plot(t, lit, ":", color=MUTED, lw=1.0, zorder=2)
    ax.plot(t, ref, "-.", color=FEM_STOP, lw=1.4, zorder=4)
    ax.plot(t, inv, "-", color=FEM_KEEP, lw=1.8, zorder=5)
    for ft in fit_t:
        ax.axvline(ft, ymin=0, ymax=0.035, color=FEM_KEEP, lw=0.8, zorder=6)
    ax.set_ylabel("sensor RMSE (°C)")
    # Headroom above the curves for the summary line.
    ax.set_ylim(0, float(np.nanmax(np.concatenate([deg, inv, ref]))) * 1.16)
    grid(ax)

    span = deg - ref
    usable = span > 0.02
    closed = np.full_like(span, np.nan)
    closed[usable] = (deg[usable] - inv[usable]) / span[usable] * 100.0
    ax2.axhline(100, color=FEM_STOP, lw=0.9, ls="-.")
    ax2.axhline(0, color=MUTED, lw=0.8)
    ax2.plot(t, closed, "-", color=FEM_KEEP, lw=1.6)
    top = max(120.0, float(np.nanmax(closed)) * 1.08) if np.any(np.isfinite(closed)) else 120.0
    ax2.fill_between(t, 0, closed, color=FEM_KEEP, alpha=0.12, lw=0)
    ax2.set_ylabel("error gap closed (%)")
    ax2.set_ylim(-20, top)
    ax2.set_xlabel("experiment time (h)")
    ax2.set_xlim(t[0], t[-1])
    grid(ax2)
    for letter, axis in (("a", ax), ("b", ax2)):
        axis.annotate(f"({letter})", xy=(0.006, 0.96), xycoords="axes fraction",
                      fontsize=8, fontweight="bold", va="top", ha="left")
    after = t >= (fit_t[0] if fit_t else t[0])
    mean_deg, mean_inv, mean_ref = (float(np.mean(a[after])) for a in (deg, inv, ref))
    ax.annotate(f"after first update, mean RMSE: degraded {mean_deg:.2f} °C, "
                f"inverted {mean_inv:.2f} °C, {ref_label} {mean_ref:.2f} °C",
                xy=(0.05, 0.97), xycoords="axes fraction", ha="left", va="top", fontsize=6.8)
    if title:
        ax.set_title(title, fontsize=7.6, loc="left", pad=6)
    handles = [
        Line2D([], [], color=STATUS_LIMIT, ls="--", lw=1.4, label="degraded, frozen"),
        Line2D([], [], color=FEM_KEEP, lw=1.8, label="degraded + live inversion"),
        Line2D([], [], color=FEM_STOP, ls="-.", lw=1.4, label=ref_label),
    ]
    if lit is not None:
        handles.append(Line2D([], [], color=MUTED, ls=":", lw=1.0, label="literature"))
    if final is not None:
        handles.append(Line2D([], [], color=MUTED, ls=":", lw=1.1,
                              label="data-preferred fit, frozen"))
    handles.append(Patch(facecolor=STATUS_LIMIT, alpha=0.09, label="error gap"))
    fig.subplots_adjust(bottom=0.17)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               ncol=len(handles), frameon=False, fontsize=6.8)
    save(fig, out_dir, name)


def fig_fem_impact(skill: list[dict], updates: list[dict], horizon_h: float,
                   out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from software.tools.paper_figures import (DOUBLE, FEM_KEEP, FEM_STOP, MUTED,
                                              STATUS_LIMIT, grid, save, style)
    style()
    t = np.array([float(r["time_hours"]) for r in skill])

    def col(key):
        return np.array([float(r[key]) for r in skill])

    deg, inv, ref = col("rmse_degraded_c"), col("rmse_inverted_c"), col("rmse_p_star_c")
    lit = col("rmse_literature_c") if "rmse_literature_c" in skill[0] else None
    fit_t = [float(u["time_hours"]) for u in updates[1:]]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(DOUBLE, DOUBLE * 0.55), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.12})
    ax.fill_between(t, ref, deg, color=STATUS_LIMIT, alpha=0.09, lw=0, zorder=1)
    ax.plot(t, deg, "--", color=STATUS_LIMIT, lw=1.4, zorder=3)
    if lit is not None:
        ax.plot(t, lit, ":", color=MUTED, lw=1.0, zorder=2)
    ax.plot(t, ref, "-.", color=FEM_STOP, lw=1.4, zorder=4)
    ax.plot(t, inv, "-", color=FEM_KEEP, lw=1.8, zorder=5)
    # Rug of fit times: each tick is a material_override the forecast received.
    for ft in fit_t:
        ax.axvline(ft, ymin=0, ymax=0.035, color=FEM_KEEP, lw=0.8, zorder=6)
    ax.set_ylabel(f"{horizon_h:g} h forecast RMSE (°C)")
    ax.set_ylim(bottom=0)
    grid(ax)

    # Share of the degraded-vs-p* gap the inversion closes. Masked where the gap
    # has not opened: a ratio over a near-zero denominator is noise, not skill.
    span = deg - ref
    usable = span > 0.02
    closed = np.full_like(span, np.nan)
    closed[usable] = (deg[usable] - inv[usable]) / span[usable] * 100.0
    ax2.axhline(100, color=FEM_STOP, lw=0.9, ls="-.")
    ax2.axhline(0, color=MUTED, lw=0.8)
    ax2.plot(t, closed, "-", color=FEM_KEEP, lw=1.6)
    ax2.fill_between(t, 0, closed, color=FEM_KEEP, alpha=0.12, lw=0)
    ax2.set_ylabel("error gap closed (%)")
    ax2.set_ylim(-20, 120)
    ax2.set_xlabel("forecast issue time (h)")
    ax2.set_xlim(t[0], t[-1])
    grid(ax2)
    for letter, axis in (("a", ax), ("b", ax2)):
        axis.annotate(f"({letter})", xy=(0.006, 0.96), xycoords="axes fraction",
                      fontsize=8, fontweight="bold", va="top", ha="left")

    after = t >= (fit_t[0] if fit_t else t[0])
    mean_deg, mean_inv, mean_ref = (float(np.mean(a[after])) for a in (deg, inv, ref))
    ax.annotate(f"after first fit, mean RMSE: degraded {mean_deg:.2f} °C, "
                f"inverted {mean_inv:.2f} °C, $p^*$ {mean_ref:.2f} °C",
                xy=(0.99, 0.97), xycoords="axes fraction", ha="right", va="top", fontsize=6.8)

    handles = [
        Line2D([], [], color=STATUS_LIMIT, ls="--", lw=1.4, label="degraded, frozen"),
        Line2D([], [], color=FEM_KEEP, lw=1.8, label="degraded + live inversion"),
        Line2D([], [], color=FEM_STOP, ls="-.", lw=1.4, label="data-calibrated $p^*$"),
        Line2D([], [], color=MUTED, ls=":", lw=1.0, label="literature"),
        Patch(facecolor=STATUS_LIMIT, alpha=0.09, label="error gap"),
    ]
    fig.subplots_adjust(bottom=0.17)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               ncol=len(handles), frameon=False, fontsize=6.8)
    save(fig, out_dir, "fig_fem_impact")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", default=DEFAULT_PATH)
    parser.add_argument("--config", default="startup.conf")
    parser.add_argument("--out", default="artifacts/inversion_recovery")
    parser.add_argument("--stage", default="all",
                        choices=["all", "calibrate", "history", "trajectory", "twin", "recursive", "replay",
                                 "skill", "plot"],
                        help="Run one stage; later stages read earlier outputs from --out.")
    # calibration
    parser.add_argument("--deep-edge", default="top",
                        help="Boundary set of the deep Dirichlet edge.")
    parser.add_argument("--deep-sensor", default="H300D30",
                        help="Drive the deep edge from this sensor; '' keeps the configured value.")
    parser.add_argument("--min-radius", type=float, default=0.02,
                        help="Leave sensors with |x| below this (pipe wall) out of fit and score.")
    parser.add_argument("--prior-weight", type=float, default=0.0,
                        help="Tikhonov weight on ln(p / literature)^2, in loss units (C^2).")
    parser.add_argument("--prior-on", default="cap_s,lambda_s",
                        help="Parameters the prior applies to.")
    parser.add_argument("--calib-windows", type=int, default=5)
    parser.add_argument("--calib-hours", type=float, default=1.0)
    parser.add_argument("--calib-every", type=int, default=4, help="Sample every n-th snapshot.")
    parser.add_argument("--calib-maxfun", type=int, default=100)
    parser.add_argument("--calib-bound", type=float, default=8.0,
                        help="p* is searched in literature x [1/b, b].")
    # degradation + replay
    parser.add_argument("--degrade", type=float, default=2.5,
                        help="Factor each parameter is moved away from p*, against the data's pull.")
    parser.add_argument("--bound", type=float, default=6.0,
                        help="Service bounds: degraded base x [1/b, b] (log-symmetric).")
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--window-stride", type=int, default=6,
                        help="20 points x stride 6 x 30 s = a ~1 h fitting window.")
    parser.add_argument("--invert-every", type=int, default=20, help="Snapshots between fits.")
    parser.add_argument("--maxfun", type=int, default=40)
    parser.add_argument("--update-every", type=int, default=40,
                        help="Recursive estimator: snapshots between updates (40 = 20 min).")
    parser.add_argument("--window-step", type=int, default=60,
                        help="Recursive estimator: spacing of its 1 h windows (60 = 30 min).")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--data", default="twin", choices=["twin", "real"],
                        help="history/trajectory: synthetic twin (known truth) or April 8 data.")
    parser.add_argument("--history-maxiter", type=int, default=15)
    parser.add_argument("--truth", default="config", choices=["config", "pstar"],
                        help="Ground truth: the lab-measured startup.conf material (default), "
                             "or the data-calibrated p*.")
    parser.add_argument("--degrade-factors", default="cap_s=2.5,k_parameter=0.4,lambda_s=0.4",
                        help="With --truth config: multipliers applied to the lab values for "
                             "the wrong starting point.")
    parser.add_argument("--twin-noise", type=float, default=0.05,
                        help="Synthetic twin: Gaussian sensor noise (C).")
    # skill
    parser.add_argument("--horizon-hours", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--first-hours", type=float, default=0.25)
    args = parser.parse_args()

    import logging
    logging.disable(logging.INFO)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = load_startup_config(args.config)
    literature = {n: float(config["fem_forecast"]["material"][n]) for n in NAMES}
    calib_path = out / "calibration.json"
    updates_path = out / "inversion_updates.csv"
    twin_path = out / "twin_updates.csv"
    skill_path = out / "forecast_skill.csv"
    stages = ["calibrate", "history", "trajectory", "plot"] if args.stage == "all" else [args.stage]
    prior_names = tuple(n for n in args.prior_on.split(",") if n in NAMES)

    marcher = None
    cadence_h = 30.0 / 3600.0
    if any(s in stages for s in ("calibrate", "history", "trajectory", "twin", "recursive",
                                "replay", "skill")):
        path = Path(args.path)
        snapshots = load_snapshots(path)
        boundaries = cached_boundary_track(config, snapshots, path, out / "boundary_track.pkl")
        if args.deep_sensor:
            boundaries = drive_deep_edge(boundaries, snapshots, boundary_set=args.deep_edge,
                                         sensor_id=args.deep_sensor)
        marcher = Marcher(args.config, snapshots, boundaries, config["fem_forecast"]["material"],
                          min_radius_m=args.min_radius)
        cadence_h = float(marcher.times[1] - marcher.times[0])
        print(f"{len(snapshots)} snapshots at {cadence_h * 3600:.0f} s, "
              f"{len(marcher.ids)} sensors ({int(marcher.score.sum())} scored), "
              f"{marcher.times[-1]:.2f} h, deep edge "
              f"{'from ' + args.deep_sensor if args.deep_sensor else 'as configured'}", flush=True)

    if "calibrate" in stages:
        length = int(round(args.calib_hours / cadence_h))
        starts = window_starts(marcher, first_h=0.25, length=length, count=args.calib_windows)
        print(f"\n=== 1. calibrate p* over {len(starts)} x {args.calib_hours:g} h windows ===",
              flush=True)
        calib = calibrate(marcher, literature, starts=starts, length=length,
                          every=args.calib_every,
                          bounds={n: (1.0 / args.calib_bound, args.calib_bound) for n in NAMES},
                          maxfun=args.calib_maxfun, prior_weight=args.prior_weight,
                          prior_names=prior_names)
        calib_path.write_text(json.dumps(calib, indent=2), encoding="utf-8")
        for n in NAMES:
            print(f"  {n:<12} literature {literature[n]:9.4g}  ->  p* {calib['p_star'][n]:9.4g}"
                  f"  (x{calib['scales_vs_literature'][n]:.3f}"
                  f"{', AT BOUND' if calib['at_bound'][n] else ''})")
        print(f"  RMSE literature {calib['rmse_literature_c']:.3f} C  ->  "
              f"p* {calib['rmse_p_star_c']:.3f} C   [{calib['elapsed_s']:.0f}s]", flush=True)
    calib = json.loads(calib_path.read_text(encoding="utf-8")) if calib_path.exists() else {
        "p_star": dict(literature), "literature": dict(literature)}
    p_star = calib["p_star"]
    if args.truth == "config":
        # The startup.conf material is the lab-measured ground truth.
        truth = dict(literature)
        factors = {n: 1.0 for n in NAMES}
        for item in args.degrade_factors.split(","):
            if "=" in item:
                key, value = item.split("=", 1)
                if key.strip() in factors:
                    factors[key.strip()] = float(value)
    else:
        truth = dict(p_star)
        factors = degrade_away(p_star, calib["literature"], args.degrade)
    degraded = {n: truth[n] * factors[n] for n in NAMES}
    print("ground truth (" + ("startup.conf lab values" if args.truth == "config" else "p*") + "): "
          + "  ".join(f"{n}={truth[n]:.4g}" for n in NAMES))
    print("degraded start: " + "  ".join(f"{n}={degraded[n]:.4g} (x{factors[n]:g})" for n in NAMES),
          flush=True)
    truth_tag = "lab" if args.truth == "config" else "pstar"

    if "replay" in stages:
        print("\n=== 2. replay the inversion service from the degraded start ===")
        for n in NAMES:
            print(f"  {n:<12} p* {p_star[n]:9.4g}   degraded x{factors[n]:.2f} -> "
                  f"{degraded[n]:9.4g}   reachable [{degraded[n] / args.bound:.4g}, "
                  f"{degraded[n] * args.bound:.4g}]", flush=True)
        replay(config, args.config, marcher.snapshots, marcher.boundaries,
               degraded=degraded, p_star=p_star, bound=args.bound,
               window_size=args.window_size, window_stride=args.window_stride,
               invert_every=args.invert_every, maxfun=args.maxfun,
               min_radius_m=args.min_radius, out_csv=updates_path,
               prior={n: literature[n] for n in prior_names}, prior_weight=args.prior_weight)
    if "twin" in stages:
        print(f"\n=== 2t. synthetic twin: truth = p*, noise {args.twin_noise:g} C ===", flush=True)
        synthetic = synthesize(marcher, p_star, noise_c=args.twin_noise, seed=7)
        replay(config, args.config, synthetic, marcher.boundaries,
               degraded=degraded, p_star=p_star, bound=args.bound,
               window_size=args.window_size, window_stride=args.window_stride,
               invert_every=args.invert_every, maxfun=args.maxfun,
               min_radius_m=args.min_radius, out_csv=twin_path,
               prior={n: literature[n] for n in prior_names}, prior_weight=args.prior_weight)

    history_path = out / f"{args.data}_{truth_tag}_history.csv"
    if "history" in stages:
        from concurrent.futures import ProcessPoolExecutor

        twin = args.data == "twin"
        print(f"\n=== 2h. recursive history estimate on {'the synthetic twin' if twin else 'real data'}"
              f" from the degraded start ===", flush=True)
        length = int(round(1.0 / cadence_h))
        with ProcessPoolExecutor(
                max_workers=4, initializer=_history_worker_init,
                initargs=(args.config, args.path, str(out / "boundary_track.pkl"),
                          args.deep_edge, args.deep_sensor, args.min_radius,
                          truth if twin else None, args.twin_noise)) as pool:
            history_estimate(pool, marcher.times, degraded=degraded, p_ref=truth,
                             bound=args.bound, every=args.calib_every, first_index=length,
                             update_every=args.update_every, maxiter=args.history_maxiter,
                             out_csv=history_path)

    if "trajectory" in stages:
        twin = args.data == "twin"
        hist = _read_csv(history_path)
        final = {n: float(hist[-1][n]) for n in NAMES}
        track_m = marcher
        if twin:
            synthetic = synthesize(marcher, truth, noise_c=args.twin_noise, seed=7)
            track_m = Marcher(args.config, synthetic, marcher.boundaries, marcher.base,
                              min_radius_m=args.min_radius)
        print(f"\n=== 3c. continuous-run error ({args.data}); reference = ground truth ===",
              flush=True)
        inverted_schedule = [(0, degraded)] + [
            (int(h["index"]), {n: float(h[n]) for n in NAMES}) for h in hist[1:]]
        curves = {
            "degraded": track_m.trajectory([(0, degraded)]),
            "inverted": track_m.trajectory(inverted_schedule),
            "reference": track_m.trajectory([(0, truth)]),
        }
        if not twin:
            # What the data alone would pick, frozen from t = 0: the floor a
            # parameter-only correction can reach on this model.
            curves["final"] = track_m.trajectory([(0, final)])
        traj_path = out / f"{args.data}_{truth_tag}_trajectory.csv"
        with traj_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["time_hours"] + [f"rmse_{k}_c" for k in curves])
            for i, t in enumerate(track_m.times):
                writer.writerow([float(t)] + [float(curves[k][i]) for k in curves])
        for k, v in curves.items():
            print(f"  {k:<10} mean RMSE {np.nanmean(v):.3f} C   final {v[-1]:.3f} C")

    if "recursive" in stages:
        from concurrent.futures import ProcessPoolExecutor

        print("\n=== 2. recursive estimate from the degraded start ===")
        for n in NAMES:
            print(f"  {n:<12} p* {p_star[n]:9.4g}   degraded x{factors[n]:.2f} -> {degraded[n]:9.4g}",
                  flush=True)
        length = int(round(1.0 / cadence_h))
        with ProcessPoolExecutor(
                max_workers=args.workers, initializer=_worker_init,
                initargs=(args.config, args.path, str(out / "boundary_track.pkl"),
                          args.deep_edge, args.deep_sensor, args.min_radius)) as pool:
            recursive(pool, marcher.times, degraded=degraded, p_star=p_star,
                      prior={n: literature[n] for n in prior_names},
                      prior_weight=args.prior_weight, bound=args.bound, length=length,
                      window_step=args.window_step, every=args.calib_every,
                      first_index=length, min_start=int(round(0.25 / cadence_h)),
                      update_every=args.update_every, maxfun=args.maxfun,
                      out_csv=updates_path)
    updates = _read_csv(updates_path) if updates_path.exists() else []

    if "skill" in stages:
        horizon = int(round(args.horizon_hours / cadence_h))
        print(f"\n=== 3. {args.horizon_hours:g} h forecast skill ===", flush=True)
        forecast_skill(marcher, updates=updates,
                       arms={"p_star": p_star, "degraded": degraded, "literature": literature},
                       horizon=horizon, eval_every=args.eval_every,
                       first_h=args.first_hours, every=1, out_csv=skill_path)

    if "plot" in stages:
        print("\n=== figures ===")
        lab = args.truth == "config"
        truth_name = "lab value (startup.conf)" if lab else "true value"
        for data in ("twin", "real"):
            hist_file = out / f"{data}_{truth_tag}_history.csv"
            traj_file = out / f"{data}_{truth_tag}_trajectory.csv"
            if not hist_file.exists():
                continue
            hist = _read_csv(hist_file)
            suffix = f"{data}" + ("" if lab else "_pstar")
            fig_parameter_evolution(
                hist, {"p_star": truth, "literature": literature}, degraded, out,
                name=f"fig_parameter_evolution_{suffix}", truth_label=truth_name,
                err_label="lab value" if lab else "truth", prior_note=False,
                show_literature=not lab,
                title=(f"Synthetic twin: soil = startup.conf lab values, noise "
                       f"{args.twin_noise:g} \u00b0C" if data == "twin"
                       else "April 8 measurements"))
            if traj_file.exists():
                fig_fem_trajectory(
                    _read_csv(traj_file), hist, out, name=f"fig_fem_impact_{suffix}",
                    ref_label=("lab values, frozen" if lab else "true soil"),
                    title=("Synthetic twin" if data == "twin" else "April 8 measurements")
                    + ": one continuous FEM run, seeded once at t = 0")
        if not updates or not skill_path.exists():
            return
        if twin_path.exists():
            fig_parameter_evolution(
                _read_csv(twin_path), calib, degraded, out, name="fig_parameter_evolution",
                truth_label="true value", err_label="truth",
                title="Synthetic twin: real geometry, sensors, initial field and forcing; "
                      f"soil = $p^*$, noise {args.twin_noise:g} °C")
        if updates:
            fig_parameter_evolution(
                updates, calib, degraded, out, name="fig_parameter_evolution_real",
                title="April 8 measurements")
        fig_fem_impact(_read_csv(skill_path), updates, args.horizon_hours, out)


if __name__ == "__main__":
    main()

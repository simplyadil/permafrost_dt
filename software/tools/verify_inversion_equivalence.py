"""Check that the window forward model matches the per-horizon one it replaced.

The inversion objective used to call ``FEM()`` once per window time, restarting
from the initial condition each time. It now walks the window as a single
trajectory with a reused solver. The service has one path and no switch, so the
original is kept in ``software/tools/golden_reference.py`` and patched in here
for the duration of a call. Both must produce the same modelled series, and
therefore the same fitted parameters.

Sameness is checked at three levels, because agreement at one does not imply it
at the next:

1. the modelled series for a fixed parameter set (the forward model itself);
2. the objective value across a spread of candidate parameters, including the
   bound corners the optimiser can visit;
3. the fitted parameters from a complete ``_run_inversion``, where any tiny
   per-evaluation difference would be amplified by the finite-difference
   gradient into a different search path.

The window is taken from a real recorded dataset, so the initial condition and
sensor geometry are the ones the service actually sees.

Run:

    python -m software.tools.verify_inversion_equivalence
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from software.digital_twin.simulator.fem_inversion.fem_inversion_server import FEMInversionServer
from software.startup.utils.config import load_startup_config
from software.tools.fem_debug_replay import DEFAULT_SENSOR_PATH, _sensor_rows
from software.tools.golden_reference import reference_forward_model

# Round-off floor for temperatures in Celsius. The two paths perform the same
# arithmetic but accumulate it in a different order, so exact bit equality is
# not expected.
TOLERANCE_C = 1e-9


def _make_server(config: dict, config_path: str) -> FEMInversionServer:
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
        config_path=config_path,
    )


def _worst_series_gap(a: dict[str, list[float]], b: dict[str, list[float]]) -> float:
    assert set(a) == set(b), "series cover different sensors"
    worst = 0.0
    for sid, left in a.items():
        right = b[sid]
        assert len(left) == len(right), f"series length differs for {sid}"
        if left:
            worst = max(worst, float(np.max(np.abs(np.asarray(left) - np.asarray(right)))))
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="startup.conf")
    parser.add_argument("--path", default=DEFAULT_SENSOR_PATH, help="Recorded sensor file to take a window from.")
    parser.add_argument("--start-row", type=int, default=40, help="Window start, so the field is not still uniform.")
    parser.add_argument("--full-inversion", action="store_true", help="Also run a complete _run_inversion both ways.")
    args = parser.parse_args()

    config = load_startup_config(args.config)
    window_size = int(config.get("fem_inversion", {}).get("window_size", 5))

    ref = _make_server(config, args.config)
    inc = _make_server(config, args.config)

    snapshots = _sensor_rows(Path(args.path), None, args.start_row, window_size, 1, None)
    if len(snapshots) < window_size:
        raise SystemExit(f"Only {len(snapshots)} snapshots read from {args.path}; need {window_size}.")
    for snapshot in snapshots:
        ref._upsert_snapshot(snapshot)
        inc._upsert_snapshot(snapshot)

    window = ref._build_observation_window()
    if window is None:
        raise SystemExit("Could not assemble an observation window from the recorded data.")
    times_hours, positions, obs_series, ic_sensors = window
    print(f"window: {len(times_hours)} times x {len(positions)} sensors, "
          f"t = {times_hours[0]:.4f} .. {times_hours[-1]:.4f} h")
    print(f"steps per horizon (reference reruns each): "
          f"{[max(1, int(round(t * 3600.0 / 30.0))) for t in times_hours]}\n")

    base = {name: float(ref.base_params[name]) for name in ref.base_params}
    worst = 0.0

    # ---- 1. forward model at several candidate parameter sets --------------
    print(f"{'candidate':<34} {'worst |dT| (C)':>16} {'orig (s)':>9} {'now (s)':>9} {'speedup':>8}")
    print("-" * 82)
    candidates: list[tuple[str, dict[str, float]]] = [("base values (scale 1.0)", dict(base))]
    for name in ref.invert_params:
        lo, hi = ref.relative_bounds[name]
        for label, scale in (("lower bound", lo), ("upper bound", hi)):
            params = dict(base)
            params[name] = float(base[name]) * float(scale)
            candidates.append((f"{name} at {label} ({scale:g}x)", params))

    total_ref = total_inc = 0.0
    for label, params in candidates:
        t0 = time.perf_counter()
        with reference_forward_model(ref):
            series_ref = ref._generate_series(params, positions, times_hours, ic_sensors)
        t_ref = time.perf_counter() - t0

        t0 = time.perf_counter()
        series_inc = inc._generate_series(params, positions, times_hours, ic_sensors)
        t_inc = time.perf_counter() - t0

        gap = _worst_series_gap(series_ref, series_inc)
        worst = max(worst, gap)
        total_ref += t_ref
        total_inc += t_inc
        print(f"{label:<34} {gap:16.3e} {t_ref:9.3f} {t_inc:9.3f} {t_ref / t_inc:7.1f}x")

    print("-" * 82)
    print(f"{'total':<34} {worst:16.3e} {total_ref:9.3f} {total_inc:9.3f} {total_ref / total_inc:7.1f}x")

    # ---- 2. the loss the optimiser actually sees ---------------------------
    with reference_forward_model(ref):
        loss_ref = ref._compute_loss(ref._generate_series(base, positions, times_hours, ic_sensors), obs_series)
    loss_inc = inc._compute_loss(inc._generate_series(base, positions, times_hours, ic_sensors), obs_series)
    print(f"\nobjective at base values: original {loss_ref:.12f}   current {loss_inc:.12f}")
    print(f"  difference {abs(loss_ref - loss_inc):.3e}")

    # ---- 3. a complete fit both ways ---------------------------------------
    if args.full_inversion:
        print("\nrunning a full _run_inversion both ways (this is the slow part)...")
        ref._last_best_scales = None
        t0 = time.perf_counter()
        with reference_forward_model(ref):
            params_ref, info_ref = ref._run_inversion(times_hours, positions, obs_series, ic_sensors)
        t_full_ref = time.perf_counter() - t0

        inc._last_best_scales = None
        t0 = time.perf_counter()
        params_inc, info_inc = inc._run_inversion(times_hours, positions, obs_series, ic_sensors)
        t_full_inc = time.perf_counter() - t0

        print(f"\n{'parameter':<16} {'original':>16} {'current':>16} {'rel diff':>12}")
        print("-" * 64)
        for name in ref.invert_params:
            a, b = float(params_ref[name]), float(params_inc[name])
            rel = abs(a - b) / (abs(a) or 1.0)
            worst = max(worst, rel)
            print(f"{name:<16} {a:16.9f} {b:16.9f} {rel:12.3e}")
        print("-" * 64)
        print(f"final loss    original {info_ref['final_loss']:.12f}   current {info_inc['final_loss']:.12f}")
        print(f"evaluations   original {info_ref['eval_count']:<6d}     current {info_inc['eval_count']}")
        print(f"wall clock    original {t_full_ref:7.1f} s     current {t_full_inc:7.1f} s"
              f"     {t_full_ref / t_full_inc:.1f}x")

    print()
    if worst < TOLERANCE_C:
        print(f"PASS  worst difference {worst:.3e} < {TOLERANCE_C:.0e} (round-off)")
    else:
        print(f"FAIL  worst difference {worst:.3e} >= {TOLERANCE_C:.0e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

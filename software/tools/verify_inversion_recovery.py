"""Does the inversion actually recover parameters it is given?

``verify_inversion_equivalence`` shows the rewritten forward model matches the
one it replaced. That is a statement about two implementations agreeing, and it
would pass just as happily if both were wrong. This script asks the other
question: given data produced by a known set of soil parameters, does the
inversion find those parameters again?

The method is a twin experiment. Take a real measured snapshot as the initial
condition and the real thermocouple positions, pick a set of "true" parameters,
run the forward model to manufacture the temperatures those parameters would
have produced, and hand the result to the inversion as if it were measured. The
global minimum of the objective now sits exactly at the true parameters and its
value there is zero, so anything the optimiser fails to recover is a failure of
the fitting machinery -- not of the physics.

That framing is also this script's limitation, and it is worth being blunt about
it: because the synthetic data comes from the same forward model the inversion
fits with, **no amount of passing here says the model is right**. Wrong boundary
conditions, a wrong freezing curve or a wrong mesh would cancel out exactly. It
tests the optimiser, the window assembly, the parameterisation and the bounds.
Model error against reality is a separate question, and one this script is
deliberately not equipped to answer.

Two separate questions get answered, and conflating them is easy:

* **Recovery** -- does the optimiser find the truth back from clean data? This
  is the correctness gate. With no noise there is no excuse: the minimum is
  exact, sits inside the bounds, and any competent optimiser must reach it. A
  failure here is a defect in the fitting machinery.
* **Practical identifiability** -- how much the objective moves when each
  parameter is swept across its whole allowed range, holding the others at
  truth. This is not about correctness at all; it is about whether the window
  carries enough signal to survive real instrument noise. A parameter whose
  entire bound range shifts predicted temperatures by less than a thermocouple
  can resolve is unrecoverable in practice no matter how good the optimiser is.

The distinction matters because the two can disagree loudly. Perfect recovery
on clean data plus a signal below the noise floor is not a contradiction -- it
means the code is right and the experiment window is too short to use it.

Run:

    python -m software.tools.verify_inversion_recovery
    python -m software.tools.verify_inversion_recovery --window-stride 30 --noise-c 0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from software.digital_twin.simulator.fem_inversion.fem_inversion_server import FEMInversionServer
from software.startup.utils.config import load_startup_config
from software.tools.fem_debug_replay import DEFAULT_SENSOR_PATH, _sensor_rows

# Thermocouple resolution, in Celsius. Used as the yardstick for whether a
# change in the objective is meaningful: a loss difference the sensors could not
# have detected is not information the inversion can act on.
SENSOR_NOISE_C = 0.1

# A parameter counts as recovered if the fitted value is within this fraction of
# the truth. Loose enough to tolerate optimiser tolerances, tight enough that a
# parameter stuck at its starting point or pinned to a bound fails.
RECOVERY_TOLERANCE = 0.02


def make_server(config: dict, config_path: str) -> FEMInversionServer:
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
        config_path=config_path,
    )


def load_window(server: FEMInversionServer, path: Path, start_row: int, stride: int):
    """Take positions, times and an initial condition from real recorded data.

    Only the geometry and the initial field are borrowed; the temperatures the
    inversion is asked to fit are manufactured in ``synthetic_observations``.
    """
    window_size = server.window_size
    limit = stride * (window_size - 1) + 1
    snapshots = _sensor_rows(path, None, start_row, limit, stride, None)
    if len(snapshots) < window_size:
        raise SystemExit(
            f"Only {len(snapshots)} snapshots read from {path} at stride {stride}; need {window_size}."
        )
    for snapshot in snapshots:
        server._upsert_snapshot(snapshot)
    window = server._build_observation_window()
    if window is None:
        raise SystemExit("Could not assemble an observation window from the recorded data.")
    return window


def true_parameters(server: FEMInversionServer, scales: dict[str, float]) -> dict[str, float]:
    """Full parameter dict for the given per-parameter scale factors.

    Built through the service's own ``_normalized_to_params`` so the test cannot
    disagree with the service about what a scale factor means.
    """
    base_inv_values = {name: float(server.base_params[name]) for name in server.invert_params}
    vector = np.array([float(scales[name]) for name in server.invert_params], dtype=float)
    return server._normalized_to_params(vector, base_inv_values)


def synthetic_observations(
    server: FEMInversionServer,
    params: dict[str, float],
    positions: list[dict],
    times_hours: list[float],
    ic_sensors: list[dict],
    noise_c: float,
    seed: int,
) -> dict[str, list[float]]:
    """Temperatures these parameters would have produced, optionally noised."""
    series = server._generate_series(params, positions, times_hours, ic_sensors)
    if noise_c <= 0.0:
        return series
    rng = np.random.default_rng(seed)
    return {
        sid: [value + float(rng.normal(0.0, noise_c)) for value in values]
        for sid, values in series.items()
    }


# ---------------------------------------------------------------------------
# IDENTIFIABILITY
# ---------------------------------------------------------------------------
def probe_identifiability(
    server: FEMInversionServer,
    positions: list[dict],
    times_hours: list[float],
    ic_sensors: list[dict],
    obs_series: dict[str, list[float]],
    truth: dict[str, float],
    samples: int,
) -> dict[str, dict]:
    """Sweep each parameter across its bounds and record how the loss responds.

    Everything else is held at the true value, so this isolates one parameter's
    influence on the objective. The span of the loss over the sweep is the signal
    the optimiser has to work with; comparing it against the noise floor says
    whether that signal is usable.
    """
    results: dict[str, dict] = {}
    # An MSE of sigma^2 is what pure sensor noise alone would contribute, so a
    # loss variation below it is indistinguishable from noise.
    noise_floor = SENSOR_NOISE_C ** 2 if server.loss == "mse" else SENSOR_NOISE_C

    for name in server.invert_params:
        lo, hi = server.relative_bounds[name]
        losses: list[float] = []
        scales = np.linspace(lo, hi, samples)
        for scale in scales:
            probe = dict(truth)
            probe[name] = float(server.base_params[name]) * float(scale)
            series = server._generate_series(probe, positions, times_hours, ic_sensors)
            losses.append(server._compute_loss(series, obs_series))

        loss_arr = np.asarray(losses, dtype=float)
        span = float(np.max(loss_arr) - np.min(loss_arr))
        ratio = span / noise_floor if noise_floor > 0 else float("inf")
        # Expressed as the RMS temperature difference the whole bound range buys,
        # which is directly comparable to what the thermocouples can resolve.
        rms_c = float(np.sqrt(span)) if server.loss == "mse" else span
        if ratio >= 10.0:
            verdict = "usable"
        elif ratio >= 1.0:
            verdict = "marginal"
        else:
            verdict = "BELOW NOISE"
        results[name] = {
            "scales": scales,
            "losses": loss_arr,
            "span": span,
            "rms_c": rms_c,
            "ratio": ratio,
            "verdict": verdict,
            "argmin_scale": float(scales[int(np.argmin(loss_arr))]),
        }
    return results


# ---------------------------------------------------------------------------
# RECOVERY
# ---------------------------------------------------------------------------
def run_recovery_case(
    server: FEMInversionServer,
    label: str,
    true_scales: dict[str, float],
    positions: list[dict],
    times_hours: list[float],
    ic_sensors: list[dict],
    noise_c: float,
    seed: int,
) -> dict:
    """Manufacture data from known parameters, then try to fit them back."""
    truth = true_parameters(server, true_scales)
    obs = synthetic_observations(server, truth, positions, times_hours, ic_sensors, noise_c, seed)

    # Cold start every case: a warm start from the previous case would smuggle in
    # the answer and make an unidentifiable parameter look recoverable.
    server._last_best_scales = None
    fitted, info = server._run_inversion(times_hours, positions, obs, ic_sensors)

    per_param = {}
    for name in server.invert_params:
        base = float(server.base_params[name])
        true_scale = float(true_scales[name])
        fitted_scale = float(fitted[name]) / base
        lo, hi = server.relative_bounds[name]
        at_bound = abs(fitted_scale - lo) < 1e-6 or abs(fitted_scale - hi) < 1e-6
        rel_error = abs(fitted[name] - truth[name]) / (abs(truth[name]) or 1.0)
        per_param[name] = {
            "true_scale": true_scale,
            "fitted_scale": fitted_scale,
            "true_value": float(truth[name]),
            "fitted_value": float(fitted[name]),
            "rel_error": rel_error,
            "at_bound": at_bound,
        }

    return {
        "label": label,
        "per_param": per_param,
        "final_loss": float(info.get("final_loss", float("nan"))),
        "eval_count": int(info.get("eval_count", 0)),
        "success": bool(info.get("success", False)),
    }


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="startup.conf")
    parser.add_argument("--path", default=DEFAULT_SENSOR_PATH, help="Recorded file supplying geometry and the IC.")
    parser.add_argument("--start-row", type=int, default=40, help="Window start, so the field is not still uniform.")
    parser.add_argument(
        "--window-stride",
        type=int,
        default=1,
        help="Rows between window snapshots. 1 is the production setting (consecutive 30 s "
        "readings); larger values widen the window without adding samples.",
    )
    parser.add_argument("--noise-c", type=float, default=0.0, help="Gaussian noise added to synthetic data (C).")
    parser.add_argument("--samples", type=int, default=9, help="Points per parameter in the identifiability sweep.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_startup_config(args.config)
    server = make_server(config, args.config)
    times_hours, positions, _measured, ic_sensors = load_window(
        server, Path(args.path), args.start_row, max(int(args.window_stride), 1)
    )

    span_minutes = (times_hours[-1] - times_hours[0]) * 60.0
    print(f"window     : {len(times_hours)} times x {len(positions)} sensors, "
          f"spanning {span_minutes:.1f} min")
    print(f"parameters : {', '.join(server.invert_params)}")
    print(f"bounds     : " + ", ".join(
        f"{n} [{server.relative_bounds[n][0]:g}, {server.relative_bounds[n][1]:g}]"
        for n in server.invert_params))
    print(f"noise      : {args.noise_c:g} C" + ("  (noise-free)" if args.noise_c <= 0 else ""))
    print()

    # The truth used for the identifiability sweep and the first recovery case.
    reference_scales = {"cap_s": 1.30, "k_parameter": 1.60, "lambda_s": 1.25}
    reference_scales = {
        name: float(np.clip(reference_scales.get(name, 1.2), *server.relative_bounds[name]))
        for name in server.invert_params
    }
    truth = true_parameters(server, reference_scales)
    obs = synthetic_observations(server, truth, positions, times_hours, ic_sensors, args.noise_c, args.seed)

    # ---- 1. recover the truth from data those parameters generated ----------
    noisy = args.noise_c > 0.0
    noise_floor = args.noise_c ** 2 if server.loss == "mse" else args.noise_c
    print("RECOVERY  (fit parameters back from the data they generated)")
    if noisy:
        print(f"          Data carries {args.noise_c:g} C of noise, so exact recovery is not")
        print("          possible and this section is diagnostic, not a pass/fail gate:")
        print("          it measures the accuracy the instrument actually permits.")
    else:
        print("          Noise-free, so the minimum is exact and inside the bounds:")
        print("          anything not recovered here is a defect.")
    print()
    print(f"{'case':<20} {'parameter':<14} {'true':>9} {'fitted':>9} {'error':>9}  status")
    print("-" * 76)

    cases = [
        ("truth above base", reference_scales),
        ("truth below base", {n: float(np.clip(0.75, *server.relative_bounds[n])) for n in server.invert_params}),
        ("truth at base", {n: float(np.clip(1.0, *server.relative_bounds[n])) for n in server.invert_params}),
    ]

    failures: list[str] = []
    for label, scales in cases:
        result = run_recovery_case(
            server, label, scales, positions, times_hours, ic_sensors, args.noise_c, args.seed
        )
        for i, name in enumerate(server.invert_params):
            info = result["per_param"][name]
            if info["rel_error"] <= RECOVERY_TOLERANCE:
                status = "recovered"
            elif info["at_bound"]:
                status = "PINNED AT BOUND"
                failures.append(f"{label}/{name}")
            elif noisy:
                # Expected: with noise on, accuracy is limited by how much signal
                # the parameter has, which the identifiability section quantifies.
                status = f"off by {info['rel_error'] * 100:.1f}% (noise-limited)"
            else:
                status = "NOT RECOVERED"
                failures.append(f"{label}/{name}")
            print(f"{label if i == 0 else '':<20} {name:<14} "
                  f"{info['true_scale']:9.3f} {info['fitted_scale']:9.3f} "
                  f"{info['rel_error'] * 100:8.2f}%  {status}")
        floor_note = ""
        if noisy and noise_floor > 0:
            # A fit that lands on the noise floor has extracted everything the
            # data contains; one far above it has not converged or cannot fit.
            floor_note = f", noise floor {noise_floor:.2e} ({result['final_loss'] / noise_floor:.2f}x)"
        print(f"{'':<20} {'-> final loss':<14} {result['final_loss']:9.2e} "
              f"in {result['eval_count']} evals, converged={result['success']}{floor_note}")

    # ---- 2. usability: would this window survive real instrument noise? ------
    print()
    print()
    print("PRACTICAL IDENTIFIABILITY  (sweep each parameter across its whole bound")
    print("                            range, others at truth; how much does the")
    print("                            prediction actually move?)")
    print()
    print(f"{'parameter':<14} {'loss span':>12} {'as RMS':>10} {'vs noise':>10}  verdict")
    print("-" * 66)
    probes = probe_identifiability(server, positions, times_hours, ic_sensors, obs, truth, args.samples)
    for name, probe in probes.items():
        print(f"{name:<14} {probe['span']:12.3e} {probe['rms_c']:9.3f}C {probe['ratio']:9.1f}x"
              f"  {probe['verdict']}")

    usable = {n for n, p in probes.items() if p["verdict"] != "BELOW NOISE"}
    blind = sorted(set(server.invert_params) - usable)

    # ---- verdict ------------------------------------------------------------
    # Only the noise-free run is a correctness gate. With noise on, how close the
    # fit lands is a property of the experiment, not of the code, so failing on
    # it would just be reporting that thermocouples are not infinitely precise.
    print()
    if failures:
        if noisy:
            print(f"FAIL  {len(failures)} parameter(s) came to rest against a bound: {', '.join(failures)}")
            print("      A bound is not a fit. Re-run without --noise-c to separate a genuine")
            print("      defect from the noise limit.")
        else:
            print(f"FAIL  {len(failures)} parameter(s) not recovered from clean data: {', '.join(failures)}")
            print("      The minimum was exact and inside the bounds, so this is a defect in the")
            print("      optimiser setup, the parameterisation or the window assembly.")
        raise SystemExit(1)

    if noisy:
        print(f"PASS  no parameter collapsed onto a bound at {args.noise_c:g} C of noise.")
        print("      Per-parameter accuracy above is what this window and instrument allow.")
    else:
        print(f"PASS  every parameter recovered to within {RECOVERY_TOLERANCE:.0%} from clean data.")
        print("      The fitting machinery is correct.")

    if blind:
        print()
        # Not labelled WARNING: that word collides with the log-level filtering
        # people habitually pipe this through, and the line then vanishes.
        print(f"WEAK SIGNAL  on this {span_minutes:.0f} min window, {', '.join(blind)} move the predicted")
        print(f"         temperatures by less than the {SENSOR_NOISE_C:g} C the thermocouples can resolve,")
        print("         even swept across their entire allowed range. Against real data the fit")
        print("         for these is driven by noise and boundary-condition error, not by the")
        print("         parameter -- which is what a fit pinned to a bound looks like.")
        print("         Try a wider window:  --window-stride 30  (about 1 h)")


if __name__ == "__main__":
    main()

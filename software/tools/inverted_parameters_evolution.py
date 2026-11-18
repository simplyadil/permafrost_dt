"""Evolution of the three FEM-inverted soil parameters, one subplot each.

Reads the ``fem_inversion`` records the inversion service writes to InfluxDB and
draws ``cap_s``, ``k_parameter`` and ``lambda_s`` as three stacked panels on a
shared time axis. Each panel carries the initial (base) value as a labelled
dashed line and, when the fit approaches one, the bound it is pressing against.

The initial values come from the config -- ``fem_forecast.material`` overlaid
with ``fem_inversion.base_params``, the same order the inversion service resolves
them in, currently ``cap_s=790``, ``k_parameter=3.0``, ``lambda_s=3.51``. The
resolved numbers are printed on every run, and ``--base`` pins them explicitly
for replotting an export that was produced under a different material set.

Why one panel per parameter and not one overlay: the three span three orders of
magnitude (cap_s ~1e3 J/kg/K, lambda_s ~3.5 W/m/K, k_parameter ~3), so a shared
axis flattens two of them into the baseline. Two y-scales on one plot is the
other wrong answer.

Why the bounds band is drawn even though the optimiser works in scale space: the
bounds are fixed multipliers on a fixed base, so base x [lo, hi] is a constant
interval in absolute units, and a trajectory that sits on the edge of it is the
signature worth seeing -- a parameter the data cannot identify, pinned at its
limit rather than fitted.

The x-axis is InfluxDB ingest time, not experiment time: ``write_inversion_
parameters`` records the parameter fields and validation metrics but no
``time_hours``, so a replay's own clock is not recoverable here. For a replay
that advances monotonically the two differ by a constant factor, and the default
axis is elapsed hours since the first record; ``--x-axis clock`` gives UTC
instead.

Outputs, under ``--out`` (default ``artifacts/inversion``):

* ``inverted_parameters_evolution.png`` -- the three panels
* ``inverted_parameters.csv`` -- the queried records, replayable with ``--csv``

    python -m software.tools.inverted_parameters_evolution
    python -m software.tools.inverted_parameters_evolution --range-hours 48
    python -m software.tools.inverted_parameters_evolution --csv artifacts/inversion/inverted_parameters.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from software.digital_twin.data_access.influx_utils import InfluxHelper
from software.startup.service_runners import build_influx_config
from software.startup.utils.config import load_startup_config

# Fixed hue per parameter, assigned in the configured order and never cycled.
# Deep steps: these are read on a white figure.
PARAM_COLORS = ["#0369a1", "#c2410c", "#15803d", "#7c3aed"]
BOUNDS_COLOR = "#cbd5e1"
BASE_COLOR = "#94a3b8"
GRID_COLOR = "#e2e8f0"
INK = "#1e293b"
INK_MUTED = "#64748b"

# Units for the parameters the inversion is allowed to touch; anything else
# falls back to a bare name rather than a guessed unit.
PARAM_UNITS = {
    "cap_s": "J/(kg K)",
    "lambda_s": "W/(m K)",
    "k_parameter": "-",
    "porosity": "-",
}


def _range_start(hours: float) -> str:
    seconds = max(int(hours * 3600), 60)
    return f"-{seconds}s"


def _parse_base_override(raw: str | None) -> dict[str, float]:
    """Parse ``--base name=value,...`` into a dict."""
    if not raw:
        return {}
    override: dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"Bad --base entry '{item}'; expected name=value.")
        name, value = item.split("=", 1)
        try:
            override[name.strip()] = float(value)
        except ValueError:
            raise SystemExit(f"Bad --base value in '{item}'; expected a number.") from None
    return override


def _base_values(config: dict, cli_override: dict[str, float] | None = None) -> dict[str, float]:
    """Initial (base) values the inversion multiplies its scales against.

    Resolution order, lowest first -- the same order ``create_fem_inversion_server``
    uses, so the line drawn here is the value the optimiser actually started from:

        fem_forecast.material  ->  fem_inversion.base_params  ->  --base

    Note there is no hardcoded fallback. These numbers exist once, in the config;
    copying the measured material set into this tool would give it a private
    version to drift out of sync with the solver's.
    """
    material = config.get("fem_forecast", {}).get("material", {})
    base = dict(material) if isinstance(material, dict) else {}
    overrides = config.get("fem_inversion", {}).get("base_params", {})
    if isinstance(overrides, dict):
        base.update(overrides)
    resolved = {key: float(value) for key, value in base.items() if isinstance(value, (int, float))}
    if cli_override:
        resolved.update(cli_override)
    return resolved


def _first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """First candidate column present.

    Influx prefixes the inversion's diagnostics with ``validation_``; a replay or
    ``inversion_updates.csv`` export names them plainly. Both feed this plot.
    """
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _relative_bounds(config: dict) -> dict[str, tuple[float, float]]:
    raw = config.get("fem_inversion", {}).get("relative_bounds", {})
    bounds: dict[str, tuple[float, float]] = {}
    if not isinstance(raw, dict):
        return bounds
    for name, pair in raw.items():
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            lo, hi = float(pair[0]), float(pair[1])
            if lo < hi:
                bounds[name] = (lo, hi)
    return bounds


def _run_replay(
    *,
    path: Path | None,
    config_path: str | Path,
    out_dir: Path,
    time_column: str | None,
    start_row: int,
    limit: int | None,
    stride: int,
    invert_every: int,
    maxiter: int | None,
    maxfun: int | None,
    exclude_sensors: set[str] | None,
) -> pd.DataFrame:
    """Drive the real inversion service over a recording and collect its output.

    This is the expensive path and the only one that produces a trajectory on the
    experiment clock: the sensor file is replayed snapshot by snapshot into the
    service, which fits the parameters every ``invert_every`` steps exactly as it
    does live. The service driver itself is imported from ``inversion_accuracy``
    rather than reimplemented -- one replay rig, two plots that agree.

    Results are flushed to CSV after every update. A full-budget run is an
    overnight job, and one that dies at hour six should not lose six hours.
    """
    # Imported here: only the replay path needs the solver stack, and importing
    # it at module load would make a plain --csv replot pay for it.
    from software.tools.fem_debug_replay import DEFAULT_SENSOR_PATH, _sensor_rows
    from software.tools.inversion_accuracy import _build_inversion, _run_inversion_now

    config = load_startup_config(config_path)
    path = path or Path(DEFAULT_SENSOR_PATH)
    snapshots = _sensor_rows(path, time_column, start_row, limit, stride, exclude_sensors)
    if not snapshots:
        raise SystemExit(f"No usable sensor rows parsed from {path}")

    inversion = _build_inversion(config, maxiter=maxiter, maxfun=maxfun)
    planned = len([i for i in range(1, len(snapshots)) if i % invert_every == 0])
    t_first = float(snapshots[0]["time_hours"])
    t_last = float(snapshots[-1]["time_hours"])
    print(
        f"Replaying {len(snapshots)} snapshots from {path.name} "
        f"(t = {t_first:.3f} .. {t_last:.3f} h)"
    )
    print(
        f"Inversion: params={inversion.invert_params}, window={inversion.window_size}, "
        f"maxiter={inversion.maxiter}, maxfun={inversion.maxfun}, every {invert_every} step(s) "
        f"-> {planned} update(s) planned"
    )
    print("Each update runs window_size x maxfun forward FEM solves; expect this to take hours.")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "inverted_parameters_replay.csv"
    rows: list[dict] = []

    for index, sensor_msg in enumerate(snapshots):
        parsed = inversion._parse_snapshot(sensor_msg)
        if parsed is not None:
            inversion._upsert_snapshot(parsed)
        if index == 0 or index % invert_every != 0:
            continue

        params, payload, elapsed = _run_inversion_now(inversion)
        if params is None:
            continue

        row = {
            "time_hours": round(float(sensor_msg["time_hours"]), 6),
            # Carried on every row so a --csv replot can still hold the final
            # step out to the end of the recording rather than to the last fit.
            "recording_end_hours": round(t_last, 6),
            "step": index,
            "elapsed_seconds": round(elapsed, 3),
            "final_loss": float(payload.get("final_loss", float("nan"))) if payload else float("nan"),
            "success": bool(payload.get("success", False)) if payload else False,
            "eval_count": int(payload.get("eval_count", 0)) if payload else 0,
        }
        row.update({name: value for name, value in params.items() if name in inversion.invert_params})
        rows.append(row)

        pretty = ", ".join(f"{k}={params[k]:.4g}" for k in inversion.invert_params if k in params)
        print(f"  * update {len(rows)}/{planned} @ t={row['time_hours']:6.3f} h  ({elapsed / 60.0:.1f} min)  {pretty}")
        remaining = planned - len(rows)
        if remaining > 0:
            mean_cost = float(np.mean([item["elapsed_seconds"] for item in rows]))
            print(f"    {remaining} left, ~{mean_cost / 60.0:.0f} min each -> ~{remaining * mean_cost / 3600.0:.1f} h remaining")

        pd.DataFrame(rows).to_csv(csv_path, index=False)

    if not rows:
        raise SystemExit(
            "The replay produced no inversion updates. With --invert-every "
            f"{invert_every} and {len(snapshots)} snapshots, none of the update points had the "
            f"{inversion.min_snapshots} snapshots the service requires."
        )
    print(f"\nWrote replay results to {csv_path}")
    return pd.DataFrame(rows)


def _load_records(
    *,
    csv_path: Path | None,
    config: dict,
    site_id: str,
    range_hours: float,
    limit: int,
) -> pd.DataFrame:
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} inversion records from {csv_path}")
        return df

    influx = InfluxHelper(build_influx_config(config.get("influxdb", {})))
    try:
        df = influx.query_inversion_parameters(
            site=site_id,
            range_start=_range_start(range_hours),
            limit=limit,
        )
    finally:
        influx.close()
    print(f"Queried {len(df)} inversion records from InfluxDB (site={site_id}, last {range_hours:g} h)")
    return df


def _prepare(df: pd.DataFrame, params: list[str], x_axis: str) -> tuple[pd.DataFrame, list[str], str]:
    if df.empty:
        raise SystemExit(
            "No inversion records found. Run the replay with --replay, widen --range-hours, "
            "or replot a previous export with --csv."
        )
    # Two kinds of input: a replay / inversion_updates export, which carries the
    # experiment clock, and an Influx query, which carries only ingest time.
    has_experiment_time = "time_hours" in df.columns
    if not has_experiment_time and "_time" not in df.columns:
        raise SystemExit(
            "Records carry neither 'time_hours' nor '_time'; cannot place them on a time axis."
        )

    present = [name for name in params if name in df.columns]
    if not present:
        raise SystemExit(
            f"None of the configured inverted parameters {params} are present in the records "
            f"(columns: {sorted(df.columns)})."
        )
    missing = [name for name in params if name not in present]
    if missing:
        print(f"Note: no records for {missing}; plotting {present}.")

    out = df.copy()
    if "_time" in out.columns:
        out["_time"] = pd.to_datetime(out["_time"], errors="coerce", utc=True)
    if has_experiment_time:
        out["time_hours"] = pd.to_numeric(out["time_hours"], errors="coerce")
        out = out.dropna(subset=["time_hours"]).sort_values("time_hours")
    else:
        out = out.dropna(subset=["_time"]).sort_values("_time")
    for name in present:
        out[name] = pd.to_numeric(out[name], errors="coerce")
    out = out.dropna(subset=present, how="all")
    if out.empty:
        raise SystemExit("Records contain no finite parameter values.")

    # 'auto' prefers the experiment clock whenever the input carries it; only an
    # Influx-only record set falls back to ingest time, and the label says so.
    if x_axis in {"auto", "experiment"} and has_experiment_time:
        out["_x"] = out["time_hours"]
        x_label = "Experiment time (h)"
    elif x_axis == "experiment":
        raise SystemExit(
            "--x-axis experiment needs a 'time_hours' column; Influx's fem_inversion records "
            "do not carry one. Use --replay or --csv with a replay export."
        )
    elif x_axis == "clock":
        out["_x"] = out["_time"]
        x_label = "Time (UTC ingest)"
    else:
        out["_x"] = (out["_time"] - out["_time"].iloc[0]).dt.total_seconds() / 3600.0
        x_label = "Time since first inversion (h, ingest clock)"
    return out, present, x_label


def _visible_bounds(
    y: np.ndarray,
    base_value: float | None,
    relative: tuple[float, float] | None,
    near_fraction: float = 0.02,
) -> list[tuple[float, float]]:
    """Bound edges the trajectory comes within ``near_fraction`` of the interval."""
    finite = y[np.isfinite(y)]
    if base_value is None or relative is None or not finite.size:
        return []
    lo, hi = relative
    width = (hi - lo) * base_value
    if width <= 0:
        return []
    edges = []
    for multiplier in (lo, hi):
        edge = base_value * multiplier
        if float(np.min(np.abs(finite - edge))) <= near_fraction * width:
            edges.append((edge, multiplier))
    return edges


def _view_limits(
    y: np.ndarray,
    base_value: float | None,
    relative: tuple[float, float] | None,
    pad_fraction: float = 0.12,
) -> tuple[float, float] | None:
    """Y-range covering the trajectory, the base line, and any bound it touches."""
    finite = y[np.isfinite(y)]
    if not finite.size:
        return None
    values = [float(finite.min()), float(finite.max())]
    if base_value is not None:
        values.append(float(base_value))
    values.extend(edge for edge, _ in _visible_bounds(y, base_value, relative))

    low, high = min(values), max(values)
    span = high - low
    pad = span * pad_fraction if span > 0 else max(abs(high), 1.0) * 0.05
    return low - pad, high + pad


def _plot(
    df: pd.DataFrame,
    params: list[str],
    *,
    x_label: str,
    base: dict[str, float],
    bounds: dict[str, tuple[float, float]],
    out_path: Path,
    anchor_x: float | None = None,
    x_end: float | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = df["_x"].to_numpy()
    # Failed solves still publish a parameter set; draw them hollow so a
    # trajectory that only moves on failed steps is not read as a fit.
    success = None
    success_column = _first_column(df, ["validation_success", "success"])
    if success_column:
        raw = df[success_column]
        if raw.dtype == bool:
            success = raw.to_numpy()
        else:
            numeric = pd.to_numeric(raw, errors="coerce")
            success = (
                raw.astype(str).str.lower().isin({"true", "1", "1.0"}).to_numpy()
                if numeric.isna().all()
                else numeric.fillna(1.0).to_numpy() > 0.5
            )

    fig, axes = plt.subplots(len(params), 1, figsize=(9.5, 2.5 * len(params) + 1.0), dpi=160, sharex=True)
    if len(params) == 1:
        axes = [axes]

    for idx, (ax, name) in enumerate(zip(axes, params)):
        color = PARAM_COLORS[idx % len(PARAM_COLORS)]
        y = df[name].to_numpy(dtype=float)

        base_value = base.get(name)
        if base_value is not None:
            ax.axhline(base_value, color=BASE_COLOR, linewidth=1.2, linestyle="--", label="initial value")
            # Name the number on the line. "2.00x base" in the summary is only
            # meaningful next to what the base actually was.
            ax.annotate(
                f"initial {base_value:g}",
                xy=(0.0, base_value),
                xycoords=ax.get_yaxis_transform(),
                xytext=(4, -11),
                textcoords="offset points",
                ha="left",
                fontsize=7.5,
                color=INK_MUTED,
            )

        # A bound is drawn only when the trajectory gets near it. Always drawing
        # both edges sets the y-range to the full feasible interval, which for
        # cap_s is 395-1580 and flattens every real movement inside it; a bound
        # the fit never approaches carries no information anyway. Pinning does,
        # so the near ones stay.
        limits = _view_limits(y, base_value, bounds.get(name))
        for edge, tag in _visible_bounds(y, base_value, bounds.get(name)):
            ax.axhline(edge, color=BOUNDS_COLOR, linewidth=1.2, linestyle=":", zorder=1)
            ax.annotate(
                f"bound ({tag:g}x)",
                # Left edge: the final value is direct-labelled on the right, and
                # a pinned trajectory puts both at the same height.
                xy=(0.0, edge),
                xycoords=ax.get_yaxis_transform(),
                xytext=(4, 3),
                textcoords="offset points",
                ha="left",
                fontsize=7.5,
                color=INK_MUTED,
            )
        if limits is not None:
            ax.set_ylim(*limits)

        # The twin runs on the initial values until the first fit lands, and on
        # each fitted set until the next one replaces it -- so the parameter in
        # force is a step function, not a line interpolated between updates.
        # Anchoring it at (t_start, initial) is what makes the trajectory start
        # where the run actually started.
        if anchor_x is not None and base_value is not None:
            step_x = np.concatenate([[anchor_x], x])
            step_y = np.concatenate([[base_value], y])
        else:
            step_x, step_y = x, y
        if x_end is not None and step_x.size and x_end > step_x[-1]:
            step_x = np.concatenate([step_x, [x_end]])
            step_y = np.concatenate([step_y, [step_y[-1]]])
        ax.plot(step_x, step_y, color=color, linewidth=2.0, drawstyle="steps-post",
                label="parameter in force", zorder=3)
        if success is None:
            ax.plot(x, y, "o", color=color, markersize=4.5, zorder=4)
        else:
            ax.plot(x[success], y[success], "o", color=color, markersize=4.5, zorder=4)
            if np.any(~success):
                ax.plot(
                    x[~success],
                    y[~success],
                    "o",
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markersize=5.5,
                    markeredgewidth=1.4,
                    zorder=4,
                    label="solver did not converge",
                )

        finite = np.isfinite(y)
        if np.any(finite):
            ax.annotate(
                f"{y[finite][-1]:.4g}",
                xy=(x[finite][-1], y[finite][-1]),
                xytext=(-4, 8),
                textcoords="offset points",
                ha="right",
                fontsize=9,
                color=INK,
            )

        unit = PARAM_UNITS.get(name)
        ax.set_title(f"{name}" + (f"  [{unit}]" if unit and unit != "-" else ""), fontsize=10, color=INK, pad=6)
        ax.grid(True, color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID_COLOR)
        ax.tick_params(labelsize=8, colors=INK_MUTED, length=3)

    axes[-1].set_xlabel(x_label, fontsize=9, color=INK_MUTED)
    fig.suptitle(
        f"Inverted soil parameters - evolution   |   {len(df)} inversion updates",
        fontsize=11,
        color=INK,
    )
    # One figure-level legend: per-axes it lands on top of the trace whenever a
    # parameter pins to its upper bound, which is exactly when the plot matters.
    seen: dict[str, object] = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            seen.setdefault(label, handle)
    if seen:
        fig.legend(
            seen.values(),
            seen.keys(),
            loc="lower center",
            ncol=len(seen),
            fontsize=8,
            frameon=False,
            labelcolor=INK_MUTED,
            bbox_to_anchor=(0.5, 0.0),
        )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def _print_summary(df: pd.DataFrame, params: list[str], base: dict[str, float]) -> None:
    print("\n--- summary ---")
    print(f"updates : {len(df)}")
    for name in params:
        y = pd.to_numeric(df[name], errors="coerce").dropna().to_numpy()
        if not y.size:
            continue
        base_value = base.get(name)
        drift = f"  ({y[-1] / base_value:.2f}x base)" if base_value else ""
        print(f"{name:<12}: first {y[0]:.4g}  final {y[-1]:.4g}  min {y.min():.4g}  max {y.max():.4g}{drift}")
    loss_column = _first_column(df, ["validation_final_loss", "final_loss"])
    if loss_column:
        loss = pd.to_numeric(df[loss_column], errors="coerce").dropna().to_numpy()
        if loss.size:
            print(f"{'final_loss':<12}: first {loss[0]:.5g}  final {loss[-1]:.5g}  min {loss.min():.5g}")
    success_column = _first_column(df, ["validation_success", "success"])
    if success_column:
        raw = df[success_column]
        numeric = pd.to_numeric(raw, errors="coerce")
        ok = (
            raw.astype(str).str.lower().isin({"true", "1", "1.0"}).to_numpy()
            if numeric.isna().all()
            else numeric.fillna(0.0).to_numpy() > 0.5
        )
        print(f"converged   : {int(ok.sum())}/{ok.size} updates")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the evolution of the FEM-inverted soil parameters, one subplot per parameter.",
    )
    parser.add_argument("--config", default="startup.conf", help="Startup config (Influx + inversion settings).")
    parser.add_argument("--out", default="artifacts/inversion", help="Output directory.")
    parser.add_argument("--site-id", default=None, help="Site ID tag (default: fem_inversion.site_id).")
    parser.add_argument("--range-hours", type=float, default=24.0, help="Lookback window in hours.")
    parser.add_argument("--limit", type=int, default=5000, help="Maximum records to fetch.")
    parser.add_argument(
        "--csv",
        default=None,
        help="Replot a previous export instead of querying InfluxDB.",
    )
    parser.add_argument(
        "--x-axis",
        choices=("auto", "experiment", "elapsed", "clock"),
        default="auto",
        help="'auto' (default) uses the experiment clock when the input carries one (--replay "
        "or a replay export) and ingest time otherwise; 'elapsed'/'clock' force ingest time.",
    )
    parser.add_argument(
        "--params",
        default=None,
        help="Comma-separated parameters to plot (default: fem_inversion.invert_params).",
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Pin the initial values the trajectories are drawn against, e.g. "
        "'cap_s=790,k_parameter=3.0,lambda_s=3.51'. Defaults to the config's "
        "fem_forecast.material overlaid with fem_inversion.base_params.",
    )

    replay_group = parser.add_argument_group(
        "replay (run the inversion service over a recording -- the slow path)"
    )
    replay_group.add_argument(
        "--replay",
        action="store_true",
        help="Replay a sensor recording through the real inversion service and plot the "
        "resulting trajectory on the experiment clock. Takes hours; results are flushed "
        "to CSV after every update.",
    )
    replay_group.add_argument(
        "--path",
        default=None,
        help="Recording to replay (default: the same April 8 export fem_debug_replay uses).",
    )
    replay_group.add_argument("--time-column", default=None, help="Override the time column header.")
    replay_group.add_argument("--start-row", type=int, default=1, help="First data row (1-based).")
    replay_group.add_argument(
        "--replay-limit", type=int, default=None,
        help="Truncate the recording to N rows. Default: the whole file.",
    )
    replay_group.add_argument("--stride", type=int, default=25, help="Use every Nth row (default 25).")
    replay_group.add_argument(
        "--invert-every", type=int, default=5,
        help="Fit every Nth replayed step (default 5).",
    )
    replay_group.add_argument("--maxiter", type=int, default=None, help="Override fem_inversion.maxiter.")
    replay_group.add_argument("--maxfun", type=int, default=None, help="Override fem_inversion.maxfun.")
    replay_group.add_argument(
        "--exclude-sensors", default=None,
        help="Comma-separated channels to drop, e.g. 'H290D165'.",
    )
    args = parser.parse_args()

    config = load_startup_config(args.config)
    inversion_cfg = config.get("fem_inversion", {})
    site_id = args.site_id or str(inversion_cfg.get("site_id", "default"))
    if args.params:
        params = [name.strip() for name in args.params.split(",") if name.strip()]
    else:
        params = list(inversion_cfg.get("invert_params", ["cap_s", "k_parameter", "lambda_s"]))

    base = _base_values(config, _parse_base_override(args.base))
    out_dir = Path(args.out)

    if args.replay:
        if args.csv:
            raise SystemExit("--replay and --csv are alternatives; pass only one.")
        df = _run_replay(
            path=Path(args.path) if args.path else None,
            config_path=args.config,
            out_dir=out_dir,
            time_column=args.time_column,
            start_row=int(args.start_row),
            limit=args.replay_limit,
            stride=max(int(args.stride), 1),
            invert_every=max(int(args.invert_every), 1),
            maxiter=args.maxiter,
            maxfun=args.maxfun,
            exclude_sensors=set(args.exclude_sensors.split(",")) if args.exclude_sensors else None,
        )
    else:
        df = _load_records(
            csv_path=Path(args.csv) if args.csv else None,
            config=config,
            site_id=site_id,
            range_hours=float(args.range_hours),
            limit=int(args.limit),
        )
    df, present, x_label = _prepare(df, params, args.x_axis)

    # State the reference the trajectories are read against, and say so loudly
    # when one is missing: a panel with no initial line is not a parameter that
    # started at zero, it is one this tool could not resolve a base for.
    resolved = ", ".join(f"{name}={base[name]:g}" for name in present if name in base)
    print(f"Initial values: {resolved or '(none resolved)'}")
    unresolved = [name for name in present if name not in base]
    if unresolved:
        print(
            f"WARNING: no initial value for {unresolved}; their panels get no reference line. "
            "Set it in fem_forecast.material / fem_inversion.base_params, or pass --base."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.csv is None and not args.replay:
        csv_path = out_dir / "inverted_parameters.csv"
        df.drop(columns=["_x"]).to_csv(csv_path, index=False)
        print(f"Wrote records to {csv_path}")

    # Anchoring only makes sense on the experiment clock, where t=0 is the start
    # of the run and the twin demonstrably held the initial values until the
    # first fit. On the ingest clock there is no such origin to anchor to.
    on_experiment_clock = x_label.startswith("Experiment time")
    anchor_x = 0.0 if on_experiment_clock else None
    x_end = None
    if on_experiment_clock:
        if "recording_end_hours" in df.columns:
            x_end = float(pd.to_numeric(df["recording_end_hours"], errors="coerce").max())
        else:
            x_end = float(df["_x"].max())

    png_path = out_dir / "inverted_parameters_evolution.png"
    _plot(
        df,
        present,
        x_label=x_label,
        base=base,
        bounds=_relative_bounds(config),
        out_path=png_path,
        anchor_x=anchor_x,
        x_end=x_end,
    )
    print(f"Wrote plot to {png_path}")

    _print_summary(df, present, base)


if __name__ == "__main__":
    main()

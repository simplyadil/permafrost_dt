"""Chain of custody for one control loop: observe -> decide -> act -> confirm -> respond.

The other tools in here argue that the twin *describes* the experiment. This one
argues that it *ran* it: that a measurement crossed a limit, the twin decided
something, a command left the software, the RS232 bridge wrote it to the
PolyScience circulator, and the physical fluid loop changed afterwards.

Prediction accuracy cannot show that. Only provenance can, so every panel is one
recorded link, all on the same experiment clock, with the event rails drawn
through all of them so a single decision can be followed down the stack:

    OBSERVE   thaw_front_metrics_v2   r_max measured, r_max forecast, vs the limit
    DECIDE    safety_alerts           the graded status ribbon and its trigger
    ACT       heater_actions          source=heater_actuation -- the command issued
    CONFIRM   heater_actions          source=polyscience -- written back by the bridge
    RESPOND   fluid_temperatures      inlet/outlet -- the loop physically changing

The two ``heater_actions`` sources are the crux. ``heater_actuation`` writes what
the twin decided; ``polyscience_control_server`` writes a *separate* row after
pushing that command down the serial link. Two rows from two processes for one
decision, and the delay between them is the actuation latency, measured rather
than asserted. A decision with no matching ``polyscience`` row is a command that
never reached hardware -- and the summary counts those, because a chain-of-custody
plot that hides its broken links is decoration.

Honest limits, both worth knowing before this figure is used as evidence:

* ``write_heater_action`` has no status field, so the bridge's verify-after-apply
  result (``applied`` / ``failed`` / the controller readback state) is logged but
  never persisted. A ``source=polyscience`` row therefore proves the command
  reached the hardware bridge and was written back -- not that the controller
  acknowledged it. Adding ``status`` to that measurement would close the gap.
* Timestamps come from two clocks. The x-axis is ``time_hours``, the experiment
  clock every service stamps its records with; latency between links is measured
  on the InfluxDB ingest clock (``_time``), which is the only one with sub-second
  resolution.

Outputs, under ``--out`` (default ``artifacts/closed_loop``):

* ``closed_loop_evidence.png``  -- the stacked chain
* ``closed_loop_events.csv``    -- one row per decision: alert, command, confirmation, latencies

    python -m software.tools.closed_loop_evidence --range-hours 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from software.digital_twin.data_access.influx_utils import InfluxHelper
from software.startup.service_runners import build_influx_config
from software.startup.utils.config import load_startup_config

MEASURED_COLOR = "#0369a1"
FORECAST_COLOR = "#7c3aed"
LIMIT_COLOR = "#b91c1c"
COMMAND_COLOR = "#c2410c"
HARDWARE_COLOR = "#15803d"
INLET_COLOR = "#c2410c"
OUTLET_COLOR = "#0369a1"
RAIL_COLOR = "#94a3b8"
GRID_COLOR = "#e2e8f0"
INK = "#1e293b"
INK_MUTED = "#64748b"

# Graded safety status -> ribbon colour. Status is a state, not a series, so it
# gets the reserved status palette and always travels with its text label.
STATUS_COLORS = {
    "ok": "#15803d",
    "warning": "#d97706",
    "alert": "#b91c1c",
    "critical": "#b91c1c",
}
UNKNOWN_STATUS_COLOR = "#94a3b8"

COMMAND_SOURCE = "heater_actuation"
HARDWARE_SOURCE = "polyscience"


def _range_start(hours: float) -> str:
    return f"-{max(int(hours * 3600), 60)}s"


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _prepare(df: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    """Sort by experiment time and coerce the columns actually needed."""
    if df is None or df.empty or "time_hours" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["time_hours"] = _numeric(out, "time_hours")
    if "_time" in out.columns:
        out["_time"] = pd.to_datetime(out["_time"], errors="coerce", utc=True)
    for column in value_columns:
        if column in out.columns:
            out[column] = _numeric(out, column)
    return out.dropna(subset=["time_hours"]).sort_values("time_hours").reset_index(drop=True)


# ---------------------------------------------------------------------------
# CHAIN OF CUSTODY
# ---------------------------------------------------------------------------
def _change_events(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Rows where the loop's output actually changed, plus the first row.

    The services re-publish their state on every sensor snapshot, so a five-hour
    run holds well over a thousand identical "heat at 10 C" rows. Every one of
    them is a command, but only the transitions are decisions -- and a rail drawn
    per row paints the figure solid grey. Repetition is summarised, not drawn.
    """
    if df.empty:
        return df
    present = [key for key in keys if key in df.columns]
    if not present:
        return df.head(1)
    signature = df[present].astype(str).agg("|".join, axis=1)
    changed = signature.ne(signature.shift())
    return df[changed].reset_index(drop=True)


def _status_spans(alerts: pd.DataFrame, t_end: float) -> list[tuple[float, float, str]]:
    """Collapse the alert stream into contiguous (start, end, status) spans."""
    if alerts.empty or "status" not in alerts.columns:
        return []
    spans: list[tuple[float, float, str]] = []
    times = alerts["time_hours"].to_numpy(dtype=float)
    statuses = [str(value) for value in alerts["status"].tolist()]
    start = times[0]
    current = statuses[0]
    for idx in range(1, len(times)):
        if statuses[idx] != current:
            spans.append((start, times[idx], current))
            start = times[idx]
            current = statuses[idx]
    spans.append((start, max(t_end, times[-1]), current))
    return spans


def _link_events(
    alerts: pd.DataFrame,
    commands: pd.DataFrame,
    hardware: pd.DataFrame,
    *,
    tolerance_seconds: float,
) -> pd.DataFrame:
    """Walk each decision along the chain and measure how long each hop took.

    Matching is on the ingest clock and forward in time only: a command can
    follow the alert that caused it, never precede it. ``merge_asof`` with
    ``direction='forward'`` encodes exactly that, and leaves the row unmatched
    when nothing arrived inside the tolerance -- which is the broken-link case
    this table exists to expose.
    """
    if alerts.empty or "_time" not in alerts.columns:
        return pd.DataFrame()

    # A decision is a transition -- the loop changing its mind -- plus anything
    # that actually tripped. Treating all 1250 re-published "ok" rows as separate
    # decisions would report a thousand broken links for one quiet run.
    decisions = _change_events(alerts, ["status", "recommended_action", "reason", "triggered"])
    if "triggered" in alerts.columns:
        triggered = alerts["triggered"].astype(str).str.lower().isin({"true", "1", "1.0"})
        if triggered.any():
            decisions = (
                pd.concat([decisions, alerts[triggered]])
                .drop_duplicates(subset=["time_hours"])
                .sort_values("time_hours")
                .reset_index(drop=True)
            )
    if decisions.empty:
        return pd.DataFrame()

    tolerance = pd.Timedelta(seconds=float(tolerance_seconds))
    left = decisions.dropna(subset=["_time"]).sort_values("_time")
    columns = {"_time": "alert_time", "time_hours": "alert_time_hours"}
    if "status" in left.columns:
        columns["status"] = "alert_status"
    if "reason" in left.columns:
        columns["reason"] = "alert_reason"
    merged = left[list(columns)].rename(columns=columns)

    for frame, prefix in ((commands, "command"), (hardware, "hardware")):
        if frame.empty or "_time" not in frame.columns:
            # Must carry the same tz-aware dtype as alert_time, or the latency
            # subtraction below raises instead of yielding NaT.
            merged[f"{prefix}_time"] = pd.Series(pd.NaT, index=merged.index, dtype="datetime64[ns, UTC]")
            continue
        right_columns = {"_time": f"{prefix}_time"}
        for name, target in (("action", f"{prefix}_action"), ("setpoint", f"{prefix}_setpoint")):
            if name in frame.columns:
                right_columns[name] = target
        right = frame.dropna(subset=["_time"]).sort_values("_time")[list(right_columns)]
        right = right.rename(columns=right_columns)
        merged = pd.merge_asof(
            merged.sort_values("alert_time"),
            right,
            left_on="alert_time",
            right_on=f"{prefix}_time",
            direction="forward",
            tolerance=tolerance,
        )

    merged["decide_to_command_s"] = (merged["command_time"] - merged["alert_time"]).dt.total_seconds()
    merged["command_to_hardware_s"] = (merged["hardware_time"] - merged["command_time"]).dt.total_seconds()
    merged["reached_hardware"] = merged["hardware_time"].notna()
    return merged


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------
def _stage_label(ax, stage: str, description: str) -> None:
    """The chain reads down the left margin: one verb per link."""
    ax.text(
        -0.085, 0.5, stage,
        transform=ax.transAxes, rotation=90, va="center", ha="center",
        fontsize=9, fontweight="bold", color=INK,
    )
    ax.set_ylabel(description, fontsize=8.5, color=INK_MUTED)


def _style(ax) -> None:
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(labelsize=8, colors=INK_MUTED, length=3)


def _plot_chain(
    *,
    thaw: pd.DataFrame,
    forecast: pd.DataFrame,
    alerts: pd.DataFrame,
    commands: pd.DataFrame,
    hardware: pd.DataFrame,
    fluid: pd.DataFrame,
    events: pd.DataFrame,
    limit_radius_m: float | None,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    frames = [frame for frame in (thaw, forecast, alerts, commands, hardware, fluid) if not frame.empty]
    if not frames:
        raise SystemExit("Nothing to plot: none of the loop measurements returned data.")
    t_min = min(float(frame["time_hours"].min()) for frame in frames)
    t_max = max(float(frame["time_hours"].max()) for frame in frames)
    if t_max <= t_min:
        t_max = t_min + 1e-3

    fig, axes = plt.subplots(
        4, 1, figsize=(11.5, 9.5), dpi=160, sharex=True,
        gridspec_kw={"height_ratios": [3.0, 0.7, 2.2, 2.2], "hspace": 0.16},
    )
    ax_obs, ax_dec, ax_act, ax_res = axes

    # --- OBSERVE -----------------------------------------------------------
    if not thaw.empty and "radius_max_m" in thaw.columns:
        ax_obs.plot(
            thaw["time_hours"], thaw["radius_max_m"] * 1000.0,
            color=MEASURED_COLOR, linewidth=2.0, label="measured r_max",
        )
    if not forecast.empty and "radius_max_m" in forecast.columns:
        ax_obs.plot(
            forecast["time_hours"], forecast["radius_max_m"] * 1000.0,
            color=FORECAST_COLOR, linewidth=1.6, linestyle="--", label="forecast r_max",
        )
    if limit_radius_m:
        ax_obs.axhline(
            limit_radius_m * 1000.0, color=LIMIT_COLOR, linewidth=1.4, linestyle="-.",
            label=f"safety limit ({limit_radius_m * 1000:.0f} mm)",
        )
        ax_obs.axhspan(
            limit_radius_m * 1000.0, ax_obs.get_ylim()[1] if ax_obs.get_ylim()[1] > limit_radius_m * 1000.0
            else limit_radius_m * 1200.0,
            color=LIMIT_COLOR, alpha=0.06, linewidth=0,
        )
    _style(ax_obs)
    _stage_label(ax_obs, "OBSERVE", "Thaw radius (mm)")
    ax_obs.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK_MUTED)

    # --- DECIDE ------------------------------------------------------------
    spans = _status_spans(alerts, t_max)
    seen_status: set[str] = set()
    for start, end, status in spans:
        color = STATUS_COLORS.get(status.lower(), UNKNOWN_STATUS_COLOR)
        ax_dec.axvspan(start, end, color=color, alpha=0.35, linewidth=0)
        # Label each state once, where it first appears: colour alone never
        # carries the meaning of a status.
        if status not in seen_status and (end - start) > (t_max - t_min) * 0.02:
            ax_dec.text(
                start + (end - start) / 2, 0.5, status,
                ha="center", va="center", fontsize=8, color=INK,
            )
            seen_status.add(status)
    ax_dec.set_ylim(0, 1)
    ax_dec.set_yticks([])
    for side in ("top", "right", "left"):
        ax_dec.spines[side].set_visible(False)
    ax_dec.spines["bottom"].set_color(GRID_COLOR)
    ax_dec.tick_params(labelsize=8, colors=INK_MUTED, length=3)
    _stage_label(ax_dec, "DECIDE", "safety status")

    # --- ACT + CONFIRM -----------------------------------------------------
    # Setpoint where the loop commands one; otherwise the actions themselves,
    # mapped to rows, so a stop/hold-only run still shows its command stream.
    command_sp = _numeric(commands, "setpoint") if not commands.empty else pd.Series(dtype=float)
    hardware_sp = _numeric(hardware, "setpoint") if not hardware.empty else pd.Series(dtype=float)
    use_setpoint = bool(command_sp.notna().any() or hardware_sp.notna().any())

    command_changes = _change_events(commands, ["action", "setpoint"])
    hardware_changes = _change_events(hardware, ["action", "setpoint"])

    if use_setpoint:
        if not commands.empty:
            ax_act.step(
                commands["time_hours"], command_sp, where="post",
                color=COMMAND_COLOR, linewidth=2.0,
                label=f"commanded setpoint ({len(commands)} msgs)",
            )
            # Markers on transitions only; one per row is a solid bar.
            ax_act.plot(
                command_changes["time_hours"], _numeric(command_changes, "setpoint"),
                "o", color=COMMAND_COLOR, markersize=6, label="command change",
            )
        if not hardware.empty:
            ax_act.plot(
                hardware_changes["time_hours"], _numeric(hardware_changes, "setpoint"), "s",
                markerfacecolor="white", markeredgecolor=HARDWARE_COLOR, markeredgewidth=1.6,
                markersize=8, label="written to PolyScience (RS232)",
            )
        else:
            ax_act.text(
                0.5, 0.12,
                f"no source={HARDWARE_SOURCE} rows in this window - hardware link unrecorded",
                transform=ax_act.transAxes, ha="center", fontsize=8.5, color=LIMIT_COLOR,
            )
        ax_act.set_ylabel("Heater setpoint (C)", fontsize=8.5, color=INK_MUTED)
    else:
        actions = sorted(
            {str(value) for frame in (commands, hardware) if not frame.empty and "action" in frame.columns
             for value in frame["action"].tolist()}
        )
        rows = {name: idx for idx, name in enumerate(actions)}
        for frame, color, marker, label, filled in (
            (command_changes, COMMAND_COLOR, "o", "command change", True),
            (hardware_changes, HARDWARE_COLOR, "s", "written to PolyScience (RS232)", False),
        ):
            if frame.empty or "action" not in frame.columns:
                continue
            y = [rows[str(value)] for value in frame["action"].tolist()]
            ax_act.plot(
                frame["time_hours"], y, marker, linestyle="none",
                color=color if filled else None,
                markerfacecolor=color if filled else "white",
                markeredgecolor=color, markeredgewidth=1.6,
                markersize=7 if filled else 9, label=label,
            )
        ax_act.set_yticks(list(rows.values()))
        ax_act.set_yticklabels(list(rows), fontsize=8)
        ax_act.set_ylim(-0.6, max(len(rows) - 0.4, 0.6))
        ax_act.set_ylabel("Heater action", fontsize=8.5, color=INK_MUTED)
    _style(ax_act)
    _stage_label(ax_act, "ACT + CONFIRM", ax_act.get_ylabel())
    if not commands.empty or not hardware.empty:
        ax_act.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK_MUTED)

    # --- RESPOND -----------------------------------------------------------
    if not fluid.empty:
        if "t_inlet_c" in fluid.columns:
            ax_res.plot(fluid["time_hours"], fluid["t_inlet_c"], color=INLET_COLOR,
                        linewidth=2.0, label="fluid inlet")
        if "t_outlet_c" in fluid.columns:
            ax_res.plot(fluid["time_hours"], fluid["t_outlet_c"], color=OUTLET_COLOR,
                        linewidth=2.0, label="fluid outlet")
        ax_res.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK_MUTED)
    _style(ax_res)
    _stage_label(ax_res, "RESPOND", "Fluid temperature (C)")
    ax_res.set_xlabel("Experiment time (h)", fontsize=9, color=INK_MUTED)

    # --- event rails -------------------------------------------------------
    # One vertical line per command, through every panel: this is what makes the
    # figure a chain rather than four plots that happen to share an axis.
    rails = list(command_changes["time_hours"]) if not command_changes.empty else []
    for ax in axes:
        for rail in rails:
            ax.axvline(rail, color=RAIL_COLOR, linewidth=0.9, alpha=0.5, zorder=0)
    for ax in axes:
        ax.set_xlim(t_min, t_max)

    # --- provenance banner -------------------------------------------------
    n_alerts = int(len(events)) if not events.empty else 0
    n_commands = int(events["command_time"].notna().sum()) if not events.empty else len(commands)
    n_hardware = int(events["reached_hardware"].sum()) if not events.empty else len(hardware)
    if not events.empty and events["decide_to_command_s"].notna().any():
        d2c = f"{float(events['decide_to_command_s'].median()):.2f}s"
    else:
        d2c = "n/a"
    if not events.empty and events["command_to_hardware_s"].notna().any():
        c2h = f"{float(events['command_to_hardware_s'].median()):.2f}s"
    else:
        c2h = "n/a"

    fig.suptitle("Closed-loop chain of custody: the twin acting on the physical rig",
                 fontsize=12.5, color=INK, y=0.985)
    fig.text(
        0.5, 0.955,
        f"{n_alerts} triggering decision(s)  ->  {n_commands} command(s) issued  ->  "
        f"{n_hardware} written to hardware   |   median decide->command {d2c}, "
        f"command->RS232 {c2h}",
        ha="center", fontsize=9.5, color=INK_MUTED,
    )

    handles = [
        Line2D([0], [0], color=RAIL_COLOR, linewidth=0.9, alpha=0.7, label="command event"),
        Patch(facecolor=STATUS_COLORS["ok"], alpha=0.35, label="status ok"),
        Patch(facecolor=STATUS_COLORS["warning"], alpha=0.35, label="status warning"),
        Patch(facecolor=STATUS_COLORS["alert"], alpha=0.35, label="status alert/critical"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               frameon=False, labelcolor=INK_MUTED, bbox_to_anchor=(0.5, 0.0))

    # Explicit margins, not tight_layout: the panels carry a fixed height ratio
    # and rotated stage labels in the left margin, which tight_layout undoes.
    fig.subplots_adjust(left=0.105, right=0.985, top=0.90, bottom=0.075, hspace=0.16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def _print_summary(events: pd.DataFrame, commands: pd.DataFrame, hardware: pd.DataFrame) -> None:
    print("\n--- chain of custody ---")
    print(f"commands issued (heater_actuation) : {len(commands)}")
    print(f"rows written back (polyscience)    : {len(hardware)}")
    if events.empty:
        print("No triggering decisions in this window; nothing to trace along the chain.")
        return

    reached = int(events["reached_hardware"].sum())
    print(f"triggering decisions               : {len(events)}")
    print(f"decisions that reached hardware    : {reached}/{len(events)}")
    for column, label in (
        ("decide_to_command_s", "decide -> command"),
        ("command_to_hardware_s", "command -> RS232 "),
    ):
        values = events[column].dropna().to_numpy(dtype=float)
        if values.size:
            print(f"{label}                : median {np.median(values):.2f}s  max {values.max():.2f}s")
    if reached < len(events):
        print(
            f"WARNING: {len(events) - reached} decision(s) have no polyscience row inside the match "
            "tolerance -- either the bridge was down, on a different site_id, or the command never "
            "reached the serial link. Raise --match-tolerance-seconds before concluding it broke."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the recorded chain of custody from measurement to physical actuation.",
    )
    parser.add_argument("--config", default="startup.conf", help="Startup config (Influx + safety settings).")
    parser.add_argument("--out", default="artifacts/closed_loop", help="Output directory.")
    parser.add_argument("--site-id", default=None, help="Site ID tag (default: safety_monitor.site_id).")
    parser.add_argument("--range-hours", type=float, default=12.0, help="Lookback window in hours.")
    parser.add_argument("--limit", type=int, default=20000, help="Maximum records per measurement.")
    parser.add_argument(
        "--limit-radius-m",
        type=float,
        default=None,
        help="Safety limit to draw (default: safety_monitor.limit_radius_m).",
    )
    parser.add_argument(
        "--match-tolerance-seconds",
        type=float,
        default=30.0,
        help="How long after a decision a command/hardware row may arrive and still be linked to it.",
    )
    args = parser.parse_args()

    config = load_startup_config(args.config)
    safety_cfg = config.get("safety_monitor", {})
    site_id = args.site_id or str(safety_cfg.get("site_id", "default"))
    limit_radius_m = args.limit_radius_m
    if limit_radius_m is None:
        raw_limit = safety_cfg.get("limit_radius_m")
        limit_radius_m = float(raw_limit) if isinstance(raw_limit, (int, float)) else None

    influx = InfluxHelper(build_influx_config(config.get("influxdb", {})))
    range_start = _range_start(float(args.range_hours))
    try:
        thaw = influx.query_thaw_front_metrics(site=site_id, range_start=range_start, limit=args.limit)
        forecast = influx.query_fem_forecast_metrics(site=site_id, range_start=range_start, limit=args.limit)
        alerts = influx.query_safety_alerts(site=site_id, range_start=range_start, limit=args.limit)
        actions = influx.query_heater_actions(site=site_id, range_start=range_start, limit=args.limit)
        fluid = influx.query_fluid_temperatures(site=site_id, range_start=range_start, limit=args.limit)
    finally:
        influx.close()

    thaw = _prepare(thaw, ["radius_max_m", "radius_min_m"])
    forecast = _prepare(forecast, ["radius_max_m"])
    alerts = _prepare(alerts, ["measured_radius_m", "forecast_radius_m", "limit_radius_m"])
    actions = _prepare(actions, ["setpoint"])
    fluid = _prepare(fluid, ["t_inlet_c", "t_outlet_c"])

    # Split by who wrote the row. Only the RS232 bridge counts as hardware; every
    # other producer (heater_actuation, or safety_monitor writing directly) is
    # software deciding. Discovered from the data rather than assumed, because
    # which services were running is exactly what this figure is reporting.
    commands = pd.DataFrame()
    hardware = pd.DataFrame()
    sources: list[str] = []
    if not actions.empty and "source" in actions.columns:
        sources = sorted({str(value) for value in actions["source"].tolist()})
        hardware = actions[actions["source"].astype(str) == HARDWARE_SOURCE].reset_index(drop=True)
        commands = actions[actions["source"].astype(str) != HARDWARE_SOURCE].reset_index(drop=True)
    elif not actions.empty:
        commands = actions

    for name, frame in (
        ("thaw_front_metrics_v2", thaw),
        ("fem_forecast_metrics", forecast),
        ("safety_alerts", alerts),
        ("heater_actions", actions),
        ("fluid_temperatures", fluid),
    ):
        print(f"{name:<24}: {len(frame)} record(s)")
    if sources:
        print(f"{'heater_actions sources':<24}: {', '.join(sources)}")
        print(f"{'  software commands':<24}: {len(commands)} row(s)")
        print(f"{'  hardware writes':<24}: {len(hardware)} row(s) (source={HARDWARE_SOURCE})")
        if hardware.empty:
            print(
                f"NOTE: no source={HARDWARE_SOURCE} rows in this window. The ACT+CONFIRM panel can "
                "show what the twin commanded but not that it reached the circulator -- either "
                "polyscience_control was not running, ran with dry_run, or used another site_id."
            )

    events = _link_events(alerts, commands, hardware, tolerance_seconds=float(args.match_tolerance_seconds))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not events.empty:
        events_path = out_dir / "closed_loop_events.csv"
        events.to_csv(events_path, index=False)
        print(f"\nWrote linked events to {events_path}")

    png_path = out_dir / "closed_loop_evidence.png"
    _plot_chain(
        thaw=thaw,
        forecast=forecast,
        alerts=alerts,
        commands=commands,
        hardware=hardware,
        fluid=fluid,
        events=events,
        limit_radius_m=limit_radius_m,
        out_path=png_path,
    )
    print(f"Wrote plot to {png_path}")

    _print_summary(events, commands, hardware)


if __name__ == "__main__":
    main()

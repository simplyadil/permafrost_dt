"""Streamlit dashboard for the permafrost digital twin."""

from __future__ import annotations

import io
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import griddata

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from software.digital_twin.data_access.influx_utils import FEM_TEMPERATURE_BASELINE_MEASUREMENT, InfluxHelper
from software.startup.service_runners import build_influx_config
from software.startup.utils.config import load_startup_config


DOMAIN_RADIUS_M = 0.24
DOMAIN_DEPTH_M = 0.30
THERMOSYPHON_RADIUS_M = 0.0075

SAFE_PCT_DEFAULT = 0.60
CAUTION_PCT_DEFAULT = 0.90

_MAX_ANIMATION_FRAMES = 40   # hard cap on Plotly frames sent to browser
_ANIMATION_GRID_RES   = 80   # lower grid resolution for animation (vs 120 for static)

# Export defaults. The browser modebar camera button uses these; a scale > 1
# multiplies the pixel dimensions so a 700x440 chart downloads at 2100x1320.
_EXPORT_MODEBAR_SCALE = 3
_EXPORT_WIDTH_PX = 1200          # canvas width used for server-side (kaleido) exports
_STATIC_FORMATS: dict[str, tuple[str, int, str]] = {
    # label -> (plotly format, scale, mime)
    "PNG - high-res (3x)": ("png", 3, "image/png"),
    "PNG - ultra (4x)": ("png", 4, "image/png"),
    "SVG - vector": ("svg", 1, "image/svg+xml"),
    "PDF - vector": ("pdf", 1, "application/pdf"),
}


# -----------------------------
# Plot export helpers
# -----------------------------

def _plotly_config(filename: str) -> dict:
    """Config for st.plotly_chart giving a high-resolution modebar PNG download.

    The camera icon in every chart's toolbar will now save a `filename`.png at
    `_EXPORT_MODEBAR_SCALE` times the on-screen pixel dimensions - zero extra
    dependencies, works fully client-side.
    """
    return {
        "displaylogo": False,
        "toImageButtonOptions": {
            "format": "png",
            "filename": filename,
            "scale": _EXPORT_MODEBAR_SCALE,
        },
    }


def _render_static_image(fig: go.Figure, fmt: str, scale: int) -> bytes:
    """Render a figure to PNG/SVG/PDF bytes via kaleido (raises if unavailable)."""
    height = int(fig.layout.height) if fig.layout.height else 600
    return fig.to_image(
        format=fmt,
        width=_EXPORT_WIDTH_PX,
        height=height,
        scale=scale,
    )


def _render_animation_gif(fig: go.Figure, scale: int = 2, duration_ms: int = 350) -> bytes:
    """Stitch a frame-animated Plotly figure into an animated GIF.

    Each Plotly frame carries only the fields that change (contour ``z`` and the
    sensor scatter positions/colours); we merge those onto a deep copy of the
    base traces, strip the play/scrub UI, render the frame to PNG with kaleido,
    then assemble the PNGs into a looping GIF with Pillow.
    """
    from PIL import Image  # local import: only needed on the GIF path

    frames = list(fig.frames or [])
    if not frames:
        raise ValueError("Figure has no animation frames to export.")

    height = int(fig.layout.height) if fig.layout.height else 500
    pil_frames: list = []
    for frame in frames:
        static = go.Figure(fig)
        static.frames = ()
        # Apply this frame's partial trace updates onto the base traces.
        target_idx = frame.traces if frame.traces is not None else range(len(frame.data))
        for idx, trace_update in zip(target_idx, frame.data):
            static.data[idx].update(trace_update)
        # Drop the animation controls and pin the timestamp into the title.
        static.update_layout(updatemenus=[], sliders=[], margin=dict(l=40, r=20, t=60, b=40))
        try:
            label = f"t = {float(frame.name):.2f} h"
        except (TypeError, ValueError):
            label = str(frame.name)
        static.update_layout(title={"text": f"Heat propagation - {label}"})
        png = static.to_image(format="png", width=_EXPORT_WIDTH_PX, height=height, scale=scale)
        pil_frames.append(Image.open(io.BytesIO(png)).convert("RGB"))

    buffer = io.BytesIO()
    pil_frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=True,
    )
    return buffer.getvalue()


def _export_controls(fig: go.Figure, *, key: str, basename: str) -> None:
    """Render an 'Export' expander with static (PNG/SVG/PDF) and GIF downloads.

    Generation is gated behind a button so no kaleido rendering happens on a
    normal page load; the produced bytes are cached in session_state so the
    download_button survives the rerun triggered by the download click.
    """
    is_animated = bool(fig.frames)
    with st.expander("Export", expanded=False):
        fmt_col, gen_col = st.columns((2, 1))
        fmt_label = fmt_col.selectbox(
            "Format", list(_STATIC_FORMATS.keys()), key=f"{key}_fmt"
        )
        if gen_col.button("Prepare file", key=f"{key}_gen"):
            fmt, scale, mime = _STATIC_FORMATS[fmt_label]
            try:
                with st.spinner("Rendering image..."):
                    st.session_state[f"{key}_blob"] = (
                        _render_static_image(fig, fmt, scale),
                        f"{basename}.{fmt}",
                        mime,
                    )
            except Exception as exc:  # kaleido missing or render failure
                st.session_state.pop(f"{key}_blob", None)
                st.warning(
                    "High-quality export needs the `kaleido` package "
                    f"(`pip install kaleido`). Details: {exc}"
                )

        blob = st.session_state.get(f"{key}_blob")
        if blob:
            data, filename, mime = blob
            st.download_button(
                f"Download {filename}",
                data,
                file_name=filename,
                mime=mime,
                key=f"{key}_dl",
            )

        if is_animated:
            st.divider()
            gif_meta_col, gif_gen_col = st.columns((2, 1))
            fps = gif_meta_col.slider(
                "GIF speed (frames/sec)", 1, 12, 3, key=f"{key}_fps"
            )
            if gif_gen_col.button("Prepare GIF", key=f"{key}_gifgen"):
                try:
                    with st.spinner(
                        f"Rendering {len(fig.frames)} frames to GIF..."
                    ):
                        st.session_state[f"{key}_gif"] = _render_animation_gif(
                            fig, duration_ms=int(1000 / fps)
                        )
                except Exception as exc:
                    st.session_state.pop(f"{key}_gif", None)
                    st.warning(
                        "GIF export needs `kaleido` and `pillow` "
                        f"(`pip install kaleido pillow`). Details: {exc}"
                    )
            gif_bytes = st.session_state.get(f"{key}_gif")
            if gif_bytes:
                st.download_button(
                    "Download animation GIF",
                    gif_bytes,
                    file_name=f"{basename}.gif",
                    mime="image/gif",
                    key=f"{key}_gifdl",
                )


def _chart(fig: go.Figure, *, key: str, basename: str, export: bool = True) -> None:
    """Render a Plotly chart full-width with a high-res modebar and export panel.

    Set ``export=False`` to skip the download expander (e.g. the safety gauge,
    which nobody exports).
    """
    st.plotly_chart(fig, use_container_width=True, config=_plotly_config(basename), key=key)
    if export:
        _export_controls(fig, key=key, basename=basename)


# -----------------------------
# Closed-loop data-flow status widget
# -----------------------------

# Ordered stages of the closed control loop. Each stage's liveness is read from
# the Influx ingest time (`_time`) of the measurement(s) it produces; freshest
# frame wins when a stage aggregates several. The last stage (Heater) feeds
# back onto the physical rig the Sensors measure - that edge closes the loop.
_LOOP_STAGES: tuple[dict, ...] = (
    {"name": "Sensors", "sources": ("sensors",)},
    {"name": "Digital twin", "sources": ("forecast_temperature", "thaw_metrics")},
    {"name": "Safety", "sources": ("safety_alerts",)},
    {"name": "Control", "sources": ("heater_actions",)},
    {"name": "Heater", "sources": ("fluid_temperatures",)},
)

_SEVERITY_RANK = {"green": 0, "yellow": 1, "red": 2}
_OVERALL_STATE = {
    "green": ("live", "dtflow-live"),
    "yellow": ("degraded", "dtflow-degraded"),
    "red": ("offline", "dtflow-offline"),
}


def _fmt_age(age_seconds: float) -> str:
    if age_seconds < 90:
        return f"{age_seconds:.0f}s"
    if age_seconds < 5400:
        return f"{age_seconds / 60:.0f}m"
    return f"{age_seconds / 3600:.1f}h"


def _stage_age_seconds(*frames: pd.DataFrame) -> float | None:
    """Seconds since the newest Influx ingest time across the given frames.

    Returns None when none of the frames carry a usable ``_time`` (i.e. the stage
    has produced nothing in the query window).
    """
    newest: pd.Timestamp | None = None
    for df in frames:
        if df is None or getattr(df, "empty", True) or "_time" not in df.columns:
            continue
        ts = pd.to_datetime(df["_time"], errors="coerce", utc=True).max()
        if pd.isna(ts):
            continue
        if newest is None or ts > newest:
            newest = ts
    if newest is None:
        return None
    return max((pd.Timestamp.now(tz="UTC") - newest).total_seconds(), 0.0)


def _led_status(age_seconds: float | None, refresh_seconds: float) -> tuple[str, str]:
    """Map a stage's data age to (led_color, short_label).

    Thresholds scale with the dashboard refresh so a slow refresh interval does
    not falsely flag a healthy stage as stale.
    """
    if age_seconds is None:
        return "red", "no data"
    fresh_limit = max(2.0 * refresh_seconds, 30.0)
    stale_limit = max(6.0 * refresh_seconds, 180.0)
    if age_seconds <= fresh_limit:
        return "green", _fmt_age(age_seconds)
    if age_seconds <= stale_limit:
        return "yellow", _fmt_age(age_seconds)
    return "red", _fmt_age(age_seconds)


def _build_data_flow_html(data: dict[str, pd.DataFrame], refresh_seconds: float) -> str:
    resolved = []
    for stage in _LOOP_STAGES:
        frames = [data.get(src) for src in stage["sources"]]
        status, label = _led_status(_stage_age_seconds(*frames), refresh_seconds)
        resolved.append({**stage, "status": status, "label": label})

    worst = max((r["status"] for r in resolved), key=lambda s: _SEVERITY_RANK[s])
    overall_label, overall_class = _OVERALL_STATE[worst]

    rows: list[str] = []
    for r in resolved:
        # A node's outgoing bus segment "flows" while the node itself is live.
        flowing = " flowing" if r["status"] == "green" else ""
        rows.append(
            f'<div class="dtflow-stage{flowing}">'
            f'<span class="dtflow-dot dot-{r["status"]}"></span>'
            f'<span class="dtflow-name">{r["name"]}</span>'
            f'<span class="dtflow-age">{r["label"]}</span>'
            '</div>'
        )

    return_active = " flowing" if worst != "red" else ""
    return (
        '<div class="dtflow">'
        '<div class="dtflow-title"><span class="dtflow-mark">⟳</span>Closed-loop control</div>'
        '<div class="dtflow-body">'
        f'<div class="dtflow-return{return_active}"></div>'
        f'{"".join(rows)}'
        '</div>'
        f'<div class="dtflow-state {overall_class}">● {overall_label}</div>'
        '</div>'
    )


# -----------------------------
# Configuration helpers
# -----------------------------

def _load_config() -> tuple[dict, dict, dict]:
    config = load_startup_config()
    viz_cfg = config.get("viz_dashboard", {})
    safety_cfg = config.get("safety_monitor", {})
    return config, viz_cfg, safety_cfg


def _build_influx() -> InfluxHelper:
    config, _, _ = _load_config()
    influx_cfg = build_influx_config(config.get("influxdb", {}))
    return InfluxHelper(influx_cfg)


def _range_start(history_hours: float) -> str:
    seconds = max(int(float(history_hours) * 3600), 60)
    return f"-{seconds}s"


# -----------------------------
# Data utilities
# -----------------------------

def _latest_value(df: pd.DataFrame, column: str) -> float | None:
    if df is None or df.empty or column not in df.columns:
        return None
    value = df[column].iloc[-1]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_text(df: pd.DataFrame, column: str, precision: int = 3) -> str:
    value = _latest_value(df, column)
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def _latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "time_hours" not in df.columns:
        return pd.DataFrame()
    latest_time = float(df["time_hours"].max())
    return df[df["time_hours"] == latest_time].copy()


def _time_step_info(df: pd.DataFrame) -> tuple[float, float, int] | None:
    """Time-step context for the currently displayed field.

    Returns ``(latest_time_h, first_time_h, n_steps)`` where ``latest_time_h`` is
    the time of the snapshot the static field is showing (and the last frame of
    the animation), or ``None`` when no timed data is available.
    """
    if df is None or df.empty or "time_hours" not in df.columns:
        return None
    times = np.sort(pd.to_numeric(df["time_hours"], errors="coerce").dropna().unique())
    if times.size == 0:
        return None
    return float(times[-1]), float(times[0]), int(times.size)


def _safe_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _percent(value: float, limit: float) -> float:
    if limit <= 0:
        return 0.0
    return 100.0 * value / limit


def _rolling_slope(time_vals: Iterable[float], radius_vals: Iterable[float]) -> float | None:
    t = np.asarray(list(time_vals), dtype=float)
    r = np.asarray(list(radius_vals), dtype=float)
    valid = np.isfinite(t) & np.isfinite(r)
    t = t[valid]
    r = r[valid]
    if t.size < 2:
        return None
    coeffs = np.polyfit(t, r, 1)
    return float(coeffs[0])


def _estimate_time_to_limit(thaw_metrics: pd.DataFrame, limit_m: float, lookback: int = 8) -> float | None:
    if thaw_metrics is None or thaw_metrics.empty:
        return None
    if "time_hours" not in thaw_metrics.columns or "radius_max_m" not in thaw_metrics.columns:
        return None

    df = thaw_metrics.copy()
    df["time_hours"] = _safe_float_series(df["time_hours"])
    df["radius_max_m"] = _safe_float_series(df["radius_max_m"])
    df = df.dropna(subset=["time_hours", "radius_max_m"])
    if df.empty:
        return None

    df = df.sort_values("time_hours")
    window = df.tail(max(lookback, 2))
    slope = _rolling_slope(window["time_hours"], window["radius_max_m"])
    if slope is None or slope <= 0:
        return float("inf")

    current_r = float(df["radius_max_m"].iloc[-1])
    remaining = limit_m - current_r
    if remaining <= 0:
        return 0.0
    return remaining / slope


def _format_time_to_limit(hours: float | None) -> tuple[str, str]:
    if hours is None:
        return "Time to limit: n/a", "neutral"
    if hours == float("inf"):
        return "System cooling - no limit breach expected", "safe"
    minutes = hours * 60.0
    if minutes < 0:
        return "Limit already exceeded", "danger"
    if minutes < 10:
        return f"CRITICAL: {minutes:.0f} minutes to limit", "danger"
    if minutes < 60:
        return f"Estimated time to limit: {minutes:.0f} minutes", "caution"
    return f"Estimated time to limit: {hours:.2f} hours", "safe"


def _status_from_percent(percent: float, safe_pct: float, caution_pct: float) -> str:
    if percent >= caution_pct * 100:
        return "ALERT"
    if percent >= safe_pct * 100:
        return "CAUTION"
    return "SAFE"


def _filter_forecast_by_horizon(df: pd.DataFrame, horizon_hours: float, tolerance: float = 0.01) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "horizon_hours" not in df.columns:
        return df.copy()
    filtered = df.copy()
    filtered["horizon_hours"] = _safe_float_series(filtered["horizon_hours"])
    filtered = filtered.dropna(subset=["horizon_hours"])
    if filtered.empty:
        return filtered
    return filtered[filtered["horizon_hours"].sub(horizon_hours).abs() <= tolerance]


def _filter_data_as_of(data: dict[str, pd.DataFrame], as_of_hours: float) -> dict[str, pd.DataFrame]:
    """Restrict fetched frames to logged data up to ``as_of_hours`` of data time.

    Display only: nothing is written back. Frames with no ``time_hours`` column
    are cut at the ingest time of the last thaw-front row inside the view.
    """
    cutoff = None
    thaw = data.get("thaw_metrics")
    if thaw is not None and not thaw.empty and {"time_hours", "_time"} <= set(thaw.columns):
        kept = thaw[_safe_float_series(thaw["time_hours"]) <= as_of_hours]
        if not kept.empty:
            cutoff = pd.to_datetime(kept["_time"], errors="coerce", utc=True).max()

    filtered: dict[str, pd.DataFrame] = {}
    for name, df in data.items():
        if df is None or df.empty:
            filtered[name] = df
        elif "time_hours" in df.columns:
            filtered[name] = df[_safe_float_series(df["time_hours"]) <= as_of_hours].copy()
        elif "_time" in df.columns and cutoff is not None:
            filtered[name] = df[pd.to_datetime(df["_time"], errors="coerce", utc=True) <= cutoff].copy()
        else:
            filtered[name] = df
    return filtered


# -----------------------------
# Visualization builders
# -----------------------------

def _build_safety_gauge(r_max: float | None, limit_m: float, safe_pct: float, caution_pct: float) -> go.Figure:
    value_m = 0.0 if r_max is None else float(r_max)
    percent = _percent(value_m, limit_m)
    if percent < safe_pct * 100:
        bar_color = "#16a34a"
    elif percent < caution_pct * 100:
        bar_color = "#f59e0b"
    else:
        bar_color = "#dc2626"

    # Gauge displayed in millimetres.
    value_mm = value_m * 1000.0
    limit_mm = limit_m * 1000.0

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value_mm,
            number={"suffix": " mm", "valueformat": ".1f", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, limit_mm]},
                "bar": {"color": bar_color},
                "bgcolor": "#0f172a",
                "borderwidth": 1,
                "bordercolor": "#1f2937",
                "steps": [
                    {"range": [0, limit_mm * safe_pct], "color": "#bbf7d0"},
                    {"range": [limit_mm * safe_pct, limit_mm * caution_pct], "color": "#fde68a"},
                    {"range": [limit_mm * caution_pct, limit_mm], "color": "#fecaca"},
                ],
                "threshold": {"line": {"color": "#dc2626", "width": 4}, "value": limit_mm},
            },
            title={"text": f"Safety gauge (limit {limit_mm:.0f} mm)", "font": {"size": 16}},
        )
    )
    fig.update_layout(height=280, margin=dict(l=40, r=40, t=70, b=10))
    return fig


def _extend_constant_along_r(field: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs along +r (axis 1), holding the outermost finite value.

    Extends each depth row's last in-coverage temperature outward to the domain
    edge, so the uncovered r-band (sensors stop at ~165 mm; the domain runs to
    240 mm) takes the value of the outermost sensor column at that depth. This is
    a horizontal (constant-in-r) hold — consistent with the insulated, zero
    radial-flux right boundary — rather than 2-D nearest-neighbour extrapolation,
    which smeared edge sensors into false corner blobs. Leading NaNs (no data at
    smaller r) and all-NaN rows are left untouched.
    """
    filled = field.copy()
    col_idx = np.arange(filled.shape[1])
    # For each cell, the index of the last valid column at or before it (-1 = none).
    last = np.where(~np.isnan(filled), col_idx[None, :], -1)
    np.maximum.accumulate(last, axis=1, out=last)
    rows = np.arange(filled.shape[0])[:, None]
    gathered = filled[rows, np.where(last >= 0, last, 0)]
    replace = np.isnan(filled) & (last >= 0)
    filled[replace] = gathered[replace]
    return filled


def _interpolate_temperature_field(
    sensors: pd.DataFrame,
    grid_resolution: int = 120,
    *,
    extend_r: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if sensors is None or sensors.empty:
        return None
    if not {"x_m", "z_m", "temperature"} <= set(sensors.columns):
        return None

    df = sensors.dropna(subset=["x_m", "z_m", "temperature"]).copy()
    if df.empty:
        return None

    x = df["x_m"].astype(float).abs().to_numpy()
    z = df["z_m"].astype(float).to_numpy()
    temp = df["temperature"].astype(float).to_numpy()

    r_grid = np.linspace(0.0, DOMAIN_RADIUS_M, grid_resolution)
    z_grid = np.linspace(0.0, DOMAIN_DEPTH_M, grid_resolution)
    R, Z = np.meshgrid(r_grid, z_grid)

    points = np.column_stack([x, z])
    method = "cubic" if len(df) >= 8 else "linear"
    try:
        T = griddata(points, temp, (R, Z), method=method)
    except Exception:
        T = griddata(points, temp, (R, Z), method="linear")

    if T is None:
        return None

    # Coverage mask: linear interpolation returns NaN *exactly* outside the
    # convex hull of the sensors. We use that as a "where do we actually have
    # data" mask so the field is never extrapolated in 2-D into uncovered
    # regions (sensors stop at r~165 mm but the domain runs to 240 mm).
    # Previously those cells were back-filled with 2-D nearest-neighbour, which
    # smeared an edge sensor's value into large false warm/cold corner blobs.
    coverage = griddata(points, temp, (R, Z), method="linear")

    # Fill interior holes of the (possibly cubic) field with nearest, then
    # re-apply the coverage mask so nothing outside the sensor hull survives.
    if np.isnan(T).any():
        T_near = griddata(points, temp, (R, Z), method="nearest")
        T = np.where(np.isnan(T), T_near, T)
    T = np.where(np.isnan(coverage), np.nan, T)

    # Extend the outermost sensor column outward in r (constant-temperature hold)
    # so the uncovered r-band mirrors the r~165 mm reading at each depth, instead
    # of being left blank — matches the insulated (zero radial-flux) boundary.
    #
    # extend_r=False leaves the uncovered band blank. That is the right choice for
    # *error* fields: holding a value constant out to the domain edge fills roughly
    # half the plot with something that was never measured, which reads as "the
    # model is wrong out here" when in truth nothing is known out here.
    if extend_r:
        T = _extend_constant_along_r(T)

    return R, Z, T


def _temperature_contour(
    sensors: pd.DataFrame,
    temp_range: tuple[float, float] = (-5.0, 30.0),
    *,
    title: str = "Temperature field with thaw front (0 C)",
    marker_name: str = "Sensors",
) -> go.Figure | None:
    latest = _latest_snapshot(sensors)
    field = _interpolate_temperature_field(latest)
    if field is None:
        return None

    snap_time = None
    if not latest.empty and "time_hours" in latest.columns:
        try:
            snap_time = float(latest["time_hours"].iloc[0])
        except (TypeError, ValueError):
            snap_time = None

    R, Z, T = field
    fig = go.Figure()

    fig.add_trace(
        go.Contour(
            x=R[0, :] * 1000.0,
            y=Z[:, 0] * 1000.0,
            z=T,
            colorscale="Jet",
            zmin=temp_range[0],
            zmax=temp_range[1],
            zmid=0.0,
            colorbar={"title": "Temperature (C)", "tick0": temp_range[0], "dtick": 5},
            contours={"showlines": False},
            connectgaps=False,
        )
    )

    fig.add_trace(
        go.Contour(
            x=R[0, :] * 1000.0,
            y=Z[:, 0] * 1000.0,
            z=T,
            showscale=False,
            contours={"start": 0.0, "end": 0.0, "coloring": "lines"},
            line={"color": "#111827", "width": 3},
            name="Thaw front (0 C)",
            connectgaps=False,
        )
    )

    if not latest.empty:
        latest = latest.copy()
        latest["x_m"] = latest["x_m"].astype(float).abs()
        fig.add_trace(
            go.Scatter(
                x=latest["x_m"] * 1000.0,
                y=latest["z_m"] * 1000.0,
                mode="markers",
                marker={
                    "size": 8,
                    "color": latest["temperature"],
                    "colorscale": "Jet",
                    "cmin": temp_range[0],
                    "cmax": temp_range[1],
                    "line": {"width": 0.5, "color": "#0f172a"},
                },
                name=marker_name,
                hovertemplate="Sensor %{text}<br>Temp: %{marker.color:.2f} C<extra></extra>",
                text=latest.get("sensor_id", ""),
            )
        )

    fig.add_shape(
        type="rect",
        x0=0.0,
        x1=THERMOSYPHON_RADIUS_M * 1000.0,
        y0=0.0,
        y1=DOMAIN_DEPTH_M * 1000.0,
        fillcolor="#111827",
        opacity=0.6,
        line_width=0,
        layer="below",
    )

    full_title = title if snap_time is None else f"{title}<br><sup>t = {snap_time:.2f} h</sup>"
    fig.update_layout(
        title=full_title,
        xaxis_title="Radial distance (mm)",
        yaxis_title="Depth (mm)",
        height=440,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, DOMAIN_RADIUS_M * 1000.0])
    fig.update_yaxes(range=[DOMAIN_DEPTH_M * 1000.0, 0.0])
    return fig


def _temperature_contour_animated(
    sensors: pd.DataFrame,
    temp_range: tuple[float, float] = (-5.0, 30.0),
    *,
    title: str = "Temperature field — heat propagation",
    marker_name: str = "Sensors",
    max_frames: int = _MAX_ANIMATION_FRAMES,
    grid_resolution: int = _ANIMATION_GRID_RES,
) -> go.Figure | None:
    """Animated Plotly figure with a play button and scrubber over all timesteps.

    Payload is kept manageable by:
    - Capping frames at max_frames (subsampled uniformly from available timesteps).
    - Using a lower grid resolution than the static version.
    - Passing only `z` in each frame for the two Contour traces (x/y are fixed on
      the base trace and are not retransmitted per frame).
    """
    if sensors is None or sensors.empty or "time_hours" not in sensors.columns:
        return None

    all_times = np.sort(sensors["time_hours"].dropna().unique())
    if len(all_times) == 0:
        return None
    if len(all_times) == 1:
        # Single snapshot — fall back to the lighter static version.
        return _temperature_contour(sensors, temp_range, title=title, marker_name=marker_name)

    # --- subsample timesteps -------------------------------------------
    n_total = len(all_times)
    if n_total > max_frames:
        indices = np.round(np.linspace(0, n_total - 1, max_frames)).astype(int)
        sampled_times = all_times[indices]
    else:
        sampled_times = all_times

    # Fixed grid axes (computed once, shared by every frame).
    r_grid = np.linspace(0.0, DOMAIN_RADIUS_M, grid_resolution)
    z_grid = np.linspace(0.0, DOMAIN_DEPTH_M, grid_resolution)
    R, Z_mesh = np.meshgrid(r_grid, z_grid)
    x_axis = (R[0, :] * 1000.0).tolist()
    y_axis = (Z_mesh[:, 0] * 1000.0).tolist()

    # --- build per-frame data -----------------------------------------
    frame_entries: list[dict] = []
    for t in sampled_times:
        snapshot = sensors[np.isclose(sensors["time_hours"], t)].copy()
        if snapshot.empty:
            continue
        field = _interpolate_temperature_field(snapshot, grid_resolution=grid_resolution)
        if field is None:
            continue
        _, _, T = field
        snap = snapshot.dropna(subset=["x_m", "z_m", "temperature"]).copy()
        snap["x_m"] = snap["x_m"].astype(float).abs()
        frame_entries.append({
            "t": float(t),
            "z": T.tolist(),                                    # 2-D list, not numpy
            "sx": (snap["x_m"] * 1000.0).tolist(),
            "sz": (snap["z_m"] * 1000.0).tolist(),
            "sc": snap["temperature"].astype(float).tolist(),
            "sid": snap["sensor_id"].astype(str).tolist() if "sensor_id" in snap.columns else [],
        })

    if not frame_entries:
        return None

    first = frame_entries[0]

    # --- base figure (first timestep) ----------------------------------
    fig = go.Figure()

    # The temperature field is a Heatmap (not a filled Contour): heatmaps update
    # in place as a raster on each frame, whereas filled contours redraw their
    # SVG paths and briefly vanish, which flashed the dark background as a black
    # frame during playback. `zsmooth` gives the smooth field look. The thin
    # 0 C thaw-front stays a Contour line overlay (a stroke, so it never blanks
    # the field).
    fig.add_trace(go.Heatmap(
        x=x_axis, y=y_axis, z=first["z"],
        colorscale="Jet",
        zmin=temp_range[0], zmax=temp_range[1], zmid=0.0,
        zsmooth="best",
        colorbar={"title": "Temperature (C)", "tick0": temp_range[0], "dtick": 5},
        hovertemplate="r %{x:.0f} mm · z %{y:.0f} mm<br>%{z:.1f} °C<extra></extra>",
        name="Temperature",
    ))

    fig.add_trace(go.Contour(
        x=x_axis, y=y_axis, z=first["z"],
        showscale=False,
        contours={"start": 0.0, "end": 0.0, "coloring": "lines"},
        line={"color": "#f8fafc", "width": 2.5},
        name="Thaw front (0 °C)",
        connectgaps=False,
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=first["sx"], y=first["sz"],
        mode="markers",
        marker={
            "size": 8,
            "color": first["sc"],
            "colorscale": "Jet",
            "cmin": temp_range[0], "cmax": temp_range[1],
            "line": {"width": 0.5, "color": "#0f172a"},
        },
        name=marker_name,
        hovertemplate="Sensor %{text}<br>Temp: %{marker.color:.2f} °C<extra></extra>",
        text=first["sid"],
    ))

    fig.add_shape(
        type="rect",
        x0=0.0, x1=THERMOSYPHON_RADIUS_M * 1000.0,
        y0=0.0, y1=DOMAIN_DEPTH_M * 1000.0,
        fillcolor="#111827", opacity=0.6, line_width=0, layer="below",
    )

    # --- animation frames (only z changes for the field + front traces) --
    frames = []
    for fe in frame_entries:
        frames.append(go.Frame(
            data=[
                go.Heatmap(z=fe["z"]),           # updates trace 0 — field, only z
                go.Contour(z=fe["z"]),           # updates trace 1 — 0 C line, only z
                go.Scatter(                      # updates trace 2 — positions + colour
                    x=fe["sx"], y=fe["sz"],
                    marker={"color": fe["sc"]},
                    text=fe["sid"],
                ),
            ],
            traces=[0, 1, 2],
            name=f"{fe['t']:.4f}",
        ))
    fig.frames = frames

    # --- scrubber slider -----------------------------------------------
    # Per-step labels are blanked to keep the track clean at up to 40 frames;
    # the live timestamp is shown by the slider's prominent `currentvalue`.
    frame_ms = 300
    slider_steps = [
        {
            "args": [
                [f.name],
                {"frame": {"duration": frame_ms, "redraw": True},
                 "mode": "immediate",
                 "transition": {"duration": 0}},
            ],
            "label": "",
            "method": "animate",
        }
        for f in frames
    ]

    n_sampled = len(frame_entries)
    subsampled = n_sampled < n_total
    subtitle = f"{n_sampled} frames" + (f" (1 of every {n_total // n_sampled} snapshots)" if subsampled else "")

    # Single toggle button: Plotly applies `args` on odd clicks (play) and
    # `args2` on even clicks (pause), so one control does both. The label is the
    # combined play/pause glyph (triangle + bars) so a single round icon conveys
    # both states.
    play_pause_button = {
        "label": "⏯",
        "method": "animate",
        "args": [None, {"frame": {"duration": frame_ms, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0}}],
        "args2": [[None], {"frame": {"duration": 0, "redraw": False},
                           "mode": "immediate",
                           "transition": {"duration": 0}}],
    }

    accent = "#2563eb"
    fig.update_layout(
        title={"text": f"{title}<br><sup>{subtitle}</sup>"},
        xaxis_title="Radial distance (mm)",
        yaxis_title="Depth (mm)",
        height=520,
        margin=dict(l=40, r=20, t=60, b=110),
        # Solid, opaque canvas so any momentary redraw can never flash the dark
        # page background through as a black frame.
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        font={"color": "#e2e8f0"},
        showlegend=False,
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.02, "y": -0.02,
            "xanchor": "left", "yanchor": "top",
            # Symmetric pad -> a small, near-circular icon button.
            "pad": {"t": 64, "b": 8, "l": 10, "r": 10},
            "bgcolor": accent,
            "bordercolor": "#1d4ed8",
            "borderwidth": 2,
            "font": {"color": "#ffffff", "size": 22},
            "buttons": [play_pause_button],
        }],
        sliders=[{
            "active": 0,
            "x": 0.12, "len": 0.88,
            "pad": {"t": 70, "b": 10},
            "bgcolor": "#334155",
            "activebgcolor": "#60a5fa",
            "bordercolor": "#475569",
            "borderwidth": 1,
            "tickcolor": "#64748b",
            "ticklen": 5,
            "minorticklen": 0,
            "currentvalue": {
                "prefix": "Time  ",
                "suffix": " h",
                "font": {"size": 16, "color": "#e2e8f0"},
                "visible": True,
                "xanchor": "left",
                "offset": 12,
            },
            "transition": {"duration": 0},
            "steps": slider_steps,
        }],
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, DOMAIN_RADIUS_M * 1000.0])
    fig.update_yaxes(range=[DOMAIN_DEPTH_M * 1000.0, 0.0])
    return fig


def _front_geometry_plot(
    thaw_points: pd.DataFrame,
    forecast_points: pd.DataFrame,
    limit_m: float,
) -> go.Figure | None:
    """The measured and forecast thaw fronts at the latest instant, against the limit.

    One profile per source: radial front position against depth, plus the
    user-set safety limit as a vertical line. That is the whole plot -- the
    question it answers is "how close is the front to the limit, and does the
    model agree about it", and anything else on these axes competes with it.

    Two things this has to get right that the earlier version did not:

    * **One horizon per curve.** ``fem_forecast_points`` is written with a
      ``horizon_hours`` field, so a single ``time_hours`` can hold several
      forecast fronts at once. Filtering on time alone merged them into one
      polyline and then let ``drop_duplicates`` keep whichever horizon happened
      to sort first, which drew a front that existed at no horizon at all.
    * **An axis scaled to the fronts.** The domain is 240 mm wide and the fronts
      live near 40 mm, so a full-domain axis pushed the entire plot into its
      leftmost sixth. The range now follows the data and the limit line.
    """
    if (thaw_points is None or thaw_points.empty) and (forecast_points is None or forecast_points.empty):
        return None

    def _clean(points: pd.DataFrame) -> pd.DataFrame:
        if points is None or points.empty:
            return pd.DataFrame()
        if not {"time_hours", "x_m", "z_m"}.issubset(points.columns):
            return pd.DataFrame()
        df = points.copy()
        for column in ("time_hours", "x_m", "z_m"):
            df[column] = _safe_float_series(df[column])
        if "horizon_hours" in df.columns:
            df["horizon_hours"] = _safe_float_series(df["horizon_hours"])
        return df.dropna(subset=["time_hours", "x_m", "z_m"])

    def _latest_front(points: pd.DataFrame) -> tuple[pd.DataFrame, float | None, float | None]:
        """Newest front as a depth-ordered profile, from a single horizon."""
        df = _clean(points)
        if df.empty:
            return pd.DataFrame(), None, None
        latest_time = float(df["time_hours"].max())
        df = df[np.isclose(df["time_hours"], latest_time, atol=1e-9)]
        horizon = None
        if "horizon_hours" in df.columns and df["horizon_hours"].notna().any():
            # The longest horizon written at this instant is the forecast the
            # safety rules are evaluated against, so it is the one to draw.
            horizon = float(df["horizon_hours"].max())
            df = df[np.isclose(df["horizon_hours"], horizon, atol=1e-9)]
        df = df.copy()
        df["x_m"] = df["x_m"].abs()
        # Deepest-first ordering, and one radius per depth: a front is a function
        # of depth, and two radii at one depth is a duplicate record, not a shape.
        df = df.sort_values(["z_m", "x_m"]).drop_duplicates(subset=["z_m"], keep="last")
        return df, latest_time, horizon

    measured, measured_time, _ = _latest_front(thaw_points)
    forecast, forecast_time, forecast_horizon = _latest_front(forecast_points)
    if measured.empty and forecast.empty:
        return None

    fig = go.Figure()

    if not measured.empty:
        fig.add_trace(
            go.Scatter(
                x=measured["x_m"] * 1000.0,
                y=measured["z_m"] * 1000.0,
                mode="lines+markers" if len(measured) > 1 else "markers",
                line={"color": "#2563eb", "width": 3},
                marker={"size": 6, "color": "#2563eb"},
                name="Measured" + (f" (t = {measured_time:.2f} h)" if measured_time is not None else ""),
                hovertemplate="measured<br>r = %{x:.1f} mm<br>depth = %{y:.0f} mm<extra></extra>",
            )
        )

    if not forecast.empty:
        label = "Forecast"
        if forecast_horizon:
            label += f" (+{forecast_horizon * 60.0:.0f} min)"
        elif forecast_time is not None:
            label += f" (t = {forecast_time:.2f} h)"
        fig.add_trace(
            go.Scatter(
                x=forecast["x_m"] * 1000.0,
                y=forecast["z_m"] * 1000.0,
                mode="lines+markers" if len(forecast) > 1 else "markers",
                line={"color": "#dc2626", "width": 3, "dash": "dash"},
                marker={"size": 6, "color": "#dc2626"},
                name=label,
                hovertemplate="forecast<br>r = %{x:.1f} mm<br>depth = %{y:.0f} mm<extra></extra>",
            )
        )

    limit_mm = float(limit_m) * 1000.0
    fig.add_vline(
        x=limit_mm,
        line_width=2,
        line_dash="dot",
        line_color="#b3261e",
        annotation_text=f"Safety limit {limit_mm:.0f} mm",
        annotation_position="top right",
        annotation_font={"color": "#b3261e", "size": 11},
    )

    # Axis follows the fronts and the limit, with headroom, instead of the full
    # 240 mm domain -- otherwise a 40 mm front occupies a sixth of the plot.
    front_max = 0.0
    for frame in (measured, forecast):
        if not frame.empty:
            front_max = max(front_max, float(frame["x_m"].max()) * 1000.0)
    x_max = max(front_max, limit_mm) * 1.25 or DOMAIN_RADIUS_M * 1000.0
    x_max = min(x_max, DOMAIN_RADIUS_M * 1000.0)

    depth_max = DOMAIN_DEPTH_M * 1000.0
    for frame in (measured, forecast):
        if not frame.empty:
            depth_max = max(depth_max, float(frame["z_m"].max()) * 1000.0)

    fig.update_layout(
        title="Thaw front: measured vs forecast",
        xaxis_title="Radial distance from pipe (mm)",
        yaxis_title="Depth (mm)",
        height=440,
        margin=dict(l=60, r=30, t=60, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
    )
    fig.update_xaxes(range=[0, x_max])
    fig.update_yaxes(range=[depth_max, 0.0], autorange=False)
    return fig


def _unified_radius_plot(
    thaw_metrics: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    heater_actions: pd.DataFrame,
    limit_m: float,
    release_m: float = 0.0,
    warn_fraction: float = 0.6,
    critical_fraction: float = 0.9,
) -> go.Figure | None:
    if thaw_metrics is None or thaw_metrics.empty:
        return None

    df_measured = thaw_metrics.copy()
    if "time_hours" not in df_measured.columns or "radius_max_m" not in df_measured.columns:
        return None
    df_measured["time_hours"] = _safe_float_series(df_measured["time_hours"])
    df_measured["radius_max_m"] = _safe_float_series(df_measured["radius_max_m"])
    has_rmin = "radius_min_m" in df_measured.columns
    if has_rmin:
        df_measured["radius_min_m"] = _safe_float_series(df_measured["radius_min_m"])
    df_measured = df_measured.dropna(subset=["time_hours", "radius_max_m"]).sort_values("time_hours")

    if len(df_measured) > 200:
        step = max(int(len(df_measured) / 200), 1)
        df_measured = df_measured.iloc[::step].copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_measured["time_hours"],
            y=df_measured["radius_max_m"] * 1000.0,
            mode="lines",
            line={"color": "#1f77b4", "width": 3},
            name="Measured r_max",
            hovertemplate="Time: %{x:.2f}h<br>r_max: %{y:.1f} mm<extra></extra>",
        )
    )

    if has_rmin:
        df_rmin = df_measured.dropna(subset=["radius_min_m"])
        if not df_rmin.empty:
            fig.add_trace(
                go.Scatter(
                    x=df_rmin["time_hours"],
                    y=df_rmin["radius_min_m"] * 1000.0,
                    mode="lines",
                    line={"color": "#17becf", "width": 3},
                    name="Measured r_min",
                    hovertemplate="Time: %{x:.2f}h<br>r_min: %{y:.1f} mm<extra></extra>",
                )
            )

    if forecast_metrics is not None and not forecast_metrics.empty and "horizon_hours" in forecast_metrics.columns:
        df_forecast = forecast_metrics.copy()
        df_forecast["time_hours"] = _safe_float_series(df_forecast["time_hours"])
        df_forecast["horizon_hours"] = _safe_float_series(df_forecast["horizon_hours"])
        df_forecast["radius_max_m"] = _safe_float_series(df_forecast["radius_max_m"])
        df_forecast = df_forecast.dropna(subset=["time_hours", "horizon_hours", "radius_max_m"])
        df_forecast["time_forecast"] = df_forecast["time_hours"] + df_forecast["horizon_hours"]
        df_forecast = df_forecast.sort_values("time_forecast")

        fig.add_trace(
            go.Scatter(
                x=df_forecast["time_forecast"],
                y=df_forecast["radius_max_m"] * 1000.0,
                mode="lines",
                line={"color": "#d62728", "width": 3, "dash": "dash"},
                name="Forecast r_max",
                hovertemplate="Forecast: %{x:.2f}h<br>r_max: %{y:.1f} mm<extra></extra>",
            )
        )

    limit_mm = limit_m * 1000.0
    fig.add_hline(
        y=limit_mm,
        line_width=4,
        line_color="#b91c1c",
        annotation_text=f"r_lim = {limit_mm:.0f} mm",
        annotation_position="top left",
        annotation_font_color="#b91c1c",
    )

    measured_max_mm = float(df_measured["radius_max_m"].max()) * 1000.0
    y_max = max(limit_mm * 1.3, measured_max_mm * 1.2)
    fig.add_hrect(
        y0=limit_mm,
        y1=y_max,
        fillcolor="rgba(239,68,68,0.12)",
        line_width=0,
    )

    # Warning / critical bands as r_max approaches r_lim. Warning is the M_warn
    # predicate; critical is display-only and takes no part in the decision. The
    # heater keeps thawing through both; only crossing r_lim stops it.
    warn_mm = warn_fraction * limit_mm
    crit_mm = critical_fraction * limit_mm
    if 0.0 < warn_mm < crit_mm < limit_mm:
        fig.add_hrect(
            y0=warn_mm,
            y1=crit_mm,
            fillcolor="rgba(234,179,8,0.10)",
            line_width=0,
        )
        fig.add_hrect(
            y0=crit_mm,
            y1=limit_mm,
            fillcolor="rgba(249,115,22,0.14)",
            line_width=0,
        )
        fig.add_hline(
            y=warn_mm,
            line_width=1.5,
            line_dash="dot",
            line_color="#a16207",
            annotation_text=f"Warning {warn_mm:.0f} mm",
            annotation_position="top left",
            annotation_font_color="#a16207",
        )
        fig.add_hline(
            y=crit_mm,
            line_width=1.5,
            line_dash="dot",
            line_color="#c2410c",
            annotation_text=f"Critical {crit_mm:.0f} mm (display only)",
            annotation_position="top left",
            annotation_font_color="#c2410c",
        )

    if release_m > 0:
        release_mm = release_m * 1000.0
        fig.add_hline(
            y=release_mm,
            line_width=1.5,
            line_dash="dash",
            line_color="#2563eb",
            annotation_text=f"Release {release_mm:.0f} mm",
            annotation_position="bottom left",
            annotation_font_color="#2563eb",
        )

    latest_time = float(df_measured["time_hours"].iloc[-1])
    latest_mm = float(df_measured["radius_max_m"].iloc[-1]) * 1000.0
    if latest_mm >= limit_mm:
        zone_label, zone_color = "ALERT (heater STOP)", "#ef4444"
    elif latest_mm >= crit_mm:
        zone_label, zone_color = "CRITICAL (heating)", "#f97316"
    elif latest_mm >= warn_mm:
        zone_label, zone_color = "WARNING (heating)", "#eab308"
    else:
        zone_label, zone_color = "OK (heating)", "#22c55e"
    fig.add_trace(
        go.Scatter(
            x=[latest_time],
            y=[latest_mm],
            mode="markers",
            name=f"Current r_max — {zone_label}",
            marker=dict(size=16, color=zone_color, line=dict(width=2, color="white")),
            hovertemplate=(
                f"NOW — {zone_label}<br>Time: {latest_time:.2f}h<br>"
                f"r_max: {latest_mm:.1f} mm<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Thaw radius evolution: measured history + forecast",
        height=520,
        margin=dict(l=70, r=50, t=70, b=70),
        legend=dict(
            x=1.02,
            y=0.02,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(15,23,42,0.65)",
            bordercolor="rgba(148,163,184,0.6)",
            borderwidth=1,
            font=dict(size=11, color="#e2e8f0"),
        ),
        hovermode="x unified",
        plot_bgcolor="rgba(15,23,42,0.08)",
        paper_bgcolor="rgba(15,23,42,0.02)",
        font=dict(color="#e2e8f0"),
    )
    fig.update_xaxes(title_text="Time (hours)", showgrid=True, gridcolor="rgba(148,163,184,0.25)")
    fig.update_yaxes(
        title_text="Thaw radius (mm)",
        range=[0, y_max],
        showgrid=True,
        gridcolor="rgba(148,163,184,0.25)",
    )
    return fig


def _volume_over_time_plot(thaw_metrics: pd.DataFrame) -> go.Figure | None:
    if thaw_metrics is None or thaw_metrics.empty:
        return None
    if "thaw_volume_m3" not in thaw_metrics.columns:
        return None

    df = thaw_metrics.copy()
    df["time_hours"] = _safe_float_series(df["time_hours"])
    df["thaw_volume_m3"] = _safe_float_series(df["thaw_volume_m3"])
    df = df.dropna(subset=["time_hours", "thaw_volume_m3"]).sort_values("time_hours")
    if df.empty:
        return None

    if len(df) > 200:
        step = max(int(len(df) / 200), 1)
        df = df.iloc[::step].copy()

    vol_cm3 = df["thaw_volume_m3"] * 1e6

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time_hours"],
        y=vol_cm3,
        mode="lines",
        line={"color": "#7c3aed", "width": 3},
        name="Thaw volume",
        hovertemplate="Time: %{x:.2f}h<br>Volume: %{y:.2f} cm³<extra></extra>",
    ))

    latest_time = float(df["time_hours"].iloc[-1])
    latest_vol = float(vol_cm3.iloc[-1])
    fig.add_trace(go.Scatter(
        x=[latest_time],
        y=[latest_vol],
        mode="markers",
        name="Current",
        marker=dict(size=14, color="#7c3aed", line=dict(width=2, color="white")),
        hovertemplate=f"NOW<br>Time: {latest_time:.2f}h<br>Volume: {latest_vol:.2f} cm³<extra></extra>",
    ))

    fig.update_layout(
        title="Thaw volume over time",
        height=400,
        margin=dict(l=70, r=50, t=70, b=70),
        legend=dict(
            x=1.02, y=0.02, xanchor="left", yanchor="bottom",
            bgcolor="rgba(15,23,42,0.65)",
            bordercolor="rgba(148,163,184,0.6)",
            borderwidth=1,
            font=dict(size=11, color="#e2e8f0"),
        ),
        hovermode="x unified",
        plot_bgcolor="rgba(15,23,42,0.08)",
        paper_bgcolor="rgba(15,23,42,0.02)",
        font=dict(color="#e2e8f0"),
    )
    fig.update_xaxes(title_text="Time (hours)", showgrid=True, gridcolor="rgba(148,163,184,0.25)")
    fig.update_yaxes(title_text="Thaw volume (cm³)", showgrid=True, gridcolor="rgba(148,163,184,0.25)")
    return fig


def _thaw_volume_3d_plot(thaw_points: pd.DataFrame) -> go.Figure | None:
    if thaw_points is None or thaw_points.empty:
        return None
    required = {"time_hours", "x_m", "z_m"}
    if not required.issubset(thaw_points.columns):
        return None

    df = thaw_points.copy()
    df["time_hours"] = _safe_float_series(df["time_hours"])
    df["x_m"] = _safe_float_series(df["x_m"]).abs()
    df["z_m"] = _safe_float_series(df["z_m"])
    df = df.dropna(subset=["time_hours", "x_m", "z_m"])
    if df.empty:
        return None

    t = float(df["time_hours"].max())
    df = df[np.isclose(df["time_hours"], t, atol=1e-9)].copy()
    df = df.sort_values("z_m").drop_duplicates(subset=["z_m"])
    if len(df) < 2:
        return None

    # Densify contour to a uniform z grid for a smooth surface
    z_raw = df["z_m"].to_numpy()
    r_raw = df["x_m"].to_numpy()
    z_dense = np.linspace(float(z_raw[0]), float(z_raw[-1]), 120)
    r_dense = np.interp(z_dense, z_raw, r_raw)

    # Convert to mm for display
    z_mm = z_dense * 1000.0
    r_mm = r_dense * 1000.0
    pipe_r_mm = THERMOSYPHON_RADIUS_M * 1000.0

    # Angular sweep
    n_theta = 72
    theta = np.linspace(0, 2 * np.pi, n_theta)

    THETA, Z_mesh = np.meshgrid(theta, z_mm)
    R_mesh = np.tile(r_mm[:, np.newaxis], (1, n_theta))
    X_outer = R_mesh * np.cos(THETA)
    Y_outer = R_mesh * np.sin(THETA)

    fig = go.Figure()

    # Outer thaw surface — coloured by depth
    fig.add_trace(go.Surface(
        x=X_outer, y=Y_outer, z=Z_mesh,
        surfacecolor=Z_mesh,
        colorscale="Blues",
        reversescale=True,
        opacity=0.72,
        showscale=False,
        name="Thaw boundary",
        hovertemplate="x: %{x:.1f} mm<br>y: %{y:.1f} mm<br>z: %{z:.1f} mm<extra>Thaw boundary</extra>",
    ))

    # Top cap (disc at shallowest depth)
    r_cap = np.linspace(pipe_r_mm, r_mm[0], 30)
    THETA_c, R_c = np.meshgrid(theta, r_cap)
    fig.add_trace(go.Surface(
        x=R_c * np.cos(THETA_c),
        y=R_c * np.sin(THETA_c),
        z=np.full_like(R_c, z_mm[0]),
        colorscale=[[0, "#3b82f6"], [1, "#3b82f6"]],
        showscale=False, opacity=0.72,
        name="Top cap",
    ))

    # Bottom cap (disc at deepest depth)
    r_cap_b = np.linspace(pipe_r_mm, r_mm[-1], 30)
    THETA_cb, R_cb = np.meshgrid(theta, r_cap_b)
    fig.add_trace(go.Surface(
        x=R_cb * np.cos(THETA_cb),
        y=R_cb * np.sin(THETA_cb),
        z=np.full_like(R_cb, z_mm[-1]),
        colorscale=[[0, "#3b82f6"], [1, "#3b82f6"]],
        showscale=False, opacity=0.72,
        name="Bottom cap",
    ))

    # Heating pipe (dark inner cylinder)
    z_pipe = np.linspace(z_mm[0], z_mm[-1], 40)
    THETA_p, Z_p = np.meshgrid(theta, z_pipe)
    fig.add_trace(go.Surface(
        x=pipe_r_mm * np.cos(THETA_p),
        y=pipe_r_mm * np.sin(THETA_p),
        z=Z_p,
        colorscale=[[0, "#1e293b"], [1, "#334155"]],
        showscale=False, opacity=1.0,
        name="Heating pipe",
    ))

    fig.update_layout(
        title=f"3D thaw volume  (t = {t:.2f} h)",
        height=560,
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(
            xaxis_title="x (mm)",
            yaxis_title="y (mm)",
            zaxis_title="Depth (mm)",
            zaxis=dict(autorange="reversed"),
            aspectmode="data",
            bgcolor="rgba(241,245,249,0.0)",
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.8)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _model_diagnostics_plot(
    thaw_metrics: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    acceptable_error_pct: float = 5.0,
) -> go.Figure | None:
    if thaw_metrics is None or thaw_metrics.empty or forecast_metrics is None or forecast_metrics.empty:
        return None

    measured = thaw_metrics.copy()
    if "time_hours" not in measured.columns or "radius_max_m" not in measured.columns:
        return None
    measured["time_hours"] = _safe_float_series(measured["time_hours"]).round(2)
    measured["radius_max_m"] = _safe_float_series(measured["radius_max_m"])
    measured = measured.dropna(subset=["time_hours", "radius_max_m"])

    forecast = forecast_metrics.copy()
    forecast["time_hours"] = _safe_float_series(forecast["time_hours"])
    forecast["horizon_hours"] = _safe_float_series(forecast.get("horizon_hours"))
    forecast["radius_max_m"] = _safe_float_series(forecast["radius_max_m"])
    forecast = forecast.dropna(subset=["time_hours", "horizon_hours", "radius_max_m"])
    if forecast.empty or measured.empty:
        return None

    forecast["time_forecast"] = (forecast["time_hours"] + forecast["horizon_hours"]).round(2)
    merged = pd.merge(
        measured[["time_hours", "radius_max_m"]],
        forecast[["time_forecast", "radius_max_m"]],
        left_on="time_hours",
        right_on="time_forecast",
        suffixes=("_measured", "_forecast"),
    )
    if merged.empty:
        return None

    # Relative error as a percentage of the measured radius. Rows where the
    # measured radius is ~0 (no meaningful denominator) are dropped rather than
    # blowing up to a huge/inf percentage.
    denom = merged["radius_max_m_measured"].abs()
    merged["radius_error_pct"] = np.where(
        denom > 1e-6,
        (merged["radius_max_m_measured"] - merged["radius_max_m_forecast"]).abs() / denom * 100.0,
        np.nan,
    )
    merged = merged.dropna(subset=["radius_error_pct"])
    if merged.empty:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=merged["time_hours"],
            y=merged["radius_error_pct"],
            mode="lines+markers",
            line={"color": "#6b7280", "width": 2},
            name="Forecast error",
            hovertemplate="t=%{x:.2f}h<br>error: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_hline(
        y=acceptable_error_pct,
        line_color="#22c55e",
        line_dash="dash",
        annotation_text=f"Acceptable (<{acceptable_error_pct:.0f}%)",
    )
    max_err = float(merged["radius_error_pct"].max())
    fig.add_hrect(
        y0=0,
        y1=acceptable_error_pct,
        fillcolor="rgba(34,197,94,0.18)",
        line_width=0,
        annotation_text="Good",
        annotation_position="inside top left",
    )
    fig.add_hrect(
        y0=acceptable_error_pct,
        y1=max(max_err * 1.1, acceptable_error_pct * 1.5),
        fillcolor="rgba(239,68,68,0.12)",
        line_width=0,
        annotation_text="Poor",
        annotation_position="inside top left",
    )
    fig.update_layout(
        title="Forecast error in r_max (%)",
        height=300,
        margin=dict(l=40, r=20, t=40, b=30),
        xaxis_title="Time (hours)",
        yaxis_title="|r_max error| (% of measured)",
    )
    return fig


# Percent error of the FEM temperature field against the measured field.
# Mirrors the reference MATLAB post-processing (Final_DT/main.m), which writes a
# nodal "abs_relative_error(percent)" zone to Simulation_results.plt every step:
#
#     T_error = abs((T - (T_experimental + 273.15)) ./ T * 100)      % T in Kelvin
#
# The Kelvin denominator is kept so the numbers line up with the Tecplot output.
# It makes the percentages look small — a 1 C miss near freezing is ~0.37% — so
# the absolute error in C is carried alongside and surfaced in the hover/tiles.
_KELVIN_OFFSET = 273.15

# Categorical slots 1 and 2 (blue, orange) and the single-hue blue sequential
# ramp, light-mode steps.
_ERROR_SERIES_MEAN = "#2a78d6"
_ERROR_SERIES_MAX = "#eb6834"
_ERROR_SEQUENTIAL = [
    [0.0, "#cde2fb"], [0.2, "#9ec5f4"], [0.4, "#6da7ec"],
    [0.6, "#3987e5"], [0.8, "#256abf"], [1.0, "#104281"],
]
# Same single hue, reversed for the animated panel's dark canvas: near-zero error
# recedes toward the surface and high error reads as the bright end, which is what
# "lightest step means near zero" becomes once the surface is dark.
_ERROR_SEQUENTIAL_DARK = [
    [0.0, "#0d366b"], [0.2, "#184f95"], [0.4, "#256abf"],
    [0.6, "#3987e5"], [0.8, "#86b6ef"], [1.0, "#cde2fb"],
]
# Diverging blue<->red about a neutral gray zero, for the signed bias field:
# blue = FEM colder than measured, red = FEM warmer. Equal step count per arm so
# the midpoint is true zero. Blue arm is the documented sequential ramp; the red
# arm mirrors its lightness profile.
_BIAS_DIVERGING = [
    [0.0, "#104281"], [0.125, "#256abf"], [0.25, "#3987e5"], [0.375, "#86b6ef"],
    [0.5, "#f0efec"],
    [0.625, "#f0a09c"], [0.75, "#e34948"], [0.875, "#b8302f"], [1.0, "#7d1f1e"],
]


def _fem_error_frame(
    sensors: pd.DataFrame,
    forecast_temperature: pd.DataFrame,
) -> pd.DataFrame | None:
    """Pair measured and FEM temperatures per sensor position and timestep.

    Both measurements carry the same ``sensor_id`` tag and ``time_hours`` field —
    the FEM field is sampled at the observed sensor positions before it is
    written (see ``_sample_field_at_sensor_grid``), so they join directly.

    Returns a frame with ``temperature_obs``/``temperature_fem`` (C),
    ``abs_error_c`` and ``error_pct``, or None when the two streams don't
    overlap yet.
    """
    required = {"time_hours", "sensor_id", "temperature", "x_m", "z_m"}
    if sensors is None or forecast_temperature is None:
        return None
    if sensors.empty or forecast_temperature.empty:
        return None
    if not required <= set(sensors.columns) or not required <= set(forecast_temperature.columns):
        return None

    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # dt is 36 s (0.01 h), so 2 decimals is the native time resolution and
        # collapses float round-trip noise without merging distinct steps.
        out["time_hours"] = _safe_float_series(out["time_hours"]).round(2)
        out["temperature"] = _safe_float_series(out["temperature"])
        out["x_m"] = _safe_float_series(out["x_m"]).abs()
        out["z_m"] = _safe_float_series(out["z_m"])
        out["sensor_id"] = out["sensor_id"].astype(str)
        out = out.dropna(subset=["time_hours", "temperature", "x_m", "z_m"])
        return out.drop_duplicates(subset=["time_hours", "sensor_id"], keep="last")

    obs = _prep(sensors)
    fem = _prep(forecast_temperature)
    if obs.empty or fem.empty:
        return None

    merged = pd.merge(
        obs[["time_hours", "sensor_id", "x_m", "z_m", "temperature"]],
        fem[["time_hours", "sensor_id", "temperature"]],
        on=["time_hours", "sensor_id"],
        suffixes=("_obs", "_fem"),
    )
    if merged.empty:
        return None

    # Signed: positive = FEM warmer than measured. The reference metric takes an
    # absolute value, which hides whether the model runs hot or cold — the single
    # most diagnostic thing about a drifting solve.
    merged["signed_error_c"] = merged["temperature_fem"] - merged["temperature_obs"]
    merged["abs_error_c"] = merged["signed_error_c"].abs()
    denom = (merged["temperature_fem"] + _KELVIN_OFFSET).abs()
    merged["error_pct"] = np.where(
        denom > 1e-9,
        merged["abs_error_c"] / denom * 100.0,
        np.nan,
    )
    merged = merged.dropna(subset=["error_pct"])
    return merged if not merged.empty else None


def _pipe_outlet_validation_plot(validation: pd.DataFrame) -> go.Figure | None:
    """Measured inlet/outlet against the pipe model's simulated outlet.

    The validation question is whether the U-shaped heat-exchanger model, driven
    by the measured inlet, reproduces the measured outlet -- so the two outlet
    series carry the same axis and differ by mark: measurement as open circles,
    model as a continuous line. Inlet is drawn for context; it is the forcing,
    not something the model predicts.
    """
    if validation is None or validation.empty:
        return None
    required = {"time_hours", "t_inlet_c", "t_outlet_meas_c", "t_outlet_sim_c"}
    if not required <= set(validation.columns):
        return None

    frame = validation.copy()
    for column in required:
        frame[column] = _safe_float_series(frame[column])
    frame = frame.dropna(subset=list(required)).sort_values("time_hours")
    frame = frame.drop_duplicates(subset=["time_hours"], keep="last")
    if frame.empty:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["time_hours"],
            y=frame["t_inlet_c"],
            mode="lines",
            name="Measured inlet",
            line=dict(color="#3987e5", width=2),
            hovertemplate="Inlet: %{y:.2f} °C<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["time_hours"],
            y=frame["t_outlet_meas_c"],
            mode="markers",
            name="Measured outlet",
            marker=dict(
                size=8,
                color="rgba(0,0,0,0)",
                line=dict(color="#d95926", width=1.6),
            ),
            hovertemplate="Outlet measured: %{y:.2f} °C<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["time_hours"],
            y=frame["t_outlet_sim_c"],
            mode="lines",
            name="Simulated outlet",
            line=dict(color="#199e70", width=2),
            hovertemplate="Outlet simulated: %{y:.2f} °C<extra></extra>",
        )
    )

    fig.update_layout(
        title="Pipe model validation — U-shaped heat exchanger outlet",
        height=520,
        margin=dict(l=70, r=50, t=70, b=70),
        legend=dict(
            x=0.99,
            y=0.99,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(15,23,42,0.65)",
            bordercolor="rgba(148,163,184,0.6)",
            borderwidth=1,
            font=dict(size=11, color="#e2e8f0"),
        ),
        hovermode="x unified",
        plot_bgcolor="rgba(15,23,42,0.08)",
        paper_bgcolor="rgba(15,23,42,0.02)",
        font=dict(color="#e2e8f0"),
    )
    fig.update_xaxes(title_text="Time (hours)", showgrid=True, gridcolor="rgba(148,163,184,0.25)")
    fig.update_yaxes(title_text="Temperature (°C)", showgrid=True, gridcolor="rgba(148,163,184,0.25)")
    return fig


def _pipe_outlet_error_summary(validation: pd.DataFrame) -> dict[str, float] | None:
    """Fit quality of the simulated outlet against the measured one."""
    if validation is None or validation.empty:
        return None
    if not {"t_outlet_meas_c", "t_outlet_sim_c"} <= set(validation.columns):
        return None
    meas = _safe_float_series(validation["t_outlet_meas_c"])
    sim = _safe_float_series(validation["t_outlet_sim_c"])
    err = (sim - meas).dropna()
    if err.empty:
        return None
    return {
        "rmse_c": float(np.sqrt(np.mean(np.square(err)))),
        "mae_c": float(np.mean(np.abs(err))),
        "bias_c": float(np.mean(err)),
        "max_abs_c": float(np.max(np.abs(err))),
        "n": float(len(err)),
    }


def _fem_variant_error_summary(
    sensors: pd.DataFrame,
    forecast_temperature: pd.DataFrame,
) -> dict[str, float] | None:
    """Aggregate error of one FEM variant against the measured field.

    Used to score the calibrated run and the uncalibrated (no-inversion) twin on
    the same measured data, so the two numbers are directly comparable.
    """
    frame = _fem_error_frame(sensors, forecast_temperature)
    if frame is None or frame.empty:
        return None
    return {
        "rmse_c": float(np.sqrt(np.mean(np.square(frame["signed_error_c"])))),
        "mae_c": float(frame["abs_error_c"].mean()),
        "bias_c": float(frame["signed_error_c"].mean()),
        "n": float(len(frame)),
    }


def _fem_error_timeseries_plot(error_frame: pd.DataFrame) -> go.Figure | None:
    """Mean and worst-sensor percent error over time.

    Both series share the percent scale, so they sit on one axis. Absolute error
    in C rides along in the hover rather than on a second y-axis.
    """
    if error_frame is None or error_frame.empty:
        return None

    grouped = (
        error_frame.groupby("time_hours")
        .agg(
            mean_pct=("error_pct", "mean"),
            max_pct=("error_pct", "max"),
            mean_c=("abs_error_c", "mean"),
            max_c=("abs_error_c", "max"),
        )
        .reset_index()
        .sort_values("time_hours")
    )
    if grouped.empty:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=grouped["time_hours"],
            y=grouped["max_pct"],
            mode="lines",
            name="Worst sensor",
            line={"color": _ERROR_SERIES_MAX, "width": 2},
            customdata=grouped["max_c"],
            hovertemplate="Worst sensor: %{y:.3f}%%  (%{customdata:.2f} °C)<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["time_hours"],
            y=grouped["mean_pct"],
            mode="lines",
            name="Mean over sensors",
            line={"color": _ERROR_SERIES_MEAN, "width": 2},
            customdata=grouped["mean_c"],
            hovertemplate="Mean: %{y:.3f}%%  (%{customdata:.2f} °C)<extra></extra>",
        )
    )

    # Direct-label the endpoint only — the axis and hover carry the rest.
    last = grouped.iloc[-1]
    fig.add_trace(
        go.Scatter(
            x=[last["time_hours"]],
            y=[last["mean_pct"]],
            mode="markers+text",
            marker={"size": 9, "color": _ERROR_SERIES_MEAN, "line": {"width": 2, "color": "#fcfcfb"}},
            text=[f" {last['mean_pct']:.2f}%"],
            textposition="middle right",
            textfont={"color": "#52514e"},
            # Scatter text is clipped at the axis boundary by default, which cuts
            # the endpoint label no matter how wide the right margin is.
            cliponaxis=False,
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=(
            "Model drift over time — reference % metric"
            "<br><sup>Flat = tracking the data · rising = drifting apart</sup>"
        ),
        height=340,
        # r=90 leaves room for the endpoint direct label; at r=30 it clipped.
        margin=dict(l=55, r=90, t=64, b=96),
        xaxis_title="Time (hours)",
        yaxis_title="|error| (%)",
        hovermode="x unified",
        # Below the plot: a right-anchored legend at y=1.0 sat on top of the
        # centred subtitle, which spans nearly the full width at this size.
        legend={"orientation": "h", "yanchor": "top", "y": -0.34, "xanchor": "center", "x": 0.5},
        plot_bgcolor="rgba(15,23,42,0.08)",
        paper_bgcolor="rgba(15,23,42,0.02)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(15,23,42,0.10)", gridwidth=1, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(15,23,42,0.10)", gridwidth=1, rangemode="tozero", zeroline=False)
    return fig


def _fem_error_field_plot(error_frame: pd.DataFrame) -> go.Figure | None:
    """Spatial percent-error field at the latest paired timestep.

    The direct analogue of the reference's Tecplot zone: one nodal error map per
    step. Reuses the sensor-hull interpolation so nothing is extrapolated into
    uncovered regions.
    """
    if error_frame is None or error_frame.empty:
        return None

    snap_time = float(error_frame["time_hours"].max())
    latest = error_frame[error_frame["time_hours"] == snap_time]
    if latest.empty:
        return None

    # _interpolate_temperature_field keys off a "temperature" column; feed it the
    # error so the coverage masking and constant-in-r hold apply unchanged.
    field = _interpolate_temperature_field(
        latest[["x_m", "z_m", "error_pct"]].rename(columns={"error_pct": "temperature"})
    )
    if field is None:
        return None

    R, Z, E = field
    # |error| is non-negative by construction; cubic interpolation can undershoot
    # slightly below zero between sensors, which would surface as a negative
    # percentage in the hover. Clamp rather than let that leak into the readout.
    E = np.clip(E, 0.0, None)
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=R[0, :] * 1000.0,
            y=Z[:, 0] * 1000.0,
            z=E,
            colorscale=_ERROR_SEQUENTIAL,
            zmin=0.0,
            colorbar={"title": "|error|<br>(% of abs. T)"},
            contours={"showlines": False},
            connectgaps=False,
            hovertemplate="r %{x:.0f} mm · z %{y:.0f} mm<br>%{z:.3f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=latest["x_m"] * 1000.0,
            y=latest["z_m"] * 1000.0,
            mode="markers",
            marker={"size": 8, "color": "rgba(0,0,0,0)", "line": {"width": 2, "color": "#fcfcfb"}},
            name="Sensors",
            customdata=np.stack(
                [latest["error_pct"], latest["abs_error_c"], latest["temperature_obs"], latest["temperature_fem"]],
                axis=-1,
            ),
            text=latest["sensor_id"],
            hovertemplate=(
                "Sensor %{text}<br>error: %{customdata[0]:.3f}%% "
                "(%{customdata[1]:.2f} °C)<br>"
                "measured %{customdata[2]:.2f} °C · FEM %{customdata[3]:.2f} °C<extra></extra>"
            ),
        )
    )

    fig.add_shape(
        type="rect",
        x0=0.0,
        x1=THERMOSYPHON_RADIUS_M * 1000.0,
        y0=0.0,
        y1=DOMAIN_DEPTH_M * 1000.0,
        fillcolor="#111827",
        opacity=0.6,
        line_width=0,
        layer="below",
    )
    fig.update_layout(
        title=(
            "Where the model disagrees — reference % metric"
            f"<br><sup>t = {snap_time:.2f} h · darker = larger disagreement · "
            "magnitude only, no direction</sup>"
        ),
        xaxis_title="Radial distance (mm)",
        yaxis_title="Depth (mm)",
        height=440,
        margin=dict(l=55, r=20, t=60, b=40),
        showlegend=False,
    )
    fig.update_xaxes(range=[0, DOMAIN_RADIUS_M * 1000.0])
    fig.update_yaxes(range=[DOMAIN_DEPTH_M * 1000.0, 0.0])
    return fig


def _fem_pairing_report(
    sensors: pd.DataFrame,
    forecast_temperature: pd.DataFrame,
    error_frame: pd.DataFrame | None,
) -> dict[str, float | int] | None:
    """How much of the available FEM/measured data actually paired up.

    Guards the failure mode in ``fem_forecast_server._process_message``: the FEM
    field snapshot is timestamped ``t`` on the plain path but ``t + step_hours``
    on the ``_run_forecast_with_field`` path (taken whenever include_field,
    save_field_csv, or *any* sticky material_override is active). A one-step
    offset makes the join silently drop rows rather than fail, so low coverage
    here means "the two streams are misaligned", not "no data".
    """
    if sensors is None or forecast_temperature is None:
        return None
    if sensors.empty or forecast_temperature.empty:
        return None
    if "time_hours" not in sensors.columns or "time_hours" not in forecast_temperature.columns:
        return None

    obs_times = set(np.round(_safe_float_series(sensors["time_hours"]).dropna(), 2))
    fem_times = set(np.round(_safe_float_series(forecast_temperature["time_hours"]).dropna(), 2))
    if not obs_times or not fem_times:
        return None

    paired_times = set()
    if error_frame is not None and not error_frame.empty:
        paired_times = set(np.round(error_frame["time_hours"], 2))

    # Coverage alone cannot see the offset this is meant to catch: shifting a
    # uniformly-sampled stream by exactly one sampling interval still overlaps
    # almost perfectly, and step_hours *is* the message spacing in tracking mode.
    # So test the alignment directly — re-pair with the FEM stream nudged one
    # step each way and see which lag fits best. If a shifted pairing fits
    # materially better than the zero-lag one, the streams are off by a step.
    fem_sorted = np.sort(np.fromiter(fem_times, dtype=float))
    step = float(np.median(np.diff(fem_sorted))) if len(fem_sorted) > 1 else 0.0
    rmse_by_lag: dict[int, float] = {}
    if step > 0:
        for lag in (-1, 0, 1):
            shifted = forecast_temperature.copy()
            shifted["time_hours"] = _safe_float_series(shifted["time_hours"]) + lag * step
            candidate = error_frame if lag == 0 else _fem_error_frame(sensors, shifted)
            if candidate is None or candidate.empty:
                continue
            rmse_by_lag[lag] = float(np.sqrt(np.mean(np.square(candidate["signed_error_c"]))))

    best_lag = min(rmse_by_lag, key=rmse_by_lag.get) if rmse_by_lag else 0
    zero_rmse = rmse_by_lag.get(0)
    misaligned = (
        best_lag != 0
        and zero_rmse is not None
        and rmse_by_lag[best_lag] < 0.8 * zero_rmse
    )

    return {
        "obs_steps": len(obs_times),
        "fem_steps": len(fem_times),
        "paired_steps": len(paired_times),
        "coverage": len(paired_times) / max(len(fem_times), 1),
        "best_lag": best_lag,
        "step_hours": step,
        "rmse_at_zero_lag": zero_rmse if zero_rmse is not None else float("nan"),
        "rmse_at_best_lag": rmse_by_lag.get(best_lag, float("nan")),
        "misaligned": bool(misaligned),
    }


def _fem_error_metrics_plot(error_frame: pd.DataFrame) -> go.Figure | None:
    """RMSE and signed bias in °C — the improved companion to the percent chart.

    Two curves in the same physical unit on one axis:

    * **RMSE** — magnitude of disagreement, in degrees a reader can judge.
    * **Signed bias** — mean of (FEM − measured), so direction is visible.

    Read together they separate a systematic offset from scatter: RMSE ≈ |bias|
    means the whole field is shifted, RMSE >> |bias| means it is noisy about the
    right mean. The reference's ``abs(...)`` percent metric can distinguish
    neither, and its Kelvin denominator compresses everything into 0-2%.
    """
    if error_frame is None or error_frame.empty:
        return None

    grouped = (
        error_frame.groupby("time_hours")
        .agg(
            rmse_c=("signed_error_c", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            bias_c=("signed_error_c", "mean"),
            worst_c=("abs_error_c", "max"),
            n=("signed_error_c", "size"),
        )
        .reset_index()
        .sort_values("time_hours")
    )
    if grouped.empty:
        return None

    fig = go.Figure()
    fig.add_hline(y=0.0, line_color="rgba(15,23,42,0.35)", line_width=1)
    fig.add_trace(
        go.Scatter(
            x=grouped["time_hours"],
            y=grouped["rmse_c"],
            mode="lines",
            name="RMSE",
            line={"color": _ERROR_SERIES_MAX, "width": 2},
            customdata=grouped["worst_c"],
            hovertemplate="RMSE: %{y:.2f} °C  (worst %{customdata:.2f} °C)<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["time_hours"],
            y=grouped["bias_c"],
            mode="lines",
            name="Bias (FEM − measured)",
            line={"color": _ERROR_SERIES_MEAN, "width": 2},
            hovertemplate="Bias: %{y:+.2f} °C<extra></extra>",
        )
    )

    last = grouped.iloc[-1]
    fig.add_trace(
        go.Scatter(
            x=[last["time_hours"]],
            y=[last["rmse_c"]],
            mode="markers+text",
            marker={"size": 9, "color": _ERROR_SERIES_MAX, "line": {"width": 2, "color": "#fcfcfb"}},
            text=[f" {last['rmse_c']:.2f} °C"],
            textposition="middle right",
            textfont={"color": "#52514e"},
            cliponaxis=False,
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # The FEM initial condition is seeded from the first sensor snapshot, so if
    # that step is in view its error is ~0 by construction, not by accuracy.
    # Flag it instead of dropping it — a mid-window view's first point is a
    # genuine measurement and must not be silently discarded.
    if len(grouped) >= 4:
        first_rmse = float(grouped["rmse_c"].iloc[0])
        median_rmse = float(grouped["rmse_c"].median())
        if median_rmse > 0 and first_rmse < 0.2 * median_rmse:
            fig.add_annotation(
                x=float(grouped["time_hours"].iloc[0]),
                y=first_rmse,
                text="IC seeded from data",
                showarrow=True,
                arrowhead=0,
                arrowcolor="rgba(15,23,42,0.45)",
                ax=30,
                ay=-34,
                font={"size": 11, "color": "#52514e"},
            )

    fig.update_layout(
        title=(
            "Model error in °C — how big, and which way"
            "<br><sup>RMSE ≈ |bias| → uniform offset · RMSE ≫ |bias| → scatter</sup>"
        ),
        height=340,
        margin=dict(l=55, r=95, t=64, b=96),
        xaxis_title="Time (hours)",
        yaxis_title="Error (°C)",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "top", "y": -0.34, "xanchor": "center", "x": 0.5},
        plot_bgcolor="rgba(15,23,42,0.08)",
        paper_bgcolor="rgba(15,23,42,0.02)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(15,23,42,0.10)", gridwidth=1, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(15,23,42,0.10)", gridwidth=1, zeroline=False)
    return fig


def _fem_bias_field_plot(error_frame: pd.DataFrame) -> go.Figure | None:
    """Signed error field in °C at the latest paired timestep.

    Three departures from ``_fem_error_field_plot``:

    * **Signed, not absolute** — a diverging blue/red scale about a gray zero, so
      "model too cold" and "model too warm" are distinguishable regions rather
      than one undifferentiated magnitude blob.
    * **Symmetric range** about 0, so the midpoint is always true zero.

    The band past the outermost sensor (~165 mm of a 240 mm domain) is filled by
    the same constant-in-r hold the temperature contours use, so this map covers
    the domain like its neighbours. A rule at the last sensor column marks where
    measurement stops — everything right of it is held, not observed.
    """
    if error_frame is None or error_frame.empty:
        return None

    snap_time = float(error_frame["time_hours"].max())
    latest = error_frame[error_frame["time_hours"] == snap_time]
    if latest.empty:
        return None

    field = _interpolate_temperature_field(
        latest[["x_m", "z_m", "signed_error_c"]].rename(columns={"signed_error_c": "temperature"}),
    )
    if field is None:
        return None

    R, Z, E = field
    limit = float(np.nanmax(np.abs(E))) if np.isfinite(E).any() else 1.0
    if not np.isfinite(limit) or limit <= 0.0:
        limit = 1.0

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=R[0, :] * 1000.0,
            y=Z[:, 0] * 1000.0,
            z=E,
            colorscale=_BIAS_DIVERGING,
            zmin=-limit, zmax=limit, zmid=0.0,
            colorbar={"title": "FEM − measured (°C)"},
            contours={"showlines": False},
            connectgaps=False,
            hovertemplate="r %{x:.0f} mm · z %{y:.0f} mm<br>%{z:+.2f} °C<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=latest["x_m"] * 1000.0,
            y=latest["z_m"] * 1000.0,
            mode="markers",
            # Dark ring, not the usual surface-coloured one: the diverging fill is
            # palest exactly where it matters most (near zero), and a white ring
            # disappears into it.
            marker={"size": 8, "color": "rgba(0,0,0,0)", "line": {"width": 1.5, "color": "#0f172a"}},
            name="Sensors",
            text=latest["sensor_id"],
            customdata=np.stack(
                [latest["signed_error_c"], latest["temperature_obs"], latest["temperature_fem"]],
                axis=-1,
            ),
            hovertemplate=(
                "Sensor %{text}<br>FEM − measured: %{customdata[0]:+.2f} °C<br>"
                "measured %{customdata[1]:.2f} °C · FEM %{customdata[2]:.2f} °C<extra></extra>"
            ),
        )
    )

    fig.add_shape(
        type="rect",
        x0=0.0, x1=THERMOSYPHON_RADIUS_M * 1000.0,
        y0=0.0, y1=DOMAIN_DEPTH_M * 1000.0,
        fillcolor="#111827", opacity=0.6, line_width=0, layer="below",
    )

    # Mark where the sensors stop. Right of this the field is a constant-in-r
    # hold of the outermost column, so it is extrapolation, not measurement.
    r_last_mm = float(latest["x_m"].max()) * 1000.0
    if r_last_mm < DOMAIN_RADIUS_M * 1000.0 - 1.0:
        fig.add_vline(
            x=r_last_mm,
            line_color="rgba(15,23,42,0.55)",
            line_width=1,
            annotation_text="last sensor · held beyond",
            annotation_position="top right",
            annotation_font={"size": 11, "color": "#52514e"},
            # Inside the plot the label lands on the z=10 mm sensor row and its
            # first glyph sits under the rule; lift it into the title margin.
            annotation_yshift=12,
            annotation_xshift=4,
        )

    fig.update_layout(
        title=(
            "Which way the model is wrong — signed bias"
            f"<br><sup>t = {snap_time:.2f} h · red = FEM too warm · "
            "blue = FEM too cold · pale = agreement</sup>"
        ),
        xaxis_title="Radial distance (mm)",
        yaxis_title="Depth (mm)",
        height=460,
        # t=78 keeps the two-line title clear of the "last sensor" rule label,
        # which sits just above the plot frame.
        margin=dict(l=55, r=20, t=78, b=40),
        showlegend=False,
    )
    fig.update_xaxes(range=[0, DOMAIN_RADIUS_M * 1000.0])
    fig.update_yaxes(range=[DOMAIN_DEPTH_M * 1000.0, 0.0])
    return fig


def _fem_error_field_animated(
    error_frame: pd.DataFrame,
    *,
    title: str = "Where the model disagrees — reference % metric",
    hint: str = "darker = larger disagreement · magnitude only, no direction",
    value_col: str = "error_pct",
    signed: bool = False,
    colorbar_title: str = "|error|<br>(% of abs. T)",
    max_frames: int = _MAX_ANIMATION_FRAMES,
    grid_resolution: int = _ANIMATION_GRID_RES,
) -> go.Figure | None:
    """Animated error field with play button and scrubber.

    Same construction as ``_temperature_contour_animated`` — capped frame count,
    coarser grid, and only ``z`` retransmitted per frame — applied to the paired
    error field instead of temperature. Serves both the absolute percent map
    (``signed=False``) and the signed °C bias map (``signed=True``, which uses a
    diverging scale symmetric about zero).

    The colour range is fixed across every frame. A per-frame autoscale would
    renormalise the ramp on each step and make a shrinking error look constant
    (or a growing one look flat), which is the one thing this animation exists
    to show.
    """
    if error_frame is None or error_frame.empty or "time_hours" not in error_frame.columns:
        return None
    if value_col not in error_frame.columns:
        return None

    all_times = np.sort(error_frame["time_hours"].dropna().unique())
    if len(all_times) == 0:
        return None
    if len(all_times) == 1:
        return _fem_bias_field_plot(error_frame) if signed else _fem_error_field_plot(error_frame)

    n_total = len(all_times)
    if n_total > max_frames:
        indices = np.round(np.linspace(0, n_total - 1, max_frames)).astype(int)
        sampled_times = all_times[indices]
    else:
        sampled_times = all_times

    r_grid = np.linspace(0.0, DOMAIN_RADIUS_M, grid_resolution)
    z_grid = np.linspace(0.0, DOMAIN_DEPTH_M, grid_resolution)
    R, Z_mesh = np.meshgrid(r_grid, z_grid)
    x_axis = (R[0, :] * 1000.0).tolist()
    y_axis = (Z_mesh[:, 0] * 1000.0).tolist()

    frame_entries: list[dict] = []
    for t in sampled_times:
        snapshot = error_frame[np.isclose(error_frame["time_hours"], t)]
        if snapshot.empty:
            continue
        field = _interpolate_temperature_field(
            snapshot[["x_m", "z_m", value_col]].rename(columns={value_col: "temperature"}),
            grid_resolution=grid_resolution,
        )
        if field is None:
            continue
        _, _, E = field
        if not signed:
            E = np.clip(E, 0.0, None)
        frame_entries.append({
            "t": float(t),
            "z": E.tolist(),
            "sx": (snapshot["x_m"].astype(float) * 1000.0).tolist(),
            "sz": (snapshot["z_m"].astype(float) * 1000.0).tolist(),
            "sid": snapshot["sensor_id"].astype(str).tolist(),
            "cd": np.stack(
                [
                    snapshot[value_col].astype(float),
                    snapshot["abs_error_c"].astype(float),
                    snapshot["temperature_obs"].astype(float),
                    snapshot["temperature_fem"].astype(float),
                ],
                axis=-1,
            ).tolist(),
        })

    if not frame_entries:
        return None

    if signed:
        limit = max(float(np.nanmax(np.abs(np.asarray(fe["z"], dtype=float)))) for fe in frame_entries)
        if not np.isfinite(limit) or limit <= 0.0:
            limit = 1.0
        zmin, zmax = -limit, limit
        colorscale = _BIAS_DIVERGING
        field_hover = "r %{x:.0f} mm · z %{y:.0f} mm<br>%{z:+.2f} °C<extra></extra>"
        sensor_hover = (
            "Sensor %{text}<br>FEM − measured: %{customdata[0]:+.2f} °C<br>"
            "measured %{customdata[2]:.2f} °C · FEM %{customdata[3]:.2f} °C<extra></extra>"
        )
    else:
        zmax = max(float(np.nanmax(np.asarray(fe["z"], dtype=float))) for fe in frame_entries)
        if not np.isfinite(zmax) or zmax <= 0.0:
            zmax = 1.0
        zmin = 0.0
        colorscale = _ERROR_SEQUENTIAL_DARK
        field_hover = "r %{x:.0f} mm · z %{y:.0f} mm<br>%{z:.3f}%<extra></extra>"
        sensor_hover = (
            "Sensor %{text}<br>error: %{customdata[0]:.3f}%% "
            "(%{customdata[1]:.2f} °C)<br>"
            "measured %{customdata[2]:.2f} °C · FEM %{customdata[3]:.2f} °C<extra></extra>"
        )
    first = frame_entries[0]

    fig = go.Figure()
    # Heatmap rather than filled Contour for the same reason as the temperature
    # animation: filled contours redraw their SVG paths per frame and briefly
    # blank the panel.
    heatmap_kwargs = {"zmid": 0.0} if signed else {}
    fig.add_trace(go.Heatmap(
        x=x_axis, y=y_axis, z=first["z"],
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        zsmooth="best",
        colorbar={"title": colorbar_title},
        hovertemplate=field_hover,
        name="Error",
        **heatmap_kwargs,
    ))
    fig.add_trace(go.Scatter(
        x=first["sx"], y=first["sz"],
        mode="markers",
        marker={"size": 8, "color": "rgba(0,0,0,0)", "line": {"width": 2, "color": "#0b1220"}},
        name="Sensors",
        text=first["sid"],
        customdata=first["cd"],
        hovertemplate=sensor_hover,
    ))

    fig.add_shape(
        type="rect",
        x0=0.0, x1=THERMOSYPHON_RADIUS_M * 1000.0,
        y0=0.0, y1=DOMAIN_DEPTH_M * 1000.0,
        fillcolor="#111827", opacity=0.6, line_width=0, layer="below",
    )

    frames = [
        go.Frame(
            data=[
                go.Heatmap(z=fe["z"]),
                go.Scatter(x=fe["sx"], y=fe["sz"], text=fe["sid"], customdata=fe["cd"]),
            ],
            traces=[0, 1],
            name=f"{fe['t']:.4f}",
        )
        for fe in frame_entries
    ]
    fig.frames = frames

    frame_ms = 300
    slider_steps = [
        {
            "args": [
                [f.name],
                {"frame": {"duration": frame_ms, "redraw": True},
                 "mode": "immediate",
                 "transition": {"duration": 0}},
            ],
            "label": "",
            "method": "animate",
        }
        for f in frames
    ]

    n_sampled = len(frame_entries)
    # Stating the ratio reads as "1 of every 1 snapshots" whenever the cap only
    # trims a little (50 snapshots -> 40 frames); name the totals instead.
    subtitle = f"{hint} · {n_sampled} frames"
    if n_sampled < n_total:
        subtitle += f" sampled from {n_total} snapshots"
    subtitle += " · colour scale fixed across frames"

    play_pause_button = {
        "label": "⏯",
        "method": "animate",
        "args": [None, {"frame": {"duration": frame_ms, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0}}],
        "args2": [[None], {"frame": {"duration": 0, "redraw": False},
                           "mode": "immediate",
                           "transition": {"duration": 0}}],
    }

    accent = "#2563eb"
    fig.update_layout(
        title={"text": f"{title}<br><sup>{subtitle}</sup>"},
        xaxis_title="Radial distance (mm)",
        yaxis_title="Depth (mm)",
        height=520,
        margin=dict(l=55, r=20, t=60, b=110),
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        font={"color": "#e2e8f0"},
        showlegend=False,
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.02, "y": -0.02,
            "xanchor": "left", "yanchor": "top",
            "pad": {"t": 64, "b": 8, "l": 10, "r": 10},
            "bgcolor": accent,
            "bordercolor": "#1d4ed8",
            "borderwidth": 2,
            "font": {"color": "#ffffff", "size": 22},
            "buttons": [play_pause_button],
        }],
        sliders=[{
            "active": 0,
            # x=0.12 put the "Time <n> h" readout underneath the play button; the
            # readout is left-anchored to the track start, so the track has to
            # begin clear of the button.
            "x": 0.26, "len": 0.74,
            "pad": {"t": 70, "b": 10},
            "bgcolor": "#334155",
            "activebgcolor": "#60a5fa",
            "bordercolor": "#475569",
            "borderwidth": 1,
            "tickcolor": "#64748b",
            "ticklen": 5,
            "minorticklen": 0,
            "currentvalue": {
                "prefix": "Time  ",
                "suffix": " h",
                "font": {"size": 16, "color": "#e2e8f0"},
                "visible": True,
                "xanchor": "left",
                "offset": 12,
            },
            "transition": {"duration": 0},
            "steps": slider_steps,
        }],
    )
    fig.update_xaxes(range=[0, DOMAIN_RADIUS_M * 1000.0])
    fig.update_yaxes(range=[DOMAIN_DEPTH_M * 1000.0, 0.0])
    return fig


_SEVERITY_LEVEL = {"ok": 0, "warning": 1, "critical": 2, "alert": 3}
_SEVERITY_COLOR = {
    "ok": "#22c55e",
    "warning": "#eab308",
    "critical": "#f97316",
    "alert": "#ef4444",
}


def _alerts_series(df: pd.DataFrame) -> go.Figure | None:
    if df is None or df.empty or "time_hours" not in df.columns:
        return None

    plot_df = df.copy()
    plot_df["time_hours"] = _safe_float_series(plot_df["time_hours"])

    # Graded severity: with the dual-limit rule engine, status ranges over
    # ok/warning/critical/alert. Map to an ordinal level so warnings are visible
    # as their own tier instead of collapsing onto the triggered 0/1 axis. Fall
    # back to triggered for legacy single-metric alerts that only emit alert/ok.
    if "status" in plot_df.columns:
        status = plot_df["status"].astype(str).str.lower()
    elif "triggered" in plot_df.columns:
        status = plot_df["triggered"].map(lambda t: "alert" if bool(t) else "ok")
    else:
        return None
    plot_df["status"] = status
    plot_df["severity"] = status.map(_SEVERITY_LEVEL).fillna(0)
    plot_df = plot_df.dropna(subset=["time_hours"]).sort_values("time_hours")
    if plot_df.empty:
        return None

    fig = go.Figure()
    for level_name in ("ok", "warning", "critical", "alert"):
        sub = plot_df[plot_df["status"] == level_name]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["time_hours"],
                y=sub["severity"],
                mode="markers",
                name=level_name,
                marker=dict(size=9, color=_SEVERITY_COLOR[level_name]),
                hovertemplate=f"{level_name}<br>t=%{{x:.2f}}h<extra></extra>",
            )
        )
    fig.update_layout(
        title="Safety status timeline",
        height=280,
        xaxis_title="time (hours)",
        showlegend=True,
    )
    fig.update_yaxes(
        title_text="severity",
        tickmode="array",
        tickvals=[0, 1, 2, 3],
        ticktext=["ok", "warning", "critical", "alert"],
        range=[-0.5, 3.5],
    )
    return fig


_RHO_FLUID = 1110.0
_CP_FLUID  = 2400.0
_V_DOT     = 0.5 / 1000 / 60   # m³/s  (0.5 L/min)
_RHO_ICE   = 917.0
_ICE_FRAC  = 0.39
_LF        = 334_000.0


def _compute_energy_metrics(
    fluid_temps: pd.DataFrame,
    thaw_metrics: pd.DataFrame,
) -> dict | None:
    if fluid_temps is None or fluid_temps.empty:
        return None
    required = {"time_hours", "t_inlet_c", "t_outlet_c"}
    if not required <= set(fluid_temps.columns):
        return None

    df = fluid_temps.copy()
    df["time_hours"] = _safe_float_series(df["time_hours"])
    df["t_inlet_c"] = _safe_float_series(df["t_inlet_c"])
    df["t_outlet_c"] = _safe_float_series(df["t_outlet_c"])
    df = df.dropna(subset=list(required)).sort_values("time_hours")
    if len(df) < 2:
        return None

    t_h = df["time_hours"].to_numpy()
    t_s = t_h * 3600.0
    Q = np.clip(
        _RHO_FLUID * _V_DOT * _CP_FLUID * (df["t_inlet_c"].to_numpy() - df["t_outlet_c"].to_numpy()),
        0.0, None,
    )
    Ein = np.concatenate([[0.0], cumulative_trapezoid(Q, t_s)])

    V_thaw = None
    if thaw_metrics is not None and not thaw_metrics.empty and "thaw_volume_m3" in thaw_metrics.columns:
        tm = thaw_metrics.copy()
        tm["time_hours"] = _safe_float_series(tm["time_hours"])
        tm["thaw_volume_m3"] = _safe_float_series(tm["thaw_volume_m3"])
        tm = tm.dropna(subset=["time_hours", "thaw_volume_m3"]).sort_values("time_hours")
        if not tm.empty:
            V_thaw = np.interp(t_h, tm["time_hours"].to_numpy(), tm["thaw_volume_m3"].to_numpy())

    with np.errstate(divide="ignore", invalid="ignore"):
        if V_thaw is not None:
            Er  = np.where(V_thaw > 1e-9, Ein / V_thaw / 1e6, np.nan)
            eta = np.where(Ein > 0, _RHO_ICE * _ICE_FRAC * V_thaw * _LF / Ein * 100.0, np.nan)
        else:
            Er  = np.full_like(Ein, np.nan)
            eta = np.full_like(Ein, np.nan)

    return {"t_h": t_h, "Ein": Ein, "V_thaw": V_thaw, "Er": Er, "eta": eta}


def _energy_ein_vs_vthaw(metrics: dict) -> go.Figure | None:
    V_thaw = metrics.get("V_thaw")
    if V_thaw is None:
        return None
    fig = go.Figure(go.Scatter(
        x=V_thaw * 1000.0, y=metrics["Ein"] / 1e6,
        mode="lines", line={"color": "#38bdf8", "width": 2},
    ))
    fig.update_layout(
        title="Cumulative energy input vs thaw volume",
        xaxis_title="V_thaw (L)", yaxis_title="E_in (MJ)",
        height=400, margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig


def _energy_specific(metrics: dict) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=metrics["t_h"], y=metrics["Er"],
        mode="lines", line={"color": "#fb923c", "width": 2},
    ))
    fig.update_layout(
        title="Specific energy consumption",
        xaxis_title="Time (h)", yaxis_title="E_r (MJ/m³)",
        height=400, margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig


def _energy_vthaw_time(metrics: dict) -> go.Figure | None:
    V_thaw = metrics.get("V_thaw")
    if V_thaw is None:
        return None
    fig = go.Figure(go.Scatter(
        x=metrics["t_h"], y=V_thaw * 1000.0,
        mode="lines", line={"color": "#4ade80", "width": 2},
    ))
    fig.update_layout(
        title="Thaw volume over time",
        xaxis_title="Time (h)", yaxis_title="V_thaw (L)",
        height=400, margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig


def _energy_eta(metrics: dict) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=metrics["t_h"], y=metrics["eta"],
        mode="lines", line={"color": "#a78bfa", "width": 2},
    ))
    fig.update_layout(
        title="Melting efficiency",
        xaxis_title="Time (h)", yaxis_title="η_melt (%)",
        height=400, margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig


def _apply_theme() -> None:
    st.set_page_config(page_title="Permafrost DT Dashboard", layout="wide")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&family=IBM+Plex+Mono:wght@400;600&display=swap');
        html, body, [class*="css"]  {
          font-family: 'Space Grotesk', sans-serif;
          background: radial-gradient(circle at 15% 15%, #f1f5f9 0%, #e2e8f0 45%, #cbd5f5 100%);
        }
        h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.5px; }
        .block-container { padding-top: 2rem; }
        .status-badge {
          padding: 0.35rem 0.8rem;
          border-radius: 999px;
          font-weight: 600;
          letter-spacing: 0.05em;
          display: inline-block;
        }
        .status-safe { background: #bbf7d0; color: #14532d; }
        .status-caution { background: #fde68a; color: #854d0e; }
        .status-alert { background: #fecaca; color: #7f1d1d; }
        .status-stopped { background: #dbeafe; color: #1e3a8a; }
        .banner {
          border-radius: 14px;
          padding: 0.75rem 1rem;
          margin-bottom: 1rem;
          border: 1px solid rgba(15,23,42,0.1);
        }
        .banner-warning { background: #fee2e2; color: #7f1d1d; }
        .banner-info { background: #e0f2fe; color: #0c4a6e; }

        /* ---- Closed-loop signal-bus widget ----
           A borderless diagram: status dots strung on a vertical bus, the flow
           animating along each segment, and a bracket on the left closing the
           loop back to the top. Base = light; dark layered via prefers-color-scheme. */
        .dtflow { max-width: 300px; margin: 0.15rem 0 0.5rem; --rail: #cbd5e1; }
        .dtflow-title {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.66rem; font-weight: 600; letter-spacing: 0.12em;
          text-transform: uppercase; color: #475569;
          display: flex; align-items: center; gap: 7px; margin-bottom: 0.7rem;
        }
        .dtflow-mark { font-size: 0.95rem; line-height: 1; color: #16a34a;
                       animation: dtflow-spin 3.2s linear infinite; }
        .dtflow-body { position: relative; padding-left: 24px; }
        .dtflow-stage {
          position: relative;
          display: grid; grid-template-columns: 16px 1fr auto;
          align-items: center; gap: 11px; min-height: 32px;
        }
        /* bus segment linking this node's dot to the next one */
        .dtflow-stage:not(:last-child)::before {
          content: ''; position: absolute; left: 7px; top: 22px; bottom: -10px;
          width: 2px; background: var(--rail);
        }
        .dtflow-stage.flowing:not(:last-child)::before {
          background: repeating-linear-gradient(180deg, #22c55e 0 5px, transparent 5px 11px);
          background-size: 2px 11px;
          animation: dtflow-move 0.7s linear infinite;
        }
        .dtflow-dot {
          width: 12px; height: 12px; border-radius: 50%; justify-self: center;
          position: relative; z-index: 1;
        }
        .dot-green  { background: #22c55e;
                      box-shadow: 0 0 0 3px rgba(34,197,94,0.18), 0 0 10px rgba(34,197,94,0.95);
                      animation: dtflow-pulse 1.6s ease-in-out infinite; }
        .dot-yellow { background: #eab308;
                      box-shadow: 0 0 0 3px rgba(234,179,8,0.16), 0 0 9px rgba(234,179,8,0.85); }
        .dot-red    { background: #ef4444;
                      box-shadow: 0 0 0 3px rgba(239,68,68,0.14), 0 0 8px rgba(239,68,68,0.70); }
        .dtflow-name { font-size: 0.8rem; color: #1e293b; font-weight: 500; }
        .dtflow-age  { font-size: 0.64rem; color: #64748b; font-variant-numeric: tabular-nums; }
        /* left bracket: output loops back up to the sensors (input) */
        .dtflow-return {
          position: absolute; left: 3px; top: 15px; bottom: 15px; width: 11px;
          border: 2px solid var(--rail); border-right: none; border-radius: 8px 0 0 8px;
        }
        .dtflow-return::before {
          content: '\\25B2'; position: absolute; top: -9px; left: -4px;
          font-size: 8px; color: #94a3b8;
        }
        .dtflow-return::after {
          content: '\\21BA'; position: absolute; bottom: -10px; left: -5px;
          font-size: 12px; color: #94a3b8;
        }
        .dtflow-return.flowing { border-color: #22c55e; }
        .dtflow-return.flowing::before { color: #22c55e; }
        .dtflow-state {
          margin-top: 0.65rem; padding-left: 24px;
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.68rem; font-weight: 600; letter-spacing: 0.06em;
        }
        .dtflow-live     { color: #15803d; }
        .dtflow-degraded { color: #a16207; }
        .dtflow-offline  { color: #b91c1c; }

        /* Dark theme overrides (colour only). */
        @media (prefers-color-scheme: dark) {
          .dtflow { --rail: #334155; }
          .dtflow-title { color: #94a3b8; }
          .dtflow-mark  { color: #4ade80; }
          .dtflow-name  { color: #e2e8f0; }
          .dtflow-age   { color: #94a3b8; }
          .dtflow-return::before,
          .dtflow-return::after { color: #64748b; }
          .dtflow-live     { color: #4ade80; }
          .dtflow-degraded { color: #fbbf24; }
          .dtflow-offline  { color: #f87171; }
        }
        @keyframes dtflow-move  { from { background-position-y: 0; } to { background-position-y: 11px; } }
        @keyframes dtflow-pulse { 0%,100% { filter: brightness(1); } 50% { filter: brightness(1.4); } }
        @keyframes dtflow-spin  { to { transform: rotate(360deg); } }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Distinct colors for inverted-parameter trajectories.
_PARAM_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def _inverted_parameters_plot(
    inversion: pd.DataFrame,
    param_names: Iterable[str],
) -> go.Figure | None:
    """Evolution of the FEM-inverted soil parameters over time.

    One stacked subplot per parameter, sharing the time axis. Parameters span
    very different magnitudes (e.g. porosity ~0.4 vs. cap_s ~1e3), so each gets
    its own y-axis rather than a shared overlay. Values are the absolute inverted
    quantities written by fem_inversion (base x scale); the config's
    ``relative_bounds`` live in scale space and are intentionally not drawn here.

    The x-axis is Influx ingest time (the inversion record carries no
    ``time_hours`` field), which for a replay advances monotonically with the run.
    """
    if inversion is None or inversion.empty or "_time" not in inversion.columns:
        return None

    present = [p for p in param_names if p in inversion.columns]
    if not present:
        return None

    df = inversion.copy()
    df["_time"] = pd.to_datetime(df["_time"], errors="coerce")
    df = df.dropna(subset=["_time"]).sort_values("_time")
    if df.empty:
        return None

    fig = make_subplots(
        rows=len(present),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=present,
    )
    for idx, param in enumerate(present, start=1):
        color = _PARAM_COLORS[(idx - 1) % len(_PARAM_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=df["_time"],
                y=_safe_float_series(df[param]),
                mode="lines+markers",
                line={"color": color, "width": 2},
                marker={"size": 5, "color": color},
                name=param,
                showlegend=False,
                hovertemplate=f"{param}: %{{y:.4g}}<br>%{{x|%H:%M:%S}}<extra></extra>",
            ),
            row=idx,
            col=1,
        )
    fig.update_xaxes(title_text="Time (UTC ingest)", row=len(present), col=1)
    fig.update_layout(
        title="Inverted soil parameters - evolution",
        height=max(280, 210 * len(present)),
        margin=dict(l=70, r=40, t=70, b=60),
        showlegend=False,
    )
    return fig


# -----------------------------
# Main app
# -----------------------------

def main() -> None:
    _apply_theme()
    config, viz_cfg, safety_cfg = _load_config()
    site_id = str(viz_cfg.get("site_id", "default"))
    history_hours = float(viz_cfg.get("history_hours", 2.0))
    max_points = int(viz_cfg.get("max_points", 3000))
    refresh_seconds = int(viz_cfg.get("refresh_seconds", 20))
    r_limit = float(viz_cfg.get("r_limit_m", safety_cfg.get("limit_radius_m", 0.1)))
    release_fraction = float(viz_cfg.get("release_fraction", safety_cfg.get("release_fraction", 0.5)))
    warn_fraction = float(viz_cfg.get("warn_fraction", safety_cfg.get("warn_fraction", 0.6)))
    critical_fraction = float(viz_cfg.get("critical_fraction", safety_cfg.get("critical_fraction", 0.9)))
    safe_pct = float(viz_cfg.get("safe_pct", SAFE_PCT_DEFAULT))
    caution_pct = float(viz_cfg.get("caution_pct", CAUTION_PCT_DEFAULT))
    invert_params = list(config.get("fem_inversion", {}).get("invert_params", ["cap_s", "k_parameter", "lambda_s"]))

    st.title("Permafrost Digital Twin - Live Dashboard")
    st.caption("Safety-first monitoring, FEM forecasting, and control actions.")

    with st.sidebar:
        st.header("Dashboard Controls")
        site_id = st.text_input("Site ID", site_id)
        history_hours = st.slider("History window (hours)", 0.5, 72.0, history_hours, 0.5)
        max_points = st.slider("Max points", 500, 12000, max_points, 500)
        refresh_seconds = st.slider("Refresh every (seconds)", 5, 120, refresh_seconds, 5)
        st.divider()
        r_limit = st.number_input("Limit max r_limit (m)", min_value=0.01, max_value=1.0, value=r_limit, step=0.01)
        r_release = release_fraction * r_limit
        st.caption(f"Release radius: {r_release * 1000.0:.0f} mm ({release_fraction:.1f} x r_lim)")
        safe_pct = st.slider("Gauge warning band (%)", 60, 90, int(safe_pct * 100), 1) / 100.0
        caution_pct = st.slider("Gauge critical band (%, display only)", 85, 99, int(caution_pct * 100), 1) / 100.0
        forecast_horizon_min = st.slider("Forecast horizon (minutes)", 5, 120, 10, 5)
        as_of_hours = st.number_input("View as of data time (h, 0 = live)", min_value=0.0, value=0.0, step=0.25)
        st.divider()
        animate_field = st.checkbox(
            "Animate field plots",
            value=False,
            help=f"Plays a scrubber over the last {_MAX_ANIMATION_FRAMES} sampled snapshots. "
                 "Disable for a lighter page load.",
        )
        if st.button("Refresh now"):
            st.cache_data.clear()
            if hasattr(st, "rerun"):
                st.rerun()
            else:  # pragma: no cover - backward compatibility
                st.experimental_rerun()
        st.caption("Tip: increase refresh interval if Influx is slow.")
        st.divider()
        st.markdown("**Danger zone**")
        if st.button("Wipe all InfluxDB data", type="primary"):
            st.session_state["_confirm_wipe"] = True
        if st.session_state.get("_confirm_wipe"):
            st.warning("Permanently deletes ALL data from the bucket. Cannot be undone.")
            wipe_col1, wipe_col2 = st.columns(2)
            with wipe_col1:
                if st.button("Confirm", type="primary"):
                    influx_cfg = config.get("influxdb", {})
                    try:
                        resp = requests.post(
                            f"{influx_cfg.get('url', 'http://localhost:8086')}/api/v2/delete"
                            f"?org={influx_cfg.get('org', 'permafrost')}&bucket={influx_cfg.get('bucket', 'permafrost_data')}",
                            headers={
                                "Authorization": f"Token {influx_cfg.get('token', 'permafrost')}",
                                "Content-Type": "application/json",
                            },
                            json={"start": "1970-01-01T00:00:00Z", "stop": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
                            timeout=10,
                        )
                        if resp.status_code == 204:
                            st.session_state["_confirm_wipe"] = False
                            st.cache_data.clear()
                            st.success("Data wiped.")
                            st.rerun()
                        else:
                            st.error(f"Delete failed: HTTP {resp.status_code}")
                    except Exception as exc:
                        st.error(f"Error: {exc}")
            with wipe_col2:
                if st.button("Cancel"):
                    st.session_state["_confirm_wipe"] = False
                    st.rerun()

    @st.cache_data(ttl=refresh_seconds)
    def _fetch_data(site_id: str, history_hours: float, max_points: int, as_of_hours: float) -> dict[str, pd.DataFrame]:
        influx = _build_influx()
        range_start = _range_start(history_hours)
        # The front plots draw the latest snapshot. Live needs the newest rows; a
        # historical view filters by data time afterwards, so it needs them all.
        point_limit = 0 if as_of_hours > 0 else max_points
        data = {
            "sensors": influx.query_sensor_temperature_2d(
                site=site_id,
                limit=max_points,
                range_start=range_start,
            ),
            "forecast_temperature": influx.query_fem_temperature_2d(
                site=site_id,
                limit=max_points,
                range_start=range_start,
            ),
            # Uncalibrated twin of the same forecast (fixed material parameters,
            # inversion overrides ignored). Empty unless fem_forecast.run_baseline
            # is enabled.
            "forecast_temperature_baseline": influx.query_fem_temperature_2d(
                site=site_id,
                limit=max_points,
                range_start=range_start,
                measurement=FEM_TEMPERATURE_BASELINE_MEASUREMENT,
            ),
            "thaw_metrics": influx.query_thaw_front_metrics(
                site=site_id,
                limit=max_points,
                range_start=range_start,
            ),
            "thaw_points": influx.query_thaw_front_points(
                site=site_id,
                limit=point_limit,
                range_start=range_start,
                newest=True,
            ),
            "forecast_metrics": influx.query_fem_forecast_metrics(
                site=site_id,
                limit=max_points,
                range_start=range_start,
            ),
            "forecast_points": influx.query_fem_forecast_points(
                site=site_id,
                limit=point_limit,
                range_start=range_start,
                newest=True,
            ),
            "safety_alerts": influx.query_safety_alerts(
                site=site_id,
                limit=max_points,
                range_start=range_start,
            ),
            "heater_actions": influx.query_heater_actions(
                site=site_id,
                limit=max_points,
                range_start=range_start,
            ),
            # Measured inlet/outlet plus the pipe model's simulated outlet, one
            # row per fluid sample. Empty unless the pipe_boundary service runs
            # with publish_outlet_validation enabled.
            "pipe_outlet_validation": influx.query_pipe_outlet_validation(
                site=site_id,
                limit=max_points,
                range_start=range_start,
            ),
            "fluid_temperatures": influx.query_fluid_temperatures(
                site=site_id,
                limit=max_points,
                range_start=range_start,
            ),
            "inversion": influx.query_inversion_parameters(
                site=site_id,
                limit=max_points,
                range_start=range_start,
            ),
        }
        influx.close()
        return data

    data = _fetch_data(site_id, history_hours, max_points, as_of_hours)
    if as_of_hours > 0:
        data = _filter_data_as_of(data, as_of_hours)
        st.info(f"Historical view: logged data up to t = {as_of_hours:.2f} h. Set the sidebar time to 0 for live data.")
    thaw_metrics = data["thaw_metrics"]
    forecast_horizon_hours = float(forecast_horizon_min) / 60.0
    forecast_metrics = _filter_forecast_by_horizon(data["forecast_metrics"], forecast_horizon_hours)
    if forecast_metrics is not None and forecast_metrics.empty:
        forecast_metrics = data["forecast_metrics"]
    safety_alerts = data["safety_alerts"]
    heater_actions = data["heater_actions"]

    current_r_max = _latest_value(thaw_metrics, "radius_max_m")
    current_volume_m3 = _latest_value(thaw_metrics, "thaw_volume_m3")
    current_percent = _percent(current_r_max or 0.0, r_limit) if current_r_max is not None else 0.0
    remaining_margin = None if current_r_max is None else max(r_limit - current_r_max, 0.0)

    time_to_limit = _estimate_time_to_limit(thaw_metrics, r_limit)
    ttl_text, ttl_level = _format_time_to_limit(time_to_limit)

    safety_state = _status_from_percent(current_percent, safe_pct, caution_pct)
    last_action = "n/a"
    last_action_time = "n/a"
    if heater_actions is not None and not heater_actions.empty:
        if "action" in heater_actions.columns:
            last_action = str(heater_actions["action"].iloc[-1]).upper()
        if "time_hours" in heater_actions.columns:
            try:
                last_action_time = f"{float(heater_actions['time_hours'].iloc[-1]):.2f}h"
            except (TypeError, ValueError):
                last_action_time = "n/a"

    if last_action.lower() == "stop":
        safety_state = "STOPPED"

    alert_forecast_minutes = None
    if forecast_metrics is not None and not forecast_metrics.empty and "horizon_hours" in forecast_metrics.columns and "radius_max_m" in forecast_metrics.columns:
        df_forecast = forecast_metrics.copy()
        df_forecast["time_hours"] = _safe_float_series(df_forecast["time_hours"])
        df_forecast["horizon_hours"] = _safe_float_series(df_forecast["horizon_hours"])
        df_forecast["radius_max_m"] = _safe_float_series(df_forecast["radius_max_m"])
        df_forecast = df_forecast.dropna(subset=["time_hours", "horizon_hours", "radius_max_m"])
        if not df_forecast.empty and current_r_max is not None:
            df_forecast["time_forecast"] = df_forecast["time_hours"] + df_forecast["horizon_hours"]
            breach = df_forecast[df_forecast["radius_max_m"] >= r_limit]
            if not breach.empty:
                breach_time = breach["time_forecast"].min()
                if current_r_max is not None and current_r_max < r_limit:
                    now_time = float(thaw_metrics["time_hours"].max()) if thaw_metrics is not None and not thaw_metrics.empty else 0.0
                    alert_forecast_minutes = max((breach_time - now_time) * 60.0, 0.0)

    tab_overview, tab_fields, tab_thaw, tab_params, tab_energy = st.tabs(
        ["Overview & Safety", "Temperature Fields", "Thaw Front", "Model Parameters", "Energy Analysis"]
    )

    with tab_overview:
        gauge_col, spacer_col, flow_col = st.columns((1.5, 0.2, 1))
        with gauge_col:
            _chart(
                _build_safety_gauge(current_r_max, r_limit, safe_pct, caution_pct),
                key="safety_gauge",
                basename="safety_gauge",
                export=False,
            )
            if current_r_max is not None:
                volume_str = f" | volume: {current_volume_m3 * 1e6:.1f} cm³" if current_volume_m3 is not None else ""
                st.caption(
                    f"r_max = {current_r_max * 1000.0:.1f} mm | {current_percent:.0f}% of limit | "
                    f"margin: {remaining_margin * 1000.0:.1f} mm{volume_str}"
                )
            else:
                st.caption("r_max not available yet.")
        with flow_col:
            # The loop LEDs report how fresh the data is now, which a past view can't show.
            if as_of_hours > 0:
                st.caption("Live loop status is hidden in the historical view.")
            else:
                st.markdown(_build_data_flow_html(data, refresh_seconds), unsafe_allow_html=True)

        st.divider()

        bottom_left, bottom_right = st.columns((1, 1))
        with bottom_left:
            fig = _alerts_series(safety_alerts)
            if fig:
                _chart(fig, key="safety_timeline", basename="safety_status_timeline")
            else:
                st.info("No safety alerts yet.")
        with bottom_right:
            if heater_actions is not None and not heater_actions.empty:
                st.subheader("Heater actions")
                display_actions = heater_actions.tail(20).copy()
                keep_cols = [col for col in ["time_hours", "action", "reason", "setpoint", "triggered"] if col in display_actions.columns]
                display_actions = display_actions[keep_cols]
                if "time_hours" in display_actions.columns:
                    display_actions["time_hours"] = display_actions["time_hours"].apply(
                        lambda val: f"{float(val):.2f}" if val is not None else ""
                    )
                if "setpoint" in display_actions.columns:
                    display_actions["setpoint"] = display_actions["setpoint"].apply(
                        lambda val: f"{float(val):.2f}" if val is not None else ""
                    )
                st.dataframe(display_actions, use_container_width=True, hide_index=True)
            else:
                st.info("No heater actions yet.")

    with tab_fields:
        with st.container(border=True):
            st.markdown("**Current time step**")
            step_info = _time_step_info(data["sensors"])
            if step_info is None:
                st.caption("Waiting for the first sensor snapshot...")
            else:
                latest_h, first_h, n_steps = step_info
                step_cols = st.columns(3)
                step_cols[0].metric(
                    "Animation frame" if animate_field else "Snapshot time",
                    f"{latest_h:.2f} h",
                )
                step_cols[1].metric("Snapshots loaded", f"{n_steps}")
                step_cols[2].metric("Time span", f"{first_h:.2f} - {latest_h:.2f} h")
                if animate_field:
                    st.caption(
                        "Animating: the field plots below scrub across all "
                        f"{n_steps} snapshots. Latest snapshot is at t = {latest_h:.2f} h "
                        "(use each plot's slider to pick a step)."
                    )
                else:
                    st.caption(
                        f"Showing the latest snapshot (t = {latest_h:.2f} h). "
                        "Enable 'Animate temperature field' in the sidebar to scrub earlier steps."
                    )

        temp_left, temp_right = st.columns((1, 1))
        with temp_left:
            fig = (
                _temperature_contour_animated(data["sensors"])
                if animate_field
                else _temperature_contour(data["sensors"])
            )
            if fig:
                _chart(fig, key="temp_field_measured", basename="temperature_field_measured")
            else:
                st.info("No sensor snapshots yet.")
        with temp_right:
            forecast_field_fig = (
                _temperature_contour_animated(
                    data["forecast_temperature"],
                    title="Forecasted temperature field — heat propagation",
                    marker_name="Forecast samples",
                )
                if animate_field
                else _temperature_contour(
                    data["forecast_temperature"],
                    title="Forecasted temperature field with thaw front (0 C)",
                    marker_name="Forecast samples",
                )
            )
            if forecast_field_fig:
                _chart(
                    forecast_field_fig,
                    key="temp_field_forecast",
                    basename="temperature_field_forecast",
                )
            else:
                st.info("No forecasted temperature field snapshots yet.")

        st.divider()
        st.subheader("What the inversion service is buying")
        st.markdown(
            "The same forecast run twice off the same mesh, boundary conditions and initial "
            "field — the **only** difference is where the soil parameters come from. The plot "
            "below sits directly under the calibrated one above so the two fields line up "
            "column-wise:\n\n"
            "- **Above right:** parameters calibrated live by the FEM inversion service.\n"
            "- **Below right:** the configured fixed parameters, inversion output ignored."
        )

        baseline_temperature = data.get("forecast_temperature_baseline")
        has_baseline = isinstance(baseline_temperature, pd.DataFrame) and not baseline_temperature.empty
        base_left, base_right = st.columns((1, 1))
        with base_left:
            if not has_baseline:
                st.info(
                    "No baseline field yet. Set `fem_forecast.run_baseline` to true in "
                    "startup.conf and restart the FEM forecast service — it then marches a "
                    "second solver that never receives the inversion's material_override."
                )
            else:
                calibrated_err = _fem_variant_error_summary(data["sensors"], data["forecast_temperature"])
                baseline_err = _fem_variant_error_summary(data["sensors"], baseline_temperature)
                if calibrated_err is None or baseline_err is None:
                    st.info(
                        "Both fields are streaming, but they have not paired with the measured "
                        "data over this time range yet."
                    )
                else:
                    gain_c = baseline_err["rmse_c"] - calibrated_err["rmse_c"]
                    st.metric(
                        "RMSE vs measured — with inversion",
                        f"{calibrated_err['rmse_c']:.2f} °C",
                        delta=f"{-gain_c:+.2f} °C vs fixed parameters",
                        delta_color="inverse",
                    )
                    st.metric(
                        "RMSE vs measured — fixed parameters",
                        f"{baseline_err['rmse_c']:.2f} °C",
                    )
                    st.caption(
                        f"Mean abs. error {calibrated_err['mae_c']:.2f} °C vs "
                        f"{baseline_err['mae_c']:.2f} °C, bias {calibrated_err['bias_c']:+.2f} °C vs "
                        f"{baseline_err['bias_c']:+.2f} °C, over "
                        f"{int(calibrated_err['n'])} and {int(baseline_err['n'])} paired points. "
                        "A negative delta means the inversion is improving the forecast; positive "
                        "means the calibrated parameters are currently doing worse than the fixed ones."
                    )
        with base_right:
            if has_baseline:
                baseline_field_fig = (
                    _temperature_contour_animated(
                        baseline_temperature,
                        title="Forecast without inversion — fixed parameters",
                        marker_name="Baseline samples",
                    )
                    if animate_field
                    else _temperature_contour(
                        baseline_temperature,
                        title="Forecast without inversion (fixed parameters), thaw front (0 C)",
                        marker_name="Baseline samples",
                    )
                )
                if baseline_field_fig:
                    _chart(
                        baseline_field_fig,
                        key="temp_field_forecast_baseline",
                        basename="temperature_field_forecast_baseline",
                    )
                else:
                    st.info("No baseline temperature field snapshots yet.")

        st.divider()
        st.subheader("FEM model error vs measured data")
        st.markdown(
            "**Is the simulation matching reality?** Each FEM node value is compared against the "
            "thermocouple reading at the same position and time.\n\n"
            "This first pair reproduces the reference MATLAB post-processing exactly "
            "(`main.m` → `Simulation_results.plt`), so the numbers are directly comparable to the "
            "Tecplot output:"
        )
        st.latex(r"\text{error} = \frac{|T_{FEM} - T_{measured}|}{T_{FEM}} \times 100 \quad [T \text{ in Kelvin}]")
        st.caption(
            "⚠️ Dividing by absolute temperature compresses everything into roughly 0–2%: a 1 °C "
            "miss near freezing reads as just 0.37%. Use these two plots for continuity with the "
            "MATLAB results, and the °C plots below to actually judge the model. Error in °C is "
            "shown in every hover and tile here too."
        )

        error_frame = _fem_error_frame(data["sensors"], data["forecast_temperature"])
        if error_frame is None:
            st.info(
                "No paired FEM/measured snapshots yet. Needs the fem_forecast service running "
                "with field snapshots enabled, over the same time range as the sensor data."
            )
        else:
            latest_time = float(error_frame["time_hours"].max())
            latest_rows = error_frame[error_frame["time_hours"] == latest_time]
            worst = latest_rows.loc[latest_rows["error_pct"].idxmax()]
            err_col1, err_col2, err_col3 = st.columns(3)
            err_col1.metric(
                "Mean error",
                f"{latest_rows['error_pct'].mean():.3f}%",
                delta=f"{latest_rows['abs_error_c'].mean():.2f} °C",
                delta_color="off",
                help="Averaged over every paired sensor at the latest timestep. "
                     "The °C figure below it is the same error unnormalised.",
            )
            err_col2.metric(
                "Worst sensor",
                f"{worst['error_pct']:.3f}%",
                delta=f"{worst['abs_error_c']:.2f} °C at {worst['sensor_id']}",
                delta_color="off",
                help="Largest single-sensor disagreement at this timestep, and where it is. "
                     "A mean that looks fine can still hide a bad front location.",
            )
            err_col3.metric(
                "Sensors paired",
                f"{len(latest_rows)}",
                delta=f"t = {latest_time:.2f} h",
                delta_color="off",
                help="How many sensors had both a measurement and an FEM value at this "
                     "timestep. A drop here means the two streams stopped lining up.",
            )

            err_left, err_right = st.columns(2)
            with err_left:
                fig = (
                    _fem_error_field_animated(error_frame)
                    if animate_field
                    else _fem_error_field_plot(error_frame)
                )
                if fig:
                    _chart(fig, key="fem_error_field", basename="fem_error_field")
                else:
                    st.info("Not enough paired points to interpolate an error field.")
            with err_right:
                fig = _fem_error_timeseries_plot(error_frame)
                if fig:
                    _chart(fig, key="fem_error_series", basename="fem_error_over_time")
                else:
                    st.info("No error history yet.")

            with st.expander("Per-sensor error table", expanded=False):
                table = latest_rows[
                    ["sensor_id", "x_m", "z_m", "temperature_obs", "temperature_fem", "abs_error_c", "error_pct"]
                ].sort_values("error_pct", ascending=False)
                st.dataframe(table, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download FEM error CSV",
                    error_frame.to_csv(index=False).encode("utf-8"),
                    file_name="fem_error.csv",
                    mime="text/csv",
                )

            st.divider()
            st.subheader("Improved error view — °C and signed")
            st.markdown(
                "**Same comparison, a metric you can act on.** Identical pairing to the plots above, "
                "with three changes:\n\n"
                "- **In °C, not normalised** — 0.4 °C means 0.4 °C, instead of a percentage of "
                "absolute temperature that is always near zero.\n"
                "- **Signed** — positive means the FEM runs *warmer* than the thermocouples, "
                "negative *colder*. Taking the absolute value, as the reference does, throws away "
                "the single most diagnostic fact about a drifting solve.\n"
                "- **RMSE against bias** — if they track each other the whole field is shifted by a "
                "constant (suspect a boundary condition or material property); if RMSE is much "
                "larger the model is scattered about roughly the right mean (suspect resolution or "
                "noise)."
            )
            st.caption(
                "Past the last sensor column the field is held constant in radius, the same "
                "convention as the temperature contours — the marked rule on the map is where "
                "measurement stops and extrapolation begins."
            )

            report = _fem_pairing_report(data["sensors"], data["forecast_temperature"], error_frame)
            if report is not None and report["misaligned"]:
                st.warning(
                    f"**Streams look misaligned by {report['best_lag']:+d} step** "
                    f"({report['step_hours'] * 60.0:.1f} min): re-pairing at that lag drops RMSE from "
                    f"{report['rmse_at_zero_lag']:.2f} °C to {report['rmse_at_best_lag']:.2f} °C. "
                    "The FEM field is timestamped `t + step_hours` on the "
                    "`_run_forecast_with_field` path (triggered by `include_field`, "
                    "`save_field_csv`, or a sticky `material_override`) and `t` otherwise — see "
                    "`fem_forecast_server._process_message`. Until that is reconciled the errors "
                    "below compare offset instants and read worse than the model really is."
                )
            elif report is not None and report["coverage"] < 0.5:
                st.warning(
                    f"Only {report['paired_steps']} of {report['fem_steps']} FEM snapshots paired "
                    f"with a measurement ({report['coverage']:.0%}). Errors below are computed only "
                    "from the rows that did pair."
                )

            imp_left, imp_right = st.columns(2)
            with imp_left:
                fig = (
                    _fem_error_field_animated(
                        error_frame,
                        title="Which way the model is wrong — signed bias",
                        hint="red = FEM too warm · blue = FEM too cold · pale = agreement",
                        value_col="signed_error_c",
                        signed=True,
                        colorbar_title="FEM − measured (°C)",
                    )
                    if animate_field
                    else _fem_bias_field_plot(error_frame)
                )
                if fig:
                    _chart(fig, key="fem_bias_field", basename="fem_bias_field")
                else:
                    st.info("Not enough paired points to interpolate a bias field.")
            with imp_right:
                fig = _fem_error_metrics_plot(error_frame)
                if fig:
                    _chart(fig, key="fem_error_metrics", basename="fem_error_metrics")
                else:
                    st.info("No error history yet.")

            rmse_now = float(np.sqrt(np.mean(np.square(latest_rows["signed_error_c"]))))
            bias_now = float(latest_rows["signed_error_c"].mean())
            imp_col1, imp_col2, imp_col3 = st.columns(3)
            systematic_pct = (abs(bias_now) / rmse_now * 100.0) if rmse_now > 1e-9 else 0.0
            imp_col1.metric(
                "RMSE",
                f"{rmse_now:.2f} °C",
                help="Typical size of the disagreement across all sensors at this timestep, "
                     "in degrees. Penalises large individual misses more than a plain average.",
            )
            imp_col2.metric(
                "Bias (FEM − measured)",
                f"{bias_now:+.2f} °C",
                delta="model runs warm" if bias_now > 0 else "model runs cold",
                delta_color="off",
                help="Average signed error. Positive means the simulation is hotter than the "
                     "thermocouples everywhere on average — the signature of too much heat "
                     "entering, or too little latent heat absorbed at the front.",
            )
            imp_col3.metric(
                "Systematic share",
                f"{systematic_pct:.0f}%",
                delta="offset" if systematic_pct > 60 else "mostly scatter",
                delta_color="off",
                help="|bias| ÷ RMSE. Near 100% the whole field is shifted by a near-constant "
                     "amount, which points at a boundary condition or a material property. "
                     "Near 0% the model is right on average but noisy in space.",
            )

        st.divider()

        fig = _front_geometry_plot(data["thaw_points"], data["forecast_points"], r_limit)
        if fig:
            _chart(fig, key="front_geometry", basename="thaw_front_geometry")
        else:
            st.info("No thaw-front points yet.")

        with st.expander("3D thaw volume", expanded=False):
            fig = _thaw_volume_3d_plot(data["thaw_points"])
            if fig:
                _chart(fig, key="thaw_3d", basename="thaw_volume_3d")
            else:
                st.info("No thaw-front points yet.")

    with tab_thaw:
        fig = _unified_radius_plot(
            thaw_metrics,
            forecast_metrics,
            heater_actions,
            r_limit,
            r_release,
            warn_fraction,
            critical_fraction,
        )
        if fig:
            _chart(fig, key="radius_evolution", basename="thaw_radius_evolution")
        else:
            st.info("No thaw-front metrics yet.")

        fig = _volume_over_time_plot(thaw_metrics)
        if fig:
            _chart(fig, key="volume_over_time", basename="thaw_volume_over_time")
        else:
            st.info("No thaw volume data yet.")

        diagnostics = _model_diagnostics_plot(thaw_metrics, forecast_metrics)
        if diagnostics:
            _chart(diagnostics, key="model_diagnostics", basename="forecast_error")
        else:
            st.info("No model validation metrics yet.")

        st.divider()

        export_col1, export_col2 = st.columns(2)
        with export_col1:
            if thaw_metrics is not None and not thaw_metrics.empty:
                st.download_button(
                    "Download measured metrics CSV",
                    thaw_metrics.to_csv(index=False).encode("utf-8"),
                    file_name="thaw_metrics.csv",
                    mime="text/csv",
                )
        with export_col2:
            if forecast_metrics is not None and not forecast_metrics.empty:
                st.download_button(
                    "Download forecast metrics CSV",
                    forecast_metrics.to_csv(index=False).encode("utf-8"),
                    file_name="forecast_metrics.csv",
                    mime="text/csv",
                )

    with tab_params:
        st.subheader("Inverted soil parameters")
        st.caption(
            "Online FEM inversion (fem_inversion) calibrating soil parameters from the "
            "observed temperature field. Requires the fem_inversion service to be running."
        )
        fig = _inverted_parameters_plot(data.get("inversion"), invert_params)
        if fig:
            _chart(fig, key="inverted_params", basename="inverted_parameters")
        else:
            st.info("No inverted parameters yet. Ensure the fem_inversion service is running.")

        inversion_df = data.get("inversion")
        if inversion_df is not None and not inversion_df.empty:
            st.download_button(
                "Download inverted parameters CSV",
                inversion_df.to_csv(index=False).encode("utf-8"),
                file_name="inverted_parameters.csv",
                mime="text/csv",
            )

        st.divider()
        st.subheader("Pipe model validation")
        st.caption(
            "The pipe_boundary service calibrates the U-shaped heat-exchanger model against the "
            "measured inlet, then feeds its interface temperature to the FEM as a Robin boundary. "
            "This is the check on that model: driven by the measured inlet, does it reproduce the "
            "measured outlet? Any gap here propagates straight into the FEM boundary condition."
        )

        pipe_validation = data.get("pipe_outlet_validation")
        pipe_fig = _pipe_outlet_validation_plot(pipe_validation)
        if pipe_fig is None:
            st.info(
                "No pipe outlet validation data yet. Needs the pipe_boundary service running with "
                "`publish_outlet_validation` enabled in startup.conf, and a fluid inlet/outlet "
                "stream on the fluid exchange."
            )
        else:
            _chart(pipe_fig, key="pipe_outlet_validation", basename="pipe_outlet_validation")

            pipe_err = _pipe_outlet_error_summary(pipe_validation)
            if pipe_err is not None:
                err_cols = st.columns(4)
                err_cols[0].metric("Outlet RMSE", f"{pipe_err['rmse_c']:.2f} °C")
                err_cols[1].metric("Outlet MAE", f"{pipe_err['mae_c']:.2f} °C")
                err_cols[2].metric("Bias (sim − meas)", f"{pipe_err['bias_c']:+.2f} °C")
                err_cols[3].metric("Worst |error|", f"{pipe_err['max_abs_c']:.2f} °C")
                st.caption(
                    f"Over {int(pipe_err['n'])} paired samples. A positive bias means the model "
                    "runs the loop hotter than the rig does, so the FEM sees a pipe wall warmer "
                    "than reality."
                )

            st.download_button(
                "Download pipe outlet validation CSV",
                pipe_validation.to_csv(index=False).encode("utf-8"),
                file_name="pipe_outlet_validation.csv",
                mime="text/csv",
            )

    with tab_energy:
        st.subheader("Energy & Efficiency Metrics")
        st.caption(
            "Requires fluid inlet/outlet temperatures published by the CR1000 publisher. "
            "Configure `fluid_fields` in `startup.conf` under `cr1000_publisher`."
        )
        energy = _compute_energy_metrics(data.get("fluid_temperatures"), thaw_metrics)
        if energy is None:
            st.info(
                "No fluid temperature data in InfluxDB yet. "
                "Make sure `fluid_fields` is configured in `cr1000_publisher` and the CR1000 publisher is running."
            )
        else:
            ein_col, eta_col = st.columns(2)
            with ein_col:
                st.metric("Max E_in", f"{energy['Ein'].max() / 1e6:.3f} MJ")
            with eta_col:
                eta_latest = float(energy["eta"][np.isfinite(energy["eta"])][-1]) if np.any(np.isfinite(energy["eta"])) else None
                st.metric("Latest η_melt", f"{eta_latest:.1f} %" if eta_latest is not None else "n/a")

            row1_left, row1_right = st.columns(2)
            with row1_left:
                fig = _energy_ein_vs_vthaw(energy)
                if fig:
                    _chart(fig, key="energy_ein_vthaw", basename="energy_ein_vs_vthaw")
                else:
                    st.info("Thaw volume not available (thaw front metrics needed).")
            with row1_right:
                _chart(_energy_specific(energy), key="energy_specific", basename="specific_energy")

            row2_left, row2_right = st.columns(2)
            with row2_left:
                fig = _energy_vthaw_time(energy)
                if fig:
                    _chart(fig, key="energy_vthaw_time", basename="thaw_volume_time")
                else:
                    st.info("Thaw volume not available.")
            with row2_right:
                _chart(_energy_eta(energy), key="energy_eta", basename="melting_efficiency")

            if data.get("fluid_temperatures") is not None and not data["fluid_temperatures"].empty:
                st.download_button(
                    "Download fluid temperatures CSV",
                    data["fluid_temperatures"].to_csv(index=False).encode("utf-8"),
                    file_name="fluid_temperatures.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()

"""Streamlit dashboard for the permafrost digital twin."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.interpolate import griddata

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from software.digital_twin.data_access.influx_utils import InfluxHelper
from software.startup.service_runners import build_influx_config
from software.startup.utils.config import load_startup_config


DOMAIN_RADIUS_M = 0.24
DOMAIN_DEPTH_M = 0.30
THERMOSYPHON_RADIUS_M = 0.0075

SAFE_PCT_DEFAULT = 0.80
CAUTION_PCT_DEFAULT = 0.95


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


# -----------------------------
# Visualization builders
# -----------------------------

def _build_safety_gauge(r_max: float | None, limit_m: float, safe_pct: float, caution_pct: float) -> go.Figure:
    value = 0.0 if r_max is None else float(r_max)
    percent = _percent(value, limit_m)
    if percent < safe_pct * 100:
        bar_color = "#16a34a"
    elif percent < caution_pct * 100:
        bar_color = "#f59e0b"
    else:
        bar_color = "#dc2626"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": " m", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, max(limit_m * 1.1, 0.01)]},
                "bar": {"color": bar_color},
                "bgcolor": "#0f172a",
                "borderwidth": 1,
                "bordercolor": "#1f2937",
                "steps": [
                    {"range": [0, limit_m * safe_pct], "color": "#bbf7d0"},
                    {"range": [limit_m * safe_pct, limit_m * caution_pct], "color": "#fde68a"},
                    {"range": [limit_m * caution_pct, limit_m], "color": "#fecaca"},
                ],
                "threshold": {"line": {"color": "#dc2626", "width": 4}, "value": limit_m},
            },
            title={"text": "Safety gauge", "font": {"size": 16}},
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10))
    return fig


def _interpolate_temperature_field(sensors: pd.DataFrame, grid_resolution: int = 120) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
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

    return R, Z, T


def _temperature_contour(sensors: pd.DataFrame, temp_range: tuple[float, float] = (-2.0, 30.0)) -> go.Figure | None:
    latest = _latest_snapshot(sensors)
    field = _interpolate_temperature_field(latest)
    if field is None:
        return None

    R, Z, T = field
    fig = go.Figure()

    fig.add_trace(
        go.Contour(
            x=R[0, :],
            y=Z[:, 0],
            z=T,
            colorscale="RdBu_r",
            zmin=temp_range[0],
            zmax=temp_range[1],
            zmid=0.0,
            colorbar={"title": "Temperature (C)"},
            contours={"showlines": False},
        )
    )

    fig.add_trace(
        go.Contour(
            x=R[0, :],
            y=Z[:, 0],
            z=T,
            showscale=False,
            contours={"start": 0.0, "end": 0.0, "coloring": "lines"},
            line={"color": "#111827", "width": 3},
            name="Thaw front (0 C)",
        )
    )

    if not latest.empty:
        latest = latest.copy()
        latest["x_m"] = latest["x_m"].astype(float).abs()
        fig.add_trace(
            go.Scatter(
                x=latest["x_m"],
                y=latest["z_m"],
                mode="markers",
                marker={
                    "size": 8,
                    "color": latest["temperature"],
                    "colorscale": "RdBu_r",
                    "cmin": temp_range[0],
                    "cmax": temp_range[1],
                    "line": {"width": 0.5, "color": "#0f172a"},
                },
                name="Sensors",
                hovertemplate="Sensor %{text}<br>Temp: %{marker.color:.2f} C<extra></extra>",
                text=latest.get("sensor_id", ""),
            )
        )

    fig.add_shape(
        type="rect",
        x0=0.0,
        x1=THERMOSYPHON_RADIUS_M,
        y0=0.0,
        y1=DOMAIN_DEPTH_M,
        fillcolor="#111827",
        opacity=0.6,
        line_width=0,
        layer="below",
    )

    fig.update_layout(
        title="Temperature field with thaw front (0 C)",
        xaxis_title="Radial distance (m)",
        yaxis_title="Depth (m)",
        height=440,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, DOMAIN_RADIUS_M])
    fig.update_yaxes(range=[DOMAIN_DEPTH_M, 0.0])
    return fig


def _front_geometry_plot(thaw_points: pd.DataFrame, forecast_points: pd.DataFrame, limit_m: float) -> go.Figure | None:
    if (thaw_points is None or thaw_points.empty) and (forecast_points is None or forecast_points.empty):
        return None

    fig = go.Figure()
    if thaw_points is not None and not thaw_points.empty:
        latest_time = thaw_points["time_hours"].max()
        df = thaw_points[thaw_points["time_hours"] == latest_time].copy()
        if "x_m" in df.columns:
            df["x_m"] = df["x_m"].astype(float).abs()
        fig.add_trace(
            go.Scatter(
                x=df["x_m"],
                y=df["z_m"],
                mode="lines+markers",
                line={"color": "#2563eb", "width": 2},
                name=f"Measured (t={latest_time:.2f}h)",
            )
        )
    if forecast_points is not None and not forecast_points.empty:
        latest_time = forecast_points["time_hours"].max()
        df = forecast_points[forecast_points["time_hours"] == latest_time].copy()
        if "x_m" in df.columns:
            df["x_m"] = df["x_m"].astype(float).abs()
        fig.add_trace(
            go.Scatter(
                x=df["x_m"],
                y=df["z_m"],
                mode="lines+markers",
                line={"color": "#dc2626", "width": 2, "dash": "dash"},
                name=f"Forecast (t={latest_time:.2f}h)",
            )
        )

    fig.add_vline(
        x=limit_m,
        line_width=3,
        line_color="#dc2626",
        annotation_text="Safety limit",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Thaw-front evolution and forecast",
        xaxis_title="Radial distance (m)",
        yaxis_title="Depth (m)",
        height=440,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, DOMAIN_RADIUS_M])
    fig.update_yaxes(range=[DOMAIN_DEPTH_M, 0.0])
    return fig


def _unified_radius_plot(
    thaw_metrics: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    heater_actions: pd.DataFrame,
    limit_m: float,
) -> go.Figure | None:
    if thaw_metrics is None or thaw_metrics.empty:
        return None

    df_measured = thaw_metrics.copy()
    df_measured["time_hours"] = _safe_float_series(df_measured["time_hours"])
    df_measured["radius_max_m"] = _safe_float_series(df_measured["radius_max_m"])
    df_measured["radius_avg_m"] = _safe_float_series(df_measured["radius_avg_m"])
    df_measured = df_measured.dropna(subset=["time_hours"])

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df_measured["time_hours"],
            y=df_measured["radius_max_m"],
            mode="lines+markers",
            line={"color": "#2563eb", "width": 2},
            name="Measured r_max",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df_measured["time_hours"],
            y=df_measured["radius_avg_m"],
            mode="lines",
            line={"color": "#22d3ee", "width": 1.5},
            name="Measured r_avg",
        ),
        secondary_y=False,
    )

    df_forecast = pd.DataFrame()
    if forecast_metrics is not None and not forecast_metrics.empty and "horizon_hours" in forecast_metrics.columns:
        df_forecast = forecast_metrics.copy()
        df_forecast["time_hours"] = _safe_float_series(df_forecast["time_hours"])
        df_forecast["horizon_hours"] = _safe_float_series(df_forecast["horizon_hours"])
        df_forecast["radius_max_m"] = _safe_float_series(df_forecast["radius_max_m"])
        df_forecast["radius_avg_m"] = _safe_float_series(df_forecast["radius_avg_m"])
        df_forecast = df_forecast.dropna(subset=["time_hours", "horizon_hours"])
        df_forecast["time_forecast"] = df_forecast["time_hours"] + df_forecast["horizon_hours"]

        fig.add_trace(
            go.Scatter(
                x=df_forecast["time_forecast"],
                y=df_forecast["radius_max_m"],
                mode="lines",
                line={"color": "#dc2626", "width": 2, "dash": "dash"},
                name="Forecast r_max",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=df_forecast["time_forecast"],
                y=df_forecast["radius_avg_m"],
                mode="lines",
                line={"color": "#f97316", "width": 1.5, "dash": "dash"},
                name="Forecast r_avg",
            ),
            secondary_y=False,
        )

    fig.add_hline(
        y=limit_m,
        line_width=3,
        line_color="#dc2626",
        annotation_text="SAFETY LIMIT",
        annotation_position="bottom right",
    )

    measured_max = df_measured["radius_max_m"].max(skipna=True)
    if not np.isfinite(measured_max):
        measured_max = limit_m
    y_max = max(limit_m * 1.2, float(measured_max) * 1.2)
    fig.add_hrect(
        y0=limit_m,
        y1=y_max,
        fillcolor="rgba(220,38,38,0.12)",
        line_width=0,
    )

    if heater_actions is not None and not heater_actions.empty and "time_hours" in heater_actions.columns:
        for _, row in heater_actions.iterrows():
            try:
                action_time = float(row["time_hours"])
            except (TypeError, ValueError):
                continue
            action = str(row.get("action", "")).lower()
            if action == "stop":
                fig.add_vline(x=action_time, line_color="#ef4444", line_width=2, line_dash="dot")
            elif action == "hold":
                fig.add_vline(x=action_time, line_color="#38bdf8", line_width=1, line_dash="dot")
            elif action:
                fig.add_vline(x=action_time, line_color="#a855f7", line_width=1)

    if df_measured["radius_max_m"].notna().sum() > 2:
        dr_dt = np.gradient(df_measured["radius_max_m"], df_measured["time_hours"])
        fig.add_trace(
            go.Scatter(
                x=df_measured["time_hours"],
                y=dr_dt,
                mode="lines",
                line={"color": "#7c3aed", "width": 1},
                name="dr/dt",
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title="Thaw radius evolution: measured + forecast",
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Time (hours)")
    fig.update_yaxes(title_text="Radius (m)", secondary_y=False)
    fig.update_yaxes(title_text="dr/dt (m/hr)", secondary_y=True)
    return fig


def _model_diagnostics_plot(thaw_metrics: pd.DataFrame, forecast_metrics: pd.DataFrame) -> go.Figure | None:
    if thaw_metrics is None or thaw_metrics.empty or forecast_metrics is None or forecast_metrics.empty:
        return None

    measured = thaw_metrics.copy()
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

    merged["radius_error"] = (merged["radius_max_m_measured"] - merged["radius_max_m_forecast"]).abs()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=merged["time_hours"],
            y=merged["radius_error"],
            mode="lines+markers",
            line={"color": "#f97316", "width": 2},
            name="Radius error",
        )
    )
    fig.add_hline(y=0.01, line_color="#facc15", line_dash="dash", annotation_text="1 cm threshold")
    fig.update_layout(
        title="Model quality: radius error",
        height=300,
        margin=dict(l=40, r=20, t=40, b=30),
        xaxis_title="Time (hours)",
        yaxis_title="|r_max(measured) - r_max(forecast)| (m)",
    )
    return fig


def _alerts_series(df: pd.DataFrame) -> go.Figure | None:
    if df is None or df.empty or "time_hours" not in df.columns:
        return None
    fig = px.scatter(
        df,
        x="time_hours",
        y="triggered",
        color="status" if "status" in df.columns else None,
        title="Safety alert timeline",
    )
    fig.update_layout(height=280, xaxis_title="time (hours)", yaxis_title="triggered")
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
        </style>
        """,
        unsafe_allow_html=True,
    )


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
    safe_pct = float(viz_cfg.get("safe_pct", SAFE_PCT_DEFAULT))
    caution_pct = float(viz_cfg.get("caution_pct", CAUTION_PCT_DEFAULT))

    st.title("Permafrost Digital Twin - Live Dashboard")
    st.caption("Safety-first monitoring, FEM forecasting, and control actions.")

    with st.sidebar:
        st.header("Dashboard Controls")
        site_id = st.text_input("Site ID", site_id)
        history_hours = st.slider("History window (hours)", 0.5, 12.0, history_hours, 0.5)
        max_points = st.slider("Max points", 500, 12000, max_points, 500)
        refresh_seconds = st.slider("Refresh every (seconds)", 5, 120, refresh_seconds, 5)
        st.divider()
        r_limit = st.number_input("Safety limit r_limit (m)", min_value=0.01, max_value=1.0, value=r_limit, step=0.01)
        safe_pct = st.slider("Safe threshold (%)", 60, 90, int(safe_pct * 100), 1) / 100.0
        caution_pct = st.slider("Alert threshold (%)", 85, 99, int(caution_pct * 100), 1) / 100.0
        forecast_horizon_min = st.slider("Forecast horizon (minutes)", 5, 120, 30, 5)
        if st.button("Refresh now"):
            st.cache_data.clear()
            st.experimental_rerun()
        st.caption("Tip: increase refresh interval if Influx is slow.")

    @st.cache_data(ttl=refresh_seconds)
    def _fetch_data(site_id: str, history_hours: float, max_points: int) -> dict[str, pd.DataFrame]:
        influx = _build_influx()
        range_start = _range_start(history_hours)
        data = {
            "sensors": influx.query_sensor_temperature_2d(
                site=site_id,
                limit=max_points,
                range_start=range_start,
            ),
            "thaw_metrics": influx.query_thaw_front_metrics(
                site=site_id,
                limit=max_points,
                range_start=range_start,
            ),
            "thaw_points": influx.query_thaw_front_points(
                site=site_id,
                limit=max_points,
                range_start=range_start,
            ),
            "forecast_metrics": influx.query_fem_forecast_metrics(
                site=site_id,
                limit=max_points,
                range_start=range_start,
            ),
            "forecast_points": influx.query_fem_forecast_points(
                site=site_id,
                limit=max_points,
                range_start=range_start,
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
        }
        influx.close()
        return data

    data = _fetch_data(site_id, history_hours, max_points)
    thaw_metrics = data["thaw_metrics"]
    forecast_horizon_hours = float(forecast_horizon_min) / 60.0
    forecast_metrics = _filter_forecast_by_horizon(data["forecast_metrics"], forecast_horizon_hours)
    if forecast_metrics is not None and forecast_metrics.empty:
        forecast_metrics = data["forecast_metrics"]
    safety_alerts = data["safety_alerts"]
    heater_actions = data["heater_actions"]

    current_r_max = _latest_value(thaw_metrics, "radius_max_m")
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
    if forecast_metrics is not None and not forecast_metrics.empty and "horizon_hours" in forecast_metrics.columns:
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

    if alert_forecast_minutes is not None:
        st.markdown(
            f"<div class='banner banner-warning'><strong>FORECAST:</strong> Limit breach predicted in "
            f"{alert_forecast_minutes:.0f} minutes. Recommended action: stop or reduce heating.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='banner banner-info'><strong>Forecast:</strong> No breach predicted within the current horizon.</div>",
            unsafe_allow_html=True,
        )

    top_left, top_right = st.columns((1.2, 1))
    with top_left:
        st.plotly_chart(_build_safety_gauge(current_r_max, r_limit, safe_pct, caution_pct), use_container_width=True)
        if current_r_max is not None:
            st.caption(
                f"r_max = {current_r_max:.3f} m | {current_percent:.0f}% of limit | "
                f"margin: {remaining_margin:.3f} m"
            )
        else:
            st.caption("r_max not available yet.")

    with top_right:
        if safety_state == "SAFE":
            badge_class = "status-safe"
        elif safety_state == "CAUTION":
            badge_class = "status-caution"
        elif safety_state == "ALERT":
            badge_class = "status-alert"
        else:
            badge_class = "status-stopped"
        st.markdown(f"<div class='status-badge {badge_class}'>SYSTEM STATUS: {safety_state}</div>", unsafe_allow_html=True)
        st.markdown(f"**Heater action:** {last_action} at {last_action_time}")
        if ttl_level == "danger":
            st.error(ttl_text)
        elif ttl_level == "caution":
            st.warning(ttl_text)
        else:
            st.success(ttl_text)
        st.caption(f"Last updated: {datetime.utcnow().strftime('%H:%M:%S UTC')}")

    st.divider()

    mid_left, mid_right = st.columns((1.15, 1))
    with mid_left:
        fig = _temperature_contour(data["sensors"])
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sensor snapshots yet.")
    with mid_right:
        fig = _front_geometry_plot(data["thaw_points"], data["forecast_points"], r_limit)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No thaw-front points yet.")

    st.divider()

    fig = _unified_radius_plot(thaw_metrics, forecast_metrics, heater_actions, r_limit)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No thaw-front metrics yet.")

    diagnostics = _model_diagnostics_plot(thaw_metrics, forecast_metrics)
    if diagnostics:
        st.plotly_chart(diagnostics, use_container_width=True)
    else:
        st.info("No model validation metrics yet.")

    bottom_left, bottom_right = st.columns((1, 1))
    with bottom_left:
        fig = _alerts_series(safety_alerts)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No safety alerts yet.")
    with bottom_right:
        if heater_actions is not None and not heater_actions.empty:
            st.subheader("Heater actions")
            st.dataframe(heater_actions.tail(20), use_container_width=True)
        else:
            st.info("No heater actions yet.")

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


if __name__ == "__main__":
    main()

"""Visualization helpers and data bridge for the permafrost dashboard."""

from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from software.digital_twin.data_access.influx_utils import InfluxConfig, InfluxHelper
from software.digital_twin.simulator.fdm.fdm_server import PhysicsParams
from software.digital_twin.visualization.viz_gateway.viz_gateway_server import VizGatewayServer
from software.utils.logging_setup import get_logger

PlotBuilder = Callable[[Dict[str, Any]], Tuple[go.Figure, str]]


# -------------------------------- #
# Plot helpers 
# -------------------------------- #

def plot_temperature_contour(time: np.ndarray, depth: np.ndarray, T: np.ndarray) -> go.Figure:
    """Plot temperature distribution over time and depth."""
    fig = go.Figure(
        data=[
            go.Contour(
                x=time,
                y=depth,
                z=T,
                colorscale="jet",
                reversescale=False,
                contours=dict(coloring="heatmap"),
                colorbar=dict(title="Temperature (°C)"),
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        title="Temperature distribution over time and depth",
        xaxis_title="Time (days)",
        yaxis_title="Depth (m)",
    )
    return fig


def plot_temperature_heatmap(time: np.ndarray, depth: np.ndarray, T: np.ndarray) -> go.Figure:
    """Plot temperature distribution when contour is underdetermined."""
    fig = go.Figure(
        data=[
            go.Heatmap(
                x=time,
                y=depth,
                z=T,
                colorscale="RdBu",
                reversescale=True,
                colorbar=dict(title="Temperature (°C)"),
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        title="Temperature distribution over time and depth",
        xaxis_title="Time (days)",
        yaxis_title="Depth (m)",
    )
    return fig


def plot_temperature_profiles(
    x: np.ndarray,
    time: np.ndarray,
    T: np.ndarray,
    *,
    sample_indices: Iterable[int],
) -> go.Figure:
    """Plot temperature time series at selected depths."""
    fig = go.Figure()
    for idx in sample_indices:
        if idx >= len(x):
            continue
        fig.add_trace(
            go.Scatter(
                x=time,
                y=T[:, idx],
                mode="lines",
                name=f"z={x[idx]:.1f}m",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        title="Temperature vs time at selected depths",
        xaxis_title="Time (days)",
        yaxis_title="Temperature (°C)",
    )
    return fig


def plot_boundary_heat_flux(
    time: np.ndarray,
    q_surface: np.ndarray,
    q_bottom: np.ndarray,
) -> go.Figure:
    """Plot boundary heat flux evolution."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time,
            y=q_surface,
            mode="lines",
            name="Surface flux",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=q_bottom,
            mode="lines",
            name="Bottom flux",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Boundary heat flux evolution",
        xaxis_title="Time (days)",
        yaxis_title="Heat flux (W/m²)",
    )
    return fig


def plot_pinn_vs_fdm(
    x: np.ndarray,
    temperature_pinn: np.ndarray,
    temperature_fdm: np.ndarray,
    *,
    time_label: float,
) -> go.Figure:
    """Plot PINN and FDM temperature comparison at a single time slice."""
    if x.size and (temperature_pinn.size == x.size) and (temperature_fdm.size == x.size):
        order = np.argsort(x)
        x = x[order]
        temperature_pinn = temperature_pinn[order]
        temperature_fdm = temperature_fdm[order]
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=("PINN", "FDM"),
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=temperature_pinn,
            mode="lines",
            name="PINN",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=temperature_fdm,
            mode="lines",
            name="FDM",
            line=dict(dash="dash"),
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        template="plotly_dark",
        title=f"Temperature comparison at t={time_label:.1f} days",
        xaxis_title="Depth (m)",
        yaxis_title="Temperature (°C)",
    )
    fig.update_xaxes(title_text="Depth (m)", row=1, col=1)
    fig.update_xaxes(title_text="Depth (m)", row=1, col=2)
    return fig


def plot_parameter_inversion(
    x: np.ndarray,
    lambda_pred: np.ndarray,
    lambda_true: np.ndarray,
) -> go.Figure:
    """Plot thermal conductivity inversion result."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=lambda_pred,
            mode="lines",
            name="Predicted λ",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=lambda_true,
            mode="lines",
            name="True λ",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Thermal conductivity inversion result",
        xaxis_title="Depth (m)",
        yaxis_title="Thermal conductivity (W/m·K)",
    )
    return fig


def plot_real_vs_predicted_time_series(
    time: np.ndarray,
    T_true: np.ndarray,
    T_pred: np.ndarray,
    *,
    depth_value: float,
) -> go.Figure:
    """Plot real vs predicted temperature time series at one depth."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time,
            y=T_true,
            mode="lines",
            name="True",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=T_pred,
            mode="lines",
            name="Predicted",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title=f"Temperature time series at depth {depth_value:.2f} m",
        xaxis_title="Time (days)",
        yaxis_title="Temperature (°C)",
    )
    return fig


def plot_error_field(
    x: np.ndarray,
    time: np.ndarray,
    error_matrix: np.ndarray,
) -> go.Figure:
    """Plot prediction error field."""
    fig = go.Figure(
        data=[
            go.Heatmap(
                x=time,
                y=x,
                z=error_matrix,
                colorscale="Hot",
                colorbar=dict(title="|Error| (°C)"),
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        title="Prediction error field",
        xaxis_title="Time (days)",
        yaxis_title="Depth (m)",
    )
    return fig


def plot_residual_heatmap(
    x: np.ndarray,
    time: np.ndarray,
    residuals: np.ndarray,
) -> go.Figure:
    """Plot physics residual heatmap."""
    fig = go.Figure(
        data=[
            go.Heatmap(
                x=x,
                y=time,
                z=residuals,
                colorscale="RdBu",
                colorbar=dict(title="PDE residual"),
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        title="Physics residuals (∂T/∂t - λ∂²T/∂x²)",
        xaxis_title="Depth (m)",
        yaxis_title="Time (days)",
    )
    return fig


# --------------------------------------------------------------------------- #
# Shared utilities
# --------------------------------------------------------------------------- #

def _safe_array(values: Iterable[Any]) -> np.ndarray:
    if values is None:
        return np.array([])
    return np.asarray(list(values), dtype=float)


def _matrix_to_time_major(matrix: Iterable[Iterable[Any]]) -> np.ndarray:
    if matrix is None:
        return np.empty((0, 0))
    arr = np.asarray([list(row) for row in matrix], dtype=float)
    if arr.size == 0:
        return np.empty((0, 0))
    return arr.T


def _evenly_spaced_indices(length: int, max_count: int) -> List[int]:
    if length <= 0:
        return []
    if length <= max_count:
        return list(range(length))
    step = max(length // max_count, 1)
    indices = list(range(0, length, step))
    return indices[:max_count]


def _maybe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def placeholder_figure(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color="#94a3b8", size=16), align="center")
    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
    )
    return fig


def render_summary_lines(summary: Dict[str, Any]) -> List[str]:
    if not summary:
        return ["No data yet."]
    history = summary.get("history_available") or {}
    lines = [
        f"FDM points: {summary.get('fdm_points', 0)}",
        f"PINN points: {summary.get('pinn_points', 0)}",
        f"Comparison pairs: {summary.get('comparison_pairs', 0)}",
        "History snapshots: "
        + f"forward={'yes' if history.get('pinn_forward') else 'no'}, "
        + f"inversion={'yes' if history.get('pinn_inversion') else 'no'}",
    ]
    return lines


# --------------------------------------------------------------------------- #
# Figure builders (payload -> figure, description)
# --------------------------------------------------------------------------- #

def build_temperature_evolution(payload: Dict[str, Any]) -> Tuple[go.Figure, str]:
    grid = (payload.get("fdm") or {}).get("grid") or {}
    depth = _safe_array(grid.get("depth_m"))
    time = _safe_array(grid.get("time_days"))
    T = _matrix_to_time_major(grid.get("temperature"))
    if depth.size == 0 or time.size == 0 or T.size == 0:
        return (
            placeholder_figure("Temperature Evolution", "No FDM temperature data available yet."),
            "Waiting for FDM simulation results.",
        )
    if not np.isfinite(T).any():
        return (
            placeholder_figure("Temperature Evolution", "FDM temperature grid has no finite values."),
            "Waiting for valid FDM simulation results.",
        )
    finite_time = time[np.isfinite(time)]
    time_offset = float(finite_time.min()) if finite_time.size else 0.0
    time_zeroed = time - time_offset
    T_plot = T.T
    if time.size < 2 or depth.size < 2:
        figure = plot_temperature_heatmap(time_zeroed, depth, T_plot)
        description = (
            f"Heatmap built from {T.shape[0]} time steps and {T.shape[1]} depth levels "
            f"(t0={time_offset:.2f}d)."
        )
        return figure, description
    figure = plot_temperature_contour(time_zeroed, depth, T_plot)
    description = (
        f"Contour built from {T.shape[0]} time steps and {T.shape[1]} depth levels "
        f"(t0={time_offset:.2f}d)."
    )
    return figure, description


def build_temperature_profiles(payload: Dict[str, Any]) -> Tuple[go.Figure, str]:
    grid = (payload.get("fdm") or {}).get("grid") or {}
    x = _safe_array(grid.get("depth_m"))
    time = _safe_array(grid.get("time_days"))
    T = _matrix_to_time_major(grid.get("temperature"))
    if x.size == 0 or time.size == 0 or T.size == 0:
        return (
            placeholder_figure("Temperature Profiles", "No FDM temperature data available yet."),
            "Waiting for FDM simulation results.",
        )
    target_depths = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    T_sorted = T[:, sort_idx]
    T_interp = np.vstack([np.interp(target_depths, x_sorted, row) for row in T_sorted])
    sample_indices = list(range(len(target_depths)))
    figure = plot_temperature_profiles(target_depths, time, T_interp, sample_indices=sample_indices)
    description = f"Time series for {len(sample_indices)} depth levels."
    return figure, description


def build_boundary_heat_flux(payload: Dict[str, Any]) -> Tuple[go.Figure, str]:
    flux = payload.get("boundary_flux") or {}
    time = _safe_array(flux.get("time_days"))
    q_surface = _safe_array(flux.get("q_surface"))
    q_bottom = _safe_array(flux.get("q_bottom"))
    if time.size == 0 or q_surface.size == 0 or q_bottom.size == 0:
        return (
            placeholder_figure("Boundary Heat Flux", "No boundary heat flux samples available."),
            "Boundary flux diagnostics have not been recorded yet.",
        )
    figure = plot_boundary_heat_flux(time, q_surface, q_bottom)
    description = f"Flux series with {time.size} samples."
    return figure, description


def build_pinn_vs_fdm(payload: Dict[str, Any]) -> Tuple[go.Figure, str]:
    comparison = payload.get("comparison") or {}
    snapshots = comparison.get("snapshots") or []
    if not snapshots:
        return (
            placeholder_figure("PINN vs FDM", "Comparison data unavailable."),
            "Waiting for overlapping PINN and FDM samples.",
        )
    snapshot = snapshots[-1]
    x = _safe_array(snapshot.get("depth_m"))
    pinn = _safe_array(snapshot.get("pinn"))
    fdm = _safe_array(snapshot.get("fdm"))
    time_value = float(snapshot.get("time_days", 0.0))
    if x.size == 0 or pinn.size == 0 or fdm.size == 0:
        return (
            placeholder_figure("PINN vs FDM", "Comparison snapshot is empty."),
            "Waiting for overlapping PINN and FDM samples.",
        )
    figure = plot_pinn_vs_fdm(x, pinn, fdm, time_label=time_value)
    description = f"Side-by-side PINN and FDM profiles at t={time_value:.2f} days."
    return figure, description


def build_parameter_inversion(payload: Dict[str, Any]) -> Tuple[go.Figure, str]:
    inversion = payload.get("inversion") or {}
    parameters = inversion.get("parameters") or {}
    param_keys = ["lambda_f", "C_f", "eta", "b", "T_nabla"]
    param_labels = {
        "lambda_f": "Thermal conductivity (λ_f)",
        "C_f": "Heat capacity (C_f)",
        "eta": "Porosity (η)",
        "b": "Unfrozen water exponent (b)",
        "T_nabla": "Freezing temperature (T∇)",
    }
    predicted_values = []
    true_values = []
    physics_defaults = PhysicsParams()
    for key in param_keys:
        predicted_values.append(_maybe_float(parameters.get(key)))
        true_candidate = _maybe_float(parameters.get(f"{key}_true"))
        if true_candidate is None:
            true_candidate = _maybe_float(getattr(physics_defaults, key, None))
        true_values.append(true_candidate)

    if all(value is None for value in predicted_values):
        return (
            placeholder_figure("Parameter Inversion", "Inversion results not available yet."),
            "Awaiting inversion output.",
        )

    figure = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[param_labels.get(key, key) for key in param_keys] + [""],
    )
    for idx, key in enumerate(param_keys):
        row = (idx // 3) + 1
        col = (idx % 3) + 1
        pred = predicted_values[idx]
        true = true_values[idx]
        figure.add_trace(
            go.Bar(
                x=["Predicted", "True"],
                y=[pred if pred is not None else 0.0, true if true is not None else 0.0],
                marker_color=["#38bdf8", "#f97316"],
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        figure.update_yaxes(title_text=param_labels.get(key, key), row=row, col=col)
    figure.update_layout(
        template="plotly_dark",
        title="Parameter inversion result",
        showlegend=False,
    )
    description = "Predicted parameters compared against baseline values."
    return figure, description


def build_time_series(payload: Dict[str, Any]) -> Tuple[go.Figure, str]:
    comparison = payload.get("comparison") or {}
    series = comparison.get("per_depth_series") or []
    if not series:
        return (
            placeholder_figure("Real vs Predicted", "Time-series comparison unavailable."),
            "Waiting for overlapping PINN and FDM samples.",
        )
    selected_depth = _maybe_float(payload.get("selected_depth"))
    entry = series[0]
    if selected_depth is not None:
        for candidate in series:
            depth_value = _maybe_float(candidate.get("depth_m"))
            if depth_value is not None and abs(depth_value - selected_depth) <= 1e-6:
                entry = candidate
                break
    time = _safe_array(entry.get("time_days"))
    fdm = _safe_array(entry.get("fdm"))
    pinn = _safe_array(entry.get("pinn"))
    depth_value = float(entry.get("depth_m", 0.0))
    if time.size == 0 or fdm.size == 0 or pinn.size == 0:
        return (
            placeholder_figure("Real vs Predicted", "Time-series comparison unavailable."),
            "Waiting for overlapping PINN and FDM samples.",
        )
    figure = plot_real_vs_predicted_time_series(time, fdm, pinn, depth_value=depth_value)
    description = f"Depth {depth_value:.2f} m with {time.size} time samples."
    return figure, description


def build_error_field(payload: Dict[str, Any]) -> Tuple[go.Figure, str]:
    grid = (payload.get("comparison") or {}).get("abs_error_grid") or {}
    x = _safe_array(grid.get("depth_m"))
    time = _safe_array(grid.get("time_days"))
    errors = _matrix_to_time_major(grid.get("abs_error"))
    if x.size == 0 or time.size == 0 or errors.size == 0:
        return (
            placeholder_figure("Error Field", "Error field data unavailable."),
            "Waiting for overlapping PINN and FDM samples.",
        )
    figure = plot_error_field(x, time, errors)
    description = f"Error heatmap across {errors.shape[0]} time steps and {errors.shape[1]} depths."
    return figure, description


def build_residual_heatmap(payload: Dict[str, Any]) -> Tuple[go.Figure, str]:
    residuals = payload.get("residuals") or {}
    x = _safe_array(residuals.get("depth_m"))
    time = _safe_array(residuals.get("time_days"))
    matrix = _matrix_to_time_major(residuals.get("residuals"))
    if x.size == 0 or time.size == 0 or matrix.size == 0:
        return (
            placeholder_figure("Residual Heatmap", "Residual diagnostics not available yet."),
            "TODO: integrate PINN residual logging.",
        )
    figure = plot_residual_heatmap(x, time, matrix)
    description = f"Residual heatmap covering {matrix.shape[0]} time steps."
    return figure, description


FIGURE_BUILDERS: Dict[str, Tuple[str, PlotBuilder]] = {
    "temperature_contour": ("Temperature Evolution (Contour)", build_temperature_evolution),
    "temperature_profiles": ("Temperature Profiles", build_temperature_profiles),
    "boundary_heat_flux": ("Boundary Heat Flux", build_boundary_heat_flux),
    "pinn_vs_fdm": ("PINN vs FDM Comparison", build_pinn_vs_fdm),
    "parameter_inversion": ("Parameter Inversion", build_parameter_inversion),
    "time_series": ("Real vs Predicted Time Series", build_time_series),
    "error_field": ("Error Field", build_error_field),
    "residual_heatmap": ("Residual Heatmap", build_residual_heatmap),
}


# --------------------------------------------------------------------------- #
# Data bridge (Influx + gateway helper)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class VisualizationConfig:
    host: str = "0.0.0.0"
    port: int = 8501
    refresh_seconds: float = 60.0
    fetch_limit: int = 20000
    output_dir: Path = Path("software/outputs")


class VizDataBridge:
    """Helper responsible for querying InfluxDB and assembling payloads."""

    def __init__(
        self,
        influx_config: InfluxConfig | None = None,
        *,
        fetch_limit: int = 20000,
        output_dir: str | Path = "software/outputs",
    ) -> None:
        self.logger = get_logger("VizDataBridge")
        self.influx_config = influx_config or InfluxConfig()
        self.fetch_limit = int(fetch_limit)
        self.output_dir = Path(output_dir)
        self.influx = InfluxHelper(self.influx_config)
        self.gateway = VizGatewayServer(influx_config=self.influx_config, output_dir=str(self.output_dir))
        self.gateway.influx = self.influx

    def load_payload(self) -> Dict[str, Any]:
        now = datetime.utcnow().isoformat()
        base_payload: Dict[str, Any] = {"timestamp": now, "status": "no_data"}
        try:
            fdm_raw = self.influx.query_model_temperature("fdm_simulation", limit=self.fetch_limit)
            pinn_raw = self.influx.query_model_temperature("pinn_forward", limit=self.fetch_limit)
            inversion_msg = self._load_inversion_message()
        except Exception as exc:  # pragma: no cover - dependent on infrastructure
            error_message = f"Influx query failed: {exc}"
            self.logger.error(error_message, exc_info=True)
            base_payload.update({"status": "error", "error": error_message})
            return base_payload
        fdm_df = self.gateway._prepare_dataframe(fdm_raw)
        pinn_df = self.gateway._prepare_dataframe(pinn_raw)
        try:
            payload = self.gateway._build_payload(fdm_df, pinn_df, inversion_msg=inversion_msg)
        except Exception as exc:  # pragma: no cover - defensive programming
            error_message = f"Payload assembly failed: {exc}"
            self.logger.error(error_message, exc_info=True)
            return {"timestamp": now, "status": "error", "error": error_message}
        payload["fetch_timestamp"] = now
        payload["boundary_flux"] = self._load_boundary_flux()
        payload["residuals"] = self._load_residuals()
        payload["status"] = "ok"
        return payload

    def _load_boundary_flux(self) -> Dict[str, List[float]]:
        try:
            df = self.influx.query_boundary_flux(limit=self.fetch_limit)
        except Exception as exc:  # pragma: no cover - defensive programming
            self.logger.error("Failed to query boundary flux data: %s", exc, exc_info=True)
            return {"time_days": [], "q_surface": [], "q_bottom": []}
        if df is None or df.empty:
            return {"time_days": [], "q_surface": [], "q_bottom": []}
        df = df.copy()
        if "time_days" not in df.columns and "_time" in df.columns:
            first_time = df["_time"].min()
            df["time_days"] = (df["_time"] - first_time).dt.total_seconds() / 86400.0
        def _find_column(candidates: Iterable[str], fallback_match: str | None = None) -> str | None:
            for candidate in candidates:
                if candidate in df.columns:
                    return candidate
            if fallback_match:
                for column in df.columns:
                    if fallback_match in column.lower():
                        return column
            return None
        surface_col = _find_column(["q_surface", "surface_flux"], fallback_match="surface")
        bottom_col = _find_column(["q_bottom", "bottom_flux"], fallback_match="bottom")
        if surface_col is None or bottom_col is None:
            self.logger.warning("Boundary flux data missing expected columns (found=%s)", sorted(df.columns))
            return {"time_days": [], "q_surface": [], "q_bottom": []}
        df = df.sort_values("time_days")
        return {
            "time_days": df["time_days"].astype(float).tolist(),
            "q_surface": df[surface_col].astype(float).tolist(),
            "q_bottom": df[bottom_col].astype(float).tolist(),
        }

    def _load_residuals(self) -> Dict[str, List[float]]:
        try:
            df = self.influx.query_pinn_residuals(limit=self.fetch_limit)
        except Exception as exc:  # pragma: no cover - defensive programming
            self.logger.error("Failed to query PINN residual data: %s", exc, exc_info=True)
            return {"time_days": [], "depth_m": [], "residuals": []}
        if df is None or df.empty:
            return {"time_days": [], "depth_m": [], "residuals": []}
        df = df.copy()
        if "time_days" not in df.columns and "_time" in df.columns:
            first_time = df["_time"].min()
            df["time_days"] = (df["_time"] - first_time).dt.total_seconds() / 86400.0
        value_column = None
        for candidate in ("residual", "pde_residual", "value"):
            if candidate in df.columns:
                value_column = candidate
                break
        if value_column is None:
            inferred = [column for column in df.columns if "residual" in column.lower()]
            if inferred:
                value_column = inferred[0]
        if value_column is None:
            self.logger.warning("Residual query lacks residual value column (columns=%s)", sorted(df.columns))
            return {"time_days": [], "depth_m": [], "residuals": []}
        df = df.dropna(subset=["time_days", "depth_m", value_column])
        if df.empty:
            return {"time_days": [], "depth_m": [], "residuals": []}
        pivot = (
            df.pivot_table(index="depth_m", columns="time_days", values=value_column, aggfunc="mean")
            .sort_index()
            .sort_index(axis=1)
        )
        depth_values = [float(depth) for depth in pivot.index.to_list()]
        time_values = [float(ts) for ts in pivot.columns.to_list()]
        matrix = pivot.to_numpy().tolist()
        return {"time_days": time_values, "depth_m": depth_values, "residuals": matrix}

    def _load_inversion_message(self) -> Dict[str, Any]:
        try:
            df = self.influx.query_inversion_parameters(limit=1)
        except Exception as exc:  # pragma: no cover - defensive programming
            self.logger.error("Failed to query inversion parameters: %s", exc, exc_info=True)
            return {}
        if df is None or df.empty:
            return {}
        row = df.iloc[-1]
        params: Dict[str, float] = {}
        for key in ("lambda_f", "C_f", "eta", "b", "T_nabla"):
            value = _maybe_float(row.get(key))
            if value is not None:
                params[key] = value
        timestamp_value = row.get("timestamp")
        missing_timestamp = False
        if isinstance(timestamp_value, float):
            missing_timestamp = math.isnan(timestamp_value)
        elif isinstance(timestamp_value, str):
            missing_timestamp = not timestamp_value.strip()
        else:
            missing_timestamp = timestamp_value is None
        if missing_timestamp and "_time" in row:
            timestamp_candidate = row["_time"]
            timestamp_value = timestamp_candidate.isoformat() if hasattr(timestamp_candidate, "isoformat") else None
        validation = {}
        for candidate in ("mean_abs_error", "max_abs_error"):
            for prefix in ("validation_", "val_", ""):
                col_name = f"{prefix}{candidate}" if prefix else candidate
                if col_name in row:
                    val = _maybe_float(row.get(col_name))
                    if val is not None:
                        validation[candidate] = val
                    break
        validation_overall = validation or None
        message: Dict[str, Any] = {
            "timestamp": timestamp_value,
            "status": row.get("status", "inverted"),
            "parameters": params,
            "validation_overall": validation_overall,
        }
        return {key: value for key, value in message.items() if value not in (None, {}, [])}

    def close(self) -> None:
        if self.influx is not None:
            self.influx.close()


# --------------------------------------------------------------------------- #
# Compatibility wrapper that launches the Streamlit app via CLI
# --------------------------------------------------------------------------- #

class VisualizationService:
    """Compatibility shim that starts the Streamlit dashboard via subprocess."""

    def __init__(
        self,
        influx_config: InfluxConfig | None = None,
        *,
        host: str = "0.0.0.0",
        port: int = 8501,
        refresh_seconds: float = 60.0,
        fetch_limit: int = 20000,
        output_dir: str | Path = "software/outputs",
    ) -> None:
        self.logger = get_logger("VisualizationService")
        self.influx_config = influx_config or InfluxConfig()
        self.config = VisualizationConfig(
            host=host,
            port=int(port),
            refresh_seconds=float(refresh_seconds),
            fetch_limit=int(fetch_limit),
            output_dir=Path(output_dir),
        )
        self.script_path = Path(__file__).with_name("streamlit_viz_app.py")

    def start(self) -> None:
        env = os.environ.copy()
        env.setdefault("STREAMLIT_SERVER_ADDRESS", self.config.host)
        env.setdefault("STREAMLIT_SERVER_PORT", str(self.config.port))
        env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
        env["PERMAFROST_VIZ_FETCH_LIMIT"] = str(self.config.fetch_limit)
        env["PERMAFROST_VIZ_REFRESH_SECONDS"] = str(self.config.refresh_seconds)
        env["PERMAFROST_VIZ_OUTPUT_DIR"] = str(self.config.output_dir)
        env["PERMAFROST_VIZ_INFLUX_CONFIG"] = json.dumps(asdict(self.influx_config))
        command = ["streamlit", "run", str(self.script_path)]
        try:
            subprocess.run(command, check=False, env=env)
        except FileNotFoundError as exc:  # pragma: no cover - depends on runtime env
            raise RuntimeError("Streamlit executable not found in PATH. Install streamlit to continue.") from exc

    def close(self) -> None:
        # Streamlit handles its own shutdown; nothing to close explicitly.
        self.logger.info("VisualizationService shutdown requested.")

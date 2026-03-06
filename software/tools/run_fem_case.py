"""Simple stakeholder-facing FEM runner."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import griddata

from software.digital_twin.simulator.fem_forecast.fem_solver import FEMMeshConfig, FEMSolver2D, MaterialProps, SolverConfig
from software.startup.utils.config import load_startup_config

DOMAIN_RADIUS_M = 0.24
DOMAIN_DEPTH_M = 0.30


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _interpolate_temperature_field(
    points_for_field: list[dict[str, Any]],
    grid_resolution: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not points_for_field:
        return None

    x = np.array([float(item["x_m"]) for item in points_for_field], dtype=float)
    z = np.array([float(item["z_m"]) for item in points_for_field], dtype=float)
    t = np.array([float(item["temperature"]) for item in points_for_field], dtype=float)

    valid = np.isfinite(x) & np.isfinite(z) & np.isfinite(t)
    x = np.abs(x[valid])
    z = z[valid]
    t = t[valid]
    if x.size == 0:
        return None

    r_grid = np.linspace(0.0, DOMAIN_RADIUS_M, grid_resolution)
    z_grid = np.linspace(0.0, DOMAIN_DEPTH_M, grid_resolution)
    R, Z = np.meshgrid(r_grid, z_grid)

    points = np.column_stack([x, z])
    method = "cubic" if len(points) >= 8 else "linear"
    try:
        T = griddata(points, t, (R, Z), method=method)
    except Exception:
        T = griddata(points, t, (R, Z), method="linear")

    if T is None:
        return None
    return R, Z, T


def _write_temperature_plot(path: Path, result: dict[str, Any], temp_range: tuple[float, float] = (-1.0, 60.0)) -> None:
    """Write compact heatmap-style temperature plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate plots.") from exc

    points_for_field = result.get("field_nodes", result.get("temperatures", []))
    if not isinstance(points_for_field, list) or not points_for_field:
        raise ValueError("No temperature field data available to plot.")

    field = _interpolate_temperature_field(points_for_field)
    if field is None:
        raise ValueError("Unable to interpolate temperature field.")

    R, Z, T = field
    x_min, x_max = float(np.nanmin(R)), float(np.nanmax(R))
    z_min, z_max = float(np.nanmin(Z)), float(np.nanmax(Z))

    fig, ax = plt.subplots(figsize=(3.3, 3.1), dpi=160)
    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
    t_masked = np.ma.masked_invalid(T)

    im = ax.imshow(
        t_masked,
        cmap=cmap,
        vmin=temp_range[0],
        vmax=temp_range[1],
        extent=[x_min, x_max, z_max, z_min],
        interpolation="none",
        aspect="auto",
    )
    ax.set_title("Estimated Result")
    ax.set_xlabel("Locations (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_max, z_min)
    ax.tick_params(length=2)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Temperature (C)")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _resolve_mesh_dir(mesh_dir_cfg: object) -> Path:
    if not isinstance(mesh_dir_cfg, str) or not mesh_dir_cfg:
        raise ValueError("fem_forecast.mesh_dir must be configured.")
    mesh_dir = Path(mesh_dir_cfg)
    if not mesh_dir.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        mesh_dir = repo_root / mesh_dir
    return mesh_dir


def _material_from_dict(material_dict: dict[str, Any]) -> MaterialProps:
    defaults = MaterialProps()
    return MaterialProps(
        porosity=float(material_dict.get("porosity", defaults.porosity)),
        rho_s=float(material_dict.get("rho_s", defaults.rho_s)),
        rho_w=float(material_dict.get("rho_w", defaults.rho_w)),
        rho_i=float(material_dict.get("rho_i", defaults.rho_i)),
        cap_s=float(material_dict.get("cap_s", defaults.cap_s)),
        cap_w=float(material_dict.get("cap_w", defaults.cap_w)),
        cap_i=float(material_dict.get("cap_i", defaults.cap_i)),
        latent_heat=float(material_dict.get("latent_heat", defaults.latent_heat)),
        k_parameter=float(material_dict.get("k_parameter", defaults.k_parameter)),
        tr_k=float(material_dict.get("tr_k", defaults.tr_k)),
        lambda_thawed=float(material_dict.get("lambda_thawed", defaults.lambda_thawed)),
        lambda_frozen=float(material_dict.get("lambda_frozen", defaults.lambda_frozen)),
    )


def _solver_from_dict(solver_dict: dict[str, Any]) -> SolverConfig:
    defaults = SolverConfig()
    return SolverConfig(
        dt_seconds=float(solver_dict.get("dt_seconds", defaults.dt_seconds)),
        max_iterations=int(solver_dict.get("max_iterations", defaults.max_iterations)),
        tol_r=float(solver_dict.get("tol_r", defaults.tol_r)),
        temperature_offset_k=float(solver_dict.get("temperature_offset_k", defaults.temperature_offset_k)),
        initial_temperature_c=float(solver_dict.get("initial_temperature_c", defaults.initial_temperature_c)),
        thaw_threshold_c=float(solver_dict.get("thaw_threshold_c", defaults.thaw_threshold_c)),
    )


def _sample_positions(
    solver: FEMSolver2D,
    positions: list[dict[str, Any]],
) -> list[dict[str, float | str]]:
    node_xy = np.asarray(solver.node, dtype=float)
    x_abs = np.abs(node_xy[:, 0])
    z = node_xy[:, 1]
    zero_c_in_solver_units = float(solver.solver.temperature_offset_k)

    sampled: list[dict[str, float | str]] = []
    for idx, position in enumerate(positions):
        x_m = float(position["x_m"])
        z_m = float(position["z_m"])
        sensor_id = str(position.get("sensor_id", f"P{idx + 1}"))

        distances = (x_abs - abs(x_m)) ** 2 + (z - z_m) ** 2
        nearest_idx = int(np.argmin(distances))
        sampled.append(
            {
                "sensor_id": sensor_id,
                "x_m": x_m,
                "z_m": z_m,
                "temperature": float(solver.T[nearest_idx]) - zero_c_in_solver_units,
            }
        )
    return sampled


def _default_boundary_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    bc_cfg = config.get("boundary_condition_builder", {})
    boundaries = bc_cfg.get("boundaries", []) if isinstance(bc_cfg, dict) else []
    if not isinstance(boundaries, list) or not boundaries:
        raise ValueError("No default boundaries found in startup config.")
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "time_hours": 0.0,
        "boundaries": boundaries,
    }


def FEM(
    params: dict[str, float],
    positions: list[dict[str, float | str]],
    boundary_snapshot: dict[str, Any] | None = None,
    config_path: str | Path = "startup.conf",
    horizon_hours: float | None = None,
    return_field: bool = False,
) -> dict[str, Any]:
    """Compute temperatures for selected positions using material params override."""
    if not isinstance(params, dict):
        raise ValueError("params must be a dict.")
    if not isinstance(positions, list) or not positions:
        raise ValueError("positions must be a non-empty list.")

    config = load_startup_config(config_path)
    fem_cfg = config.get("fem_forecast", {})
    if not isinstance(fem_cfg, dict):
        raise ValueError("fem_forecast config must be an object.")

    mesh = FEMMeshConfig(
        mesh_dir=_resolve_mesh_dir(fem_cfg.get("mesh_dir")),
        node_file=str(fem_cfg.get("node_file", "DT_model_nodes.txt")),
        element_file=str(fem_cfg.get("element_file", "DT_model_elements.txt")),
        edge_sets=fem_cfg.get("edge_sets", {}) if isinstance(fem_cfg.get("edge_sets"), dict) else {},
        node_sets=fem_cfg.get("node_sets", {}) if isinstance(fem_cfg.get("node_sets"), dict) else {},
    )
    solver_cfg = _solver_from_dict(fem_cfg.get("solver", {}) if isinstance(fem_cfg.get("solver"), dict) else {})

    base_material = fem_cfg.get("material", {}) if isinstance(fem_cfg.get("material"), dict) else {}
    merged_material = dict(base_material)
    merged_material.update(params)

    solver = FEMSolver2D(
        mesh=mesh,
        material=_material_from_dict(merged_material),
        solver=solver_cfg,
    )

    snapshot = boundary_snapshot or _default_boundary_snapshot(config)
    if horizon_hours is None:
        run_horizon_hours = float(fem_cfg.get("horizon_hours", 1.0))
    else:
        run_horizon_hours = float(horizon_hours)
    if run_horizon_hours <= 0.0:
        raise ValueError("horizon_hours must be > 0.")

    solver.advance(snapshot, horizon_hours=run_horizon_hours)
    radius_max_m, radius_avg_m, thaw_points = solver.compute_thaw_metrics()

    result: dict[str, Any] = {
        "timestamp": snapshot.get("timestamp"),
        "time_hours": float(snapshot.get("time_hours", 0.0)),
        "horizon_hours": run_horizon_hours,
        "radius_max_m": radius_max_m,
        "radius_avg_m": radius_avg_m,
        "points": thaw_points,
        "temperatures": _sample_positions(solver, positions),
    }
    if return_field:
        zero_c_in_solver_units = float(solver.solver.temperature_offset_k)
        field_points: list[dict[str, float]] = []
        for idx in range(solver.node.shape[0]):
            field_points.append(
                {
                    "x_m": abs(float(solver.node[idx, 0])),
                    "z_m": float(solver.node[idx, 1]),
                    "temperature": float(solver.T[idx]) - zero_c_in_solver_units,
                }
            )
        result["field_nodes"] = field_points
    return result


def main() -> None:
    params = {
        "porosity": 0.01,
        "rho_s": 2700.0,
        "rho_w": 1000.0,
        "rho_i": 920.0,
        "cap_s": 790.0,
        "cap_w": 4180.0,
        "cap_i": 2090.0,
        "latent_heat": 334000.0,
        "k_parameter": 5.0,
        "tr_k": 273.15,
        "lambda_thawed": 1.64,
        "lambda_frozen": 2.96,
    }
    depths_mm = (10, 50, 90, 170, 250, 290)
    widths_mm = (15, 30, 45, 60, 75, 90, 120)
    positions = [
        {
            "sensor_id": f"D{depth_mm}-W{width_mm}",
            "x_m": width_mm / 1000.0,
            "z_m": depth_mm / 1000.0,
        }
        for depth_mm in depths_mm
        for width_mm in widths_mm
    ]
    horizon_hours = 1.0

    result = FEM(params, positions, horizon_hours=horizon_hours, return_field=True)
    out_path = Path("artifacts/fem_cases/manual_case_result.json")
    plot_path = Path("artifacts/fem_cases/manual_case_plot.png")
    _write_temperature_plot(plot_path, result)

    json_result = dict(result)
    json_result.pop("field_nodes", None)
    _write_json(out_path, json_result)
    print(f"Saved FEM result to: {out_path}")
    print(f"Saved temperature plot to: {plot_path}")


if __name__ == "__main__":
    main()

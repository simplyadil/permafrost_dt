"""FEM forecast service for 2D thaw-front predictions."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from software.digital_twin.communication.messaging import RabbitMQClient, RabbitMQConfig, resolve_queue_config
from software.digital_twin.data_access.influx_utils import InfluxConfig, InfluxHelper
from software.digital_twin.simulator.fem_forecast.fem_solver import (
    FEMMeshConfig,
    FEMSolver2D,
    MaterialProps,
    SolverConfig,
)
from software.utils.logging_setup import get_logger

BOUNDARY_QUEUE = "permafrost.record.boundary.2d"
FORECAST_QUEUE = "permafrost.record.fem_forecast.2d"
BOUNDARY_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "boundary_condition_message.json"
FORECAST_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "fem_forecast_message.json"


@dataclass
class ForecastResult:
    radius_max_m: float | None
    radius_avg_m: float | None
    points: list[dict[str, float]]


@dataclass
class FieldSnapshot:
    """Snapshot of FEM temperature field at simulation end."""
    node_xy: np.ndarray      # shape (numnode, 2)
    temperature: np.ndarray  # shape (numnode,)
    time_hours: float
    units: str               # "K" for Kelvin


class FEMForecastServer:
    """Consumes boundary snapshots and publishes FEM forecasts."""
    SENSOR_DEPTHS_MM = (10, 50, 90, 170, 250, 290)
    SENSOR_WIDTHS_MM = (15, 30, 45, 60, 75, 90, 120)

    def __init__(
        self,
        influx_config: InfluxConfig | None = None,
        input_config: RabbitMQConfig | None = None,
        output_config: RabbitMQConfig | None = None,
        *,
        input_queue: str | None = None,
        output_queue: str | None = None,
        horizon_hours: float = 1.0,
        site_id: str = "default",
        mesh_dir: str | Path | None = None,
        node_file: str = "DT_model_nodes.txt",
        element_file: str = "DT_model_elements.txt",
        edge_sets: dict[str, str] | None = None,
        node_sets: dict[str, str] | None = None,
        material: dict | None = None,
        solver: dict | None = None,
        save_field_csv_default: bool = True,
        include_field_default: bool = False,
        field_csv_dir: str | None = None,
    ) -> None:
        self.logger = get_logger("FEMForecastServer")
        self.influx_config = influx_config or InfluxConfig()
        self.input_config = resolve_queue_config(
            input_config,
            queue=input_queue or BOUNDARY_QUEUE,
            schema_path=BOUNDARY_SCHEMA,
        )
        self.output_config = resolve_queue_config(
            output_config,
            queue=output_queue or FORECAST_QUEUE,
            schema_path=FORECAST_SCHEMA,
        )
        self.horizon_hours = float(horizon_hours)
        self.site_id = site_id
        self.mesh_dir = Path(mesh_dir) if mesh_dir is not None else None
        self.node_file = node_file
        self.element_file = element_file
        self.edge_sets = edge_sets or {}
        self.node_sets = node_sets or {}
        self.material_config = material or {}
        self.solver_config = solver or {}
        self.save_field_csv_default = self._as_bool(save_field_csv_default, True)
        self.include_field_default = self._as_bool(include_field_default, False)
        self.field_csv_dir = field_csv_dir

        self.mq_in: Optional[RabbitMQClient] = None
        self.mq_out: Optional[RabbitMQClient] = None
        self.influx: Optional[InfluxHelper] = None
        self._solver: Optional[FEMSolver2D] = None

    @staticmethod
    def _as_bool(value: object, default: bool) -> bool:
        """Parse bool-like values from message/config without surprising truthiness."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            parsed = value.strip().lower()
            if parsed in {"1", "true", "yes", "y", "on"}:
                return True
            if parsed in {"0", "false", "no", "n", "off"}:
                return False
        return default

    def _material_from_dict(self, material_dict: dict) -> MaterialProps:
        """Construct MaterialProps from dict with same defaults as _build_solver."""
        return MaterialProps(
            porosity=float(material_dict.get("porosity", 0.42)),
            rho_s=float(material_dict.get("rho_s", 2700.0)),
            rho_w=float(material_dict.get("rho_w", 1000.0)),
            rho_i=float(material_dict.get("rho_i", 920.0)),
            cap_s=float(material_dict.get("cap_s", 790.0)),
            cap_w=float(material_dict.get("cap_w", 4180.0)),
            cap_i=float(material_dict.get("cap_i", 2090.0)),
            latent_heat=float(material_dict.get("latent_heat", 3.34e5)),
            k_parameter=float(material_dict.get("k_parameter", 5.0)),
            tr_k=float(material_dict.get("tr_k", 273.15)),
            lambda_thawed=float(material_dict.get("lambda_thawed", 1.64)),
            lambda_frozen=float(material_dict.get("lambda_frozen", 2.96)),
        )

    def _write_field_csv(self, field: FieldSnapshot, horizon_hours: float, field_csv_dir: str | None = None) -> str:
        """Append one horizontal row per timestep using D*-W* sensor-style columns."""
        if field_csv_dir is None:
            field_csv_dir = "artifacts/fem_fields"
        
        field_path = Path(field_csv_dir)
        field_path.mkdir(parents=True, exist_ok=True)
        
        site_str = str(self.site_id).replace("/", "_").replace(" ", "_")
        filename = f"fem_field_{site_str}.csv"
        filepath = field_path / filename
        write_header = (not filepath.exists()) or filepath.stat().st_size == 0

        # Map FEM nodal field to the requested sensor-style horizontal columns.
        # Dxx-Wyy means depth xx mm and width yy mm.
        points = np.asarray(field.node_xy, dtype=float)
        x_abs = np.abs(points[:, 0])
        z = points[:, 1]
        sensor_labels = [
            f"D{depth_mm}-W{width_mm}"
            for depth_mm in self.SENSOR_DEPTHS_MM
            for width_mm in self.SENSOR_WIDTHS_MM
        ]
        row_values: list[float] = []
        zero_c_in_solver_units = self._solver._to_kelvin(0.0) if self._solver is not None else 0.0
        for depth_mm in self.SENSOR_DEPTHS_MM:
            z_target = depth_mm / 1000.0
            for width_mm in self.SENSOR_WIDTHS_MM:
                x_target = width_mm / 1000.0
                distances = (x_abs - x_target) ** 2 + (z - z_target) ** 2
                nearest_idx = int(np.argmin(distances))
                temp_c = float(field.temperature[nearest_idx]) - float(zero_c_in_solver_units)
                row_values.append(temp_c)

        with open(filepath, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Time (h)", *sensor_labels])
            writer.writerow([field.time_hours, *row_values])

        self.logger.info("Appended field snapshot to %s", filepath)
        return str(filepath)

    def _sample_field_at_sensor_grid(self, field: FieldSnapshot) -> list[dict[str, float | str]]:
        """Sample FEM field at the configured D*-W* grid using nearest FEM node."""
        points = np.asarray(field.node_xy, dtype=float)
        x_abs = np.abs(points[:, 0])
        z = points[:, 1]
        zero_c_in_solver_units = self._solver._to_kelvin(0.0) if self._solver is not None else 0.0

        sampled: list[dict[str, float | str]] = []
        for depth_mm in self.SENSOR_DEPTHS_MM:
            z_target = depth_mm / 1000.0
            for width_mm in self.SENSOR_WIDTHS_MM:
                x_target = width_mm / 1000.0
                distances = (x_abs - x_target) ** 2 + (z - z_target) ** 2
                nearest_idx = int(np.argmin(distances))
                temp_c = float(field.temperature[nearest_idx]) - float(zero_c_in_solver_units)
                sampled.append(
                    {
                        "sensor_id": f"D{depth_mm}-W{width_mm}",
                        "x_m": float(x_target),
                        "z_m": float(z_target),
                        "temperature": temp_c,
                    }
                )
        return sampled

    def _build_solver(self) -> FEMSolver2D:
        if self.mesh_dir is None:
            raise ValueError("mesh_dir must be configured for FEM forecasting.")

        mesh = FEMMeshConfig(
            mesh_dir=self.mesh_dir,
            node_file=self.node_file,
            element_file=self.element_file,
            edge_sets=self.edge_sets,
            node_sets=self.node_sets,
        )
        material = MaterialProps(
            porosity=float(self.material_config.get("porosity", 0.42)),
            rho_s=float(self.material_config.get("rho_s", 2700.0)),
            rho_w=float(self.material_config.get("rho_w", 1000.0)),
            rho_i=float(self.material_config.get("rho_i", 920.0)),
            cap_s=float(self.material_config.get("cap_s", 790.0)),
            cap_w=float(self.material_config.get("cap_w", 4180.0)),
            cap_i=float(self.material_config.get("cap_i", 2090.0)),
            latent_heat=float(self.material_config.get("latent_heat", 3.34e5)),
            k_parameter=float(self.material_config.get("k_parameter", 5.0)),
            tr_k=float(self.material_config.get("tr_k", 273.15)),
            lambda_thawed=float(self.material_config.get("lambda_thawed", 1.64)),
            lambda_frozen=float(self.material_config.get("lambda_frozen", 2.96)),
        )
        solver = SolverConfig(
            dt_seconds=float(self.solver_config.get("dt_seconds", 36.0)),
            max_iterations=int(self.solver_config.get("max_iterations", 25)),
            tol_r=float(self.solver_config.get("tol_r", 1e-6)),
            temperature_offset_k=float(self.solver_config.get("temperature_offset_k", 0.0)),
            initial_temperature_c=float(self.solver_config.get("initial_temperature_c", -1.0)),
            thaw_threshold_c=float(self.solver_config.get("thaw_threshold_c", 0.0)),
        )
        return FEMSolver2D(mesh=mesh, material=material, solver=solver)

    def _run_forecast(self, boundary_snapshot: dict) -> ForecastResult:
        if self._solver is None:
            self._solver = self._build_solver()
        self._solver.advance(boundary_snapshot, horizon_hours=self.horizon_hours)
        radius_max_m, radius_avg_m, points = self._solver.compute_thaw_metrics()
        return ForecastResult(
            radius_max_m=radius_max_m,
            radius_avg_m=radius_avg_m,
            points=points,
        )

    def _run_forecast_with_field(
        self,
        boundary_snapshot: dict,
        material_override: dict | None = None,
        include_field: bool = True,
        save_field_csv: bool = True,
        field_csv_dir: str | None = None,
    ) -> tuple[ForecastResult, FieldSnapshot | None, str | None]:
        """Run forecast with optional field snapshot and CSV output.
        
        Returns:
            (ForecastResult, FieldSnapshot or None, csv_path or None)
        """
        if self._solver is None:
            self._solver = self._build_solver()
        
        # Apply material override if provided
        if material_override is not None:
            merged = {**self.material_config, **material_override}
            self._solver.material = self._material_from_dict(merged)
            self.logger.debug("Applied material override to solver")
        
        # Run forecast
        self._solver.advance(boundary_snapshot, horizon_hours=self.horizon_hours)
        radius_max_m, radius_avg_m, points = self._solver.compute_thaw_metrics()
        result = ForecastResult(
            radius_max_m=radius_max_m,
            radius_avg_m=radius_avg_m,
            points=points,
        )
        
        # Optionally capture field snapshot
        field_snapshot = None
        if include_field or save_field_csv:
            field_snapshot = FieldSnapshot(
                node_xy=self._solver.node.copy(),
                temperature=self._solver.T.copy(),
                time_hours=float(self._solver.time_hours or 0.0),
                units="K",  # FEMSolver2D stores Kelvin internally
            )
        
        # Optionally save to CSV
        csv_path = None
        if save_field_csv and field_snapshot is not None:
            csv_path = self._write_field_csv(field_snapshot, self.horizon_hours, field_csv_dir)
        
        return result, field_snapshot, csv_path

    def _process_message(self, msg: dict) -> None:
        if self.influx is None or self.mq_out is None:
            raise RuntimeError("Dependencies not initialised. Did you call setup()?")  # pragma: no cover

        try:
            time_hours = float(msg["time_hours"])
            boundary_count = len(msg.get("boundaries", []))
            field_snapshot: FieldSnapshot | None = None
            
            # Extract optional field/material control flags
            include_field = self._as_bool(msg.get("include_field"), self.include_field_default)
            save_field_csv = self._as_bool(msg.get("save_field_csv"), self.save_field_csv_default)
            field_csv_dir = msg.get("field_csv_dir", self.field_csv_dir)
            material_override = msg.get("material_override")
            
            # Choose forecast path based on flags
            if include_field or save_field_csv or material_override is not None:
                result, field_snapshot, csv_path = self._run_forecast_with_field(
                    msg,
                    material_override=material_override,
                    include_field=include_field,
                    save_field_csv=save_field_csv,
                    field_csv_dir=field_csv_dir,
                )
                if csv_path:
                    self.logger.info("Field CSV written to %s", csv_path)
            else:
                # Default path: existing behavior unchanged
                result = self._run_forecast(msg)
            
            self.logger.info(
                "Received boundary snapshot (t=%.2fh, boundaries=%d)",
                time_hours,
                boundary_count,
            )
            start_time = datetime.utcnow()
            if result.radius_max_m is None or result.radius_avg_m is None:
                self.logger.warning(
                    "FEM forecast returned None metrics (t=%.2fh, points=%d). Publishing anyway.",
                    time_hours,
                    len(result.points),
                )
            payload = {
                "timestamp": msg.get("timestamp") or datetime.utcnow().isoformat(),
                "time_hours": time_hours,
                "horizon_hours": self.horizon_hours,
                "status": "forecasted",
                "radius_max_m": result.radius_max_m,
                "radius_avg_m": result.radius_avg_m,
                "points": result.points,
            }
            self.mq_out.publish(payload)
            self.influx.write_fem_forecast_metrics(
                time_hours=time_hours,
                horizon_hours=self.horizon_hours,
                radius_max_m=result.radius_max_m,
                radius_avg_m=result.radius_avg_m,
                points=result.points,
                site=self.site_id,
            )
            if field_snapshot is None and self._solver is not None:
                field_snapshot = FieldSnapshot(
                    node_xy=self._solver.node.copy(),
                    temperature=self._solver.T.copy(),
                    time_hours=float(self._solver.time_hours or 0.0),
                    units="K",
                )
            if field_snapshot is not None:
                self.influx.write_fem_temperature_snapshot_2d(
                    time_hours=float(field_snapshot.time_hours),
                    horizon_hours=self.horizon_hours,
                    sensors=self._sample_field_at_sensor_grid(field_snapshot),
                    site=self.site_id,
                )
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            self.logger.info(
                "FEM forecast published (t=%.2fh, points=%d, elapsed=%.2fs)",
                time_hours,
                len(result.points),
                elapsed,
            )
        except Exception as exc:
            self.logger.error("FEM forecast FAILED: %s", exc, exc_info=True)

    def setup(self) -> None:
        if self.mq_in is None:
            self.mq_in = RabbitMQClient(self.input_config)
        if self.mq_out is None:
            self.mq_out = RabbitMQClient(self.output_config)
        if self.influx is None:
            self.influx = InfluxHelper(self.influx_config)
        self.logger.info(
            "Dependencies initialised (in=%s, out=%s)",
            self.input_config.queue,
            self.output_config.queue,
        )

    def start(self) -> None:
        if self.mq_in is None or self.mq_out is None or self.influx is None:
            self.setup()

        self.logger.info("Listening for boundary snapshots on %s", self.input_config.queue)
        try:
            self.mq_in.consume(callback=self._process_message)
        except KeyboardInterrupt:  # pragma: no cover - runtime behaviour
            self.logger.info("Interrupt received; shutting down")
        finally:
            self.close()

    def stop(self) -> None:
        if self.mq_in and self.mq_in.channel:
            self.mq_in.channel.stop_consuming()

    def close(self) -> None:
        self.stop()
        if self.mq_in is not None:
            self.mq_in.disconnect()
        if self.mq_out is not None:
            self.mq_out.disconnect()
        if self.influx is not None:
            self.influx.close()
        self.logger.info("Shutdown complete")


if __name__ == "__main__":
    FEMForecastServer().start()

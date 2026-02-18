"""FEM forecast service for 2D thaw-front predictions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

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


class FEMForecastServer:
    """Consumes boundary snapshots and publishes FEM forecasts."""

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

        self.mq_in: Optional[RabbitMQClient] = None
        self.mq_out: Optional[RabbitMQClient] = None
        self.influx: Optional[InfluxHelper] = None
        self._solver: Optional[FEMSolver2D] = None

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

    def _process_message(self, msg: dict) -> None:
        if self.influx is None or self.mq_out is None:
            raise RuntimeError("Dependencies not initialised. Did you call setup()?")  # pragma: no cover

        try:
            time_hours = float(msg["time_hours"])
            boundary_count = len(msg.get("boundaries", []))
            self.logger.info(
                "Received boundary snapshot (t=%.2fh, boundaries=%d)",
                time_hours,
                boundary_count,
            )
            start_time = datetime.utcnow()
            result = self._run_forecast(msg)
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

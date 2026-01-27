"""Boundary condition builder service for 2D FEM inputs."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional

from software.digital_twin.communication.messaging import RabbitMQClient, RabbitMQConfig, resolve_queue_config
from software.utils.logging_setup import get_logger

SENSOR_QUEUE = "permafrost.record.sensors.2d"
BOUNDARY_QUEUE = "permafrost.record.boundary.2d"
SENSOR_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "sensor_message_2d.json"
BOUNDARY_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "boundary_condition_message.json"
ALLOWED_BC_TYPES = {"dirichlet", "neumann", "robin"}


class BoundaryConditionBuilderServer:
    """Builds boundary condition snapshots from sensor messages and static config."""

    def __init__(
        self,
        input_config: RabbitMQConfig | None = None,
        output_config: RabbitMQConfig | None = None,
        *,
        input_queue: str | None = None,
        output_queue: str | None = None,
        boundaries: list[dict] | None = None,
        sensor_overrides: list[dict] | None = None,
    ) -> None:
        self.logger = get_logger("BoundaryConditionBuilderServer")
        self.input_config = resolve_queue_config(
            input_config,
            queue=input_queue or SENSOR_QUEUE,
            schema_path=SENSOR_SCHEMA,
        )
        self.output_config = resolve_queue_config(
            output_config,
            queue=output_queue or BOUNDARY_QUEUE,
            schema_path=BOUNDARY_SCHEMA,
        )
        self.boundaries = boundaries or []
        self.sensor_overrides = sensor_overrides or []
        self._validate_boundaries()
        self._validate_overrides()

        self.mq_in: Optional[RabbitMQClient] = None
        self.mq_out: Optional[RabbitMQClient] = None

    def _validate_boundaries(self) -> None:
        for boundary in self.boundaries:
            bc_type = boundary.get("bc_type")
            boundary_set = boundary.get("boundary_set")
            if not boundary_set or not isinstance(boundary_set, str):
                raise ValueError("Each boundary entry must include a string boundary_set.")
            if bc_type not in ALLOWED_BC_TYPES:
                raise ValueError(f"Unsupported bc_type '{bc_type}'. Allowed: {sorted(ALLOWED_BC_TYPES)}")

    def _validate_overrides(self) -> None:
        for override in self.sensor_overrides:
            if "boundary_set" not in override or "sensor_id" not in override:
                raise ValueError("Sensor overrides must include boundary_set and sensor_id.")

    def _apply_sensor_overrides(self, boundaries: list[dict], sensors: list[dict]) -> None:
        if not boundaries or not sensors or not self.sensor_overrides:
            return

        temp_by_sensor = {}
        for sensor in sensors:
            try:
                sensor_id = str(sensor["sensor_id"])
                temperature = float(sensor["temperature"])
            except (KeyError, TypeError, ValueError):
                continue
            temp_by_sensor[sensor_id] = temperature

        for override in self.sensor_overrides:
            boundary_set = override.get("boundary_set")
            sensor_id = override.get("sensor_id")
            if boundary_set is None or sensor_id is None:
                continue
            if sensor_id not in temp_by_sensor:
                continue

            for boundary in boundaries:
                if boundary.get("boundary_set") != boundary_set:
                    continue
                if boundary.get("bc_type") not in {"dirichlet", "robin"}:
                    continue
                boundary["temperature"] = temp_by_sensor[sensor_id]

    def _build_snapshot(self, msg: dict) -> dict:
        time_hours = float(msg["time_hours"])
        boundaries = deepcopy(self.boundaries)
        sensors = msg.get("sensors", [])
        self._apply_sensor_overrides(boundaries, sensors)
        return {
            "timestamp": msg.get("timestamp") or datetime.utcnow().isoformat(),
            "time_hours": time_hours,
            "boundaries": boundaries,
        }

    def _process_message(self, msg: dict) -> None:
        snapshot = self._build_snapshot(msg)
        if self.mq_out is None:
            raise RuntimeError("Output client not initialised. Did you call setup()?")  # pragma: no cover
        self.mq_out.publish(snapshot)
        sensor_count = len(msg.get("sensors", []))
        self.logger.info(
            "Published boundary snapshot (t=%.2fh, boundaries=%d, sensors=%d)",
            snapshot["time_hours"],
            len(snapshot["boundaries"]),
            sensor_count,
        )

    def setup(self) -> None:
        if self.mq_in is None:
            self.mq_in = RabbitMQClient(self.input_config)
        if self.mq_out is None:
            self.mq_out = RabbitMQClient(self.output_config)
        self.logger.info(
            "Dependencies initialised (in=%s, out=%s)",
            self.input_config.queue,
            self.output_config.queue,
        )

    def start(self) -> None:
        if self.mq_in is None or self.mq_out is None:
            self.setup()

        self.logger.info("Listening for sensor updates on %s", self.input_config.queue)
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
        self.logger.info("Shutdown complete")


if __name__ == "__main__":
    BoundaryConditionBuilderServer().start()

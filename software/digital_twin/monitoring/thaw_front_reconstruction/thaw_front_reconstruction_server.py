"""Thaw front reconstruction service for 2D sensor inputs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from software.digital_twin.communication.messaging import RabbitMQClient, RabbitMQConfig, resolve_queue_config
from software.digital_twin.data_access.influx_utils import InfluxConfig, InfluxHelper
from software.utils.logging_setup import get_logger

SENSOR_QUEUE = "permafrost.record.sensors.2d"
THAW_FRONT_QUEUE = "permafrost.record.thaw_front.2d"
SENSOR_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "sensor_message_2d.json"
THAW_FRONT_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "thaw_front_message.json"


class ThawFrontReconstructionServer:
    """Compute thaw front metrics and points from 2D sensor snapshots."""

    def __init__(
        self,
        influx_config: InfluxConfig | None = None,
        input_config: RabbitMQConfig | None = None,
        output_config: RabbitMQConfig | None = None,
        *,
        input_queue: str | None = None,
        output_queue: str | None = None,
        thaw_threshold_c: float = 0.0,
        site_id: str = "default",
    ) -> None:
        self.logger = get_logger("ThawFrontReconstructionServer")
        self.influx_config = influx_config or InfluxConfig()
        self.input_config = resolve_queue_config(
            input_config,
            queue=input_queue or SENSOR_QUEUE,
            schema_path=SENSOR_SCHEMA,
        )
        self.output_config = resolve_queue_config(
            output_config,
            queue=output_queue or THAW_FRONT_QUEUE,
            schema_path=THAW_FRONT_SCHEMA,
        )
        self.thaw_threshold_c = float(thaw_threshold_c)
        self.site_id = site_id

        self.mq_in: Optional[RabbitMQClient] = None
        self.mq_out: Optional[RabbitMQClient] = None
        self.influx: Optional[InfluxHelper] = None

    def _extract_thaw_points(self, sensors: list[dict]) -> list[dict]:
        thawed = []
        for sensor in sensors:
            try:
                temperature = float(sensor["temperature"])
                if temperature < self.thaw_threshold_c:
                    continue
                x_m = float(sensor["x_m"])
                z_m = float(sensor["z_m"])
            except (KeyError, TypeError, ValueError):
                continue
            thawed.append({"x_m": x_m, "z_m": z_m})
        return thawed

    def _compute_radius_metrics(self, points: list[dict]) -> tuple[float | None, float | None]:
        if not points:
            return None, None
        radii = [abs(float(point["x_m"])) for point in points]
        return max(radii), sum(radii) / len(radii)

    def _process_message(self, msg: dict) -> None:
        if self.influx is None or self.mq_out is None:
            raise RuntimeError("Dependencies not initialised. Did you call setup()?")  # pragma: no cover

        time_hours = float(msg["time_hours"])
        sensors = msg.get("sensors", [])
        thaw_points = self._extract_thaw_points(sensors)
        radius_max_m, radius_avg_m = self._compute_radius_metrics(thaw_points)

        payload = {
            "timestamp": msg.get("timestamp") or datetime.utcnow().isoformat(),
            "time_hours": time_hours,
            "radius_max_m": radius_max_m,
            "radius_avg_m": radius_avg_m,
            "points": thaw_points,
        }

        self.mq_out.publish(payload)
        self.influx.write_thaw_front_metrics(
            time_hours=time_hours,
            radius_max_m=radius_max_m,
            radius_avg_m=radius_avg_m,
            points=thaw_points,
            site=self.site_id,
        )
        self.logger.info(
            "Thaw front updated (t=%.2fh, points=%d)",
            time_hours,
            len(thaw_points),
        )

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
        if self.influx is not None:
            self.influx.close()
        self.logger.info("Shutdown complete")


if __name__ == "__main__":
    ThawFrontReconstructionServer().start()

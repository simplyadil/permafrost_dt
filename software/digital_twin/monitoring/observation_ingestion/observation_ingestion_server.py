"""2D sensor ingestion server."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from software.digital_twin.communication.messaging import RabbitMQClient, RabbitMQConfig, resolve_queue_config
from software.digital_twin.data_access.influx_utils import InfluxConfig, InfluxHelper
from software.utils.logging_setup import get_logger

SENSOR_QUEUE = "permafrost.record.sensors.2d"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "sensor_message_2d.json"


class ObservationIngestion2DServer:
    """Consumes 2D sensor data and writes it into InfluxDB."""

    def __init__(
        self,
        rabbitmq_config: RabbitMQConfig | None = None,
        influx_config: InfluxConfig | None = None,
        *,
        queue: str | None = None,
        site_id: str = "default",
    ) -> None:
        self.logger = get_logger("ObservationIngestion2DServer")
        self.rabbitmq_config = resolve_queue_config(
            rabbitmq_config,
            queue=queue or SENSOR_QUEUE,
            schema_path=SCHEMA_PATH,
        )
        self.influx_config = influx_config or InfluxConfig()
        self.site_id = site_id
        self.mq_client: Optional[RabbitMQClient] = None
        self.db: Optional[InfluxHelper] = None

    def _process_message(self, msg: dict) -> None:
        if self.db is None:
            raise RuntimeError("Influx helper not initialised. Did you call setup()?")  # pragma: no cover

        try:
            time_hours = float(msg["time_hours"])
            raw_sensors = msg.get("sensors", [])
            if not raw_sensors:
                self.logger.warning("Sensor message contained no readings (t=%.2fh)", time_hours)
                return

            cleaned = []
            for sensor in raw_sensors:
                try:
                    sensor_id = str(sensor["sensor_id"])
                    x_m = float(sensor["x_m"])
                    z_m = float(sensor["z_m"])
                    y_m_raw = sensor.get("y_m")
                    y_m = float(y_m_raw) if y_m_raw is not None else None
                    temperature = float(sensor["temperature"])
                except (KeyError, TypeError, ValueError) as exc:
                    self.logger.warning("Skipping invalid sensor payload: %s", exc)
                    continue

                cleaned.append(
                    {
                        "sensor_id": sensor_id,
                        "x_m": x_m,
                        "y_m": y_m,
                        "z_m": z_m,
                        "temperature": temperature,
                    }
                )

            if not cleaned:
                self.logger.warning("No valid sensor readings found (t=%.2fh)", time_hours)
                return

            self.db.write_sensor_snapshot_2d(time_hours=time_hours, sensors=cleaned, site=self.site_id)
            self.logger.info(
                "Persisted 2D sensor snapshot (t=%.2fh, sensors=%d)",
                time_hours,
                len(cleaned),
            )
        except Exception as exc:
            self.logger.error("Failed to process 2D sensor message: %s", exc, exc_info=True)

    def setup(self) -> None:
        if self.mq_client is None:
            self.mq_client = RabbitMQClient(self.rabbitmq_config)
        if self.db is None:
            self.db = InfluxHelper(self.influx_config)
        self.logger.info("Dependencies initialised (queue=%s)", self.rabbitmq_config.queue)

    def start(self) -> None:
        if self.mq_client is None or self.db is None:
            self.setup()
        self.logger.info("Listening for 2D sensor updates on %s", self.rabbitmq_config.queue)
        try:
            self.mq_client.consume(callback=self._process_message)
        except KeyboardInterrupt:  # pragma: no cover - runtime behaviour
            self.logger.info("Interrupt received; shutting down")
        finally:
            self.close()

    def stop(self) -> None:
        if self.mq_client and self.mq_client.channel:
            self.mq_client.channel.stop_consuming()

    def close(self) -> None:
        self.stop()
        if self.mq_client is not None:
            self.mq_client.disconnect()
        if self.db is not None:
            self.db.close()
        self.logger.info("Shutdown complete")


if __name__ == "__main__":
    ObservationIngestion2DServer().start()

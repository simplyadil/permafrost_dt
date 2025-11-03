"""Sensor ingestion server."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from software.digital_twin.communication.messaging import RabbitMQClient, RabbitMQConfig, resolve_queue_config
from software.digital_twin.data_access.influx_utils import InfluxConfig, InfluxHelper
from software.utils.logging_setup import get_logger

SENSOR_QUEUE = "permafrost.record.sensors.state"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "sensor_message.json"


class ObservationIngestionServer:
    """
    Consumes physical twin sensor data from RabbitMQ
    and writes it into InfluxDB for the digital twin pipeline.
    """

    def __init__(
        self,
        rabbitmq_config: RabbitMQConfig | None = None,
        influx_config: InfluxConfig | None = None,
    ) -> None:
        self.logger = get_logger("ObservationIngestionServer")
        self.rabbitmq_config = resolve_queue_config(
            rabbitmq_config,
            queue=SENSOR_QUEUE,
            schema_path=SCHEMA_PATH,
        )
        self.influx_config = influx_config or InfluxConfig()
        self.mq_client: Optional[RabbitMQClient] = None
        self.db: Optional[InfluxHelper] = None
        self.logger.info("ObservationIngestionServer configured.")
        self._logged_snapshot = False

    # -----------------------------------------------------
    # CALLBACK: when new message arrives
    # -----------------------------------------------------
    def _process_message(self, msg: dict) -> None:
        """
        Processes a validated message:
        - Writes temperature profile by depth into InfluxDB
        """
        if self.db is None:
            raise RuntimeError("Influx helper not initialised. Did you call setup()?")

        try:
            time_days = msg["time_days"]

            sensor_pairs = sorted(
                (
                    float(depth_key.replace("temperature_", "").replace("m", "")),
                    float(temp_value),
                )
                for depth_key, temp_value in msg.items()
                if depth_key.startswith("temperature_")
            )

            if not sensor_pairs:
                self.logger.warning("Sensor message contained no depth readings (t=%.2fd)", time_days)
                return

            depths, temperatures = zip(*sensor_pairs)
            if hasattr(self.db, "write_depth_series"):
                self.db.write_depth_series(
                    measurement="sensor_temperature",
                    time_days=float(time_days),
                    depths=depths,
                    temperatures=temperatures,
                )
            else:  # Fallback for legacy helpers used in tests
                for depth_m, temp in zip(depths, temperatures):
                    self.db.write_temperature(
                        time_days=float(time_days),
                        depth=float(depth_m),
                        temperature=float(temp),
                    )

            if not getattr(self, "_logged_snapshot", False):
                self.logger.info(
                    "Persisted sensor snapshot (t=%.2fd, depths=%d)",
                    float(time_days),
                    len(depths),
                )
                self._logged_snapshot = True
        except Exception as exc:
            self.logger.error("Failed to process sensor message: %s", exc, exc_info=True)

    # -----------------------------------------------------
    # LIFECYCLE
    # -----------------------------------------------------
    def setup(self) -> None:
        """Initialise transport and data access clients."""

        if self.mq_client is None:
            self.mq_client = RabbitMQClient(self.rabbitmq_config)
        if self.db is None:
            self.db = InfluxHelper(self.influx_config)
        self.logger.info("Dependencies initialised (queue=%s)", self.rabbitmq_config.queue)

    def start(self) -> None:
        """Start consuming messages."""

        if self.mq_client is None or self.db is None:
            self.setup()

        self.logger.info("Listening for sensor updates on %s", SENSOR_QUEUE)
        try:
            self.mq_client.consume(callback=self._process_message)
        except KeyboardInterrupt:  # pragma: no cover - runtime behaviour
            self.logger.info("Interrupt received; shutting down")
        finally:
            self.close()

    def stop(self) -> None:
        """Signal the server to stop consuming."""

        if self.mq_client and self.mq_client.channel:
            self.mq_client.channel.stop_consuming()

    def close(self) -> None:
        """Close resources gracefully."""

        self.stop()
        if self.mq_client is not None:
            self.mq_client.disconnect()
        if self.db is not None:
            self.db.close()
        self.logger.info("Shutdown complete")


if __name__ == "__main__":
    ObservationIngestionServer().start()

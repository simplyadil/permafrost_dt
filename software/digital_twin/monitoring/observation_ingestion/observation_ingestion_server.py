"""Sensor ingestion server."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from software.digital_twin.communication.logger import setup_logger
from software.digital_twin.communication.messaging import RabbitMQClient, RabbitMQConfig
from software.digital_twin.data_access.influx_utils import InfluxConfig, InfluxHelper

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
        self.logger = setup_logger("ObservationIngestionServer")
        # Legacy direct fallback without schema reconciliation:
        # base_config = rabbitmq_config or RabbitMQConfig(schema_path=SCHEMA_PATH)
        if rabbitmq_config is not None and rabbitmq_config.schema_path is None:
            rabbitmq_config = RabbitMQConfig(
                host=rabbitmq_config.host,
                queue=rabbitmq_config.queue,
                schema_path=SCHEMA_PATH,
                username=rabbitmq_config.username,
                password=rabbitmq_config.password,
            )
        base_config = rabbitmq_config or RabbitMQConfig(schema_path=SCHEMA_PATH)
        self.rabbitmq_config = base_config.with_queue(SENSOR_QUEUE)
        self.influx_config = influx_config or InfluxConfig()
        self.mq_client: Optional[RabbitMQClient] = None
        self.db: Optional[InfluxHelper] = None
        self._running = False
        self.logger.info("ObservationIngestionServer configured.")

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

            # Write each depth reading as an individual point
            for depth_key, temp_value in msg.items():
                if depth_key.startswith("temperature_"):
                    depth_m = float(depth_key.replace("temperature_", "").replace("m", ""))
                    self.db.write_temperature(
                        time_days=time_days,
                        depth=depth_m,
                        temperature=temp_value,
                    )

            self.logger.info("Processed sensor message at t=%s days.", time_days)
        except Exception as exc:
            self.logger.error("Error processing message: %s", exc, exc_info=True)

    # -----------------------------------------------------
    # LIFECYCLE
    # -----------------------------------------------------
    def setup(self) -> None:
        """Initialise transport and data access clients."""

        if self.mq_client is None:
            self.mq_client = RabbitMQClient(self.rabbitmq_config)
        if self.db is None:
            self.db = InfluxHelper(self.influx_config)
        self._running = True
        self.logger.info("ObservationIngestionServer setup complete.")

    def start(self) -> None:
        """Start consuming messages."""

        if self.mq_client is None or self.db is None:
            self.setup()

        self.logger.info("ObservationIngestionServer listening on %s", SENSOR_QUEUE)
        try:
            self.mq_client.consume(callback=self._process_message)
        except KeyboardInterrupt:  # pragma: no cover - runtime behaviour
            self.logger.info("ObservationIngestionServer interrupted. Shutting down...")
        finally:
            self.close()

    def stop(self) -> None:
        """Signal the server to stop consuming."""

        self._running = False
        if self.mq_client and self.mq_client.channel:
            self.mq_client.channel.stop_consuming()

    def close(self) -> None:
        """Close resources gracefully."""

        self.stop()
        if self.mq_client is not None:
            self.mq_client.disconnect()
        if self.db is not None:
            self.db.close()
        self.logger.info("ObservationIngestionServer shutdown complete.")


if __name__ == "__main__":
    ObservationIngestionServer().start()

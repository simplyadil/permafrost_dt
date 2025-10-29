from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from software.digital_twin.communication.logger import setup_logger
from software.digital_twin.communication.messaging import RabbitMQClient, RabbitMQConfig
# Legacy import without access to depth parser kept for reference:
# from software.digital_twin.data_access.influx_utils import InfluxConfig, InfluxHelper
from software.digital_twin.data_access.influx_utils import (
    InfluxConfig,
    InfluxHelper,
    _parse_depth_value,
)


SURFACE_DEPTH_METERS = 0.0
BOUNDARY_QUEUE = "permafrost.record.boundary.state"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "boundary_forcing_message.json"


@dataclass(frozen=True)
class BoundaryMessage:
    timestamp: str
    time_days: float
    Ts: float


class BoundaryForcingServer:
    """Extract boundary forcing signals from InfluxDB observations."""

    def __init__(
        self,
        polling_interval: int = 5,
        rabbitmq_config: RabbitMQConfig | None = None,
        influx_config: InfluxConfig | None = None,
    ) -> None:
        self.logger = setup_logger("BoundaryForcingServer")
        self.polling_interval = polling_interval
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
        self.rabbitmq_config = base_config.with_queue(BOUNDARY_QUEUE)
        self.influx_config = influx_config or InfluxConfig()
        self.mq_client: Optional[RabbitMQClient] = None
        self.influx: Optional[InfluxHelper] = None
        self._running = False
        self.logger.info("BoundaryForcingServer initialised with polling interval %ss.", polling_interval)

    def _latest_surface_sample(self, df: pd.DataFrame) -> Optional[pd.Series]:
        if df.empty:
            return None

        working = df.copy()
        if "depth_m" not in working.columns and "depth" in working.columns:
            working["depth_m"] = working["depth"].apply(_parse_depth_value)
        elif "depth_m" in working.columns:
            working["depth_m"] = working["depth_m"].apply(_parse_depth_value)

        if "depth_m" not in working.columns:
            self.logger.warning("Depth data unavailable in Influx row.")
            return None

        surface_rows = working[working["depth_m"] == SURFACE_DEPTH_METERS]
        if surface_rows.empty:
            self.logger.warning("No surface temperature data found.")
            return None

        return surface_rows[["time_days", "temperature"]].iloc[[-1]].squeeze()

    def compute_boundary_forcing(self, df: pd.DataFrame) -> Optional[BoundaryMessage]:
        """Return the most recent surface temperature sample as a message payload."""

        latest_sample = self._latest_surface_sample(df)
        if latest_sample is None:
            return None

        message = BoundaryMessage(
            timestamp=datetime.utcnow().isoformat(),
            time_days=float(latest_sample["time_days"]),
            Ts=float(latest_sample["temperature"]),
        )
        return message

    def publish_boundary_forcing(self, message: BoundaryMessage) -> None:
        payload = {
            "timestamp": message.timestamp,
            "time_days": message.time_days,
            "Ts": message.Ts,
        }
        if self.mq_client is None:
            raise RuntimeError("RabbitMQ client not initialised. Did you call setup()?")
        self.mq_client.publish(payload)
        self.logger.info(
            "Published boundary forcing: Ts=%s°C @ t=%s",
            message.Ts,
            message.time_days,
        )

    def run_once(self) -> Optional[BoundaryMessage]:
        """Execute a single polling iteration."""

        if self.influx is None:
            raise RuntimeError("Influx client not initialised. Did you call setup()?")

        dataframe = self.influx.query_temperature(limit=200)
        message = self.compute_boundary_forcing(dataframe)
        if message is not None:
            self.publish_boundary_forcing(message)
        return message

    def setup(self) -> None:
        """Connect to RabbitMQ and InfluxDB."""

        if self.mq_client is None:
            self.mq_client = RabbitMQClient(self.rabbitmq_config)
        if self.influx is None:
            self.influx = InfluxHelper(self.influx_config)
        self._running = True
        self.logger.info("BoundaryForcingServer setup complete.")

    def start(self) -> None:
        """Begin polling loop until stopped."""

        if self.mq_client is None or self.influx is None:
            self.setup()

        self.logger.info("BoundaryForcingServer polling for boundary data...")
        try:
            while self._running:
                try:
                    self.run_once()
                except Exception as exc:  # pragma: no cover - integration behaviour
                    self.logger.error("Error during boundary forcing computation: %s", exc)
                time.sleep(self.polling_interval)
        except KeyboardInterrupt:  # pragma: no cover - runtime behaviour
            self.logger.info("BoundaryForcingServer interrupted. Shutting down...")
        finally:
            self.close()

    def stop(self) -> None:
        """Stop the polling loop without closing transports."""

        self._running = False

    def close(self) -> None:
        """Gracefully close external connections."""

        self.stop()
        if self.mq_client is not None:
            self.mq_client.disconnect()
        if self.influx is not None:
            self.influx.close()
        self.logger.info("BoundaryForcingServer shutdown complete.")


if __name__ == "__main__":
    server = BoundaryForcingServer(polling_interval=10)
    server.start()

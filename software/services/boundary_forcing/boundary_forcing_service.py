# software/services/boundary_forcing/boundary_forcing_service.py
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from software.services.common.influx_utils import InfluxHelper
from software.services.common.logger import setup_logger
from software.services.common.messaging import RabbitMQClient


SURFACE_DEPTH_METERS = 0.0


@dataclass(frozen=True)
class BoundaryMessage:
    timestamp: str
    time_days: float
    Ts: float


class BoundaryForcingService:
    """Extract boundary forcing signals from InfluxDB observations."""

    def __init__(self, polling_interval: int = 5) -> None:
        self.logger = setup_logger("boundary_forcing_service")
        self.polling_interval = polling_interval
        self.mq_client = RabbitMQClient(
            queue="permafrost.boundary.forcing",
            schema_path="software/services/common/schemas/boundary_forcing_message.json",
        )
        self.influx = InfluxHelper()
        self.logger.info("boundary_forcing_service initialized.")

    def _latest_surface_sample(self, df: pd.DataFrame) -> Optional[pd.Series]:
        if df.empty:
            return None

        surface_rows = df[df["depth"] == SURFACE_DEPTH_METERS]
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
        self.mq_client.publish(payload)
        self.logger.info(
            "Published boundary forcing: Ts=%s°C @ t=%s",
            message.Ts,
            message.time_days,
        )

    def run_once(self) -> Optional[BoundaryMessage]:
        """Execute a single polling iteration."""

        dataframe = self.influx.query_temperature(limit=200)
        message = self.compute_boundary_forcing(dataframe)
        if message is not None:
            self.publish_boundary_forcing(message)
        return message

    def run(self) -> None:
        self.logger.info("boundary_forcing_service started polling for new data...")
        while True:
            try:
                self.run_once()
            except Exception as exc:  # pragma: no cover - integration behaviour
                self.logger.error("Error during boundary forcing computation: %s", exc)
            time.sleep(self.polling_interval)


if __name__ == "__main__":
    service = BoundaryForcingService(polling_interval=10)
    service.run()

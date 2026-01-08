from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from software.digital_twin.communication.messaging import RabbitMQClient, RabbitMQConfig, resolve_queue_config
from software.digital_twin.data_access.influx_utils import (
    InfluxConfig,
    InfluxHelper,
    _parse_depth_value,
)
from software.utils.logging_setup import get_logger


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
        *,
        synthetic_enabled: bool = False,
        synthetic_start_day: float = 0.0,
        synthetic_step_days: float = 1.0,
    ) -> None:
        self.logger = get_logger("BoundaryForcingServer")
        self.polling_interval = polling_interval
        self.rabbitmq_config = resolve_queue_config(
            rabbitmq_config,
            queue=BOUNDARY_QUEUE,
            schema_path=SCHEMA_PATH,
        )
        self.influx_config = influx_config or InfluxConfig()
        self.mq_client: Optional[RabbitMQClient] = None
        self.influx: Optional[InfluxHelper] = None
        self._running = False
        self._synthetic_enabled = synthetic_enabled
        self._synthetic_time_days = synthetic_start_day
        self._synthetic_step_days = synthetic_step_days
        self._last_published_time_days: Optional[float] = None
        self.logger.info(
            "Configured (poll_interval=%ss, synthetic=%s)",
            polling_interval,
            "on" if synthetic_enabled else "off",
        )
        self._logged_publish = False

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
        if self._last_published_time_days is not None and message.time_days <= self._last_published_time_days:
            self.logger.warning(
                "Skipping boundary forcing (non-increasing time_days %.2f <= %.2f)",
                message.time_days,
                self._last_published_time_days,
            )
            return None
        if self._synthetic_enabled:
            self._synthetic_time_days = float(message.time_days) + self._synthetic_step_days
        return message

    @staticmethod
    def _sinusoid_air_temp(t_days: float) -> float:
        """Seasonal surface temperature approximation reused for synthetic forcing."""
        return 4.03 + 16.11 * math.sin((2 * math.pi * t_days / 365.0) - 1.709)

    def _generate_synthetic_boundary(self) -> Optional[BoundaryMessage]:
        if not self._synthetic_enabled:
            return None

        t_day = self._synthetic_time_days
        if self._last_published_time_days is not None and t_day <= self._last_published_time_days:
            t_day = self._last_published_time_days + self._synthetic_step_days
            self._synthetic_time_days = t_day
        Ts = self._sinusoid_air_temp(t_day)
        self._synthetic_time_days += self._synthetic_step_days
        message = BoundaryMessage(
            timestamp=datetime.utcnow().isoformat(),
            time_days=float(t_day),
            Ts=float(Ts),
        )
        self.logger.debug(
            "Generated synthetic boundary forcing (t=%.2fd, Ts=%.2f°C)",
            message.time_days,
            message.Ts,
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
        self._last_published_time_days = message.time_days
        if not getattr(self, "_logged_publish", False):
            queue_name = getattr(self.rabbitmq_config, "queue", "unknown")
            self.logger.info(
                "Published boundary conditions (t=%.2fd, Ts=%.2f°C) to %s",
                message.time_days,
                message.Ts,
                queue_name,
            )
            self._logged_publish = True

    def run_once(self) -> Optional[BoundaryMessage]:
        """Execute a single polling iteration."""

        if self._synthetic_enabled:
            message = self._generate_synthetic_boundary()
            if message is not None:
                self.publish_boundary_forcing(message)
            else:
                self.logger.debug("Synthetic boundary forcing unavailable.")
            return message

        if self.influx is None:
            raise RuntimeError("Influx client not initialised. Did you call setup()?")  # pragma: no cover

        dataframe = self.influx.query_temperature(limit=200)
        message = self.compute_boundary_forcing(dataframe)
        if message is not None:
            self.publish_boundary_forcing(message)
        else:
            self.logger.debug("No boundary forcing generated this cycle.")
        return message

    def setup(self) -> None:
        """Connect to RabbitMQ and InfluxDB."""

        if self.mq_client is None:
            self.mq_client = RabbitMQClient(self.rabbitmq_config)
        if self.influx is None:
            self.influx = InfluxHelper(self.influx_config)
        self._running = True
        queue_name = getattr(self.rabbitmq_config, "queue", "unknown")
        self.logger.info("Dependencies initialised (queue=%s)", queue_name)

    def start(self) -> None:
        """Begin polling loop until stopped."""

        if self.mq_client is None or self.influx is None:
            self.setup()

        self.logger.info("Polling boundary data every %ss", self.polling_interval)
        try:
            while self._running:
                try:
                    self.run_once()
                except Exception as exc:  # pragma: no cover - integration behaviour
                    self.logger.error("Boundary forcing computation failed: %s", exc, exc_info=True)
                time.sleep(self.polling_interval)
        except KeyboardInterrupt:  # pragma: no cover - runtime behaviour
            self.logger.info("Interrupt received; shutting down")
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
        self.logger.info("Shutdown complete")


if __name__ == "__main__":
    server = BoundaryForcingServer(polling_interval=10)
    server.start()

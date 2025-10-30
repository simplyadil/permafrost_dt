"""Integration-style checks for the BoundaryForcingServer."""

from __future__ import annotations

import logging

import pandas as pd

from software.digital_twin.monitoring.boundary_forcing.boundary_forcing_server import (
    BoundaryForcingServer,
    BoundaryMessage,
)


logger = logging.getLogger(__name__)


class _InfluxStub:
    """Captures boundary queries without relying on an external InfluxDB."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self._dataframe = dataframe

    def query_temperature(self, limit: int = 200) -> pd.DataFrame:  # pragma: no cover - trivial
        return self._dataframe

    def close(self) -> None:  # pragma: no cover - unused cleanup
        pass


class _MQStub:
    """Collects published payloads for assertions."""

    def __init__(self) -> None:
        self.messages: list[dict] = []

    def publish(self, payload: dict) -> None:
        self.messages.append(payload)


def _make_server(dataframe: pd.DataFrame) -> BoundaryForcingServer:
    """Instantiate a server without triggering external IO."""

    server = BoundaryForcingServer.__new__(BoundaryForcingServer)  # type: ignore[call-arg]
    server.logger = logging.getLogger("BoundaryForcingServerTest")
    server.polling_interval = 1
    server.rabbitmq_config = None  # type: ignore[attr-defined]
    server.influx_config = None  # type: ignore[attr-defined]
    server.mq_client = _MQStub()
    server.influx = _InfluxStub(dataframe)
    server._running = False  # type: ignore[attr-defined]
    server._synthetic_enabled = False  # type: ignore[attr-defined]
    server._synthetic_time_days = 0.0  # type: ignore[attr-defined]
    server._synthetic_step_days = 1.0  # type: ignore[attr-defined]
    return server


def test_compute_boundary_message_uses_latest_surface_sample() -> None:
    df = pd.DataFrame(
        {
            "depth_m": [0.0, 1.0, 2.0],
            "time_days": [1.0, 1.0, 1.0],
            "temperature": [-5.0, -4.0, -3.0],
        }
    )

    server = _make_server(df)
    result = server.compute_boundary_forcing(df)

    assert isinstance(result, BoundaryMessage), "Expected a BoundaryMessage instance."
    assert result.time_days == 1.0, "Surface sample should use latest time_days."
    assert result.Ts == -5.0, "Surface temperature should match depth 0.0."
    logger.info("Boundary sample computed successfully: %s", result)
    logger.info("test_compute_boundary_message_uses_latest_surface_sample passed successfully.")


def test_run_once_publishes_surface_temperature() -> None:
    df = pd.DataFrame(
        {
            "depth_m": [0.0, 1.0, 2.0],
            "time_days": [2.0, 2.0, 2.0],
            "temperature": [-6.0, -4.5, -3.0],
        }
    )

    server = _make_server(df)
    message = server.run_once()

    assert isinstance(message, BoundaryMessage), "run_once should return a BoundaryMessage."
    assert server.mq_client.messages, "Expected a boundary message to be published."
    payload = server.mq_client.messages[0]
    assert payload["Ts"] == message.Ts, "Published payload Ts mismatch."
    assert payload["time_days"] == message.time_days, "Published payload time mismatch."
    logger.info("Boundary forcing published payload: %s", payload)
    logger.info("test_run_once_publishes_surface_temperature passed successfully.")


def test_run_once_generates_synthetic_boundary_when_no_data() -> None:
    df = pd.DataFrame()

    server = _make_server(df)
    server._synthetic_enabled = True  # type: ignore[attr-defined]
    server._synthetic_time_days = 2.5  # type: ignore[attr-defined]
    server._synthetic_step_days = 0.5  # type: ignore[attr-defined]

    message = server.run_once()

    assert isinstance(message, BoundaryMessage), "Synthetic fallback should yield a BoundaryMessage."
    assert server.mq_client.messages, "Expected a synthetic boundary message to be published."

    expected_ts = BoundaryForcingServer._sinusoid_air_temp(2.5)
    assert message.Ts == expected_ts
    assert message.time_days == 2.5
    assert server._synthetic_time_days == 3.0  # type: ignore[attr-defined]
    logger.info("Synthetic boundary forcing payload: %s", server.mq_client.messages[0])
    logger.info("test_run_once_generates_synthetic_boundary_when_no_data passed successfully.")

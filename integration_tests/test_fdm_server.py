"""Integration-style tests for the FDMServer boundary handling."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

import software.digital_twin.simulator.fdm.fdm_server as fdm_module
from software.digital_twin.simulator.fdm.fdm_server import FDMServer


logger = logging.getLogger(__name__)


@dataclass
class _ModelWrite:
    measurement: str
    time_days: float
    depth: float
    temperature: float


class _InfluxStub:
    """Captures model writes for assertion."""

    def __init__(self) -> None:
        self.writes: list[_ModelWrite] = []

    def write_model_temperature(
        self,
        *,
        measurement: str,
        time_days: float,
        depth: float,
        temperature: float,
        site: str = "default",
        extra_tags: dict | None = None,
    ) -> None:
        self.writes.append(_ModelWrite(measurement, time_days, depth, temperature))

    def close(self) -> None:  # pragma: no cover - unused cleanup
        pass


class _PublisherStub:
    """Captures RabbitMQ messages."""

    def __init__(self) -> None:
        self.messages: list[dict] = []

    def publish(self, payload: dict) -> None:
        self.messages.append(payload)


def _make_server() -> tuple[FDMServer, _InfluxStub, _PublisherStub]:
    """Instantiate a server without touching external dependencies."""

    influx = _InfluxStub()
    publisher = _PublisherStub()

    server = FDMServer.__new__(FDMServer)  # type: ignore[call-arg]
    server.logger = logging.getLogger("FDMServerTest")
    server.influx = influx
    server.mq_out = _PublisherStub()  # Separate publisher for control messages
    server.mq_sensor = publisher  # Use provided publisher for sensor messages
    server.phys = fdm_module.PhysicsParams()
    server.grid = fdm_module.GridParams()
    server.x = np.linspace(0.0, server.grid.Lx, server.grid.Nx)
    server.dx = server.x[1] - server.x[0]
    server.current_time_days = None
    server.T = None
    server.theta_prev = None
    server.bottom_bc_temp = 1.0
    server.mq_in = None  # type: ignore[attr-defined]
    server._running = False  # type: ignore[attr-defined]
    server.sensor_depths = (0, 1, 2, 3, 4, 5)
    return server, influx, publisher


def test_on_boundary_forcing_initialises_state_and_notifies() -> None:
    server, influx, publisher = _make_server()

    message = {"time_days": 1.0, "Ts": -5.0}

    server._on_boundary_forcing(message)  # type: ignore[attr-defined]

    assert len(influx.writes) == server.grid.Nx, "Expected one write per grid depth."
    assert publisher.messages and publisher.messages[0]["status"] == "ready", "Ready notification missing."
    assert server.current_time_days == 1.0, "Server time should advance to message time."
    assert isinstance(server.T, np.ndarray), "Temperature profile should be initialised."
    logger.info("FDM wrote %d depth points at t=%s.", len(influx.writes), server.current_time_days)
    logger.info("test_on_boundary_forcing_initialises_state_and_notifies passed successfully.")

"""Integration-style checks for the ObservationIngestionServer."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from software.digital_twin.monitoring.observation_ingestion.observation_ingestion_server import (
    ObservationIngestionServer,
)

import os
import sys

# Ensure repository root is on sys.path so `software` package imports work when running this test directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logger = logging.getLogger(__name__)


@dataclass
class _WriteRecord:
    time_days: float
    depth: float
    temperature: float


class _InfluxRecorder:
    """Captures writes for assertion without needing a live InfluxDB instance."""

    def __init__(self) -> None:
        self.records: list[_WriteRecord] = []

    def write_temperature(self, *, time_days: float, depth: float, temperature: float) -> None:
        self.records.append(_WriteRecord(time_days, depth, temperature))


def _make_server() -> ObservationIngestionServer:
    server = ObservationIngestionServer.__new__(ObservationIngestionServer)  # type: ignore[call-arg]
    server.logger = logging.getLogger("ObservationIngestionServerTest")
    server.mq_client = None  # type: ignore[attr-defined]
    server.db = _InfluxRecorder()
    return server


def test_process_message_persists_each_depth_reading() -> None:
    server = _make_server()

    payload = {
        "time_days": 0.5,
        "temperature_0m": -10.1,
        "temperature_1m": -8.3,
    }

    server._process_message(payload)

    assert len(server.db.records) == 2, "Expected two depth samples to be recorded."
    assert server.db.records[0] == _WriteRecord(0.5, 0.0, -10.1), "Surface record mismatch."
    assert server.db.records[1] == _WriteRecord(0.5, 1.0, -8.3), "Deeper record mismatch."
    logger.info("Observation ingestion persisted records: %s", server.db.records)
    logger.info("test_process_message_persists_each_depth_reading passed successfully.")

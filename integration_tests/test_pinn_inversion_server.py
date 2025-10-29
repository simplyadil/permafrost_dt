"""Integration-style checks for PINNInversionServer orchestration."""

from __future__ import annotations

import logging

import software.digital_twin.simulator.pinn_inversion.pinn_inversion_server as pinn_inversion_module


logger = logging.getLogger(__name__)


class _ImmediateThread:
    def __init__(self, target, daemon=True) -> None:  # pragma: no cover - trivial wrapper
        self._target = target

    def start(self) -> None:
        self._target()


def test_on_message_spawns_inversion_worker() -> None:
    server = pinn_inversion_module.PINNInversionServer.__new__(pinn_inversion_module.PINNInversionServer)  # type: ignore[call-arg]
    server.logger = logging.getLogger("PINNInversionServerTest")
    server._running = True  # type: ignore[attr-defined]
    server.mq_client = None  # type: ignore[attr-defined]
    server.out_publisher = None  # type: ignore[attr-defined]
    tasks: list[str] = []

    def fake_run() -> None:
        tasks.append("inversion")

    server.run_inversion = fake_run  # type: ignore[assignment]

    original_thread = pinn_inversion_module.Thread
    pinn_inversion_module.Thread = _ImmediateThread  # type: ignore[assignment]
    try:
        server._on_message({"status": "trained"})
    finally:
        pinn_inversion_module.Thread = original_thread  # type: ignore[assignment]

    assert tasks == ["inversion"], "Expected inversion worker to execute exactly once."
    logger.info("PINN inversion worker executed successfully.")
    logger.info("test_on_message_spawns_inversion_worker passed successfully.")

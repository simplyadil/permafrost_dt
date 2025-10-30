"""Integration-style checks for the PINNForwardServer orchestration."""

from __future__ import annotations

import logging

import software.digital_twin.simulator.pinn_forward.pinn_forward_server as pinn_forward_module


logger = logging.getLogger(__name__)


class _ThreadStub:
    """Executes the target synchronously for deterministic testing."""

    def __init__(self, target, daemon=True) -> None:  # pragma: no cover - trivial wrapper
        self._target = target

    def start(self) -> None:
        self._target()


def test_on_message_with_ready_triggers_training_thread() -> None:
    server = pinn_forward_module.PINNForwardServer.__new__(pinn_forward_module.PINNForwardServer)  # type: ignore[call-arg]
    server.logger = logging.getLogger("PINNForwardServerTest")
    server._running = True  # type: ignore[attr-defined]
    server.mq_client = None  # type: ignore[attr-defined]
    server.influx = None  # type: ignore[attr-defined]
    server.enable_training = True  # Add the missing attribute
    writes: list[str] = []

    def fake_train() -> None:
        writes.append("trained")

    server.train_from_influx = fake_train  # type: ignore[assignment]

    original_thread = pinn_forward_module.Thread
    pinn_forward_module.Thread = _ThreadStub  # type: ignore[assignment]
    try:
        server._on_message({"status": "ready"})
    finally:
        pinn_forward_module.Thread = original_thread  # type: ignore[assignment]

    assert writes == ["trained"], "Expected the training routine to be invoked."
    logger.info("PINN forward training stub executed successfully.")
    logger.info("test_on_message_with_ready_triggers_training_thread passed successfully.")

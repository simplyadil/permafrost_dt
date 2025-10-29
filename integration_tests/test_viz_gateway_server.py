"""Integration-style checks for the VizGatewayServer."""

from __future__ import annotations

import logging

import software.digital_twin.visualization.viz_gateway.viz_gateway_server as viz_module


logger = logging.getLogger(__name__)


class _InlineThread:
    def __init__(self, target, args=(), daemon=True) -> None:  # pragma: no cover - trivial wrapper
        self._target = target
        self._args = args

    def start(self) -> None:
        self._target(*self._args)


def test_on_message_triggers_aggregation() -> None:
    server = viz_module.VizGatewayServer.__new__(viz_module.VizGatewayServer)  # type: ignore[call-arg]
    server.logger = logging.getLogger("VizGatewayServerTest")
    server.influx = None  # type: ignore[attr-defined]
    server.mq_client = None  # type: ignore[attr-defined]
    server.out_publisher = None  # type: ignore[attr-defined]
    calls: list[dict] = []

    def fake_aggregate(msg: dict | None = None) -> None:
        calls.append(msg or {})

    server.aggregate_and_publish = fake_aggregate  # type: ignore[assignment]

    original_thread = viz_module.Thread
    viz_module.Thread = _InlineThread  # type: ignore[assignment]
    try:
        payload = {"status": "inverted", "parameters": {"lambda_f": 1.5}}
        server._on_message(payload)
    finally:
        viz_module.Thread = original_thread  # type: ignore[assignment]

    assert calls == [payload], "Visualization aggregation should be invoked once with payload."
    logger.info("Viz gateway aggregation triggered with payload: %s", payload)
    logger.info("test_on_message_triggers_aggregation passed successfully.")

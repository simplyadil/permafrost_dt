"""Heater actuation service for the 2D thaw-front digital twin."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from software.digital_twin.communication.messaging import RabbitMQClient, RabbitMQConfig, resolve_queue_config
from software.digital_twin.data_access.influx_utils import InfluxConfig, InfluxHelper
from software.utils.logging_setup import get_logger

ALERT_QUEUE = "permafrost.record.safety_alerts.2d"
COMMAND_QUEUE = "permafrost.command.heater.2d"
ACTION_QUEUE = "permafrost.record.heater_actions.2d"

ALERT_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "safety_alert_message.json"
COMMAND_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "heater_command_message.json"
ACTION_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "heater_action_message.json"


class HeaterActuationServer:
    """Translate safety alerts into heater actuation commands."""

    def __init__(
        self,
        influx_config: InfluxConfig | None = None,
        input_config: RabbitMQConfig | None = None,
        command_config: RabbitMQConfig | None = None,
        action_config: RabbitMQConfig | None = None,
        *,
        input_queue: str | None = None,
        command_queue: str | None = None,
        action_queue: str | None = None,
        action_on_alert: str = "stop",
        action_on_ok: str = "hold",
        setpoint_on_alert: float | None = 0.0,
        setpoint_on_ok: float | None = None,
        publish_on_change: bool = False,
        site_id: str = "default",
    ) -> None:
        self.logger = get_logger("HeaterActuationServer")
        self.influx_config = influx_config or InfluxConfig()
        self.input_config = resolve_queue_config(
            input_config,
            queue=input_queue or ALERT_QUEUE,
            schema_path=ALERT_SCHEMA,
        )
        self.command_config = resolve_queue_config(
            command_config,
            queue=command_queue or COMMAND_QUEUE,
            schema_path=COMMAND_SCHEMA,
        )
        self.action_config = resolve_queue_config(
            action_config,
            queue=action_queue or ACTION_QUEUE,
            schema_path=ACTION_SCHEMA,
        )
        self.action_on_alert = action_on_alert
        self.action_on_ok = action_on_ok
        self.setpoint_on_alert = self._coerce_setpoint(setpoint_on_alert)
        self.setpoint_on_ok = self._coerce_setpoint(setpoint_on_ok)
        self.publish_on_change = publish_on_change
        self.site_id = site_id

        self.mq_in: Optional[RabbitMQClient] = None
        self.mq_command: Optional[RabbitMQClient] = None
        self.mq_action: Optional[RabbitMQClient] = None
        self.influx: Optional[InfluxHelper] = None
        self._last_action: Optional[str] = None
        self._last_setpoint: Optional[float | None] = None

    @staticmethod
    def _coerce_setpoint(value: float | None) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _build_payload(self, msg: dict) -> dict:
        time_hours = float(msg["time_hours"])
        triggered = bool(msg.get("triggered"))
        reason = str(msg.get("reason", "unknown"))
        action = self.action_on_alert if triggered else self.action_on_ok
        setpoint = self.setpoint_on_alert if triggered else self.setpoint_on_ok

        return {
            "timestamp": msg.get("timestamp") or datetime.utcnow().isoformat(),
            "time_hours": time_hours,
            "action": action,
            "setpoint": setpoint,
            "triggered": triggered,
            "status": "issued",
            "source": "safety_monitor",
            "reason": reason,
        }

    def _should_publish(self, action: str, setpoint: float | None) -> bool:
        if not self.publish_on_change:
            return True
        if self._last_action is None:
            return True
        return action != self._last_action or setpoint != self._last_setpoint

    def _process_message(self, msg: dict) -> None:
        if self.mq_command is None or self.mq_action is None or self.influx is None:
            raise RuntimeError("Dependencies not initialised. Did you call setup()?")  # pragma: no cover

        payload = self._build_payload(msg)
        action = payload["action"]
        setpoint = payload["setpoint"]
        if not self._should_publish(action, setpoint):
            self.logger.info(
                "Heater action unchanged (t=%.2fh, action=%s); skipping publish",
                payload["time_hours"],
                action,
            )
            return

        self.mq_command.publish(payload)
        self.mq_action.publish(payload)
        self.influx.write_heater_action(
            time_hours=payload["time_hours"],
            action=action,
            setpoint=setpoint,
            triggered=payload["triggered"],
            reason=payload["reason"],
            source=payload["source"],
            site=self.site_id,
        )
        self._last_action = action
        self._last_setpoint = setpoint

        self.logger.info(
            "Heater command issued (t=%.2fh, action=%s, setpoint=%s)",
            payload["time_hours"],
            action,
            "n/a" if setpoint is None else f"{float(setpoint):.3f}",
        )

    def setup(self) -> None:
        if self.mq_in is None:
            self.mq_in = RabbitMQClient(self.input_config)
        if self.mq_command is None:
            self.mq_command = RabbitMQClient(self.command_config)
        if self.mq_action is None:
            self.mq_action = RabbitMQClient(self.action_config)
        if self.influx is None:
            self.influx = InfluxHelper(self.influx_config)
        self.logger.info(
            "Dependencies initialised (in=%s, command=%s, action=%s)",
            self.input_config.queue,
            self.command_config.queue,
            self.action_config.queue,
        )
        self.logger.info(
            "Heater policy (alert=%s, ok=%s)",
            self.action_on_alert,
            self.action_on_ok,
        )

    def start(self) -> None:
        if self.mq_in is None or self.mq_command is None or self.mq_action is None or self.influx is None:
            self.setup()

        self.logger.info("Listening for safety alerts on %s", self.input_config.queue)
        try:
            self.mq_in.consume(callback=self._process_message)
        except KeyboardInterrupt:  # pragma: no cover - runtime behaviour
            self.logger.info("Interrupt received; shutting down")
        finally:
            self.close()

    def stop(self) -> None:
        if self.mq_in and self.mq_in.channel:
            self.mq_in.channel.stop_consuming()

    def close(self) -> None:
        self.stop()
        if self.mq_in is not None:
            self.mq_in.disconnect()
        if self.mq_command is not None:
            self.mq_command.disconnect()
        if self.mq_action is not None:
            self.mq_action.disconnect()
        if self.influx is not None:
            self.influx.close()
        self.logger.info("Shutdown complete")


if __name__ == "__main__":
    HeaterActuationServer().start()

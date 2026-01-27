"""Safety monitoring service for the 2D thaw-front digital twin."""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from software.digital_twin.communication.messaging import RabbitMQClient, RabbitMQConfig, resolve_queue_config
from software.digital_twin.data_access.influx_utils import InfluxConfig, InfluxHelper
from software.utils.logging_setup import get_logger

THAW_FRONT_QUEUE = "permafrost.record.thaw_front.2d"
FORECAST_QUEUE = "permafrost.record.fem_forecast.2d"
ALERT_QUEUE = "permafrost.record.safety_alerts.2d"

THAW_FRONT_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "thaw_front_message.json"
FORECAST_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "fem_forecast_message.json"
ALERT_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "safety_alert_message.json"

ALLOWED_METRICS = {"max", "avg"}


class SafetyMonitorServer:
    """Evaluate thaw-front metrics against a safety limit and emit alerts."""

    def __init__(
        self,
        influx_config: InfluxConfig | None = None,
        thaw_input_config: RabbitMQConfig | None = None,
        forecast_input_config: RabbitMQConfig | None = None,
        output_config: RabbitMQConfig | None = None,
        *,
        thaw_input_queue: str | None = None,
        forecast_input_queue: str | None = None,
        output_queue: str | None = None,
        limit_radius_m: float = 0.1,
        metric: str = "max",
        publish_on_change: bool = False,
        site_id: str = "default",
    ) -> None:
        self.logger = get_logger("SafetyMonitorServer")
        self.influx_config = influx_config or InfluxConfig()
        self.thaw_input_config = resolve_queue_config(
            thaw_input_config,
            queue=thaw_input_queue or THAW_FRONT_QUEUE,
            schema_path=THAW_FRONT_SCHEMA,
        )
        self.forecast_input_config = resolve_queue_config(
            forecast_input_config,
            queue=forecast_input_queue or FORECAST_QUEUE,
            schema_path=FORECAST_SCHEMA,
        )
        self.output_config = resolve_queue_config(
            output_config,
            queue=output_queue or ALERT_QUEUE,
            schema_path=ALERT_SCHEMA,
        )
        metric = metric.lower().strip()
        if metric not in ALLOWED_METRICS:
            raise ValueError(f"Unsupported metric '{metric}'. Allowed: {sorted(ALLOWED_METRICS)}")

        self.metric = metric
        self.limit_radius_m = float(limit_radius_m)
        self.publish_on_change = bool(publish_on_change)
        self.site_id = site_id

        self._lock = threading.Lock()
        self._last_thaw: Optional[dict] = None
        self._last_forecast: Optional[dict] = None
        self._last_triggered: Optional[bool] = None
        self._threads: list[threading.Thread] = []

        self.mq_thaw: Optional[RabbitMQClient] = None
        self.mq_forecast: Optional[RabbitMQClient] = None
        self.mq_out: Optional[RabbitMQClient] = None
        self.influx: Optional[InfluxHelper] = None

    def _metric_field(self) -> str:
        return "radius_max_m" if self.metric == "max" else "radius_avg_m"

    def _parse_metric(self, msg: dict, *, source: str) -> dict:
        try:
            time_hours = float(msg["time_hours"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Missing time_hours in {source} message.")

        metric_field = self._metric_field()
        raw_value = msg.get(metric_field)
        radius_m = None if raw_value is None else float(raw_value)
        timestamp = msg.get("timestamp") or datetime.utcnow().isoformat()

        return {
            "time_hours": time_hours,
            "radius_m": radius_m,
            "timestamp": timestamp,
        }

    def _evaluate_and_publish(self, *, event_time_hours: float, event_timestamp: str) -> None:
        if self.mq_out is None or self.influx is None:
            raise RuntimeError("Dependencies not initialised. Did you call setup()?")  # pragma: no cover

        with self._lock:
            thaw = self._last_thaw
            forecast = self._last_forecast

        measured_radius = thaw["radius_m"] if thaw else None
        forecast_radius = forecast["radius_m"] if forecast else None

        reasons: list[str] = []
        if measured_radius is not None and measured_radius >= self.limit_radius_m:
            reasons.append("measured")
        if forecast_radius is not None and forecast_radius >= self.limit_radius_m:
            reasons.append("forecast")

        triggered = bool(reasons)
        if self.publish_on_change and self._last_triggered is not None and triggered == self._last_triggered:
            return
        self._last_triggered = triggered

        if not reasons:
            reason = "none"
        elif len(reasons) == 2:
            reason = "both"
        else:
            reason = reasons[0]

        payload = {
            "timestamp": event_timestamp,
            "time_hours": event_time_hours,
            "limit_radius_m": self.limit_radius_m,
            "measured_radius_m": measured_radius,
            "forecast_radius_m": forecast_radius,
            "triggered": triggered,
            "status": "alert" if triggered else "ok",
            "reason": reason,
            "metric": self.metric,
        }

        self.mq_out.publish(payload)
        self.influx.write_safety_alert(
            time_hours=event_time_hours,
            limit_radius_m=self.limit_radius_m,
            measured_radius_m=measured_radius,
            forecast_radius_m=forecast_radius,
            triggered=triggered,
            reason=reason,
            metric=self.metric,
            site=self.site_id,
        )
        self.logger.info(
            "Safety status (t=%.2fh, measured=%s, forecast=%s, limit=%.3fm, status=%s)",
            event_time_hours,
            "n/a" if measured_radius is None else f"{measured_radius:.3f}m",
            "n/a" if forecast_radius is None else f"{forecast_radius:.3f}m",
            self.limit_radius_m,
            payload["status"],
        )

    def _handle_thaw(self, msg: dict) -> None:
        data = self._parse_metric(msg, source="thaw_front")
        with self._lock:
            self._last_thaw = data
        self._evaluate_and_publish(
            event_time_hours=data["time_hours"],
            event_timestamp=data["timestamp"],
        )

    def _handle_forecast(self, msg: dict) -> None:
        data = self._parse_metric(msg, source="forecast")
        with self._lock:
            self._last_forecast = data
        self._evaluate_and_publish(
            event_time_hours=data["time_hours"],
            event_timestamp=data["timestamp"],
        )

    def _consume_thaw(self) -> None:
        if self.mq_thaw is None:
            raise RuntimeError("Thaw input client not initialised.")  # pragma: no cover
        self.mq_thaw.consume(callback=self._handle_thaw)

    def _consume_forecast(self) -> None:
        if self.mq_forecast is None:
            raise RuntimeError("Forecast input client not initialised.")  # pragma: no cover
        self.mq_forecast.consume(callback=self._handle_forecast)

    def setup(self) -> None:
        if self.mq_thaw is None:
            self.mq_thaw = RabbitMQClient(self.thaw_input_config)
        if self.mq_forecast is None:
            self.mq_forecast = RabbitMQClient(self.forecast_input_config)
        if self.mq_out is None:
            self.mq_out = RabbitMQClient(self.output_config)
        if self.influx is None:
            self.influx = InfluxHelper(self.influx_config)
        self.logger.info(
            "Dependencies initialised (thaw=%s, forecast=%s, out=%s)",
            self.thaw_input_config.queue,
            self.forecast_input_config.queue,
            self.output_config.queue,
        )
        self.logger.info(
            "Safety limit configured (limit=%.3fm, metric=%s, publish_on_change=%s)",
            self.limit_radius_m,
            self.metric,
            self.publish_on_change,
        )

    def start(self) -> None:
        if self.mq_thaw is None or self.mq_forecast is None or self.mq_out is None or self.influx is None:
            self.setup()

        self.logger.info("Listening for thaw-front updates on %s", self.thaw_input_config.queue)
        self.logger.info("Listening for FEM forecasts on %s", self.forecast_input_config.queue)

        thaw_thread = threading.Thread(target=self._consume_thaw, name="safety_monitor_thaw", daemon=True)
        forecast_thread = threading.Thread(
            target=self._consume_forecast,
            name="safety_monitor_forecast",
            daemon=True,
        )
        self._threads = [thaw_thread, forecast_thread]
        for thread in self._threads:
            thread.start()

        try:
            while any(thread.is_alive() for thread in self._threads):
                for thread in self._threads:
                    thread.join(timeout=0.5)
        except KeyboardInterrupt:  # pragma: no cover - runtime behaviour
            self.logger.info("Interrupt received; shutting down")
        finally:
            self.close()

    def stop(self) -> None:
        if self.mq_thaw and self.mq_thaw.channel:
            self.mq_thaw.channel.stop_consuming()
        if self.mq_forecast and self.mq_forecast.channel:
            self.mq_forecast.channel.stop_consuming()

    def close(self) -> None:
        self.stop()
        if self.mq_thaw is not None:
            self.mq_thaw.disconnect()
        if self.mq_forecast is not None:
            self.mq_forecast.disconnect()
        if self.mq_out is not None:
            self.mq_out.disconnect()
        if self.influx is not None:
            self.influx.close()
        self.logger.info("Shutdown complete")


if __name__ == "__main__":
    SafetyMonitorServer().start()

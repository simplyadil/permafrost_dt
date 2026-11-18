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
ALLOWED_MODES = {"single", "dual", "predictive"}


class SafetyMonitorServer:
    """Evaluate thaw-front metrics against a safety limit and emit alerts.

    Two modes:

    * ``single`` (legacy) — compare one reduced metric (``max``/``avg``) against a
      single ``limit_radius_m`` and trigger when it is exceeded.
    * ``dual`` — the closed-loop thaw-control rule engine. It uses both the outer
      thaw radius ``r_max`` and the least-thawed radius ``r_min`` against an upper
      and lower limit:

        - ``r_max >= limit_max``           -> STOP heating (status ``alert``). This
          is the safety cut and overrides everything, including a low ``r_min``.
        - otherwise                        -> keep HEATING ("keep thawing"), whether
          ``r_min`` is above or below its floor.
        - ``r_max`` in [warn, critical)·limit_max -> status ``warning`` (still heating)
        - ``r_max`` in [critical, 1)·limit_max     -> status ``critical`` (still heating)
        - ``r_min < limit_min``            -> flagged ``rmin_below_floor`` (informational;
          reinforces that heating should continue — never stops the heater).

      The emitted alert carries a ``recommended_action`` (``stop``/``heat``) that the
      heater-actuation service maps directly onto the PolyScience power state.
    """

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
        mode: str = "single",
        limit_min_radius_m: float = 0.0,
        warn_fraction: float = 0.6,
        critical_fraction: float = 0.9,
        publish_on_change: bool = False,
        site_id: str = "default",
        confirm_updates: int = 3,
        release_fraction: float = 0.85,
        forecast_max_age_hours: float = 0.05,
        measurement_max_age_hours: float = 0.05,
        fail_safe_on_stale_measurement: bool = True,
        latch_alert: bool = True,
        f2_action: str = "stop",
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
        mode = mode.lower().strip()
        if mode not in ALLOWED_MODES:
            raise ValueError(f"Unsupported mode '{mode}'. Allowed: {sorted(ALLOWED_MODES)}")
        metric = metric.lower().strip()
        if mode == "single" and metric not in ALLOWED_METRICS:
            raise ValueError(f"Unsupported metric '{metric}'. Allowed: {sorted(ALLOWED_METRICS)}")

        self.mode = mode
        self.metric = metric
        self.limit_radius_m = float(limit_radius_m)
        self.limit_min_radius_m = float(limit_min_radius_m)
        self.warn_fraction = float(warn_fraction)
        self.critical_fraction = float(critical_fraction)
        self.publish_on_change = bool(publish_on_change)
        self.site_id = site_id

        # --- predictive mode -------------------------------------------------
        # Confirmation applies to the forecast predicates only. A measured
        # breach is never debounced: the measurement is the reference signal, and
        # delaying the safety cut to wait for agreement with itself is the one
        # thing this service must not do.
        self.confirm_updates = max(int(confirm_updates), 1)
        # Hysteresis. Entry is at warn/critical, release at a lower fraction, so
        # a metric hovering on a threshold cannot walk the actuator back and
        # forth. Clamped at or below the warn fraction: a release band above the
        # entry band would latch permanently.
        self.release_fraction = min(float(release_fraction), float(warn_fraction))
        # Staleness, in *data* time (both streams carry time_hours), so replay is
        # deterministic and a slow wall clock cannot fabricate an alert.
        self.forecast_max_age_hours = float(forecast_max_age_hours)
        self.measurement_max_age_hours = float(measurement_max_age_hours)
        # Asymmetric degradation: a missing forecast falls back to the measured
        # ladder, a missing measurement fails safe. Losing the prediction costs
        # lead time; losing the measurement means flying blind on the reference.
        self.fail_safe_on_stale_measurement = bool(fail_safe_on_stale_measurement)
        # Thaw is effectively irreversible on this timescale, so r_max does not
        # come back down and an auto-release after a measured breach would only
        # ever fire on noise. Latch it and require an operator to clear it.
        self.latch_alert = bool(latch_alert)
        f2_action = str(f2_action).lower().strip()
        if f2_action not in {"stop", "reduce"}:
            raise ValueError(f"f2_action must be 'stop' or 'reduce' (got '{f2_action}').")
        self.f2_action = f2_action

        self._lock = threading.Lock()
        self._last_thaw: Optional[dict] = None
        self._last_forecast: Optional[dict] = None
        self._last_triggered: Optional[bool] = None
        self._last_change_key: Optional[tuple] = None
        self._threads: list[threading.Thread] = []
        self._f1_streak = 0
        self._f2_streak = 0
        self._last_forecast_event_h: Optional[float] = None
        self._alert_latched = False
        self._elevated = False

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

    @staticmethod
    def _opt_float(value) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _parse_dual(self, msg: dict, *, source: str) -> dict:
        try:
            time_hours = float(msg["time_hours"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Missing time_hours in {source} message.")

        return {
            "time_hours": time_hours,
            "r_max": self._opt_float(msg.get("radius_max_m")),
            "r_min": self._opt_float(msg.get("radius_min_m")),
            "timestamp": msg.get("timestamp") or datetime.utcnow().isoformat(),
        }

    def _parse_predictive(self, msg: dict, *, source: str) -> dict:
        """Everything the predictive rules need, from either stream.

        One parser for both queues: the thaw message carries the measured radii,
        the forecast message additionally carries the two lookahead branches.
        Missing keys stay None rather than raising, so a forecast service that
        publishes no lookahead degrades to the measured ladder instead of taking
        the monitor down.
        """
        try:
            time_hours = float(msg["time_hours"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Missing time_hours in {source} message.")

        def _series(key):
            raw = msg.get(key)
            if not isinstance(raw, (list, tuple)) or not raw:
                return None
            return [self._opt_float(v) for v in raw]

        return {
            "time_hours": time_hours,
            "r_max": self._opt_float(msg.get("radius_max_m")),
            "r_min": self._opt_float(msg.get("radius_min_m")),
            "timestamp": msg.get("timestamp") or datetime.utcnow().isoformat(),
            "lookahead_hours": self._opt_float(msg.get("lookahead_hours")),
            "times_h": _series("lookahead_times_h"),
            "keep_m": _series("lookahead_radius_keep_m"),
            "stop_m": _series("lookahead_radius_stop_m"),
        }

    @staticmethod
    def _any_over(series, limit_m: float) -> bool:
        """Does the branch reach the limit anywhere in the sampled horizon?

        Anywhere, not only at the endpoint: a front that crosses and then stalls
        just below the limit at the final sample has still breached, and reading
        the endpoint alone would miss it.
        """
        if not series:
            return False
        return any(v is not None and v >= limit_m for v in series)

    @staticmethod
    def _first_crossing(times, series, limit_m: float):
        if not times or not series:
            return None
        for t, v in zip(times, series):
            if v is not None and v >= limit_m:
                return float(t)
        return None

    def _evaluate_and_publish_predictive(self, *, event_time_hours: float,
                                         event_timestamp: str, source: str) -> None:
        """Apply the M1/M2/F1/F2 rules and publish the decision with its evidence.

        Four predicates, from two independent sources::

            M1  measured r_max >= limit                the safety cut
            M2  measured r_max >= warn * limit         measurement is elevated
            F1  forecast breaches within the horizon if heating continues
            F2  forecast breaches *even if the heater is cut now*

        F2 is the last-safe-moment test, and the one only a forward model can
        answer -- a trend extrapolation has no boundary condition to change.

        The combination rule is deliberately asymmetric. Stopping trips on
        1-of-2 (M1 or F2): either source alone seeing a breach is enough to
        justify the safe action. Continuing to heat needs 2-of-2 agreement. The
        discretionary middle action, ``reduce``, is the only one requiring
        corroboration from both -- which is what keeps a false forecast alarm
        cheap, and is why the predictive arm can run at a useful lead at all.
        """
        if self.mq_out is None or self.influx is None:
            raise RuntimeError("Dependencies not initialised. Did you call setup()?")  # pragma: no cover

        with self._lock:
            thaw = self._last_thaw
            forecast = self._last_forecast

        limit_max = self.limit_radius_m
        limit_min = self.limit_min_radius_m

        m_rmax = thaw["r_max"] if thaw else None
        m_rmin = thaw["r_min"] if thaw else None

        # Ages in data time, not wall clock: both streams carry time_hours, so a
        # replay behaves exactly as the live run did, and a slow consumer cannot
        # invent an alert.
        meas_age = (event_time_hours - thaw["time_hours"]) if thaw else None
        fc_age = (event_time_hours - forecast["time_hours"]) if forecast else None

        # Freshness and content are separate questions. Before anything thaws the
        # reconstruction legitimately reports no front at all, and reading that
        # absence as a lost sensor would fail safe and stop the heater at t = 0 --
        # every run would begin by shutting itself down. Staleness is therefore
        # decided by age alone; a missing radius simply means no predicate about
        # the front can be true yet.
        measurement_fresh = bool(
            thaw is not None and meas_age is not None
            and meas_age <= self.measurement_max_age_hours
        )
        measurement_ok = bool(measurement_fresh and m_rmax is not None)
        forecast_available = bool(
            forecast is not None and fc_age is not None
            and fc_age <= self.forecast_max_age_hours
            and forecast.get("keep_m") and forecast.get("stop_m")
        )

        keep_m = forecast.get("keep_m") if forecast_available else None
        stop_m = forecast.get("stop_m") if forecast_available else None
        times_h = forecast.get("times_h") if forecast_available else None

        # --- predicates -------------------------------------------------------
        m1 = bool(measurement_ok and m_rmax >= limit_max)
        m2 = bool(measurement_ok and m_rmax >= self.warn_fraction * limit_max)
        m_critical = bool(measurement_ok and m_rmax >= self.critical_fraction * limit_max)
        f1_raw = self._any_over(keep_m, limit_max)
        f2_raw = self._any_over(stop_m, limit_max)

        # Confirmation, advanced only when a *new* forecast arrives. The monitor
        # is called on every message from either stream, so counting on both
        # would reach the threshold in half the intended time.
        # Staleness and confirmation are separate concerns, and conflating them
        # breaks both. Staleness *gates the output*: an expired forecast votes on
        # nothing, whichever stream noticed it expired. Confirmation counts only
        # forecast messages, and only ones that continue an unbroken stream -- a
        # gap longer than the staleness window starts a new streak rather than
        # extending an old one, because "three consecutive updates" has to mean
        # consecutive in time, not merely three in a row eventually.
        #
        # Resetting the counter on every stale-looking message instead would
        # clobber it from the thaw queue, which normally advances the clock past
        # the window before the next forecast lands -- the streak would then
        # never exceed one and the predictive arm would be silently dead.
        if source == "forecast":
            gap = (None if self._last_forecast_event_h is None
                   else event_time_hours - self._last_forecast_event_h)
            continuous = gap is not None and gap <= self.forecast_max_age_hours
            if not forecast_available:
                self._f1_streak = 0
                self._f2_streak = 0
            elif continuous:
                self._f1_streak = self._f1_streak + 1 if f1_raw else 0
                self._f2_streak = self._f2_streak + 1 if f2_raw else 0
            else:
                self._f1_streak = 1 if f1_raw else 0
                self._f2_streak = 1 if f2_raw else 0
            self._last_forecast_event_h = event_time_hours

        f1 = bool(forecast_available and self._f1_streak >= self.confirm_updates)
        f2 = bool(forecast_available and self._f2_streak >= self.confirm_updates)

        # --- hysteresis on the measured side ----------------------------------
        if m2:
            self._elevated = True
        elif measurement_ok and m_rmax < self.release_fraction * limit_max:
            self._elevated = False
        elevated = self._elevated

        if m1:
            self._alert_latched = True
        latched = bool(self._alert_latched and self.latch_alert)

        # --- decision ---------------------------------------------------------
        # `blind` only once a measurement has actually been seen: at startup the
        # forecast can arrive first, and failing safe on an empty history would
        # stop the heater before the experiment begins. See measurement_fresh
        # above for why a *present but empty* front does not count as blind.
        blind = bool(thaw is not None and not measurement_fresh)
        rmin_below_floor = bool(m_rmin is not None and m_rmin < limit_min)

        ttb_hours = None
        crossing = self._first_crossing(times_h, keep_m, limit_max)
        if crossing is not None:
            ttb_hours = max(crossing - event_time_hours, 0.0)

        if blind and self.fail_safe_on_stale_measurement:
            # Blind on the reference signal. Losing the forecast costs lead time;
            # losing the measurement means we cannot see the thing being limited.
            status, action, reason = "alert", "stop", "measurement_stale"
        elif m1:
            status, action, reason = "alert", "stop", "rmax_over_limit"
        elif latched:
            status, action, reason = "alert", "stop", "rmax_over_limit_latched"
        elif f2:
            status, action = "committed", self.f2_action
            reason = "breach_unavoidable"
        elif elevated and f1:
            status, action, reason = "pre_alert", "reduce", "rmax_elevated_and_forecast_breach"
        elif f1:
            status, action, reason = "predictive_warning", "heat", "forecast_breach_within_horizon"
        elif m_critical:
            status, action, reason = "critical", "heat", "rmax_critical"
        elif m2:
            status, action, reason = "warning", "heat", "rmax_warning"
        elif rmin_below_floor:
            status, action, reason = "ok", "heat", "rmin_below_floor"
        elif thaw is None:
            status, action, reason = "ok", "heat", "awaiting_measurement"
        else:
            status, action, reason = "ok", "heat", "nominal"

        triggered = action == "stop"

        change_key = (status, action, reason)
        if self.publish_on_change and self._last_change_key is not None and change_key == self._last_change_key:
            return
        self._last_change_key = change_key

        keep_end = keep_m[-1] if keep_m else None
        stop_end = stop_m[-1] if stop_m else None

        payload = {
            "timestamp": event_timestamp,
            "time_hours": event_time_hours,
            "limit_radius_m": limit_max,
            "measured_radius_m": m_rmax,
            "forecast_radius_m": keep_end,
            "triggered": triggered,
            "status": status,
            "reason": reason,
            "metric": "predictive",
            "r_max_m": m_rmax,
            "r_min_m": m_rmin,
            "limit_min_radius_m": limit_min,
            "warn_fraction": self.warn_fraction,
            "critical_fraction": self.critical_fraction,
            "rmin_below_floor": rmin_below_floor,
            "recommended_action": action,
            # Evidence. Every decision carries the predicates that produced it,
            # both branch endpoints and the age of each input, so an alert can be
            # audited afterwards without re-running anything.
            "predicate_m1": m1,
            "predicate_m2": m2,
            "predicate_f1": f1,
            "predicate_f2": f2,
            "forecast_radius_keep_m": keep_end,
            "forecast_radius_stop_m": stop_end,
            "time_to_breach_hours": ttb_hours,
            "lookahead_hours": (forecast.get("lookahead_hours") if forecast else None),
            "measurement_age_hours": meas_age,
            "forecast_age_hours": fc_age,
            "forecast_available": forecast_available,
            "confirm_count": max(self._f1_streak, self._f2_streak),
            "confirm_required": self.confirm_updates,
            "latched": latched,
        }

        self.mq_out.publish(payload)
        self.influx.write_safety_alert(
            time_hours=event_time_hours,
            limit_radius_m=limit_max,
            measured_radius_m=m_rmax,
            forecast_radius_m=keep_end,
            triggered=triggered,
            reason=reason,
            metric="predictive",
            status=status,
            r_max_m=m_rmax,
            r_min_m=m_rmin,
            limit_min_radius_m=limit_min,
            rmin_below_floor=rmin_below_floor,
            recommended_action=action,
            site=self.site_id,
        )
        self.logger.info(
            "Predictive status (t=%.2fh, r_max=%s, keep=%s, stop=%s, ttb=%s, "
            "M1=%d M2=%d F1=%d F2=%d, status=%s, action=%s%s)",
            event_time_hours,
            "n/a" if m_rmax is None else f"{m_rmax:.3f}m",
            "n/a" if keep_end is None else f"{keep_end:.3f}m",
            "n/a" if stop_end is None else f"{stop_end:.3f}m",
            "n/a" if ttb_hours is None else f"{ttb_hours * 60:.0f}min",
            m1, m2, f1, f2, status, action,
            "" if forecast_available else ", forecast unavailable (degraded)",
        )

    def clear_latch(self) -> None:
        """Operator acknowledgement: allow heating again after a measured breach.

        Deliberately not automatic. A measured breach latches because thaw does
        not reverse on this timescale, so an auto-release rule would only ever
        fire on noise in the reconstruction.
        """
        self._alert_latched = False
        self._elevated = False
        self.logger.warning("Safety alert latch cleared by operator request.")

    def _evaluate_and_publish_dual(self, *, event_time_hours: float, event_timestamp: str) -> None:
        """Apply the dual-limit thaw-control rules and publish the decision.

        Safety-first priority: if the outer thaw radius (measured or forecast)
        reaches ``limit_max`` the heater is stopped regardless of ``r_min``.
        Otherwise heating continues, with graded warning statuses as ``r_max``
        approaches the limit and an informational flag when ``r_min`` sits below
        its floor.
        """
        if self.mq_out is None or self.influx is None:
            raise RuntimeError("Dependencies not initialised. Did you call setup()?")  # pragma: no cover

        with self._lock:
            thaw = self._last_thaw
            forecast = self._last_forecast

        m_rmax = thaw["r_max"] if thaw else None
        m_rmin = thaw["r_min"] if thaw else None
        f_rmax = forecast["r_max"] if forecast else None

        limit_max = self.limit_radius_m
        limit_min = self.limit_min_radius_m

        over_measured = m_rmax is not None and m_rmax >= limit_max
        over_forecast = f_rmax is not None and f_rmax >= limit_max
        triggered = bool(over_measured or over_forecast)

        ratio = (m_rmax / limit_max) if (m_rmax is not None and limit_max > 0) else None
        if triggered:
            status = "alert"
        elif ratio is not None and ratio >= self.critical_fraction:
            status = "critical"
        elif ratio is not None and ratio >= self.warn_fraction:
            status = "warning"
        else:
            status = "ok"

        rmin_below_floor = bool(m_rmin is not None and m_rmin < limit_min)
        recommended_action = "stop" if triggered else "heat"

        if over_measured:
            reason = "rmax_over_limit"
        elif over_forecast:
            reason = "forecast_rmax_over_limit"
        elif status == "critical":
            reason = "rmax_critical"
        elif status == "warning":
            reason = "rmax_warning"
        elif rmin_below_floor:
            reason = "rmin_below_floor"
        else:
            reason = "nominal"

        change_key = (status, recommended_action)
        if self.publish_on_change and self._last_change_key is not None and change_key == self._last_change_key:
            return
        self._last_change_key = change_key

        payload = {
            "timestamp": event_timestamp,
            "time_hours": event_time_hours,
            "limit_radius_m": limit_max,
            "measured_radius_m": m_rmax,
            "forecast_radius_m": f_rmax,
            "triggered": triggered,
            "status": status,
            "reason": reason,
            "metric": "dual",
            "r_max_m": m_rmax,
            "r_min_m": m_rmin,
            "limit_min_radius_m": limit_min,
            "warn_fraction": self.warn_fraction,
            "critical_fraction": self.critical_fraction,
            "rmin_below_floor": rmin_below_floor,
            "recommended_action": recommended_action,
        }

        self.mq_out.publish(payload)
        self.influx.write_safety_alert(
            time_hours=event_time_hours,
            limit_radius_m=limit_max,
            measured_radius_m=m_rmax,
            forecast_radius_m=f_rmax,
            triggered=triggered,
            reason=reason,
            metric="dual",
            status=status,
            r_max_m=m_rmax,
            r_min_m=m_rmin,
            limit_min_radius_m=limit_min,
            rmin_below_floor=rmin_below_floor,
            recommended_action=recommended_action,
            site=self.site_id,
        )
        self.logger.info(
            "Thaw-control status (t=%.2fh, r_max=%s, r_min=%s, limit_max=%.3fm, "
            "limit_min=%.3fm, status=%s, action=%s%s)",
            event_time_hours,
            "n/a" if m_rmax is None else f"{m_rmax:.3f}m",
            "n/a" if m_rmin is None else f"{m_rmin:.3f}m",
            limit_max,
            limit_min,
            status,
            recommended_action,
            ", rmin<floor" if rmin_below_floor else "",
        )

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
        if self.mode == "predictive":
            data = self._parse_predictive(msg, source="thaw_front")
            with self._lock:
                self._last_thaw = data
            self._evaluate_and_publish_predictive(
                event_time_hours=data["time_hours"],
                event_timestamp=data["timestamp"],
                source="thaw_front",
            )
            return
        if self.mode == "dual":
            data = self._parse_dual(msg, source="thaw_front")
            with self._lock:
                self._last_thaw = data
            self._evaluate_and_publish_dual(
                event_time_hours=data["time_hours"],
                event_timestamp=data["timestamp"],
            )
            return
        data = self._parse_metric(msg, source="thaw_front")
        with self._lock:
            self._last_thaw = data
        self._evaluate_and_publish(
            event_time_hours=data["time_hours"],
            event_timestamp=data["timestamp"],
        )

    def _handle_forecast(self, msg: dict) -> None:
        if self.mode == "predictive":
            data = self._parse_predictive(msg, source="forecast")
            with self._lock:
                self._last_forecast = data
            self._evaluate_and_publish_predictive(
                event_time_hours=data["time_hours"],
                event_timestamp=data["timestamp"],
                source="forecast",
            )
            return
        if self.mode == "dual":
            data = self._parse_dual(msg, source="forecast")
            with self._lock:
                self._last_forecast = data
            self._evaluate_and_publish_dual(
                event_time_hours=data["time_hours"],
                event_timestamp=data["timestamp"],
            )
            return
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
        if self.mode == "dual":
            self.logger.info(
                "Dual-limit thaw control (limit_max=%.3fm, limit_min=%.3fm, "
                "warn=%.0f%%, critical=%.0f%%, publish_on_change=%s)",
                self.limit_radius_m,
                self.limit_min_radius_m,
                self.warn_fraction * 100.0,
                self.critical_fraction * 100.0,
                self.publish_on_change,
            )
        else:
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

"""Unit tests for the predictive safety rule engine (M1/M2/F1/F2).

The rule engine is pure logic over two dicts, so none of this touches the FEM,
RabbitMQ or InfluxDB -- the message bus and database are replaced by recorders
and the whole file runs in well under a second.

What is covered here is the decision table and the mechanics that surround it:
confirmation, hysteresis, staleness and the alert latch. What is *not* covered
here is whether the forecast is any good -- that is a numerics question, and it
belongs to ``software/tools/verify_prediction_skill.py``. The two fail
independently: a failure here means the rules mishandled their inputs, a failure
there means the model moved.
"""

from __future__ import annotations

import pytest

from software.digital_twin.control.safety_monitor.safety_monitor_server import (
    SafetyMonitorServer,
)

LIMIT = 0.040
WARN = 0.6        # 24 mm
CRITICAL = 0.9    # 36 mm


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
class _Recorder:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def publish(self, message: dict) -> None:
        self.messages.append(message)

    @property
    def last(self) -> dict | None:
        return self.messages[-1] if self.messages else None


class _NullInflux:
    def __getattr__(self, _name):
        def _noop(*_args, **_kwargs):
            return None
        return _noop


def make_monitor(**overrides) -> SafetyMonitorServer:
    """Monitor in predictive mode with its outputs captured.

    ``confirm_updates=1`` by default so the decision table can be exercised one
    row at a time; the confirmation tests override it explicitly.
    """
    kwargs = dict(
        mode="predictive",
        limit_radius_m=LIMIT,
        limit_min_radius_m=0.015,
        warn_fraction=WARN,
        critical_fraction=CRITICAL,
        confirm_updates=1,
        release_fraction=0.5,
        forecast_max_age_hours=0.05,
        measurement_max_age_hours=0.05,
    )
    kwargs.update(overrides)
    monitor = SafetyMonitorServer(**kwargs)
    monitor.mq_out = _Recorder()
    monitor.influx = _NullInflux()
    return monitor


def thaw_msg(t_h: float, r_max_mm: float, r_min_mm: float = 20.0) -> dict:
    return {
        "timestamp": f"t={t_h:.4f}",
        "time_hours": t_h,
        "radius_max_m": r_max_mm / 1000.0,
        "radius_min_m": r_min_mm / 1000.0,
    }


def forecast_msg(t_h: float, keep_mm: list[float] | None, stop_mm: list[float] | None,
                 horizon_h: float = 1.0) -> dict:
    """A forecast carrying both lookahead branches, sampled over the horizon."""
    n = len(keep_mm) if keep_mm else 0
    step = horizon_h / n if n else 0.0
    return {
        "timestamp": f"t={t_h:.4f}",
        "time_hours": t_h,
        "horizon_hours": 0.17,
        "radius_max_m": None,
        "lookahead_hours": horizon_h,
        "lookahead_valid_at_hours": t_h + horizon_h,
        "lookahead_times_h": [t_h + (k + 1) * step for k in range(n)],
        "lookahead_radius_keep_m": None if keep_mm is None else [v / 1000.0 for v in keep_mm],
        "lookahead_radius_stop_m": None if stop_mm is None else [v / 1000.0 for v in stop_mm],
    }


def drive(monitor, t_h, r_max_mm, keep_mm, stop_mm, r_min_mm=20.0) -> dict:
    """One full cycle: measurement then forecast, as the live service sees them."""
    monitor._handle_thaw(thaw_msg(t_h, r_max_mm, r_min_mm))
    monitor._handle_forecast(forecast_msg(t_h, keep_mm, stop_mm))
    return monitor.mq_out.last


# ---------------------------------------------------------------------------
# THE DECISION TABLE
# ---------------------------------------------------------------------------
def test_nominal_keeps_heating():
    alert = drive(make_monitor(), 1.0, r_max_mm=10.0, keep_mm=[12, 14], stop_mm=[11, 11])
    assert alert["status"] == "ok"
    assert alert["recommended_action"] == "heat"
    assert alert["triggered"] is False


def test_measured_warning_still_heats():
    alert = drive(make_monitor(), 1.0, r_max_mm=26.0, keep_mm=[27, 28], stop_mm=[26, 26])
    assert alert["status"] == "warning"
    assert alert["recommended_action"] == "heat"


def test_measured_critical_still_heats():
    alert = drive(make_monitor(), 1.0, r_max_mm=37.0, keep_mm=[37, 38], stop_mm=[37, 37])
    assert alert["status"] == "critical"
    assert alert["recommended_action"] == "heat"


def test_m1_measured_breach_stops():
    alert = drive(make_monitor(), 1.0, r_max_mm=41.0, keep_mm=[41, 42], stop_mm=[41, 41])
    assert alert["status"] == "alert"
    assert alert["recommended_action"] == "stop"
    assert alert["triggered"] is True
    assert alert["reason"] == "rmax_over_limit"
    assert alert["predicate_m1"] is True


def test_f1_alone_below_warn_only_flags():
    """Forecast sees a breach the measurement does not corroborate: log, don't act."""
    alert = drive(make_monitor(), 1.0, r_max_mm=10.0, keep_mm=[20, 45], stop_mm=[11, 12])
    assert alert["status"] == "predictive_warning"
    assert alert["recommended_action"] == "heat"
    assert alert["predicate_f1"] is True
    assert alert["predicate_m2"] is False


def test_m2_and_f1_reduces():
    """Both sources agree it is coming, and it is still avoidable -> reduce."""
    alert = drive(make_monitor(), 1.0, r_max_mm=30.0, keep_mm=[33, 44], stop_mm=[31, 32])
    assert alert["status"] == "pre_alert"
    assert alert["recommended_action"] == "reduce"
    assert alert["triggered"] is False
    assert alert["predicate_m2"] is True
    assert alert["predicate_f1"] is True
    assert alert["predicate_f2"] is False


def test_f2_stops_even_though_measurement_is_low():
    """The last-safe-moment test: cutting heat now no longer avoids the breach."""
    alert = drive(make_monitor(), 1.0, r_max_mm=30.0, keep_mm=[35, 46], stop_mm=[33, 41])
    assert alert["status"] == "committed"
    assert alert["recommended_action"] == "stop"
    assert alert["triggered"] is True
    assert alert["reason"] == "breach_unavoidable"
    assert alert["predicate_f2"] is True


def test_f2_can_be_configured_to_reduce_instead_of_stop():
    monitor = make_monitor(f2_action="reduce")
    alert = drive(monitor, 1.0, r_max_mm=30.0, keep_mm=[35, 46], stop_mm=[33, 41])
    assert alert["status"] == "committed"
    assert alert["recommended_action"] == "reduce"
    assert alert["triggered"] is False


def test_m1_outranks_f2():
    """A measured breach is the safety cut and reports as such, not as committed."""
    alert = drive(make_monitor(), 1.0, r_max_mm=41.0, keep_mm=[42, 46], stop_mm=[41, 42])
    assert alert["status"] == "alert"
    assert alert["reason"] == "rmax_over_limit"


def test_crossing_anywhere_in_the_horizon_counts_not_just_the_endpoint():
    """A branch that crosses then falls back below the limit has still breached."""
    alert = drive(make_monitor(), 1.0, r_max_mm=30.0, keep_mm=[44, 38], stop_mm=[31, 30])
    assert alert["predicate_f1"] is True
    assert alert["status"] == "pre_alert"


# ---------------------------------------------------------------------------
# CONFIRMATION
# ---------------------------------------------------------------------------
def test_forecast_predicates_need_confirmation():
    monitor = make_monitor(confirm_updates=3)
    for _ in range(2):
        alert = drive(monitor, 1.0, 30.0, keep_mm=[35, 45], stop_mm=[31, 32])
        assert alert["predicate_f1"] is False, "fired before the streak was met"
    alert = drive(monitor, 1.0, 30.0, keep_mm=[35, 45], stop_mm=[31, 32])
    assert alert["predicate_f1"] is True
    assert alert["confirm_count"] == 3
    assert alert["confirm_required"] == 3


def test_a_single_clear_update_resets_the_streak():
    monitor = make_monitor(confirm_updates=3)
    drive(monitor, 1.0, 30.0, keep_mm=[35, 45], stop_mm=[31, 32])
    drive(monitor, 1.0, 30.0, keep_mm=[35, 45], stop_mm=[31, 32])
    drive(monitor, 1.0, 30.0, keep_mm=[30, 31], stop_mm=[30, 30])   # clears
    alert = drive(monitor, 1.0, 30.0, keep_mm=[35, 45], stop_mm=[31, 32])
    assert alert["predicate_f1"] is False


def test_measurement_messages_do_not_advance_the_streak():
    """Both queues call the engine; only a new forecast may confirm a forecast."""
    monitor = make_monitor(confirm_updates=3)
    monitor._handle_forecast(forecast_msg(1.0, keep_mm=[35, 45], stop_mm=[31, 32]))
    for _ in range(5):
        monitor._handle_thaw(thaw_msg(1.0, 30.0))
    assert monitor.mq_out.last["predicate_f1"] is False
    assert monitor.mq_out.last["confirm_count"] == 1


def test_measured_breach_is_never_debounced():
    """The reference signal trips on the first sample, whatever the streak says."""
    monitor = make_monitor(confirm_updates=5)
    monitor._handle_thaw(thaw_msg(1.0, 41.0))
    assert monitor.mq_out.last["recommended_action"] == "stop"


# ---------------------------------------------------------------------------
# HYSTERESIS AND THE LATCH
# ---------------------------------------------------------------------------
def test_elevated_state_releases_only_below_the_release_band():
    monitor = make_monitor(release_fraction=0.5)          # 20 mm
    drive(monitor, 1.0, 26.0, keep_mm=[27, 28], stop_mm=[26, 26])   # elevated
    # Between release and warn: still elevated, so a forecast breach reduces.
    alert = drive(monitor, 1.1, 22.0, keep_mm=[30, 45], stop_mm=[23, 24])
    assert alert["status"] == "pre_alert"
    # Below the release band: elevated clears, so the same forecast only flags.
    drive(monitor, 1.2, 18.0, keep_mm=[19, 20], stop_mm=[18, 18])
    alert = drive(monitor, 1.3, 18.0, keep_mm=[30, 45], stop_mm=[19, 20])
    assert alert["status"] == "predictive_warning"


def test_release_fraction_cannot_exceed_warn_fraction():
    """A release band above the entry band would latch elevated permanently."""
    monitor = make_monitor(release_fraction=0.95)
    assert monitor.release_fraction == pytest.approx(WARN)


def test_measured_breach_latches():
    monitor = make_monitor()
    drive(monitor, 1.0, 41.0, keep_mm=[42, 43], stop_mm=[41, 41])
    alert = drive(monitor, 1.1, 10.0, keep_mm=[11, 12], stop_mm=[10, 10])
    assert alert["status"] == "alert"
    assert alert["reason"] == "rmax_over_limit_latched"
    assert alert["latched"] is True


def test_operator_can_clear_the_latch():
    monitor = make_monitor()
    drive(monitor, 1.0, 41.0, keep_mm=[42, 43], stop_mm=[41, 41])
    monitor.clear_latch()
    alert = drive(monitor, 1.1, 10.0, keep_mm=[11, 12], stop_mm=[10, 10])
    assert alert["status"] == "ok"
    assert alert["recommended_action"] == "heat"


def test_latching_can_be_disabled():
    monitor = make_monitor(latch_alert=False)
    drive(monitor, 1.0, 41.0, keep_mm=[42, 43], stop_mm=[41, 41])
    alert = drive(monitor, 1.1, 10.0, keep_mm=[11, 12], stop_mm=[10, 10])
    assert alert["status"] == "ok"


# ---------------------------------------------------------------------------
# STALENESS -- the two failures are deliberately not symmetric
# ---------------------------------------------------------------------------
def test_stale_forecast_degrades_to_the_measured_ladder():
    monitor = make_monitor(forecast_max_age_hours=0.05)
    monitor._handle_forecast(forecast_msg(1.0, keep_mm=[35, 45], stop_mm=[33, 44]))
    monitor._handle_thaw(thaw_msg(1.5, 26.0))             # 0.5 h later
    alert = monitor.mq_out.last
    assert alert["forecast_available"] is False
    assert alert["predicate_f1"] is False
    assert alert["predicate_f2"] is False
    assert alert["status"] == "warning"                    # measured ladder only
    assert alert["recommended_action"] == "heat"


def test_stale_measurement_fails_safe():
    monitor = make_monitor(measurement_max_age_hours=0.05)
    monitor._handle_thaw(thaw_msg(1.0, 10.0))
    monitor._handle_forecast(forecast_msg(1.5, keep_mm=[11, 12], stop_mm=[10, 10]))
    alert = monitor.mq_out.last
    assert alert["status"] == "alert"
    assert alert["recommended_action"] == "stop"
    assert alert["reason"] == "measurement_stale"


def test_fail_safe_can_be_disabled():
    monitor = make_monitor(fail_safe_on_stale_measurement=False)
    monitor._handle_thaw(thaw_msg(1.0, 10.0))
    monitor._handle_forecast(forecast_msg(1.5, keep_mm=[11, 12], stop_mm=[10, 10]))
    assert monitor.mq_out.last["recommended_action"] == "heat"


def test_no_front_yet_is_not_a_stale_measurement():
    """Regression: before anything thaws there is no front, and that is normal.

    The reconstruction reports radius_max_m = None until the first 0 C isotherm
    appears. Treating that absence as a lost sensor makes the monitor fail safe
    on the very first snapshot, so every run would open by stopping its own
    heater.
    """
    monitor = make_monitor()
    monitor._handle_thaw({"timestamp": "t", "time_hours": 0.0,
                          "radius_max_m": None, "radius_min_m": None})
    alert = monitor.mq_out.last
    assert alert["status"] == "ok"
    assert alert["recommended_action"] == "heat"
    assert alert["predicate_m1"] is False
    assert alert["predicate_m2"] is False


def test_startup_without_a_measurement_does_not_fail_safe():
    """The forecast can arrive first; an empty history is not a stale one."""
    monitor = make_monitor()
    monitor._handle_forecast(forecast_msg(0.0, keep_mm=[1, 2], stop_mm=[1, 1]))
    alert = monitor.mq_out.last
    assert alert["status"] == "ok"
    assert alert["reason"] == "awaiting_measurement"


def test_a_forecast_without_a_lookahead_is_treated_as_unavailable():
    """An older forecast service publishes no branches; degrade, do not crash."""
    monitor = make_monitor()
    monitor._handle_thaw(thaw_msg(1.0, 26.0))
    monitor._handle_forecast({"timestamp": "t", "time_hours": 1.0, "radius_max_m": 0.026})
    alert = monitor.mq_out.last
    assert alert["forecast_available"] is False
    assert alert["status"] == "warning"


def test_interleaved_thaw_messages_do_not_clobber_the_streak():
    """Regression: a thaw message must not reset confirmation.

    The thaw queue normally advances the clock past the staleness window before
    the next forecast lands. A monitor that reset the counter whenever the
    forecast merely *looked* stale would knock it back to zero every cycle, so
    the streak could never exceed one and the whole predictive arm would be
    silently dead -- passing every other test while never firing on real data.
    """
    monitor = make_monitor(confirm_updates=3, forecast_max_age_hours=0.05)
    for i in range(3):
        t = 1.0 + i * 0.01                      # 36 s apart: a continuous stream
        monitor._handle_forecast(forecast_msg(t, keep_mm=[35, 45], stop_mm=[31, 32]))
        monitor._handle_thaw(thaw_msg(t + 0.008, 30.0))
    monitor._handle_forecast(forecast_msg(1.03, keep_mm=[35, 45], stop_mm=[31, 32]))
    alert = monitor.mq_out.last
    assert alert["predicate_f1"] is True
    assert alert["status"] == "pre_alert"


def test_a_break_in_the_forecast_stream_restarts_the_streak():
    """'Three consecutive updates' has to mean consecutive in time."""
    monitor = make_monitor(confirm_updates=2, forecast_max_age_hours=0.05)
    monitor._handle_forecast(forecast_msg(1.0, keep_mm=[35, 45], stop_mm=[31, 32]))
    monitor._handle_forecast(forecast_msg(3.0, keep_mm=[35, 45], stop_mm=[31, 32]))
    assert monitor._f1_streak == 1, "a 2 h gap extended the streak"
    monitor._handle_forecast(forecast_msg(3.01, keep_mm=[35, 45], stop_mm=[31, 32]))
    assert monitor.mq_out.last["predicate_f1"] is True


def test_stale_forecast_clears_a_partial_streak():
    monitor = make_monitor(confirm_updates=2)
    monitor._handle_forecast(forecast_msg(1.0, keep_mm=[35, 45], stop_mm=[31, 32]))
    assert monitor._f1_streak == 1
    monitor._handle_thaw(thaw_msg(2.0, 30.0))          # forecast now stale
    monitor._handle_forecast(forecast_msg(2.0, keep_mm=[35, 45], stop_mm=[31, 32]))
    assert monitor.mq_out.last["predicate_f1"] is False


# ---------------------------------------------------------------------------
# EVIDENCE
# ---------------------------------------------------------------------------
def test_alert_carries_the_evidence_for_its_decision():
    monitor = make_monitor()
    alert = drive(monitor, 1.0, r_max_mm=30.0, keep_mm=[35, 44], stop_mm=[31, 32])
    assert alert["metric"] == "predictive"
    assert alert["measured_radius_m"] == pytest.approx(0.030)
    assert alert["forecast_radius_keep_m"] == pytest.approx(0.044)
    assert alert["forecast_radius_stop_m"] == pytest.approx(0.032)
    assert alert["lookahead_hours"] == pytest.approx(1.0)
    assert alert["measurement_age_hours"] == pytest.approx(0.0)
    assert alert["forecast_age_hours"] == pytest.approx(0.0)
    for key in ("predicate_m1", "predicate_m2", "predicate_f1", "predicate_f2"):
        assert isinstance(alert[key], bool)


def test_time_to_breach_is_the_first_crossing_of_the_keep_branch():
    monitor = make_monitor()
    # 4 samples over 1 h -> 0.25 h apart; the third crosses.
    alert = drive(monitor, 1.0, r_max_mm=30.0,
                  keep_mm=[31, 35, 41, 46], stop_mm=[30, 30, 31, 31])
    assert alert["time_to_breach_hours"] == pytest.approx(0.75)


def test_time_to_breach_is_none_when_no_branch_crosses():
    alert = drive(make_monitor(), 1.0, 20.0, keep_mm=[21, 22], stop_mm=[20, 20])
    assert alert["time_to_breach_hours"] is None


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
def test_dual_mode_is_untouched_by_the_predictive_arm():
    """The published figures depend on dual; the two engines stay separate."""
    monitor = make_monitor(mode="dual")
    monitor._handle_thaw(thaw_msg(1.0, 30.0))
    alert = monitor.mq_out.last
    assert alert["metric"] == "dual"
    assert "predicate_f1" not in alert


def test_unknown_f2_action_is_rejected_at_construction():
    with pytest.raises(ValueError, match="f2_action"):
        make_monitor(f2_action="explode")

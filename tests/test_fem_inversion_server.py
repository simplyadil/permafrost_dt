"""Unit tests for the inversion service's bookkeeping.

These cover the parts that surround the fit rather than the fit itself: how
sensor snapshots are parsed and accumulated, how the fitting window is cut out
of them, when a fit is allowed to start, and how the result is packaged. None of
it touches the FEM, so the whole file runs in well under a second.

The fit itself is covered elsewhere, by two scripts that need the solver:

* ``software/tools/verify_inversion_recovery.py``    -- does it find known parameters?
* ``software/tools/verify_inversion_equivalence.py`` -- does the fast forward model
  match the one it replaced?

The distinction is worth keeping in mind when something breaks. A failure here
means the service mishandled its inputs; a failure there means the numerics
moved. They fail independently and for unrelated reasons.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from software.digital_twin.simulator.fem_inversion.fem_inversion_server import (
    DEFAULT_INVERT_PARAMS,
    INVERTIBLE_PARAMS,
    FEMInversionServer,
)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def utcnow() -> datetime:
    """Naive UTC, matching what the service stores internally.

    The service timestamps with ``datetime.utcnow()``, which returns a naive
    value, and compares it against the argument passed to
    ``_can_run_inversion``. Passing a timezone-aware datetime would raise on the
    subtraction, so the tests keep the same convention -- without the deprecated
    call, which is scheduled for removal.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def make_server(**overrides) -> FEMInversionServer:
    """Server with no message bus or database attached.

    The constructor only assembles configuration objects, so this is cheap and
    needs no running infrastructure.
    """
    kwargs = dict(window_size=5, min_snapshots=5, min_sensors=3)
    kwargs.update(overrides)
    return FEMInversionServer(**kwargs)


def server_cap() -> float:
    """Default base ``cap_s``, read off a fresh server rather than hard-coded."""
    return float(make_server().base_params["cap_s"])


def sensor(sensor_id: str, temperature: float, x_m: float = 0.05, z_m: float = 0.10) -> dict:
    return {"sensor_id": sensor_id, "x_m": x_m, "z_m": z_m, "temperature": temperature}


def snapshot(time_hours: float, sensors: list[dict]) -> dict:
    return {"timestamp": "2026-04-08T00:00:00", "time_hours": time_hours, "sensors": sensors}


def fill(server: FEMInversionServer, count: int, ids=("A", "B", "C"), start: float = 0.0) -> None:
    """Push `count` well-formed snapshots at 30 s spacing; snapshot i reads i - 1 + n."""
    for i in range(count):
        t = start + i * (30.0 / 3600.0)
        server._upsert_snapshot(
            snapshot(t, [sensor(sid, -1.0 + i + n) for n, sid in enumerate(ids)])
        )


# ---------------------------------------------------------------------------
# PARSING
# ---------------------------------------------------------------------------
class TestParseSnapshot:
    def test_accepts_a_well_formed_message(self):
        parsed = make_server()._parse_snapshot(snapshot(1.5, [sensor("A", -2.0)]))
        assert parsed is not None
        assert parsed["time_hours"] == 1.5
        assert parsed["sensors"] == [sensor("A", -2.0)]

    @pytest.mark.parametrize("bad_time", ["not-a-number", None, {}])
    def test_rejects_non_numeric_time(self, bad_time):
        msg = snapshot(0.0, [sensor("A", -2.0)])
        msg["time_hours"] = bad_time
        assert make_server()._parse_snapshot(msg) is None

    def test_rejects_missing_time(self):
        assert make_server()._parse_snapshot({"sensors": [sensor("A", -2.0)]}) is None

    @pytest.mark.parametrize("sensors", [[], "not-a-list", None])
    def test_rejects_empty_or_malformed_sensor_list(self, sensors):
        assert make_server()._parse_snapshot(snapshot(1.0, sensors)) is None

    def test_drops_individual_bad_sensors_but_keeps_the_rest(self):
        msg = snapshot(1.0, [
            sensor("good", -2.0),
            {"sensor_id": "no_temp", "x_m": 0.0, "z_m": 0.0},          # missing key
            {"sensor_id": "bad_temp", "x_m": 0.0, "z_m": 0.0, "temperature": "warm"},
        ])
        parsed = make_server()._parse_snapshot(msg)
        assert [s["sensor_id"] for s in parsed["sensors"]] == ["good"]

    def test_returns_none_when_every_sensor_is_bad(self):
        msg = snapshot(1.0, [{"sensor_id": "x", "x_m": 0.0, "z_m": 0.0, "temperature": "warm"}])
        assert make_server()._parse_snapshot(msg) is None

    def test_time_hours_is_coerced_from_a_numeric_string(self):
        msg = snapshot(0.0, [sensor("A", -2.0)])
        msg["time_hours"] = "2.5"
        assert make_server()._parse_snapshot(msg)["time_hours"] == 2.5


# ---------------------------------------------------------------------------
# SNAPSHOT BOOKKEEPING
# ---------------------------------------------------------------------------
class TestUpsertSnapshot:
    def test_appends_in_time_order_regardless_of_arrival_order(self):
        server = make_server()
        for t in (0.3, 0.1, 0.2):
            server._upsert_snapshot(snapshot(t, [sensor("A", 0.0)]))
        assert [s["time_hours"] for s in server._snapshots] == [0.1, 0.2, 0.3]

    def test_replaces_rather_than_duplicates_the_same_time(self):
        server = make_server()
        server._upsert_snapshot(snapshot(0.1, [sensor("A", -5.0)]))
        server._upsert_snapshot(snapshot(0.1, [sensor("A", 12.0)]))
        assert len(server._snapshots) == 1
        assert server._snapshots[0]["sensors"][0]["temperature"] == 12.0

    def test_trims_to_the_window_size_keeping_the_newest(self):
        server = make_server(window_size=3)
        fill(server, 6)
        assert len(server._snapshots) == 3
        times = [s["time_hours"] for s in server._snapshots]
        assert times == sorted(times)
        # the three most recent of six, at 30 s spacing
        assert times[-1] == pytest.approx(5 * 30.0 / 3600.0)


# ---------------------------------------------------------------------------
# WINDOW ASSEMBLY
# ---------------------------------------------------------------------------
class TestObservationWindow:
    def test_returns_none_before_min_snapshots(self):
        server = make_server(min_snapshots=5)
        fill(server, 4)
        assert server._build_observation_window() is None

    def test_times_are_relative_to_the_first_snapshot(self):
        server = make_server()
        fill(server, 5, start=3.0)
        times, _positions, _obs, _ic = server._build_observation_window()
        assert times[0] == pytest.approx(server.min_horizon_hours)
        assert times[-1] == pytest.approx(4 * 30.0 / 3600.0)

    def test_absolute_times_when_relative_window_is_off(self):
        server = make_server(relative_time_window=False)
        fill(server, 5, start=3.0)
        times, _positions, _obs, _ic = server._build_observation_window()
        assert times[0] == pytest.approx(3.0)

    def test_first_time_never_collapses_to_zero(self):
        """A zero horizon would make the forward model refuse to run."""
        server = make_server()
        fill(server, 5)
        times, _positions, _obs, _ic = server._build_observation_window()
        assert all(t >= server.min_horizon_hours for t in times)

    def test_keeps_only_sensors_present_in_every_snapshot(self):
        server = make_server(min_sensors=1)
        for i in range(5):
            sensors = [sensor("always", float(i)), sensor("sometimes", float(i))]
            if i == 2:
                sensors = sensors[:1]  # channel drops out for one reading
            server._upsert_snapshot(snapshot(i * 30.0 / 3600.0, sensors))
        _times, positions, obs, _ic = server._build_observation_window()
        assert [p["sensor_id"] for p in positions] == ["always"]
        assert set(obs) == {"always"}

    def test_returns_none_when_too_few_sensors_are_common(self):
        server = make_server(min_sensors=3)
        fill(server, 5, ids=("A", "B"))
        assert server._build_observation_window() is None

    def test_returns_none_when_no_sensor_is_common_to_all(self):
        server = make_server(min_sensors=1)
        for i in range(5):
            server._upsert_snapshot(snapshot(i * 30.0 / 3600.0, [sensor(f"only_{i}", float(i))]))
        assert server._build_observation_window() is None

    def test_observations_line_up_with_the_times(self):
        server = make_server(min_sensors=1)
        for i in range(5):
            server._upsert_snapshot(snapshot(i * 30.0 / 3600.0, [sensor("A", float(i))]))
        times, _positions, obs, _ic = server._build_observation_window()
        assert obs["A"] == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert len(obs["A"]) == len(times)

    def test_initial_condition_is_the_oldest_snapshot(self):
        """The IC seeds every forward run, so taking the wrong end is silently wrong."""
        server = make_server(min_sensors=1)
        for i in range(5):
            server._upsert_snapshot(snapshot(i * 30.0 / 3600.0, [sensor("A", float(i))]))
        _times, _positions, _obs, ic = server._build_observation_window()
        assert [s["temperature"] for s in ic] == [0.0]

    def test_initial_condition_is_a_copy(self):
        """The fit runs on a worker thread; a shared reference could be mutated."""
        server = make_server(min_sensors=1)
        fill(server, 5, ids=("A",))
        _times, _positions, _obs, ic = server._build_observation_window()
        ic[0]["temperature"] = 999.0
        assert server._snapshots[0]["sensors"][0]["temperature"] != 999.0


class TestSensorRadius:
    def test_wall_sensors_are_left_out_of_the_fit(self):
        server = make_server(min_sensors=1, min_sensor_radius_m=0.02)
        for i in range(5):
            server._upsert_snapshot(snapshot(i * 30.0 / 3600.0, [
                sensor("wall", 5.0, x_m=0.0), sensor("near", 4.0, x_m=0.015),
                sensor("soil", -1.0, x_m=0.03)]))
        _times, positions, obs, ic = server._build_observation_window()
        assert list(obs) == ["soil"]
        assert [p["sensor_id"] for p in positions] == ["soil"]
        # ...but they still seed the initial field.
        assert {s["sensor_id"] for s in ic} == {"wall", "near", "soil"}

    def test_default_keeps_every_sensor(self):
        server = make_server(min_sensors=1)
        for i in range(5):
            server._upsert_snapshot(snapshot(i * 30.0 / 3600.0, [
                sensor("wall", 5.0, x_m=0.0), sensor("soil", -1.0, x_m=0.03)]))
        _times, _positions, obs, _ic = server._build_observation_window()
        assert sorted(obs) == ["soil", "wall"]


class TestWindowStride:
    def test_stride_one_is_the_old_behaviour(self):
        server = make_server(min_sensors=1)
        fill(server, 9, ids=("A",))
        _times, _positions, obs, _ic = server._build_observation_window()
        assert obs["A"] == [3.0, 4.0, 5.0, 6.0, 7.0]

    def test_stride_spreads_the_same_points_over_more_data(self):
        server = make_server(min_sensors=1, window_stride=3)
        fill(server, 20, ids=("A",))
        times, _positions, obs, ic = server._build_observation_window()
        assert obs["A"] == [6.0, 9.0, 12.0, 15.0, 18.0]
        assert [s["temperature"] for s in ic] == [6.0]
        assert times[-1] == pytest.approx(12 * 30.0 / 3600.0)

    def test_window_always_ends_on_the_newest_snapshot(self):
        server = make_server(min_sensors=1, window_stride=4)
        fill(server, 30, ids=("A",))
        _times, _positions, obs, _ic = server._build_observation_window()
        assert obs["A"][-1] == 28.0

    def test_buffer_holds_exactly_one_strided_window(self):
        server = make_server(window_stride=3)
        fill(server, 40)
        assert len(server._snapshots) == (5 - 1) * 3 + 1

    def test_not_ready_until_the_strided_window_is_full(self):
        server = make_server(min_sensors=1, window_stride=3)
        fill(server, 12, ids=("A",))
        assert server._build_observation_window() is None
        fill(server, 13, ids=("A",))
        assert server._build_observation_window() is not None


# ---------------------------------------------------------------------------
# GATING
# ---------------------------------------------------------------------------
class TestGating:
    def test_blocked_until_enough_snapshots(self):
        server = make_server(min_snapshots=5)
        fill(server, 4)
        assert server._can_run_inversion(1.0, utcnow()) is False
        fill(server, 5)
        assert server._can_run_inversion(1.0, utcnow()) is True

    def test_respects_the_minimum_data_time_between_fits(self):
        server = make_server(min_update_hours=0.5)
        fill(server, 5)
        server._last_inversion_time = 1.0
        assert server._can_run_inversion(1.2, utcnow()) is False
        assert server._can_run_inversion(1.6, utcnow()) is True

    def test_respects_the_minimum_wallclock_between_fits(self):
        server = make_server(min_update_seconds=30.0)
        fill(server, 5)
        now = utcnow()
        server._last_inversion_wallclock = now
        assert server._can_run_inversion(1.0, now + timedelta(seconds=10)) is False
        assert server._can_run_inversion(1.0, now + timedelta(seconds=31)) is True

    def test_first_fit_is_not_blocked_by_either_limit(self):
        server = make_server(min_update_hours=99.0, min_update_seconds=99.0)
        fill(server, 5)
        assert server._can_run_inversion(0.0, utcnow()) is True

    def test_a_fit_in_flight_defers_instead_of_starting_a_second(self):
        server = make_server()
        fill(server, 5)
        server._inversion_running = True
        assert server._prepare_inversion_job(utcnow()) is None
        assert server._pending_recheck is True

    def test_preparing_a_job_marks_the_service_busy(self):
        server = make_server(min_sensors=1)
        fill(server, 5)
        job = server._prepare_inversion_job(utcnow())
        assert job is not None
        assert server._inversion_running is True
        assert server._pending_recheck is False

    def test_no_job_without_snapshots(self):
        assert make_server()._prepare_inversion_job(utcnow()) is None


# ---------------------------------------------------------------------------
# PARAMETERISATION
# ---------------------------------------------------------------------------
class TestParameterHandling:
    def test_non_invertible_names_are_dropped(self):
        server = make_server(invert_params=["cap_s", "porosity", "lambda_thawed"])
        assert server.invert_params == ["cap_s"]

    def test_duplicates_are_collapsed(self):
        server = make_server(invert_params=["cap_s", "cap_s", "lambda_s"])
        assert server.invert_params == ["cap_s", "lambda_s"]

    def test_falls_back_to_defaults_when_nothing_valid_remains(self):
        server = make_server(invert_params=["porosity", "rho_s"])
        assert server.invert_params == list(DEFAULT_INVERT_PARAMS)

    def test_every_default_is_invertible(self):
        assert set(DEFAULT_INVERT_PARAMS) <= set(INVERTIBLE_PARAMS)

    def test_bounds_are_taken_from_config(self):
        server = make_server(invert_params=["cap_s"], relative_bounds={"cap_s": [0.4, 3.0]})
        assert server.relative_bounds["cap_s"] == (0.4, 3.0)

    @pytest.mark.parametrize("bad", [[2.0, 1.0], [1.0, 1.0], ["x", "y"], [1.0], "nonsense", None])
    def test_unusable_bounds_fall_back_to_the_default_box(self, bad):
        server = make_server(invert_params=["cap_s"], relative_bounds={"cap_s": bad})
        assert server.relative_bounds["cap_s"] == (0.5, 1.5)

    def test_scales_are_applied_multiplicatively_to_the_base_values(self):
        server = make_server(invert_params=["cap_s", "lambda_s"])
        base = {"cap_s": 800.0, "lambda_s": 4.0}
        params = server._normalized_to_params(np.array([1.5, 0.5]), base)
        assert params["cap_s"] == pytest.approx(1200.0)
        assert params["lambda_s"] == pytest.approx(2.0)

    def test_non_inverted_parameters_pass_through_untouched(self):
        server = make_server(invert_params=["cap_s"])
        params = server._normalized_to_params(np.array([2.0]), {"cap_s": 800.0})
        assert params["porosity"] == server.base_params["porosity"]
        assert params["latent_heat"] == server.base_params["latent_heat"]


# ---------------------------------------------------------------------------
# OBJECTIVE
# ---------------------------------------------------------------------------
class TestLoss:
    def test_mse_is_the_mean_squared_residual(self):
        server = make_server(loss="mse")
        loss = server._compute_loss({"A": [1.0, 2.0]}, {"A": [0.0, 0.0]})
        assert loss == pytest.approx((1.0 + 4.0) / 2)

    def test_mae_is_the_mean_absolute_residual(self):
        server = make_server(loss="mae")
        loss = server._compute_loss({"A": [1.0, -3.0]}, {"A": [0.0, 0.0]})
        assert loss == pytest.approx((1.0 + 3.0) / 2)

    def test_a_perfect_prediction_scores_zero(self):
        server = make_server()
        assert server._compute_loss({"A": [1.0, 2.0]}, {"A": [1.0, 2.0]}) == 0.0

    def test_no_overlapping_data_is_infinite_rather_than_zero(self):
        """Averaging an empty list must not look like a perfect fit."""
        server = make_server()
        assert server._compute_loss({}, {}) == float("inf")
        assert server._compute_loss({"A": []}, {"A": []}) == float("inf")

    def test_sensors_missing_from_the_prediction_are_skipped(self):
        server = make_server()
        loss = server._compute_loss({"A": [1.0]}, {"A": [0.0], "B": [5.0]})
        assert loss == pytest.approx(1.0)

    def test_an_unknown_loss_name_is_rejected_at_construction(self):
        with pytest.raises(ValueError):
            make_server(loss="rmse")


class TestInitialScales:
    def test_cold_start_begins_at_the_measured_values(self):
        server = make_server(invert_params=["cap_s"], relative_bounds={"cap_s": [0.5, 2.0]})
        assert server._initial_scales() == [1.0]

    def test_cold_start_is_clamped_when_the_box_excludes_one(self):
        server = make_server(invert_params=["cap_s"], relative_bounds={"cap_s": [1.2, 2.0]})
        assert server._initial_scales() == [1.2]

    def test_warm_start_reuses_the_previous_fit(self):
        server = make_server(invert_params=["cap_s"], relative_bounds={"cap_s": [0.5, 2.0]})
        server._last_best_scales = [1.4]
        assert server._initial_scales() == [1.4]

    def test_warm_start_is_clamped_into_the_current_bounds(self):
        server = make_server(invert_params=["cap_s"], relative_bounds={"cap_s": [0.5, 1.1]})
        server._last_best_scales = [1.9]
        assert server._initial_scales() == [1.1]

    def test_stale_warm_start_of_the_wrong_length_is_ignored(self):
        server = make_server(invert_params=["cap_s", "lambda_s"])
        server._last_best_scales = [1.4]
        assert server._initial_scales() == [1.0, 1.0]


class TestScaleSpace:
    """The optimiser loop with the FEM replaced by a closed-form stand-in."""

    @staticmethod
    def fit(scale_space: str, target: dict[str, float], bounds=(0.2, 5.0)):
        server = make_server(
            invert_params=["cap_s", "lambda_s"],
            relative_bounds={"cap_s": list(bounds), "lambda_s": list(bounds)},
            scale_space=scale_space,
        )
        base = {n: server.base_params[n] for n in ("cap_s", "lambda_s")}

        # One "sensor" per parameter whose reading is that parameter's scale, so
        # the loss minimum sits exactly at ``target``.
        def fake_series(params, positions, times_hours, ic_sensors=None):
            return {n: [params[n] / base[n]] for n in base}

        server._generate_series = fake_series
        obs = {n: [target[n]] for n in base}
        _params, info = server._run_inversion([0.1], [], obs, [])
        return server, info

    @pytest.mark.parametrize("space", ["linear", "log"])
    def test_finds_an_interior_optimum(self, space):
        _server, info = self.fit(space, {"cap_s": 0.4, "lambda_s": 2.5})
        assert info["best_scales"]["cap_s"] == pytest.approx(0.4, rel=1e-3)
        assert info["best_scales"]["lambda_s"] == pytest.approx(2.5, rel=1e-3)

    def test_log_space_reports_scales_not_logs(self):
        server, info = self.fit("log", {"cap_s": 0.5, "lambda_s": 2.0})
        assert info["initial_scales"] == {"cap_s": 1.0, "lambda_s": 1.0}
        assert server._last_best_scales == pytest.approx([0.5, 2.0], rel=1e-3)

    def test_log_space_respects_the_bounds(self):
        _server, info = self.fit("log", {"cap_s": 0.01, "lambda_s": 50.0}, bounds=(0.25, 4.0))
        assert info["best_scales"]["cap_s"] == pytest.approx(0.25, rel=1e-6)
        assert info["best_scales"]["lambda_s"] == pytest.approx(4.0, rel=1e-6)

    def test_prior_pulls_a_flat_direction_to_the_prior_value(self):
        server = make_server(invert_params=["cap_s", "lambda_s"],
                             relative_bounds={"cap_s": [0.2, 5.0], "lambda_s": [0.2, 5.0]},
                             scale_space="log", prior_weight=1.0,
                             prior_params={"cap_s": server_cap() * 2.0})

        # Data constrains lambda_s only; cap_s is invisible to it.
        def fake_series(params, positions, times_hours, ic_sensors=None):
            return {"L": [params["lambda_s"] / server.base_params["lambda_s"]]}

        server._generate_series = fake_series
        _params, info = server._run_inversion([0.1], [], {"L": [0.5]}, [])
        assert info["best_scales"]["lambda_s"] == pytest.approx(0.5, rel=1e-3)
        assert info["best_scales"]["cap_s"] == pytest.approx(2.0, rel=1e-3)

    def test_prior_is_off_by_default(self):
        server = make_server()
        assert server._prior_penalty(dict(server.base_params)) == 0.0

    def test_prior_ignores_parameters_that_are_not_inverted(self):
        server = make_server(invert_params=["cap_s"], prior_weight=1.0,
                             prior_params={"lambda_s": 1.0, "cap_s": server_cap()})
        assert list(server.prior_params) == ["cap_s"]

    def test_an_unknown_scale_space_is_rejected(self):
        with pytest.raises(ValueError):
            make_server(scale_space="sqrt")


# ---------------------------------------------------------------------------
# OUTPUT PACKAGING
# ---------------------------------------------------------------------------
class TestBoundarySnapshot:
    BOUNDARIES = [
        {"name": "deep", "boundary_set": "top", "bc_type": "dirichlet", "temperature": -5.0},
        {"name": "pipe", "boundary_set": "left", "bc_type": "robin", "temperature": 27.0, "h_coeff": 5.0},
        {"name": "surface", "boundary_set": "bottom", "bc_type": "neumann", "heat_flux": 0.0},
    ]

    def test_carries_the_material_override(self):
        server = make_server(boundaries=self.BOUNDARIES)
        out = server._build_boundary_snapshot(snapshot(2.0, [sensor("A", 1.0)]), {"cap_s": 900.0})
        assert out["material_override"] == {"cap_s": 900.0}
        assert out["time_hours"] == 2.0

    def test_does_not_mutate_the_configured_boundaries(self):
        server = make_server(
            boundaries=self.BOUNDARIES,
            sensor_overrides=[{"boundary_set": "top", "sensor_id": "probe"}],
        )
        server._build_boundary_snapshot(snapshot(1.0, [sensor("probe", 3.5)]), {})
        assert server.boundaries[0]["temperature"] == -5.0

    def test_sensor_override_rewrites_the_matching_boundary(self):
        server = make_server(
            boundaries=self.BOUNDARIES,
            sensor_overrides=[{"boundary_set": "top", "sensor_id": "probe"}],
        )
        out = server._build_boundary_snapshot(snapshot(1.0, [sensor("probe", 3.5)]), {})
        by_set = {b["boundary_set"]: b for b in out["boundaries"]}
        assert by_set["top"]["temperature"] == 3.5
        assert by_set["left"]["temperature"] == 27.0

    def test_override_is_ignored_when_the_sensor_is_absent(self):
        server = make_server(
            boundaries=self.BOUNDARIES,
            sensor_overrides=[{"boundary_set": "top", "sensor_id": "missing"}],
        )
        out = server._build_boundary_snapshot(snapshot(1.0, [sensor("other", 3.5)]), {})
        assert out["boundaries"][0]["temperature"] == -5.0

    def test_neumann_boundaries_are_never_given_a_temperature(self):
        server = make_server(
            boundaries=self.BOUNDARIES,
            sensor_overrides=[{"boundary_set": "bottom", "sensor_id": "probe"}],
        )
        out = server._build_boundary_snapshot(snapshot(1.0, [sensor("probe", 3.5)]), {})
        by_set = {b["boundary_set"]: b for b in out["boundaries"]}
        assert "temperature" not in by_set["bottom"]

"""FEM-based inversion service for 2D sensor snapshots.

Purpose: sharpen the FEM forecast by replacing literature soil parameters with
values calibrated against the live sensor stream. The service repeatedly runs
the same forward FEM the forecast service uses, over a short window of recent
snapshots, and fits the soil parameters that minimise the mismatch with the
measured temperatures. The result is published on the inversion queue and --
when ``publish_material_override_boundary`` is on -- pushed to the forecast as
a ``material_override`` on the boundary snapshot.

Only the parameters in ``INVERTIBLE_PARAMS`` are estimated:

* ``cap_s``       -- specific heat of the solid grains
* ``k_parameter`` -- sharpness of the freezing curve around ``tr_k``
* ``lambda_s``    -- thermal conductivity of the solid grains

Everything else in ``base_params`` (porosity, densities, water/ice properties,
latent heat, ``tr_k``) is held fixed at its measured/literature value and just
travels through to the forward model. Note that the solver mixes the bulk
conductivity from ``lambda_s``/``lambda_w``/``lambda_i`` and the phase
saturations, so the old ``lambda_thawed``/``lambda_frozen`` pair no longer
exists in either model or inversion.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from random import Random
from threading import Lock, Thread
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from software.digital_twin.communication.messaging import RabbitMQClient, RabbitMQConfig, resolve_queue_config
from software.digital_twin.data_access.influx_utils import InfluxConfig, InfluxHelper
from software.digital_twin.simulator.fem_inversion.window_forward_model import WindowForwardModel
from software.utils.logging_setup import get_logger

SENSOR_QUEUE = "permafrost.record.sensors.2d"
INVERSION_QUEUE = "permafrost.record.fem_inversion"
FEM_BOUNDARY_QUEUE = "permafrost.record.boundary.2d"

SENSOR_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "sensor_message_2d.json"
INVERSION_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "fem_inversion_output_message.json"
FEM_BOUNDARY_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "boundary_condition_message.json"

# Soil parameters this service is allowed to estimate. Anything outside this
# set is fixed by measurement/literature and must not be handed to the
# optimiser, however the config asks for it.
INVERTIBLE_PARAMS = ("cap_s", "k_parameter", "lambda_s")
DEFAULT_INVERT_PARAMS = list(INVERTIBLE_PARAMS)


class FEMInversionServer:
    """Consumes 2D sensor snapshots and estimates material parameters via FEM inversion."""

    def __init__(
        self,
        influx_config: InfluxConfig | None = None,
        input_config: RabbitMQConfig | None = None,
        output_config: RabbitMQConfig | None = None,
        fem_input_config: RabbitMQConfig | None = None,
        *,
        input_queue: str | None = None,
        output_queue: str | None = None,
        fem_input_queue: str | None = None,
        site_id: str = "default",
        invert_params: list[str] | None = None,
        base_params: dict | None = None,
        relative_bounds: dict[str, list[float] | tuple[float, float]] | None = None,
        loss: str = "mse",
        window_size: int = 5,
        window_stride: int = 1,
        min_snapshots: int = 5,
        min_sensors: int = 3,
        min_sensor_radius_m: float = 0.0,
        min_update_hours: float = 0.0,
        min_update_seconds: float = 0.0,
        warmup_hours: float = 0.5,
        relative_time_window: bool = True,
        min_horizon_hours: float = 1e-3,
        maxiter: int = 40,
        maxfun: int = 80,
        scale_space: str = "linear",
        prior_params: dict | None = None,
        prior_weight: float = 0.0,
        seed: int | None = 42,
        publish_material_override_boundary: bool = False,
        boundaries: list[dict] | None = None,
        sensor_overrides: list[dict] | None = None,
        config_path: str = "startup.conf",
    ) -> None:
        self.logger = get_logger("FEMInversionServer")
        self.influx_config = influx_config or InfluxConfig()
        self.input_config = resolve_queue_config(
            input_config,
            queue=input_queue or SENSOR_QUEUE,
            schema_path=SENSOR_SCHEMA,
        )
        self.output_config = resolve_queue_config(
            output_config,
            queue=output_queue or INVERSION_QUEUE,
            schema_path=INVERSION_SCHEMA,
        )
        self.fem_input_config = resolve_queue_config(
            fem_input_config,
            queue=fem_input_queue or FEM_BOUNDARY_QUEUE,
            schema_path=FEM_BOUNDARY_SCHEMA,
        )

        self.site_id = site_id
        self.invert_params = self._resolve_invert_params(invert_params)
        self.base_params = self._build_base_params(base_params)
        self.relative_bounds = self._build_bounds(relative_bounds)
        self.loss = str(loss).strip().lower()
        if self.loss not in {"mse", "mae"}:
            raise ValueError("loss must be either 'mse' or 'mae'.")

        self.window_size = max(int(window_size), 2)
        # Every ``window_stride``-th snapshot goes into the window, so the same
        # number of points spans ``window_stride`` times more experiment. The
        # storage term (``cap_s``) and bulk conduction only show up in a sensor
        # history integrated over time; at a 30 s cadence a 20-point window is
        # ten minutes, which is mostly the seeded initial field relaxing.
        self.window_stride = max(int(window_stride), 1)
        self.min_snapshots = max(int(min_snapshots), 2)
        self.min_sensors = max(int(min_sensors), 1)
        # Sensors closer than this to the axis (|x_m|) are left out of the fit.
        # The pipe-wall thermocouples read the boundary condition, not the soil:
        # on April 8 the 12 within 2 cm carry a 1.5-3 C pipe-BC bias while the 59
        # beyond fit to ~0.2 C, and no soil parameter can remove a boundary bias
        # -- the optimiser just drives the parameters to bounds trying. They still
        # seed the initial field, which needs them.
        self.min_sensor_radius_m = max(float(min_sensor_radius_m), 0.0)
        self.min_update_hours = max(float(min_update_hours), 0.0)
        self.min_update_seconds = max(float(min_update_seconds), 0.0)
        self.warmup_hours = max(float(warmup_hours), 0.0)
        self.relative_time_window = bool(relative_time_window)
        self.min_horizon_hours = max(float(min_horizon_hours), 1e-6)
        self.maxiter = max(int(maxiter), 1)
        self.maxfun = max(int(maxfun), 1)
        # "log" optimises log(scale): halving and doubling are the same distance
        # from the start, and the finite-difference gradient is relative to the
        # current value instead of absolute. In linear space a parameter started
        # at 2x its truth has to cover four times the distance of one started at
        # 0.5x, and L-BFGS-B tends to run out of budget on the long side.
        self.scale_space = str(scale_space).strip().lower()
        # Optional Tikhonov prior: adds ``prior_weight * sum(ln(p / prior)^2)``
        # over the inverted parameters named in ``prior_params`` (absolute
        # values, independent of ``base_params`` so the prior does not move
        # with the starting point). On April 8 ``cap_s`` and ``lambda_s`` trade
        # off along a ridge -- scaling both together barely changes the
        # response -- and without a prior the fit slides down it to the bounds.
        self.prior_weight = max(float(prior_weight), 0.0)
        self.prior_params: dict[str, float] = {}
        for name, value in (prior_params or {}).items():
            if name in self.invert_params:
                try:
                    if float(value) > 0.0:
                        self.prior_params[name] = float(value)
                except (TypeError, ValueError):
                    continue
        if self.scale_space not in {"linear", "log"}:
            raise ValueError("scale_space must be either 'linear' or 'log'.")

        self.publish_material_override_boundary = bool(publish_material_override_boundary)
        self.boundaries = boundaries or []
        self.sensor_overrides = sensor_overrides or []

        self.config_path = str(config_path)
        self._window_model: WindowForwardModel | None = None
        # Calibrated boundary snapshots as they arrived, (time_hours, snapshot),
        # so a window is fitted under the boundary in force at each step rather
        # than the latest one throughout. Trimmed to the buffered window.
        self._boundary_history: list[tuple[float, dict]] = []

        self._rng = Random(seed)
        self._snapshots: list[dict] = []
        # First snapshot time ever seen, which is what ``warmup_hours`` is
        # measured from. It cannot be taken from ``_snapshots``: that list is
        # trimmed to the fitting window, so its oldest entry is only ever a few
        # snapshots old and would satisfy any warm-up immediately.
        self._first_time_hours: float | None = None
        self._warmup_announced = False
        self._last_inversion_time: float | None = None
        self._last_inversion_wallclock: datetime | None = None
        self._last_best_scales: list[float] | None = None
        self._eval_count: int = 0
        self._inversion_running = False
        self._pending_recheck = False
        self._state_lock = Lock()

        self.mq_in: Optional[RabbitMQClient] = None
        self.mq_out: Optional[RabbitMQClient] = None
        self.mq_fem_in: Optional[RabbitMQClient] = None
        self.influx: Optional[InfluxHelper] = None

    def _resolve_invert_params(self, invert_params: list[str] | None) -> list[str]:
        """Keep only parameters this service is allowed to estimate.

        Names outside ``INVERTIBLE_PARAMS`` are dropped with a warning rather
        than raising, so a stale config (e.g. one still asking for the removed
        ``lambda_thawed``, or for the now-fixed ``porosity``) degrades to the
        default set instead of taking the service down at startup.
        """
        requested = [str(name) for name in (invert_params or DEFAULT_INVERT_PARAMS)]
        resolved: list[str] = []
        rejected: list[str] = []
        for name in requested:
            if name not in INVERTIBLE_PARAMS:
                rejected.append(name)
            elif name not in resolved:
                resolved.append(name)

        if rejected:
            self.logger.warning(
                "Ignoring non-invertible parameter(s) %s; invertible set is %s",
                rejected,
                list(INVERTIBLE_PARAMS),
            )
        if not resolved:
            self.logger.warning("No invertible parameters configured; falling back to %s", DEFAULT_INVERT_PARAMS)
            return list(DEFAULT_INVERT_PARAMS)
        return resolved

    def _build_base_params(self, base_params: dict | None) -> dict[str, float]:
        # Fixed properties (measured or from literature) followed by the three
        # inverted unknowns. Only the latter are handed to the optimiser; see
        # INVERTIBLE_PARAMS and the module docstring.
        defaults = {
            "porosity": 0.43,
            "rho_s": 2650.0,
            "rho_w": 1000.0,
            "rho_i": 910.0,
            "cap_w": 4180.0,
            "cap_i": 1930.0,
            "latent_heat": 334000.0,
            "tr_k": 273.15,
            "lambda_w": 0.56,
            "lambda_i": 2.24,
            # Inverted:
            "cap_s": 790.0,
            "k_parameter": 3.0,
            "lambda_s": 3.51,
        }
        if not isinstance(base_params, dict):
            return defaults
        merged = dict(defaults)
        for key, value in base_params.items():
            try:
                merged[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return merged

    def _build_bounds(
        self,
        relative_bounds: dict[str, list[float] | tuple[float, float]] | None,
    ) -> dict[str, tuple[float, float]]:
        bounds: dict[str, tuple[float, float]] = {}
        for name in self.invert_params:
            raw = None if relative_bounds is None else relative_bounds.get(name)
            if isinstance(raw, (list, tuple)) and len(raw) == 2:
                try:
                    lo = float(raw[0])
                    hi = float(raw[1])
                except (TypeError, ValueError):
                    lo, hi = 0.5, 1.5
            else:
                lo, hi = 0.5, 1.5
            if lo >= hi:
                lo, hi = 0.5, 1.5
            bounds[name] = (lo, hi)
        return bounds

    def _parse_snapshot(self, msg: dict) -> dict | None:
        try:
            time_hours = float(msg["time_hours"])
        except (KeyError, TypeError, ValueError):
            self.logger.warning("Skipping sensor snapshot without numeric time_hours")
            return None

        sensors = msg.get("sensors", [])
        if not isinstance(sensors, list) or not sensors:
            return None

        cleaned_sensors = []
        for sensor in sensors:
            try:
                cleaned_sensors.append(
                    {
                        "sensor_id": str(sensor["sensor_id"]),
                        "x_m": float(sensor["x_m"]),
                        "z_m": float(sensor["z_m"]),
                        "temperature": float(sensor["temperature"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue

        if not cleaned_sensors:
            return None

        return {
            "timestamp": msg.get("timestamp") or datetime.utcnow().isoformat(),
            "time_hours": time_hours,
            "sensors": cleaned_sensors,
        }

    def _upsert_snapshot(self, snapshot: dict) -> None:
        time_hours = float(snapshot["time_hours"])
        if self._first_time_hours is None:
            self._first_time_hours = time_hours
        replaced = False
        for idx, existing in enumerate(self._snapshots):
            if np.isclose(float(existing["time_hours"]), time_hours, atol=1e-9):
                self._snapshots[idx] = snapshot
                replaced = True
                break
        if not replaced:
            self._snapshots.append(snapshot)

        self._snapshots.sort(key=lambda item: float(item["time_hours"]))
        if len(self._snapshots) > self._buffer_size:
            self._snapshots = self._snapshots[-self._buffer_size :]
        self._trim_boundary_history()

    def _trim_boundary_history(self) -> None:
        """Drop boundaries older than the one in force at the oldest snapshot."""
        if not self._snapshots or len(self._boundary_history) < 2:
            return
        oldest = float(self._snapshots[0]["time_hours"])
        keep_from = 0
        for idx, (t, _snap) in enumerate(self._boundary_history):
            if t <= oldest + 1e-9:
                keep_from = idx
        self._boundary_history = self._boundary_history[keep_from:]

    def _boundary_schedule(self, t0_hours: float) -> list[tuple[float, dict]] | None:
        """Boundary history on the window clock (t0 = first window snapshot)."""
        if not self._boundary_history:
            return None
        schedule: list[tuple[float, dict]] = []
        for t, snap in self._boundary_history:
            rel = t - t0_hours
            if rel <= 1e-9:
                schedule = [(0.0, snap)]
            else:
                schedule.append((rel, snap))
        return schedule or None

    @property
    def _buffer_size(self) -> int:
        """Raw snapshots needed to fill one strided window."""
        return (self.window_size - 1) * self.window_stride + 1

    def _window_snapshots(self) -> list[dict]:
        """The strided window, always ending on the newest snapshot."""
        recent = self._snapshots[-self._buffer_size :]
        return recent[::-1][:: self.window_stride][::-1]

    def _warmup_remaining_hours(self, current_time_hours: float) -> float:
        """Data-time left before the service is allowed to fit at all.

        Measured on the sensor clock, not the wall clock, so a replay warms up
        over the same stretch of *experiment* the live service would. The first
        fits of a run are the worst ones -- the window is short, the field is
        still relaxing out of its seeded initial condition, and the thaw front
        has barely moved, so the parameters are close to unidentifiable and the
        fit lands on whatever bound the noise points at. Warming up keeps those
        fits from being published as a ``material_override`` the forecast then
        has to live with, since each fit warm-starts from the last.

        If the service is started or restarted mid-experiment this counts from
        the first snapshot *it* sees, which is the intended reading: the warm-up
        is about how much data this instance has accumulated, not how long the
        rig has been running.
        """
        if self.warmup_hours <= 0.0 or self._first_time_hours is None:
            return 0.0
        elapsed = current_time_hours - self._first_time_hours
        return max(self.warmup_hours - elapsed, 0.0)

    def _can_run_inversion(self, current_time_hours: float, now_utc: datetime) -> bool:
        if len(self._window_snapshots()) < self.min_snapshots:
            return False
        if self._warmup_remaining_hours(current_time_hours) > 0.0:
            return False
        if self._last_inversion_time is not None:
            if (current_time_hours - self._last_inversion_time) < self.min_update_hours:
                return False
        if self._last_inversion_wallclock is not None:
            if (now_utc - self._last_inversion_wallclock).total_seconds() < self.min_update_seconds:
                return False
        return True

    def _build_observation_window(
        self,
    ) -> tuple[list[float], list[dict], dict[str, list[float]], list[dict]] | None:
        """Assemble the fitting window: times, positions, observations, and IC.

        The fourth element is the *first* snapshot of the window. Times are
        relative to that snapshot, and every forward FEM evaluation starts from
        it, so it is the measured initial condition the model must be seeded
        with -- otherwise the objective compares real temperature histories
        against runs that all begin from a flat field.
        """
        window = self._window_snapshots()
        if len(window) < self.min_snapshots:
            return None

        common_ids: set[str] | None = None
        sensor_position: dict[str, tuple[float, float]] = {}

        for snapshot in window:
            ids_in_snapshot = set()
            for sensor in snapshot["sensors"]:
                sid = str(sensor["sensor_id"])
                ids_in_snapshot.add(sid)
                sensor_position[sid] = (float(sensor["x_m"]), float(sensor["z_m"]))
            common_ids = ids_in_snapshot if common_ids is None else (common_ids & ids_in_snapshot)

        if not common_ids:
            return None

        sorted_ids = sorted(
            sid for sid in common_ids
            if abs(sensor_position[sid][0]) >= self.min_sensor_radius_m
        )
        if len(sorted_ids) < self.min_sensors:
            return None

        obs_series: dict[str, list[float]] = {sid: [] for sid in sorted_ids}
        times_raw = [float(item["time_hours"]) for item in window]
        t0 = times_raw[0] if self.relative_time_window else 0.0
        times_hours = [max((t - t0), self.min_horizon_hours) for t in times_raw]

        for snapshot in window:
            temperature_by_id = {str(s["sensor_id"]): float(s["temperature"]) for s in snapshot["sensors"]}
            for sid in sorted_ids:
                obs_series[sid].append(temperature_by_id[sid])

        positions = [
            {
                "sensor_id": sid,
                "x_m": sensor_position[sid][0],
                "z_m": sensor_position[sid][1],
            }
            for sid in sorted_ids
        ]

        ic_sensors = deepcopy(window[0]["sensors"])

        return times_hours, positions, obs_series, ic_sensors

    def _prepare_inversion_job(
        self, now_utc: datetime
    ) -> tuple[float, dict, list[float], list[dict], dict[str, list[float]], list[dict]] | None:
        if not self._snapshots:
            return None
        latest = self._snapshots[-1]
        latest_time = float(latest["time_hours"])

        if self._inversion_running:
            self._pending_recheck = True
            self.logger.info(
                "Inversion deferred (running=true, latest_time=%.3fh, snapshots=%d)",
                latest_time,
                len(self._snapshots),
            )
            return None

        warmup_left = self._warmup_remaining_hours(latest_time)
        if warmup_left > 0.0:
            # Said once per instance at INFO, then at DEBUG: a 30 min warm-up at
            # a 30 s cadence is 60 snapshots, and logging every one of them as a
            # skip buries the reason it is quiet in the noise of it being quiet.
            message = ("Inversion warming up (%.1f min of %.1f min remaining, "
                       "snapshots=%d); no fit will run yet")
            args = (warmup_left * 60.0, self.warmup_hours * 60.0, len(self._snapshots))
            if not self._warmup_announced:
                self._warmup_announced = True
                self.logger.info(message, *args)
            else:
                self.logger.debug(message, *args)
            return None
        if self._warmup_announced and self._last_inversion_time is None:
            self.logger.info(
                "Inversion warm-up complete at t=%.3fh; fitting is now live", latest_time
            )
            self._warmup_announced = False

        if not self._can_run_inversion(latest_time, now_utc):
            self.logger.info(
                "Inversion skipped (snapshots=%d, min_snapshots=%d, dt_hours=%.3f, dt_seconds=%.1f)",
                len(self._snapshots),
                self.min_snapshots,
                float("inf") if self._last_inversion_time is None else (latest_time - self._last_inversion_time),
                float("inf")
                if self._last_inversion_wallclock is None
                else (now_utc - self._last_inversion_wallclock).total_seconds(),
            )
            return None

        window = self._build_observation_window()
        if window is None:
            self.logger.warning("Inversion skipped: unable to assemble a valid observation window.")
            return None

        times_hours, positions, obs_series, ic_sensors = window
        boundary_schedule = self._boundary_schedule(float(self._window_snapshots()[0]["time_hours"]))
        trigger_snapshot = deepcopy(latest)
        self._inversion_running = True
        self._pending_recheck = False

        return (latest_time, trigger_snapshot, times_hours, positions, obs_series, ic_sensors,
                boundary_schedule)

    def _start_inversion_thread(
        self,
        *,
        trigger_time: float,
        trigger_snapshot: dict,
        times_hours: list[float],
        positions: list[dict],
        obs_series: dict[str, list[float]],
        ic_sensors: list[dict],
        boundary_schedule: list[tuple[float, dict]] | None = None,
    ) -> None:
        worker = Thread(
            target=self._run_inversion_job,
            kwargs={
                "trigger_time": trigger_time,
                "trigger_snapshot": trigger_snapshot,
                "times_hours": times_hours,
                "positions": positions,
                "obs_series": obs_series,
                "ic_sensors": ic_sensors,
                "boundary_schedule": boundary_schedule,
            },
            daemon=True,
        )
        worker.start()

    def _run_inversion_job(
        self,
        *,
        trigger_time: float,
        trigger_snapshot: dict,
        times_hours: list[float],
        positions: list[dict],
        obs_series: dict[str, list[float]],
        ic_sensors: list[dict],
        boundary_schedule: list[tuple[float, dict]] | None = None,
    ) -> None:
        self.logger.info(
            "Starting inversion (time=%.3fh, snapshots=%d, sensors=%d, points=%d, ic_sensors=%d)",
            trigger_time,
            len(times_hours),
            len(positions),
            len(times_hours) * len(positions),
            len(ic_sensors),
        )

        start = datetime.utcnow()
        try:
            best_params, info = self._run_inversion(times_hours, positions, obs_series, ic_sensors,
                                                    boundary_schedule=boundary_schedule)
            self._publish_result(
                sensor_msg=trigger_snapshot,
                best_params=best_params,
                info=info,
                n_sensors=len(positions),
                n_times=len(times_hours),
            )
            elapsed = (datetime.utcnow() - start).total_seconds()
            self.logger.info(
                "Inversion complete (time=%.3fh, loss=%.6f, success=%s, elapsed=%.2fs)",
                trigger_time,
                float(info.get("final_loss", float("nan"))),
                bool(info.get("success", False)),
                elapsed,
            )
        except Exception as exc:
            self.logger.error("Inversion failed: %s", exc, exc_info=True)
        finally:
            restart_job = None
            with self._state_lock:
                self._last_inversion_time = trigger_time
                self._last_inversion_wallclock = datetime.utcnow()
                self._inversion_running = False
                if self._pending_recheck:
                    restart_job = self._prepare_inversion_job(datetime.utcnow())

            if restart_job is not None:
                (
                    next_time,
                    next_snapshot,
                    next_times,
                    next_positions,
                    next_obs_series,
                    next_ic_sensors,
                    next_schedule,
                ) = restart_job
                self._start_inversion_thread(
                    trigger_time=next_time,
                    trigger_snapshot=next_snapshot,
                    times_hours=next_times,
                    positions=next_positions,
                    ic_sensors=next_ic_sensors,
                    obs_series=next_obs_series,
                    boundary_schedule=next_schedule,
                )

    def _normalized_to_params(self, scales: np.ndarray, base_inv_values: dict[str, float]) -> dict[str, float]:
        params = deepcopy(self.base_params)
        for idx, name in enumerate(self.invert_params):
            params[name] = float(base_inv_values[name]) * float(scales[idx])
        return params

    def set_boundary_snapshot(self, snapshot: dict | None) -> bool:
        """Record the boundary the forecast is marching, and fit under it.

        The inversion used to fit every window under the *configured* boundary
        while the forecast ran the pipe-calibrated one. The optimiser cannot move
        a boundary, so it absorbed the difference into the material instead --
        and the resulting parameters, applied under the other boundary, carried
        much of that compensation as error. On the April 8 data the mismatch cost
        roughly two thirds of the error reduction the inversion could deliver,
        and it left ``lambda_s`` with its optimum outside the allowed range.

        Call this whenever a calibrated boundary snapshot arrives. It is cheap
        and idempotent: an unchanged boundary does not disturb the cached solver.
        """
        if self._window_model is None:
            self._window_model = WindowForwardModel(self.config_path)
        if isinstance(snapshot, dict) and snapshot.get("boundaries"):
            try:
                t = float(snapshot["time_hours"])
            except (KeyError, TypeError, ValueError):
                t = None
            if t is not None:
                with self._state_lock:
                    self._boundary_history = [
                        item for item in self._boundary_history if item[0] < t - 1e-9
                    ] + [(t, snapshot)]
        changed = self._window_model.set_boundary(snapshot)
        if changed:
            self.logger.info("Inversion now fitting under the calibrated boundary")
        return changed

    def _generate_series(
        self,
        params: dict[str, float],
        positions: list[dict],
        times_hours: list[float],
        ic_sensors: list[dict] | None = None,
        boundary_schedule: list[tuple[float, dict]] | None = None,
    ) -> dict[str, list[float]]:
        """Forward-model the sensor histories for one candidate parameter set.

        Each horizon is integrated from the same measured initial field
        (``ic_sensors``, the first snapshot of the window) rather than from the
        uniform fallback temperature, so the residual the optimiser minimises
        reflects the parameters instead of a wrong starting state.

        The window is walked as a single trajectory with a reused solver, so the
        early time steps are integrated once rather than once per later
        observation time. The forward model this replaced is kept in
        ``software/tools/golden_reference.py`` and checked against it by
        ``software/tools/verify_inversion_equivalence.py``.
        """
        if self._window_model is None:
            self._window_model = WindowForwardModel(self.config_path)
        return self._window_model.generate_series(params, positions, times_hours, ic_sensors,
                                                  boundary_schedule=boundary_schedule)

    def _compute_loss(self, pred_series: dict[str, list[float]], obs_series: dict[str, list[float]]) -> float:
        values = []
        for sid, obs_vals in obs_series.items():
            pred_vals = pred_series.get(sid, [])
            for obs, pred in zip(obs_vals, pred_vals):
                diff = float(pred) - float(obs)
                if self.loss == "mae":
                    values.append(abs(diff))
                else:
                    values.append(diff * diff)
        if not values:
            return float("inf")
        return float(sum(values) / len(values))

    def _initial_scales(self) -> list[float]:
        if self._last_best_scales is not None and len(self._last_best_scales) == len(self.invert_params):
            scales = []
            for idx, name in enumerate(self.invert_params):
                lo, hi = self.relative_bounds[name]
                scales.append(min(max(float(self._last_best_scales[idx]), lo), hi))
            return scales

        # Cold start: begin at the base values themselves (scale 1.0), clamped
        # into the bounds in case a bound excludes 1.0. The previous behaviour
        # drew a random point in the box, which made the very first fit of a run
        # jump away from the measured material set for no reason the data
        # supports, and made two runs on identical inputs disagree. Starting at
        # the base means the first result reads as a refinement of the measured
        # values, and every later update warm-starts from the previous fit above.
        scales = []
        for name in self.invert_params:
            lo, hi = self.relative_bounds[name]
            scales.append(min(max(1.0, lo), hi))
        return scales

    def _prior_penalty(self, params: dict[str, float]) -> float:
        if self.prior_weight <= 0.0 or not self.prior_params:
            return 0.0
        return self.prior_weight * float(sum(
            np.log(float(params[name]) / prior) ** 2 for name, prior in self.prior_params.items()
        ))

    def _run_inversion(
        self,
        times_hours: list[float],
        positions: list[dict],
        obs_series: dict[str, list[float]],
        ic_sensors: list[dict] | None = None,
        boundary_schedule: list[tuple[float, dict]] | None = None,
    ) -> tuple[dict[str, float], dict[str, object]]:
        base_inv_values = {name: float(self.base_params[name]) for name in self.invert_params}
        y0 = self._initial_scales()
        bounds = [self.relative_bounds[name] for name in self.invert_params]
        use_log = self.scale_space == "log"
        if use_log:
            x0 = np.log(np.array(y0, dtype=float))
            opt_bounds = [(float(np.log(lo)), float(np.log(hi))) for lo, hi in bounds]
        else:
            x0 = np.array(y0, dtype=float)
            opt_bounds = bounds

        def to_scales(x: np.ndarray) -> np.ndarray:
            return np.exp(x) if use_log else np.asarray(x, dtype=float)

        self._eval_count = 0
        best_eval_loss = float("inf")

        def objective(x: np.ndarray) -> float:
            nonlocal best_eval_loss
            self._eval_count += 1
            scales = to_scales(x)
            params = self._normalized_to_params(scales, base_inv_values)
            if boundary_schedule:
                pred_series = self._generate_series(params, positions, times_hours, ic_sensors,
                                                    boundary_schedule=boundary_schedule)
            else:
                pred_series = self._generate_series(params, positions, times_hours, ic_sensors)
            loss_value = self._compute_loss(pred_series, obs_series) + self._prior_penalty(params)
            if np.isfinite(loss_value):
                best_eval_loss = min(best_eval_loss, loss_value)
            self.logger.info(
                "Inversion eval %03d | loss=%.6f | scales=%s",
                self._eval_count,
                float(loss_value),
                [float(v) for v in scales],
            )
            return float(loss_value)

        result = minimize(
            objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=opt_bounds,
            options={
                "maxiter": self.maxiter,
                "maxfun": self.maxfun,
            },
        )

        result_scales = to_scales(result.x)
        best_scales = [float(v) for v in result_scales]
        best_params = self._normalized_to_params(result_scales, base_inv_values)
        self._last_best_scales = best_scales

        info = {
            "success": bool(result.success),
            "message": str(result.message),
            "final_loss": float(result.fun),
            "prior_penalty": self._prior_penalty(best_params),
            "best_eval_loss": float(best_eval_loss),
            "loss_name": self.loss,
            "eval_count": int(self._eval_count),
            "best_scales": {name: best_scales[idx] for idx, name in enumerate(self.invert_params)},
            "initial_scales": {name: float(y0[idx]) for idx, name in enumerate(self.invert_params)},
            "relative_bounds": {
                name: [float(self.relative_bounds[name][0]), float(self.relative_bounds[name][1])]
                for name in self.invert_params
            },
        }
        return best_params, info

    def _build_boundary_snapshot(self, sensor_msg: dict, material_override: dict[str, float]) -> dict:
        boundaries = deepcopy(self.boundaries)

        if boundaries and self.sensor_overrides:
            temp_by_sensor = {}
            for sensor in sensor_msg.get("sensors", []):
                try:
                    temp_by_sensor[str(sensor["sensor_id"])] = float(sensor["temperature"])
                except (KeyError, TypeError, ValueError):
                    continue

            for override in self.sensor_overrides:
                boundary_set = override.get("boundary_set")
                sensor_id = override.get("sensor_id")
                if boundary_set is None or sensor_id is None:
                    continue
                if str(sensor_id) not in temp_by_sensor:
                    continue
                for boundary in boundaries:
                    if boundary.get("boundary_set") != boundary_set:
                        continue
                    if boundary.get("bc_type") not in {"dirichlet", "robin"}:
                        continue
                    boundary["temperature"] = temp_by_sensor[str(sensor_id)]

        return {
            "timestamp": sensor_msg.get("timestamp") or datetime.utcnow().isoformat(),
            "time_hours": float(sensor_msg["time_hours"]),
            "boundaries": boundaries,
            "material_override": material_override,
        }

    def _publish_result(
        self,
        *,
        sensor_msg: dict,
        best_params: dict[str, float],
        info: dict[str, object],
        n_sensors: int,
        n_times: int,
    ) -> None:
        if self.mq_out is None or self.influx is None:
            raise RuntimeError("Dependencies not initialised. Did you call setup()?")

        payload = {
            "timestamp": sensor_msg.get("timestamp") or datetime.utcnow().isoformat(),
            "status": "inverted",
            "time_hours": float(sensor_msg["time_hours"]),
            "site_id": self.site_id,
            "parameters": {name: float(best_params[name]) for name in self.invert_params},
            "loss_name": info.get("loss_name"),
            "final_loss": float(info.get("final_loss", float("nan"))),
            "success": bool(info.get("success", False)),
            "message": str(info.get("message", "")),
            "eval_count": int(info.get("eval_count", 0)),
            "n_sensors": int(n_sensors),
            "n_times": int(n_times),
            "best_scales": info.get("best_scales", {}),
            "initial_scales": info.get("initial_scales", {}),
            "relative_bounds": info.get("relative_bounds", {}),
        }
        self.mq_out.publish(payload)

        validation = {
            "final_loss": float(info.get("final_loss", float("nan"))),
            "success": 1.0 if bool(info.get("success", False)) else 0.0,
            "eval_count": float(info.get("eval_count", 0)),
            "n_sensors": float(n_sensors),
            "n_times": float(n_times),
        }
        self.influx.write_inversion_parameters(
            parameters={name: float(best_params[name]) for name in self.invert_params},
            status="inverted",
            site=self.site_id,
            validation=validation,
        )

        if self.publish_material_override_boundary:
            if self.mq_fem_in is None:
                raise RuntimeError("FEM input client is not initialised.")
            snapshot = self._build_boundary_snapshot(sensor_msg, payload["parameters"])
            self.mq_fem_in.publish(snapshot)

    def _process_message(self, msg: dict) -> None:
        snapshot = self._parse_snapshot(msg)
        if snapshot is None:
            return

        job = None
        with self._state_lock:
            self._upsert_snapshot(snapshot)
            job = self._prepare_inversion_job(datetime.utcnow())

        if job is None:
            return

        (
            trigger_time,
            trigger_snapshot,
            times_hours,
            positions,
            obs_series,
            ic_sensors,
            boundary_schedule,
        ) = job
        self._start_inversion_thread(
            trigger_time=trigger_time,
            trigger_snapshot=trigger_snapshot,
            times_hours=times_hours,
            positions=positions,
            obs_series=obs_series,
            ic_sensors=ic_sensors,
            boundary_schedule=boundary_schedule,
        )

    def setup(self) -> None:
        if self.mq_in is None:
            self.mq_in = RabbitMQClient(self.input_config)
        if self.mq_out is None:
            self.mq_out = RabbitMQClient(self.output_config)
        if self.influx is None:
            self.influx = InfluxHelper(self.influx_config)
        if self.publish_material_override_boundary and self.mq_fem_in is None:
            self.mq_fem_in = RabbitMQClient(self.fem_input_config)

        self.logger.info(
            "Dependencies initialised (in=%s, out=%s, fem_forward=%s)",
            self.input_config.queue,
            self.output_config.queue,
            "enabled" if self.publish_material_override_boundary else "disabled",
        )

    def start(self) -> None:
        if self.mq_in is None or self.mq_out is None or self.influx is None:
            self.setup()
        self.logger.info("Listening for sensor updates on %s", self.input_config.queue)
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
        if self.mq_out is not None:
            self.mq_out.disconnect()
        if self.mq_fem_in is not None:
            self.mq_fem_in.disconnect()
        if self.influx is not None:
            self.influx.close()
        self.logger.info("Shutdown complete")


if __name__ == "__main__":
    FEMInversionServer().start()

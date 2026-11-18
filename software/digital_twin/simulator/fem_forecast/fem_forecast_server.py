"""FEM forecast service for 2D thaw-front predictions."""

from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from software.digital_twin.communication.messaging import RabbitMQClient, RabbitMQConfig, resolve_queue_config
from software.digital_twin.data_access.influx_utils import (
    FEM_TEMPERATURE_BASELINE_MEASUREMENT,
    InfluxConfig,
    InfluxHelper,
)
from software.digital_twin.simulator.fem_forecast.fem_solver import (
    FEMMeshConfig,
    FEMSolver2D,
    MaterialProps,
    SolverConfig,
)
from software.utils.logging_setup import get_logger

BOUNDARY_QUEUE = "permafrost.record.boundary.2d"
FORECAST_QUEUE = "permafrost.record.fem_forecast.2d"
BOUNDARY_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "boundary_condition_message.json"
FORECAST_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "fem_forecast_message.json"


@dataclass
class ForecastResult:
    radius_max_m: float | None
    radius_avg_m: float | None
    points: list[dict[str, float]]


@dataclass
class LookaheadResult:
    """Two counterfactual branches sampled forward from the current state.

    ``keep`` continues on the boundary as it stands; ``stop`` is the same
    snapshot with the heater setpoint forced off. The difference between them is
    the effect of the decision and nothing else, which is the whole point of
    running both.

    The trajectories are published rather than reduced to a time-to-breach here:
    the breach limit belongs to the safety monitor, and duplicating it in the
    simulator would give the system two places to disagree about what "the
    limit" is.
    """
    times_h: list[float]
    radius_keep_m: list[float | None]
    radius_stop_m: list[float | None]
    horizon_hours: float
    valid_at_hours: float


@dataclass
class FieldSnapshot:
    """Snapshot of FEM temperature field at simulation end."""
    node_xy: np.ndarray      # shape (numnode, 2)
    temperature: np.ndarray  # shape (numnode,)
    time_hours: float
    units: str               # "K" for Kelvin


class FEMForecastServer:
    """Consumes boundary snapshots and publishes FEM forecasts."""
    SENSOR_DEPTHS_MM = (10, 50, 90, 170, 250, 290)
    SENSOR_WIDTHS_MM = (15, 30, 45, 60, 75, 90, 120)
    MATERIAL_KEYS = {item.name for item in fields(MaterialProps)}

    def __init__(
        self,
        influx_config: InfluxConfig | None = None,
        input_config: RabbitMQConfig | None = None,
        output_config: RabbitMQConfig | None = None,
        *,
        input_queue: str | None = None,
        output_queue: str | None = None,
        horizon_hours: float = 1.0,
        track_observed_time: bool = True,
        max_step_hours: float = 0.5,
        site_id: str = "default",
        mesh_dir: str | Path | None = None,
        node_file: str = "DT_model_nodes.txt",
        element_file: str = "DT_model_elements.txt",
        edge_sets: dict[str, str] | None = None,
        node_sets: dict[str, str] | None = None,
        material: dict | None = None,
        solver: dict | None = None,
        save_field_csv_default: bool = True,
        include_field_default: bool = False,
        sticky_material_override: bool = True,
        field_csv_dir: str | None = None,
        run_baseline: bool = False,
        enable_lookahead: bool = True,
        lookahead_hours: float = 1.0,
        lookahead_samples: int = 12,
        lookahead_stop_setpoint_c: float = 0.0,
        pipe_boundary_set: str = "left",
    ) -> None:
        self.logger = get_logger("FEMForecastServer")
        self.influx_config = influx_config or InfluxConfig()
        self.input_config = resolve_queue_config(
            input_config,
            queue=input_queue or BOUNDARY_QUEUE,
            schema_path=BOUNDARY_SCHEMA,
        )
        self.output_config = resolve_queue_config(
            output_config,
            queue=output_queue or FORECAST_QUEUE,
            schema_path=FORECAST_SCHEMA,
        )
        self.horizon_hours = float(horizon_hours)
        self.track_observed_time = self._as_bool(track_observed_time, True)
        self.max_step_hours = float(max_step_hours)
        self._last_time_hours: float | None = None
        self.site_id = site_id
        self.mesh_dir = Path(mesh_dir) if mesh_dir is not None else None
        self.node_file = node_file
        self.element_file = element_file
        self.edge_sets = edge_sets or {}
        self.node_sets = node_sets or {}
        self.material_config = material or {}
        self.solver_config = solver or {}
        self.save_field_csv_default = self._as_bool(save_field_csv_default, True)
        self.include_field_default = self._as_bool(include_field_default, False)
        self.sticky_material_override = self._as_bool(sticky_material_override, True)
        self.field_csv_dir = field_csv_dir
        # Baseline twin: a second solver marched on the same boundary snapshots
        # and the same initial condition, but always with the configured
        # material parameters -- material_override from the inversion service is
        # never applied to it. Its field is published to a separate Influx
        # measurement so the dashboard can show "with inversion" against
        # "without inversion" and the difference is attributable to the
        # calibration alone.
        self.run_baseline = self._as_bool(run_baseline, False)
        # Predictive arm. The tracked step above is a nowcast -- with
        # track_observed_time on it advances by the elapsed time since the last
        # message, so its radius is valid *at* the observation time and carries no
        # lead. The lookahead below is the genuine prediction the safety rules
        # need, and it is the only thing in the system that can answer what
        # happens if the heater is switched off now.
        self.enable_lookahead = self._as_bool(enable_lookahead, True)
        self.lookahead_hours = float(lookahead_hours)
        self.lookahead_samples = max(int(lookahead_samples), 1)
        self.lookahead_stop_setpoint_c = float(lookahead_stop_setpoint_c)
        self.pipe_boundary_set = str(pipe_boundary_set)

        self.mq_in: Optional[RabbitMQClient] = None
        self.mq_out: Optional[RabbitMQClient] = None
        self.influx: Optional[InfluxHelper] = None
        self._solver: Optional[FEMSolver2D] = None
        self._baseline_solver: Optional[FEMSolver2D] = None
        self._last_material_override: dict[str, float] | None = None
        # Observed sensor layout (from the boundary message passthrough), used to
        # sample the FEM field at the same positions as the observations so the
        # forecast plot's markers/contour match the observation plot.
        self._sensor_positions: list[dict[str, float | str]] | None = None

    @staticmethod
    def _as_bool(value: object, default: bool) -> bool:
        """Parse bool-like values from message/config without surprising truthiness."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            parsed = value.strip().lower()
            if parsed in {"1", "true", "yes", "y", "on"}:
                return True
            if parsed in {"0", "false", "no", "n", "off"}:
                return False
        return default

    def _normalize_material_override(self, material_override: object) -> dict[str, float]:
        """Validate canonical material keys and coerce values to float."""
        if material_override is None:
            return {}
        if not isinstance(material_override, dict):
            raise ValueError("material_override must be an object/dict.")

        normalized: dict[str, float] = {}
        for key, raw_value in material_override.items():
            if not isinstance(key, str):
                raise ValueError("material_override keys must be strings.")
            if key not in self.MATERIAL_KEYS:
                allowed = ", ".join(sorted(self.MATERIAL_KEYS))
                raise ValueError(f"Unknown material key '{key}'. Allowed keys: {allowed}.")
            try:
                normalized[key] = float(raw_value)
            except (TypeError, ValueError):
                raise ValueError(f"material_override.{key} must be numeric, got {raw_value!r}.")
        return normalized

    def _normalize_positions(self, positions: object) -> list[dict[str, float | str]]:
        """Validate and normalize user-requested sampling positions."""
        if positions is None:
            return []
        if not isinstance(positions, list):
            raise ValueError("positions must be a list.")

        normalized: list[dict[str, float | str]] = []
        for idx, point in enumerate(positions):
            if not isinstance(point, dict):
                raise ValueError(f"positions[{idx}] must be an object.")
            if "x_m" not in point or "z_m" not in point:
                raise ValueError(f"positions[{idx}] must include x_m and z_m.")

            try:
                x_m = float(point["x_m"])
                z_m = float(point["z_m"])
            except (TypeError, ValueError):
                raise ValueError(f"positions[{idx}] x_m and z_m must be numeric.")
            sensor_id = point.get("sensor_id", point.get("id", f"P{idx + 1}"))
            normalized.append(
                {
                    "sensor_id": str(sensor_id),
                    "x_m": x_m,
                    "z_m": z_m,
                }
            )
        return normalized

    def _material_from_dict(self, material_dict: dict) -> MaterialProps:
        """Construct MaterialProps from dict with same defaults as _build_solver."""
        return MaterialProps(
            porosity=float(material_dict.get("porosity", 0.43)),
            rho_s=float(material_dict.get("rho_s", 2650.0)),
            rho_w=float(material_dict.get("rho_w", 1000.0)),
            rho_i=float(material_dict.get("rho_i", 910.0)),
            cap_s=float(material_dict.get("cap_s", 790.0)),
            cap_w=float(material_dict.get("cap_w", 4180.0)),
            cap_i=float(material_dict.get("cap_i", 1930.0)),
            latent_heat=float(material_dict.get("latent_heat", 3.34e5)),
            k_parameter=float(material_dict.get("k_parameter", 3.0)),
            tr_k=float(material_dict.get("tr_k", 273.15)),
            lambda_s=float(material_dict.get("lambda_s", 3.51)),
            lambda_w=float(material_dict.get("lambda_w", 0.56)),
            lambda_i=float(material_dict.get("lambda_i", 2.24)),
        )

    def _write_field_csv(
        self,
        field: FieldSnapshot,
        horizon_hours: float,
        field_csv_dir: str | None = None,
        variant: str | None = None,
    ) -> str:
        """Append one horizontal row per timestep using D*-W* sensor-style columns.

        ``variant`` suffixes the filename so the baseline twin never appends to
        the calibrated run's CSV.
        """
        if field_csv_dir is None:
            field_csv_dir = "artifacts/fem_fields"

        field_path = Path(field_csv_dir)
        field_path.mkdir(parents=True, exist_ok=True)

        site_str = str(self.site_id).replace("/", "_").replace(" ", "_")
        suffix = f"_{variant}" if variant else ""
        filename = f"fem_field_{site_str}{suffix}.csv"
        filepath = field_path / filename
        write_header = (not filepath.exists()) or filepath.stat().st_size == 0

        # Map FEM nodal field to the requested sensor-style horizontal columns.
        # Dxx-Wyy means depth xx mm and width yy mm.
        points = np.asarray(field.node_xy, dtype=float)
        x_abs = np.abs(points[:, 0])
        z = points[:, 1]
        sensor_labels = [
            f"D{depth_mm}-W{width_mm}"
            for depth_mm in self.SENSOR_DEPTHS_MM
            for width_mm in self.SENSOR_WIDTHS_MM
        ]
        row_values: list[float] = []
        zero_c_in_solver_units = self._solver._to_kelvin(0.0) if self._solver is not None else 0.0
        for depth_mm in self.SENSOR_DEPTHS_MM:
            z_target = depth_mm / 1000.0
            for width_mm in self.SENSOR_WIDTHS_MM:
                x_target = width_mm / 1000.0
                distances = (x_abs - x_target) ** 2 + (z - z_target) ** 2
                nearest_idx = int(np.argmin(distances))
                temp_c = float(field.temperature[nearest_idx]) - float(zero_c_in_solver_units)
                row_values.append(temp_c)

        with open(filepath, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Time (h)", *sensor_labels])
            writer.writerow([field.time_hours, *row_values])

        self.logger.info("Appended field snapshot to %s", filepath)
        return str(filepath)

    def _sample_field_at_positions(
        self,
        field: FieldSnapshot,
        positions: list[dict[str, float | str]],
    ) -> list[dict[str, float | str]]:
        """Sample FEM field at arbitrary x/z positions using nearest FEM node."""
        points = np.asarray(field.node_xy, dtype=float)
        x_abs = np.abs(points[:, 0])
        z = points[:, 1]
        zero_c_in_solver_units = self._solver._to_kelvin(0.0) if self._solver is not None else 0.0

        sampled: list[dict[str, float | str]] = []
        for point in positions:
            x_target = float(point["x_m"])
            z_target = float(point["z_m"])
            distances = (x_abs - abs(x_target)) ** 2 + (z - z_target) ** 2
            nearest_idx = int(np.argmin(distances))
            temp_c = float(field.temperature[nearest_idx]) - float(zero_c_in_solver_units)
            sampled.append(
                {
                    "sensor_id": str(point["sensor_id"]),
                    "x_m": float(x_target),
                    "z_m": float(z_target),
                    "temperature": temp_c,
                }
            )
        return sampled

    def _sample_field_at_sensor_grid(self, field: FieldSnapshot) -> list[dict[str, float | str]]:
        """Sample the FEM field at the observed sensor positions when known.

        Using the actual sensor layout (from the boundary message passthrough)
        makes the forecast markers/contour align exactly with the observation
        plot — including the x=0 axis column, which the built-in D*-W* grid
        omits. Falls back to the built-in grid before any sensors are seen.
        """
        if self._sensor_positions:
            return self._sample_field_at_positions(field, self._sensor_positions)
        grid_positions = [
            {
                "sensor_id": f"D{depth_mm}-W{width_mm}",
                "x_m": float(width_mm / 1000.0),
                "z_m": float(depth_mm / 1000.0),
            }
            for depth_mm in self.SENSOR_DEPTHS_MM
            for width_mm in self.SENSOR_WIDTHS_MM
        ]
        return self._sample_field_at_positions(field, grid_positions)

    def _update_sensor_positions(self, sensors: object) -> None:
        """Cache the observed sensor layout from a boundary message passthrough."""
        if not isinstance(sensors, list) or not sensors:
            return
        positions: list[dict[str, float | str]] = []
        for sensor in sensors:
            try:
                positions.append(
                    {
                        "sensor_id": str(sensor["sensor_id"]),
                        "x_m": float(sensor["x_m"]),
                        "z_m": float(sensor["z_m"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        if positions:
            self._sensor_positions = positions

    def _build_solver(self) -> FEMSolver2D:
        if self.mesh_dir is None:
            raise ValueError("mesh_dir must be configured for FEM forecasting.")

        mesh = FEMMeshConfig(
            mesh_dir=self.mesh_dir,
            node_file=self.node_file,
            element_file=self.element_file,
            edge_sets=self.edge_sets,
            node_sets=self.node_sets,
        )
        material = MaterialProps(
            porosity=float(self.material_config.get("porosity", 0.43)),
            rho_s=float(self.material_config.get("rho_s", 2650.0)),
            rho_w=float(self.material_config.get("rho_w", 1000.0)),
            rho_i=float(self.material_config.get("rho_i", 910.0)),
            cap_s=float(self.material_config.get("cap_s", 790.0)),
            cap_w=float(self.material_config.get("cap_w", 4180.0)),
            cap_i=float(self.material_config.get("cap_i", 1930.0)),
            latent_heat=float(self.material_config.get("latent_heat", 3.34e5)),
            k_parameter=float(self.material_config.get("k_parameter", 3.0)),
            tr_k=float(self.material_config.get("tr_k", 273.15)),
            lambda_s=float(self.material_config.get("lambda_s", 3.51)),
            lambda_w=float(self.material_config.get("lambda_w", 0.56)),
            lambda_i=float(self.material_config.get("lambda_i", 2.24)),
        )
        solver = SolverConfig(
            dt_seconds=float(self.solver_config.get("dt_seconds", 36.0)),
            max_iterations=int(self.solver_config.get("max_iterations", 25)),
            tol_r=float(self.solver_config.get("tol_r", 1e-6)),
            temperature_offset_k=float(self.solver_config.get("temperature_offset_k", 0.0)),
            initial_temperature_c=float(self.solver_config.get("initial_temperature_c", -1.0)),
            thaw_threshold_c=float(self.solver_config.get("thaw_threshold_c", 0.0)),
        )
        return FEMSolver2D(mesh=mesh, material=material, solver=solver)

    def _maybe_seed_initial_field(self, sensors: object) -> None:
        """Seed the FEM initial condition from the first sensor snapshot.

        Builds the solver if needed and asks it to interpolate the snapshot onto
        the mesh nodes. The solver guards against re-seeding, so this is safe to
        call on every message — only the first usable snapshot takes effect.
        """
        if not sensors or not isinstance(sensors, list):
            return
        if self._solver is None:
            self._solver = self._build_solver()
        if self._solver.seed_initial_field(sensors):
            self.logger.info(
                "FEM initial condition seeded from sensor snapshot (%d sensors).",
                len(sensors),
            )
        # The baseline twin must start from the identical initial field, or the
        # two runs would differ for reasons other than the material parameters.
        if self.run_baseline:
            if self._baseline_solver is None:
                self._baseline_solver = self._build_solver()
            self._baseline_solver.seed_initial_field(sensors)

    def _resolve_step_hours(self, time_hours: float) -> float:
        """Time to advance for this message.

        When ``track_observed_time`` is set, advance by the real elapsed time
        since the previous message so simulated time tracks observed time and
        the thaw-front spread rate matches reality. A fixed horizon per message
        (the previous behaviour) integrates far more heating than the elapsed
        time when messages arrive faster than the horizon, over-spreading the
        field. The first message (or non-tracking mode) uses ``horizon_hours``.
        """
        if not self.track_observed_time or self._last_time_hours is None:
            return float(self.horizon_hours)
        step = float(time_hours) - float(self._last_time_hours)
        if step <= 0.0:
            return 0.0  # duplicate / out-of-order message: don't integrate
        if self.max_step_hours > 0.0:
            step = min(step, self.max_step_hours)
        return step

    def _run_forecast(self, boundary_snapshot: dict, step_hours: float | None = None) -> ForecastResult:
        if self._solver is None:
            self._solver = self._build_solver()
        hz = self.horizon_hours if step_hours is None else step_hours
        if hz > 0.0:
            self._solver.advance(boundary_snapshot, horizon_hours=hz)
        radius_max_m, radius_avg_m, points = self._solver.compute_thaw_metrics()
        return ForecastResult(
            radius_max_m=radius_max_m,
            radius_avg_m=radius_avg_m,
            points=points,
        )

    def _run_forecast_with_field(
        self,
        boundary_snapshot: dict,
        material_override: dict | None = None,
        include_field: bool = True,
        save_field_csv: bool = True,
        field_csv_dir: str | None = None,
        step_hours: float | None = None,
    ) -> tuple[ForecastResult, FieldSnapshot | None, str | None]:
        """Run forecast with optional field snapshot and CSV output.

        Returns:
            (ForecastResult, FieldSnapshot or None, csv_path or None)
        """
        if self._solver is None:
            self._solver = self._build_solver()

        merged = dict(self.material_config)
        if material_override is not None:
            merged.update(self._normalize_material_override(material_override))
        self._solver.material = self._material_from_dict(merged)

        # Run forecast
        hz = self.horizon_hours if step_hours is None else step_hours
        if hz > 0.0:
            self._solver.advance(boundary_snapshot, horizon_hours=hz)
        radius_max_m, radius_avg_m, points = self._solver.compute_thaw_metrics()
        result = ForecastResult(
            radius_max_m=radius_max_m,
            radius_avg_m=radius_avg_m,
            points=points,
        )
        
        # Optionally capture field snapshot
        field_snapshot = None
        if include_field or save_field_csv:
            # In tracking mode the field represents the *observation* time, not
            # observation + step. advance() tags the solver with the end of the
            # step, and _process_message corrects the solver afterwards -- but a
            # snapshot taken here has already copied the value, so it would be
            # written to Influx and CSV one step ahead of the sensor series.
            # Every downstream join is on time_hours, so that offset silently
            # mispairs or drops rows and inflates the apparent FEM error.
            snapshot_time = float(self._solver.time_hours or 0.0)
            if self.track_observed_time:
                try:
                    snapshot_time = float(boundary_snapshot["time_hours"])
                except (KeyError, TypeError, ValueError):
                    pass
            field_snapshot = FieldSnapshot(
                node_xy=self._solver.node.copy(),
                temperature=self._solver.T.copy(),
                time_hours=snapshot_time,
                units="K",  # FEMSolver2D stores Kelvin internally
            )
        
        # Optionally save to CSV
        csv_path = None
        if save_field_csv and field_snapshot is not None:
            csv_path = self._write_field_csv(field_snapshot, self.horizon_hours, field_csv_dir)
        
        return result, field_snapshot, csv_path

    # ---------------------------
    # PREDICTIVE LOOKAHEAD
    # ---------------------------
    def _stopped_boundary(self, boundary_snapshot: dict) -> dict:
        """The same boundary snapshot with the heater turned off.

        Only the pipe boundary changes: the deep frozen anchor, the insulated
        faces, the mesh and the current field are all what they were.
        """
        snapshot = deepcopy(boundary_snapshot)
        for bc in snapshot.get("boundaries", []) or []:
            if (bc.get("boundary_set") == self.pipe_boundary_set
                    and bc.get("bc_type") in {"robin", "dirichlet"}):
                # A depth-resolved profile would otherwise keep driving heat in
                # and silently override the scalar setpoint below.
                bc.pop("temperature_profile", None)
                bc["temperature"] = self.lookahead_stop_setpoint_c
        return snapshot

    def _sample_branch(self, boundary_snapshot: dict, start_hours: float) -> tuple[list[float], list[float | None]]:
        """Advance a copy of the solver, sampling the front as it goes.

        Sampling rather than jumping to the endpoint is what makes a crossing
        *time* recoverable at all -- it cannot be read off the final state. The
        solver state is restored before returning: the tracked run must not
        inherit the lookahead, or simulated time races ahead of observed time and
        every later step is wrong.
        """
        solver = self._solver
        if solver is None:
            return [], []
        saved_T = solver.T.copy()
        saved_time = solver.time_hours
        step = self.lookahead_hours / self.lookahead_samples
        times: list[float] = []
        radii: list[float | None] = []
        try:
            for k in range(1, self.lookahead_samples + 1):
                solver.advance(boundary_snapshot, horizon_hours=step)
                radius_m, _radius_avg, _points = solver.compute_thaw_metrics()
                times.append(start_hours + k * step)
                radii.append(None if radius_m is None else float(radius_m))
        finally:
            solver.T = saved_T
            solver.time_hours = saved_time
        return times, radii

    def _run_lookahead(self, boundary_snapshot: dict, time_hours: float) -> LookaheadResult | None:
        """Both counterfactual branches, or None when the arm is off/unavailable."""
        if not self.enable_lookahead or self._solver is None or self.lookahead_hours <= 0.0:
            return None
        times, keep = self._sample_branch(boundary_snapshot, time_hours)
        _times, stop = self._sample_branch(
            self._stopped_boundary(boundary_snapshot), time_hours)
        if not times:
            return None
        return LookaheadResult(
            times_h=times,
            radius_keep_m=keep,
            radius_stop_m=stop,
            horizon_hours=self.lookahead_hours,
            valid_at_hours=time_hours + self.lookahead_hours,
        )

    def _run_baseline_forecast(
        self,
        boundary_snapshot: dict,
        step_hours: float | None = None,
        save_field_csv: bool = False,
        field_csv_dir: str | None = None,
    ) -> tuple[ForecastResult, FieldSnapshot] | None:
        """Advance the uncalibrated twin and snapshot its field.

        Same mesh, same boundary snapshot, same step and same seeded initial
        condition as the calibrated run -- the only difference is that the
        material stays at its configured values, so ``material_override`` from
        the inversion service is deliberately never applied here. Any gap
        between the two fields is therefore attributable to the inversion.
        """
        if not self.run_baseline:
            return None
        if self._baseline_solver is None:
            self._baseline_solver = self._build_solver()

        hz = self.horizon_hours if step_hours is None else step_hours
        if hz > 0.0:
            self._baseline_solver.advance(boundary_snapshot, horizon_hours=hz)
        radius_max_m, radius_avg_m, points = self._baseline_solver.compute_thaw_metrics()
        field = FieldSnapshot(
            node_xy=self._baseline_solver.node.copy(),
            temperature=self._baseline_solver.T.copy(),
            time_hours=float(self._baseline_solver.time_hours or 0.0),
            units="K",
        )
        if save_field_csv:
            self._write_field_csv(field, self.horizon_hours, field_csv_dir, variant="baseline")
        return (
            ForecastResult(radius_max_m=radius_max_m, radius_avg_m=radius_avg_m, points=points),
            field,
        )

    def compute_temperatures(
        self,
        boundary_snapshot: dict,
        positions: object,
        material_override: object = None,
    ) -> dict[str, object]:
        """Public scenario API for direct FEM temperature sampling.

        Args:
            boundary_snapshot: FEM boundary payload (same shape as queue input).
            positions: list of {"x_m": float, "z_m": float, "sensor_id"?: str}.
            material_override: optional per-run material edits.
        """
        if not isinstance(boundary_snapshot, dict):
            raise ValueError("boundary_snapshot must be an object/dict.")

        normalized_positions = self._normalize_positions(positions)
        if not normalized_positions:
            raise ValueError("positions must contain at least one x_m/z_m point.")

        result, field_snapshot, _ = self._run_forecast_with_field(
            boundary_snapshot,
            material_override=material_override,
            include_field=True,
            save_field_csv=False,
            field_csv_dir=None,
        )
        if field_snapshot is None:
            raise RuntimeError("Field snapshot unavailable after FEM run.")

        try:
            base_time = float(boundary_snapshot.get("time_hours", 0.0))
        except (TypeError, ValueError):
            raise ValueError("boundary_snapshot.time_hours must be numeric.")

        return {
            "timestamp": boundary_snapshot.get("timestamp") or datetime.utcnow().isoformat(),
            "time_hours": base_time,
            "horizon_hours": self.horizon_hours,
            "radius_max_m": result.radius_max_m,
            "radius_avg_m": result.radius_avg_m,
            "points": result.points,
            "temperatures": self._sample_field_at_positions(field_snapshot, normalized_positions),
        }

    def _process_message(self, msg: dict) -> None:
        if self.influx is None or self.mq_out is None:
            raise RuntimeError("Dependencies not initialised. Did you call setup()?")  # pragma: no cover

        try:
            time_hours = float(msg["time_hours"])
            boundary_count = len(msg.get("boundaries", []))
            field_snapshot: FieldSnapshot | None = None

            # Seed the initial condition from the first sensor snapshot before
            # the solver marches forward. No-op once already seeded.
            self._maybe_seed_initial_field(msg.get("sensors"))
            # Track the observed sensor layout so the forecast is sampled at the
            # same positions as the observations (matching plot markers/contour).
            self._update_sensor_positions(msg.get("sensors"))

            # Extract optional field/material control flags
            include_field = self._as_bool(msg.get("include_field"), self.include_field_default)
            save_field_csv = self._as_bool(msg.get("save_field_csv"), self.save_field_csv_default)
            field_csv_dir = msg.get("field_csv_dir", self.field_csv_dir)
            raw_material_override = msg.get("material_override")
            material_override: dict[str, float] | None = None
            override_source = "default"
            if raw_material_override is not None:
                normalized_override = self._normalize_material_override(raw_material_override)
                if normalized_override:
                    material_override = normalized_override
                    self._last_material_override = dict(normalized_override)
                    override_source = "message"
                else:
                    self._last_material_override = None
                    override_source = "cleared"
            elif self.sticky_material_override and self._last_material_override is not None:
                material_override = dict(self._last_material_override)
                override_source = "sticky"
            
            # Advance by real elapsed time (tracking) instead of a fixed horizon.
            step_hours = self._resolve_step_hours(time_hours)

            # Choose forecast path based on flags
            if include_field or save_field_csv or material_override is not None:
                result, field_snapshot, csv_path = self._run_forecast_with_field(
                    msg,
                    material_override=material_override,
                    include_field=include_field,
                    save_field_csv=save_field_csv,
                    field_csv_dir=field_csv_dir,
                    step_hours=step_hours,
                )
                if csv_path:
                    self.logger.info("Field CSV written to %s", csv_path)
            else:
                result = self._run_forecast(msg, step_hours=step_hours)

            # In tracking mode the field now represents the observation time, so
            # report that (advance() would otherwise tag it observed_time+step).
            if self.track_observed_time and self._solver is not None:
                self._solver.time_hours = time_hours
            self._last_time_hours = time_hours

            # Predictive arm, run from the state the tracked step just produced
            # and with whatever material the inversion has calibrated. Isolated
            # like the baseline below: it feeds the safety rules, but a failure
            # here must not cost us the nowcast that the reactive rules need.
            lookahead: LookaheadResult | None = None
            try:
                lookahead = self._run_lookahead(msg, time_hours)
            except Exception as exc:
                self.logger.error("Lookahead failed: %s", exc, exc_info=True)

            # Uncalibrated reference run. Isolated in its own try block: it is a
            # diagnostic, so a failure here must not cost us the live forecast.
            baseline_field: FieldSnapshot | None = None
            if self.run_baseline:
                try:
                    baseline = self._run_baseline_forecast(
                        msg,
                        step_hours=step_hours,
                        save_field_csv=save_field_csv,
                        field_csv_dir=field_csv_dir,
                    )
                    if baseline is not None:
                        baseline_result, baseline_field = baseline
                        if self.track_observed_time:
                            if self._baseline_solver is not None:
                                self._baseline_solver.time_hours = time_hours
                            baseline_field.time_hours = time_hours
                        self.logger.info(
                            "Baseline (no-inversion) run: r_max=%s r_avg=%s | calibrated: r_max=%s r_avg=%s",
                            baseline_result.radius_max_m,
                            baseline_result.radius_avg_m,
                            result.radius_max_m,
                            result.radius_avg_m,
                        )
                except Exception as exc:
                    self.logger.error("Baseline forecast failed: %s", exc, exc_info=True)


            override_keys = sorted(material_override.keys()) if material_override else []
            self.logger.info(
                "Received boundary snapshot (t=%.2fh, boundaries=%d, override_source=%s, override_keys=%s)",
                time_hours,
                boundary_count,
                override_source,
                override_keys,
            )
            start_time = datetime.utcnow()
            if result.radius_max_m is None or result.radius_avg_m is None:
                self.logger.warning(
                    "FEM forecast returned None metrics (t=%.2fh, points=%d). Publishing anyway.",
                    time_hours,
                    len(result.points),
                )
            payload = {
                "timestamp": msg.get("timestamp") or datetime.utcnow().isoformat(),
                "time_hours": time_hours,
                "horizon_hours": self.horizon_hours,
                "status": "forecasted",
                "radius_max_m": result.radius_max_m,
                "radius_avg_m": result.radius_avg_m,
                "points": result.points,
                # Predictive arm. Absent (null) whenever the lookahead is off or
                # failed, which the safety monitor reads as "no forecast
                # evidence" and degrades to the measured ladder for.
                "lookahead_hours": None if lookahead is None else lookahead.horizon_hours,
                "lookahead_valid_at_hours": None if lookahead is None else lookahead.valid_at_hours,
                "lookahead_times_h": None if lookahead is None else lookahead.times_h,
                "lookahead_radius_keep_m": None if lookahead is None else lookahead.radius_keep_m,
                "lookahead_radius_stop_m": None if lookahead is None else lookahead.radius_stop_m,
                "radius_max_keep_m": None if lookahead is None else lookahead.radius_keep_m[-1],
                "radius_max_stop_m": None if lookahead is None else lookahead.radius_stop_m[-1],
            }
            self.mq_out.publish(payload)
            self.influx.write_fem_forecast_metrics(
                time_hours=time_hours,
                horizon_hours=self.horizon_hours,
                radius_max_m=result.radius_max_m,
                radius_avg_m=result.radius_avg_m,
                points=result.points,
                site=self.site_id,
            )
            if field_snapshot is None and self._solver is not None:
                field_snapshot = FieldSnapshot(
                    node_xy=self._solver.node.copy(),
                    temperature=self._solver.T.copy(),
                    time_hours=float(self._solver.time_hours or 0.0),
                    units="K",
                )
            if field_snapshot is not None:
                self.influx.write_fem_temperature_snapshot_2d(
                    time_hours=float(field_snapshot.time_hours),
                    horizon_hours=self.horizon_hours,
                    sensors=self._sample_field_at_sensor_grid(field_snapshot),
                    site=self.site_id,
                )
            if baseline_field is not None:
                self.influx.write_fem_temperature_snapshot_2d(
                    time_hours=float(baseline_field.time_hours),
                    horizon_hours=self.horizon_hours,
                    sensors=self._sample_field_at_sensor_grid(baseline_field),
                    site=self.site_id,
                    measurement=FEM_TEMPERATURE_BASELINE_MEASUREMENT,
                )
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            self.logger.info(
                "FEM forecast published (t=%.2fh, points=%d, elapsed=%.2fs)",
                time_hours,
                len(result.points),
                elapsed,
            )
        except Exception as exc:
            self.logger.error("FEM forecast FAILED: %s", exc, exc_info=True)

    def setup(self) -> None:
        if self.mq_in is None:
            self.mq_in = RabbitMQClient(self.input_config)
        if self.mq_out is None:
            self.mq_out = RabbitMQClient(self.output_config)
        if self.influx is None:
            self.influx = InfluxHelper(self.influx_config)
        self.logger.info(
            "Dependencies initialised (in=%s, out=%s)",
            self.input_config.queue,
            self.output_config.queue,
        )

    def start(self) -> None:
        if self.mq_in is None or self.mq_out is None or self.influx is None:
            self.setup()

        self.logger.info("Listening for boundary snapshots on %s", self.input_config.queue)
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
        if self.influx is not None:
            self.influx.close()
        self.logger.info("Shutdown complete")


if __name__ == "__main__":
    FEMForecastServer().start()

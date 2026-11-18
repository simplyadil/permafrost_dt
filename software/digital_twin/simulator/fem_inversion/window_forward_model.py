"""Incremental forward model for the inversion fitting window.

The inversion objective needs the modelled sensor histories at every time in
its observation window, for one candidate parameter set. The straightforward
way to get them -- and what ``_generate_series`` did -- is to call
``run_fem_case.FEM`` once per window time. That is correct but quadratic: each
call builds a solver from scratch and marches from the initial condition to its
own horizon, so a window of N snapshots costs 1+2+...+N time steps plus N mesh
loads, when the whole window is one trajectory that already passes through
every one of those times on its way to the last.

This class walks that trajectory once. It keeps a single solver alive across
candidate evaluations, because the mesh and its geometry cache depend only on
geometry, and it re-seeds the initial condition only when the window itself
moves. Per candidate the remaining work is: swap the material, rewind the
temperature field to the cached seeded state, and march, sampling as it passes
each observation time.

The result is identical to the per-time calls, not merely close. The state after
k steps of one continuous march is the state a fresh k-step run reaches, since
both start from the same seeded field and integrate the same equations with the
same fixed boundary snapshot and step size. ``steps_for`` reproduces the
rounding in ``FEMSolver2D.advance`` so the sample points line up exactly.
Equivalence is checked by ``verify_inversion_equivalence.py`` (development repo).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from software.digital_twin.simulator.fem_forecast.fem_solver import FEMMeshConfig, FEMSolver2D
from software.startup.utils.config import load_startup_config
from software.tools.run_fem_case import (
    _default_boundary_snapshot,
    _material_from_dict,
    _resolve_mesh_dir,
    _solver_from_dict,
)


class WindowForwardModel:
    """Forward-models sensor histories over one fitting window, reusing a solver."""

    def __init__(self, config_path: str | Path = "startup.conf") -> None:
        config = load_startup_config(config_path)
        fem_cfg = config.get("fem_forecast", {})
        if not isinstance(fem_cfg, dict):
            raise ValueError("fem_forecast config must be an object.")

        self._config = config
        self._fem_cfg = fem_cfg
        # The same base material FEM() merges candidate parameters on top of.
        base_material = fem_cfg.get("material", {})
        self._base_material = dict(base_material) if isinstance(base_material, dict) else {}
        solver_section = fem_cfg.get("solver", {})
        self._solver_cfg = _solver_from_dict(solver_section if isinstance(solver_section, dict) else {})
        self._mesh_cfg = FEMMeshConfig(
            mesh_dir=_resolve_mesh_dir(fem_cfg.get("mesh_dir")),
            node_file=str(fem_cfg.get("node_file", "DT_model_nodes.txt")),
            element_file=str(fem_cfg.get("element_file", "DT_model_elements.txt")),
            edge_sets=fem_cfg.get("edge_sets", {}) if isinstance(fem_cfg.get("edge_sets"), dict) else {},
            node_sets=fem_cfg.get("node_sets", {}) if isinstance(fem_cfg.get("node_sets"), dict) else {},
        )
        # Starting point only: the configured fallback FEM() uses. ``set_boundary``
        # replaces it with the calibrated snapshot the forecast is marching, and
        # until it is called this is what the window is fitted under. Held as one
        # object either way, so the solver sees an unchanging boundary set for the
        # whole march.
        self._snapshot = _default_boundary_snapshot(config)

        self._solver: FEMSolver2D | None = None
        self._initial_field: np.ndarray | None = None
        self._ic_signature: tuple | None = None
        self._nearest_idx: np.ndarray | None = None
        self._positions_signature: tuple | None = None

    # ------------------------------------------------------------------
    def set_boundary(self, snapshot: dict | None) -> bool:
        """Fit under ``snapshot`` instead of the configured fallback.

        Parameters fitted under one boundary and applied under another need not
        transfer: the optimiser will absorb a boundary error into the material,
        because the material is the only thing it is allowed to move. Measured on
        the April 8 data, fitting under the configured default while the forecast
        ran the pipe-calibrated boundary cost roughly two thirds of the error
        reduction the inversion could otherwise deliver.

        Returns whether the boundary actually changed, so a caller can drop the
        cached solver state only when it has to -- the mesh and its geometry
        cache survive, but a different boundary set invalidates nothing else and
        re-seeding on every identical snapshot would undo the reuse this class
        exists for.
        """
        if not isinstance(snapshot, dict) or not snapshot.get("boundaries"):
            return False
        if snapshot is self._snapshot:
            return False
        previous = self._boundary_signature(self._snapshot)
        current = self._boundary_signature(snapshot)
        self._snapshot = snapshot
        if previous == current:
            return False
        # The seeded field is unaffected -- it comes from the sensors -- but the
        # solver holds the boundary it was built with, so it has to be rebuilt.
        self._solver = None
        return True

    @staticmethod
    def _boundary_signature(snapshot: dict | None) -> tuple:
        """Just enough of a boundary set to tell two of them apart."""
        if not isinstance(snapshot, dict):
            return ()
        out = []
        for bc in snapshot.get("boundaries", []) or []:
            if not isinstance(bc, dict):
                continue
            profile = bc.get("temperature_profile")
            out.append((
                bc.get("boundary_set"),
                bc.get("bc_type"),
                bc.get("temperature"),
                bc.get("h_coeff"),
                bc.get("flux"),
                len(profile) if isinstance(profile, (list, tuple)) else None,
            ))
        return tuple(sorted(out, key=lambda item: str(item[0])))

    @property
    def dt_seconds(self) -> float:
        return float(self._solver_cfg.dt_seconds)

    def steps_for(self, horizon_hours: float) -> int:
        """Step count ``FEMSolver2D.advance`` would take for this horizon."""
        return max(1, int(round(float(horizon_hours) * 3600.0 / self.dt_seconds)))

    # ------------------------------------------------------------------
    def _ensure_solver(self, params: dict[str, float]) -> FEMSolver2D:
        merged = dict(self._base_material)
        merged.update(params)
        material = _material_from_dict(merged)
        if self._solver is None:
            self._solver = FEMSolver2D(mesh=self._mesh_cfg, material=material, solver=self._solver_cfg)
        else:
            self._solver.set_material(material)
        return self._solver

    def _ensure_initial_field(self, solver: FEMSolver2D, ic_sensors: Sequence[dict] | None) -> np.ndarray:
        """Seeded initial field for this window, recomputed only when it moves."""
        signature = self._sensor_signature(ic_sensors)
        if self._initial_field is not None and signature == self._ic_signature:
            return self._initial_field

        offset_k = float(solver.solver.temperature_offset_k)
        solver.T = np.full(solver.numnode, float(self._solver_cfg.initial_temperature_c) + offset_k)
        if isinstance(ic_sensors, list) and ic_sensors:
            solver.seed_initial_field(ic_sensors, force=True)
        self._initial_field = solver.T.copy()
        self._ic_signature = signature
        return self._initial_field

    def _ensure_nearest(self, solver: FEMSolver2D, positions: Sequence[dict]) -> np.ndarray:
        """Nearest mesh node per sensor position, matching ``_sample_positions``."""
        signature = tuple((str(p.get("sensor_id", "")), float(p["x_m"]), float(p["z_m"])) for p in positions)
        if self._nearest_idx is not None and signature == self._positions_signature:
            return self._nearest_idx

        node_xy = np.asarray(solver.node, dtype=float)
        x_abs = np.abs(node_xy[:, 0])
        z = node_xy[:, 1]
        idx = np.empty(len(positions), dtype=int)
        for i, position in enumerate(positions):
            distances = (x_abs - abs(float(position["x_m"]))) ** 2 + (z - float(position["z_m"])) ** 2
            idx[i] = int(np.argmin(distances))
        self._nearest_idx = idx
        self._positions_signature = signature
        return idx

    @staticmethod
    def _sensor_signature(sensors: Sequence[dict] | None) -> tuple:
        if not sensors:
            return ()
        return tuple(
            (str(s.get("sensor_id", "")), float(s["x_m"]), float(s["z_m"]), float(s["temperature"]))
            for s in sensors
        )

    # ------------------------------------------------------------------
    def generate_series(
        self,
        params: dict[str, float],
        positions: list[dict],
        times_hours: list[float],
        ic_sensors: list[dict] | None = None,
        boundary_schedule: Sequence[tuple[float, dict]] | None = None,
    ) -> dict[str, list[float]]:
        """Modelled temperature history at each position, for each window time.

        ``boundary_schedule`` -- ``(t_hours, snapshot)`` pairs on the window's own
        clock, sorted -- marches each step under the boundary in force at its
        start instead of one snapshot for the whole window. Over an hour-long
        window the boundaries move (fluid temperature, a sensor-driven deep
        edge), and fitting under the end-of-window boundary from the start
        hands the optimiser a boundary error to absorb into the material.
        """
        if boundary_schedule:
            return self._generate_series_scheduled(params, positions, times_hours,
                                                   ic_sensors, boundary_schedule)
        solver = self._ensure_solver(params)
        initial_field = self._ensure_initial_field(solver, ic_sensors)
        nearest = self._ensure_nearest(solver, positions)
        offset_k = float(solver.solver.temperature_offset_k)
        sensor_ids = [str(p.get("sensor_id", f"P{i + 1}")) for i, p in enumerate(positions)]

        series: dict[str, list[float]] = {sid: [] for sid in sensor_ids}
        solver.T = initial_field.copy()
        marched = 0

        for horizon_hours in times_hours:
            target = self.steps_for(horizon_hours)
            # Window times arrive sorted, so this is a forward walk. A
            # non-monotonic entry would otherwise be sampled from a state that
            # is too far advanced, so rewind rather than trust the ordering.
            if target < marched:
                solver.T = initial_field.copy()
                marched = 0
            if target > marched:
                solver.advance(
                    self._snapshot,
                    horizon_hours=(target - marched) * self.dt_seconds / 3600.0,
                )
                marched = target

            sampled = solver.T[nearest] - offset_k
            for sid, value in zip(sensor_ids, sampled):
                series[sid].append(float(value))

        return series

    def _generate_series_scheduled(
        self,
        params: dict[str, float],
        positions: list[dict],
        times_hours: list[float],
        ic_sensors: list[dict] | None,
        schedule: Sequence[tuple[float, dict]],
    ) -> dict[str, list[float]]:
        solver = self._ensure_solver(params)
        initial_field = self._ensure_initial_field(solver, ic_sensors)
        nearest = self._ensure_nearest(solver, positions)
        offset_k = float(solver.solver.temperature_offset_k)
        sensor_ids = [str(p.get("sensor_id", f"P{i + 1}")) for i, p in enumerate(positions)]
        sched_t = np.array([float(t) for t, _snap in schedule], dtype=float)
        dt_hours = self.dt_seconds / 3600.0

        series: dict[str, list[float]] = {sid: [] for sid in sensor_ids}
        solver.T = initial_field.copy()
        marched = 0
        for horizon_hours in times_hours:
            target = self.steps_for(horizon_hours)
            if target < marched:
                solver.T = initial_field.copy()
                marched = 0
            while marched < target:
                k = int(np.searchsorted(sched_t, marched * dt_hours + 1e-9, side="right")) - 1
                solver._solve_step(schedule[max(k, 0)][1])
                marched += 1
            sampled = solver.T[nearest] - offset_k
            for sid, value in zip(sensor_ids, sampled):
                series[sid].append(float(value))
        return series

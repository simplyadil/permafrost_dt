"""Pre-optimisation implementations, frozen as the reference for equivalence tests.

The FEM solver and the inversion forward model were rewritten for speed. The
rewrite is only defensible if it can be shown to compute the same numbers, and
that requires the original code to still be runnable. It does not, however,
belong in the services: production has one code path, chosen for us, with no
configuration switch that could quietly select the slow one.

So the originals live here instead. Nothing in ``software/digital_twin`` imports
this module -- only the verification and benchmarking tools do:

* ``verify_assembly_equivalence``   -- assembled matrices, old vs new
* ``verify_inversion_equivalence``  -- fitted parameters, old vs new
* ``optimization_ab_replay``        -- two full pipelines side by side

These functions are deliberately frozen. They are not maintained alongside the
solver and must not be "fixed" or tidied: their whole value is that they still
express the behaviour from before the rewrite. If the solver's physics genuinely
changes, the correct response is to retire the affected check, not to edit the
reference so that it agrees again.

Provenance: ``element_matrices`` is the element loop from ``_assemble_system``
and ``generate_series`` is ``FEMInversionServer._generate_series``, both copied
verbatim at the commit before the optimisation.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import numpy as np
import scipy.sparse as sp

from software.digital_twin.simulator.fem_forecast.fem_solver import FEMSolver2D
from software.tools.run_fem_case import FEM


# ---------------------------------------------------------------------------
# SOLVER: the original scalar element loop
# ---------------------------------------------------------------------------
def element_matrices(solver: FEMSolver2D) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Original per-element assembly of the stiffness and capacity matrices.

    One Python iteration per element per quadrature point, accumulating COO
    triplets in lists. Equivalent to ``FEMSolver2D._element_matrices``; kept for
    comparison only.
    """
    # ===== ELEMENT LOOP: Stiffness and Capacity Assembly =====
    # Assembles K (thermal conductivity) and C (heat capacity) matrices
    rows: list[int] = []
    cols: list[int] = []
    data_k: list[float] = []
    data_c: list[float] = []

    for iel in range(solver.numelem):
        sctrT = solver.element[iel, :].astype(int)
        nn = len(sctrT)
        for jel, w in enumerate(solver.W2):
            # --- Extract precomputed basis and Jacobian (from _build_geometry_cache) ---
            N = solver.elem_N[iel, jel, :]
            dNdx = solver.elem_dNdx[iel, jel, :, :]
            detJ = solver.elem_detJ[iel, jel]
            T_temp = float(N.T @ solver.T[sctrT])

            # ===== Auxiliary Relations: Phase-Change Logic =====
            exponent = solver.material.k_parameter * (T_temp - solver.material.tr_k)
            exponent = np.clip(exponent, -100, 100)
            X_exp = np.exp(exponent)
            si = 1.0 / (1.0 + X_exp)
            dsidT = -solver.material.k_parameter * X_exp / (1.0 + X_exp) ** 2
            sw = 1.0 - si

            landa = solver._bulk_conductivity(sw, si)

            rhoc_eff = (
                (1.0 - solver.material.porosity) * solver.material.rho_s * solver.material.cap_s
                + (solver.material.porosity * sw) * solver.material.rho_w * solver.material.cap_w
                + (solver.material.porosity * si) * solver.material.rho_i * solver.material.cap_i
            )
            cap_coeff = (
                rhoc_eff
                - solver.material.latent_heat * solver.material.rho_i * solver.material.porosity * dsidT
            )

            # ===== Local Element Contributions =====
            Ge = dNdx
            localK = landa * (Ge @ Ge.T) * detJ * w
            Ne = N.reshape(nn, 1)
            localC = (Ne @ Ne.T) * cap_coeff * detJ * w

            # Scatter-add into global sparse structure (COO accumulation)
            ii = np.repeat(sctrT, nn)
            jj = np.tile(sctrT, nn)
            rows.extend(ii.tolist())
            cols.extend(jj.tolist())
            data_k.extend(localK.ravel().tolist())
            data_c.extend(localC.ravel().tolist())

    # ===== Global Matrix Assembly from COO Format =====
    K_matrix = sp.coo_matrix((data_k, (rows, cols)), shape=(solver.numnode, solver.numnode)).tocsr()
    C_matrix = sp.coo_matrix((data_c, (rows, cols)), shape=(solver.numnode, solver.numnode)).tocsr()
    return K_matrix, C_matrix


@contextmanager
def reference_assembly() -> Iterator[None]:
    """Run the block with every solver assembling the original way.

    Patches the class rather than an instance because the forecast service builds
    its solver lazily, inside message handling, where a caller has no instance to
    reach. Restores the real method on the way out, including on exception, so a
    failure inside the block cannot leave the process running the slow path.
    """
    original = FEMSolver2D._element_matrices
    FEMSolver2D._element_matrices = element_matrices  # type: ignore[assignment]
    try:
        yield
    finally:
        FEMSolver2D._element_matrices = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# INVERSION: the original per-horizon forward model
# ---------------------------------------------------------------------------
def generate_series(
    params: dict[str, float],
    positions: list[dict],
    times_hours: list[float],
    ic_sensors: list[dict] | None = None,
    config_path: str = "startup.conf",
) -> dict[str, list[float]]:
    """Original forward model: one complete FEM run per window time.

    Each horizon restarts from the measured initial field and integrates from
    zero, so the early time steps are recomputed once for every later time.
    Equivalent to ``WindowForwardModel.generate_series``; kept for comparison
    only.
    """
    series = {pos["sensor_id"]: [] for pos in positions}
    for t in times_hours:
        result = FEM(
            params=params,
            positions=positions,
            boundary_snapshot=None,
            config_path=config_path,
            horizon_hours=float(t),
            return_field=False,
            sensors=ic_sensors,
        )
        for item in result["temperatures"]:
            sid = item["sensor_id"]
            series[sid].append(float(item["temperature"]))
    return series


@contextmanager
def reference_forward_model(server) -> Iterator[None]:
    """Run the block with one inversion server using the original forward model."""
    original = server._generate_series

    def _reference(params, positions, times_hours, ic_sensors=None):
        return generate_series(params, positions, times_hours, ic_sensors, server.config_path)

    server._generate_series = _reference
    try:
        yield
    finally:
        server._generate_series = original

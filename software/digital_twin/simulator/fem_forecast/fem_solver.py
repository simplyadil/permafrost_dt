"""Reusable 2D FEM solver core for thaw-front forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from software.digital_twin.simulator.fem_forecast.lagrange_basis import lagrange_basis
from software.digital_twin.simulator.fem_forecast.quadrature import quadrature
from software.utils.logging_setup import get_logger


# ---------------------------
# INPUT and PARAMETERS
# ---------------------------
@dataclass(frozen=True)
class FEMMeshConfig:
    """Mesh configuration paths (corresponds to baseline mesh loading section)."""
    mesh_dir: Path
    node_file: str
    element_file: str
    edge_sets: Mapping[str, str]
    node_sets: Mapping[str, str]


@dataclass(frozen=True)
class MaterialProps:
    """Material and thermal properties.

    Bulk thermal conductivity is not prescribed as a thawed/frozen pair any
    more. It is mixed from the three phases (solid grains, water, ice) via a
    geometric mean, so the only conductivity unknown is the grain conductivity
    ``lambda_s`` -- which is what the inversion service estimates. See
    ``_bulk_conductivity`` for the mixing law and the calibration of the
    default value.
    """
    porosity: float = 0.43
    rho_s: float = 2650.0
    rho_w: float = 1000.0
    rho_i: float = 910.0
    cap_s: float = 790.0
    cap_w: float = 4180.0
    cap_i: float = 1930.0
    latent_heat: float = 3.34e5
    k_parameter: float = 3.0
    tr_k: float = 273.15
    lambda_s: float = 3.51
    lambda_w: float = 0.56
    lambda_i: float = 2.24


@dataclass(frozen=True)
class SolverConfig:
    """Solver configuration"""
    dt_seconds: float = 30.0
    max_iterations: int = 1
    tol_r: float = 1e-6
    temperature_offset_k: float = 0.0
    initial_temperature_c: float = -7.0 #TODO read initial temp from csv, and interpolate for more accuracy
    thaw_threshold_c: float = 0.0


# ---------------------------
# FEM SOLVER CLASS
# ---------------------------
class FEMSolver2D:
    """Solve transient heat conduction with phase change on a 2D mesh.
    
    This class encapsulates the baseline's computational kernel:
      - Mesh loading (corresponds to MESHING and GEOMETRY section)
      - Geometry caching (precomputes basis functions and Jacobians)
      - System assembly (corresponds to element loop in baseline)
      - Time-stepping (corresponds to TIME-STEPPING LOOP in baseline)
    """

    # ---------------------------
    # INITIALIZATION and MESH SETUP
    # ---------------------------
    def __init__(
        self,
        mesh: FEMMeshConfig,
        material: MaterialProps,
        solver: SolverConfig,
    ) -> None:
        """Initialize solver with mesh, material, and solver configurations."""
        self.logger = get_logger("FEMSolver2D")
        self.mesh_cfg = mesh
        self.material = material
        self.solver = solver

        # Conductivities are mixed geometrically (see _bulk_conductivity), which
        # is only defined for strictly positive values -- a non-positive one
        # would silently yield a complex/NaN conductivity deep inside assembly.
        for name in ("lambda_s", "lambda_w", "lambda_i"):
            if float(getattr(material, name)) <= 0.0:
                raise ValueError(f"{name} must be > 0 (got {getattr(material, name)}).")
        # Grain term of the mixture: constant per material, so hoisted out of
        # the element/quadrature loop.
        self._lambda_solid_factor = float(material.lambda_s) ** (1.0 - float(material.porosity))

        # Load mesh arrays
        self.node = self._load_array(mesh.mesh_dir / mesh.node_file, dtype=float)
        self.element = self._load_array(mesh.mesh_dir / mesh.element_file, dtype=int)
        self.element = self._ensure_zero_based(self.element)

        # Load edge and node sets
        self.edge_sets = {
            name: self._ensure_zero_based(self._load_array(mesh.mesh_dir / path, dtype=int))
            for name, path in mesh.edge_sets.items()
        }
        self.node_sets = {
            name: self._ensure_zero_based(self._load_array(mesh.mesh_dir / path, dtype=int))
            for name, path in mesh.node_sets.items()
        }

        self.numnode = self.node.shape[0]
        self.numelem = self.element.shape[0]

        # Initialize quadrature rules
        self.W1, self.Q1 = quadrature(2, "GAUSS", 1)
        self.W2, self.Q2 = quadrature(2, "GAUSS", 2)

        # Initialize temperature field with a uniform fallback. This is replaced
        # by a spatially-interpolated field on the first sensor snapshot via
        # seed_initial_field(); _seeded guards against re-seeding once the
        # solution has started marching forward.
        initial_temp_k = self._to_kelvin(self.solver.initial_temperature_c)
        self.T = np.full(self.numnode, initial_temp_k, dtype=float)
        self._seeded = False
        self.time_hours = None
        self._build_geometry_cache()

    # ---------------------------
    # MATERIAL SWAPPING
    # ---------------------------
    def set_material(self, material: MaterialProps) -> None:
        """Replace the material properties, keeping mesh and geometry cache.

        Mesh loading and ``_build_geometry_cache`` depend only on the geometry,
        so a parameter sweep (the inversion optimiser evaluating candidate soil
        properties) does not need a new solver per candidate -- it needs this.
        Rebuilding instead costs ~79 ms of file I/O and basis evaluation per
        candidate, which for one inversion is several hundred redundant rebuilds.

        Only ``_lambda_solid_factor`` is derived from the material, so that is
        the only cached quantity to refresh. The temperature field is left
        untouched; callers restoring an initial condition must do so themselves.
        """
        for name in ("lambda_s", "lambda_w", "lambda_i"):
            if float(getattr(material, name)) <= 0.0:
                raise ValueError(f"{name} must be > 0 (got {getattr(material, name)}).")
        self.material = material
        self._lambda_solid_factor = float(material.lambda_s) ** (1.0 - float(material.porosity))

    # ---------------------------
    # INITIAL CONDITION SEEDING
    # ---------------------------
    def seed_initial_field(self, sensors: list[dict], *, force: bool = False) -> bool:
        """Seed the initial temperature field from a single sensor snapshot.

        Replaces the uniform initial condition with a per-node field built by
        spatially interpolating one snapshot of thermocouple readings onto the
        mesh nodes. This is a purely spatial operation: it needs all sensors at
        one instant, not a time history, so it works directly off the first
        streamed snapshot.

        Linear (Delaunay) interpolation is used inside the convex hull of the
        sensor positions; nodes outside that hull (e.g. unmonitored corners)
        fall back to nearest-sensor extrapolation. This mirrors the reference
        interpolation_initial.py.

        Sensor x_m/z_m are expected in the same metric frame as the mesh nodes.
        Runs at most once (guarded by ``_seeded``); later calls are no-ops so the
        forward-marching solution is never reset. ``force=True`` overrides that
        guard, for callers that deliberately rewind a reused solver to a new
        initial condition (the inversion's window forward model).

        Returns:
            True if the field was seeded, False if it kept the uniform fallback
            (already seeded, or too few usable sensors to triangulate).
        """
        if self._seeded and not force:
            return False

        points: list[tuple[float, float]] = []
        temps_c: list[float] = []
        for sensor in sensors:
            try:
                x_m = float(sensor["x_m"])
                z_m = float(sensor["z_m"])
                temp_c = float(sensor["temperature"])
            except (KeyError, TypeError, ValueError):
                continue
            points.append((x_m, z_m))
            temps_c.append(temp_c)

        # LinearNDInterpolator needs a non-degenerate triangulation (>=3
        # non-collinear points). Below that, keep the uniform IC.
        if len(points) < 3:
            self.logger.warning(
                "seed_initial_field: only %d usable sensor(s) (<3); keeping uniform IC.",
                len(points),
            )
            return False

        sensor_points = np.asarray(points, dtype=float)
        sensor_temps_c = np.asarray(temps_c, dtype=float)

        try:
            linear = LinearNDInterpolator(sensor_points, sensor_temps_c)
            nearest = NearestNDInterpolator(sensor_points, sensor_temps_c)
        except Exception as exc:  # e.g. QhullError for collinear sensors
            self.logger.warning(
                "seed_initial_field: interpolation setup failed (%s); keeping uniform IC.",
                exc,
            )
            return False

        node_temps_c = linear(self.node)
        nan_mask = np.isnan(node_temps_c)
        if np.any(nan_mask):
            node_temps_c[nan_mask] = nearest(self.node[nan_mask])

        self.T = node_temps_c.astype(float) + float(self.solver.temperature_offset_k)
        self._seeded = True
        self.logger.info(
            "Seeded initial field from %d sensors (T range %.2f..%.2f C, %d/%d nodes extrapolated).",
            len(points),
            float(np.min(node_temps_c)),
            float(np.max(node_temps_c)),
            int(np.sum(nan_mask)),
            self.numnode,
        )
        return True

    # ---------------------------
    # GEOMETRY CACHING
    # Precomputes basis functions and Jacobians for performance
    # ---------------------------

    def _build_geometry_cache(self) -> None:
        """Precompute shape/basis and geometric terms for faster assembly.
        
        This replaces the baseline's per-loop computation of:
          - Lagrange basis N(xi) and derivatives dNdxi at quadrature points
          - Jacobian J, detJ, invJ for physical space mapping
        """
        self.edge_cache: dict[str, dict[str, np.ndarray]] = {}
        self.edge_basis = []
        for j, w in enumerate(self.W1):
            xi = self.Q1[j]
            N, _ = lagrange_basis("L2", xi)
            self.edge_basis.append((N, float(w)))

        for name, edges in self.edge_sets.items():
            if edges.size == 0:
                self.edge_cache[name] = {"edges": edges, "J0": np.array([])}
                continue
            sctr = edges.astype(int)
            coords = self.node[sctr, :]
            edge_lengths = np.linalg.norm(coords[:, 1, :] - coords[:, 0, :], axis=1)
            self.edge_cache[name] = {"edges": sctr, "J0": edge_lengths / 2.0}

        self.elem_sctr = self.element.astype(int)
        nqp = len(self.W2)
        nn = self.elem_sctr.shape[1]
        self.elem_N = np.zeros((self.numelem, nqp, nn), dtype=float)
        self.elem_dNdx = np.zeros((self.numelem, nqp, nn, 2), dtype=float)
        self.elem_detJ = np.zeros((self.numelem, nqp), dtype=float)

        #TODO also compute J on gpu since our mesh is large it could be causing overhead?q
        for iel in range(self.numelem):
            sctrT = self.elem_sctr[iel]
            Xe = self.node[sctrT, :]
            for jel in range(nqp):
                pt = self.Q2[jel, :]
                N, dNdxi = lagrange_basis("Q4", pt)
                J = Xe.T @ dNdxi
                detJ = float(np.linalg.det(J))
                if detJ <= 0.0:
                    raise ValueError(f"Negative/zero detJ at elem {iel}")
                invJ = np.linalg.inv(J)
                dNdx = dNdxi @ invJ
                self.elem_N[iel, jel, :] = N
                self.elem_dNdx[iel, jel, :, :] = dNdx
                self.elem_detJ[iel, jel] = detJ

        self._build_sparsity_pattern()

    def _build_sparsity_pattern(self) -> None:
        """Precompute the fixed COO->CSR scatter map for the element matrices.

        The (row, col) triplet lists produced by the element loop depend only on
        the element connectivity, so they are identical on every assembly. The
        original code rebuilt them -- and re-sorted them in ``coo_matrix.tocsr()``
        -- for every time step. Here the pattern is built once and each assembly
        only fills the CSR ``data`` array via ``np.bincount``.

        ``rows``/``cols`` reproduce the per-element ``np.repeat(sctr, nn)`` /
        ``np.tile(sctr, nn)`` ordering exactly, so ``Ke.ravel()`` lines up with
        the scatter map without any reordering.
        """
        sctr = self.elem_sctr
        nn = sctr.shape[1]
        rows = np.repeat(sctr, nn, axis=1).ravel()
        cols = np.tile(sctr, (1, nn)).ravel()

        n = self.numnode
        linear = rows.astype(np.int64) * n + cols.astype(np.int64)
        unique, inverse = np.unique(linear, return_inverse=True)
        self._scatter_map = inverse.astype(np.int64)
        self._scatter_nnz = int(unique.size)
        self._csr_indices = (unique % n).astype(np.int32)
        unique_rows = (unique // n).astype(np.int64)
        self._csr_indptr = np.searchsorted(unique_rows, np.arange(n + 1)).astype(np.int32)

    # ---------------------------
    # HELPER METHODS
    # Utility functions for coordinate transformations and boundary identification
    # ---------------------------
    @staticmethod
    def _load_array(path: Path, dtype) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Mesh file not found at {path}")
        return np.loadtxt(str(path), dtype=dtype)

    @staticmethod
    def _ensure_zero_based(data: np.ndarray) -> np.ndarray:
        if data.size == 0:
            return data
        if data.min() == 1:
            return data - 1
        return data

    def _to_kelvin(self, temp_c: float) -> float:
        return float(temp_c) + float(self.solver.temperature_offset_k)

    def _bc_nodal_temperature_k(self, bc: dict, node_ids: np.ndarray) -> np.ndarray | None:
        """Nodal boundary temperature (Kelvin) for the given node ids.

        Supports a depth-resolved ``temperature_profile`` (list of
        ``{z_m, temperature}``) interpolated by node depth ``z``; values outside
        the profile range are clamped to its ends. Falls back to the scalar
        ``temperature``. Returns None if neither is present.
        """
        node_ids = np.asarray(node_ids, dtype=int)
        profile = bc.get("temperature_profile")
        if isinstance(profile, list) and profile:
            zs: list[float] = []
            ts: list[float] = []
            for point in profile:
                try:
                    zs.append(float(point["z_m"]))
                    ts.append(float(point["temperature"]))
                except (KeyError, TypeError, ValueError):
                    continue
            if len(zs) >= 2:
                order = np.argsort(zs)
                zs_arr = np.asarray(zs)[order]
                ts_arr = np.asarray(ts)[order]
                z_nodes = self.node[node_ids, 1]
                temps_c = np.interp(z_nodes, zs_arr, ts_arr)  # clamps at ends
                return temps_c + float(self.solver.temperature_offset_k)
            if len(zs) == 1:
                return np.full(node_ids.shape[0], self._to_kelvin(ts[0]))
        temperature = bc.get("temperature")
        if temperature is None:
            return None
        return np.full(node_ids.shape[0], self._to_kelvin(float(temperature)))

    def _bulk_conductivity(self, sw: float, si: float) -> float:
        """Three-phase bulk conductivity from a geometric-mean mixing law.

            landa = lambda_s^(1-poro) * lambda_w^(poro*sw) * lambda_i^(poro*si)

        This replaces the former ``lambda_thawed``/``lambda_frozen`` switch at
        Tr. Two reasons: the conductivity is now continuous through the
        phase-change band, following the same smooth ice/water saturations that
        already drive the capacity term instead of jumping at the threshold;
        and a single grain conductivity carries both regimes, so the inversion
        estimates one physical parameter rather than two correlated ones.

        The default ``lambda_s`` = 3.51 W/(m K) is the measured grain value. At
        poro = 0.43 the law gives 1.59 W/(m K) fully thawed (sw = 1) and
        2.89 W/(m K) fully frozen (si = 1).

        Air is not a phase here: the sample is treated as saturated, so the pore
        space is split between water and ice only and ``lambda_a`` never enters.
        """
        poro = self.material.porosity
        return float(
            self._lambda_solid_factor
            * self.material.lambda_w ** (poro * sw)
            * self.material.lambda_i ** (poro * si)
        )

    def _thaw_threshold_k(self) -> float:
        return self._to_kelvin(self.solver.thaw_threshold_c)

    def _boundary_nodes(self, boundary_set: str) -> np.ndarray:
        if boundary_set in self.node_sets:
            nodes = self.node_sets[boundary_set]
            return nodes.reshape(-1)
        if boundary_set in self.edge_sets:
            edges = self.edge_sets[boundary_set]
            return np.unique(edges.reshape(-1))
        raise KeyError(f"Unknown boundary set '{boundary_set}'")

    def _boundary_edges(self, boundary_set: str) -> np.ndarray:
        if boundary_set not in self.edge_sets:
            raise KeyError(f"Unknown boundary edge set '{boundary_set}'")
        return self.edge_sets[boundary_set]

    # ---------------------------
    # SYSTEM ASSEMBLY
    # Core FEM assembly loop corresponding to baseline TIME-STEPPING LOOP
    # ---------------------------

    #TODO  use cupy for gpu bases matmul, move away from nested loops
    def _assemble_system(
        self,
        boundary_snapshot: dict,
        t_prev: np.ndarray | None = None,
    ) -> tuple[sp.csr_matrix, np.ndarray, dict[int, float]]:
        """Assemble global stiffness matrix K and force vector f.

        ``self.T`` supplies the temperature at which the nonlinear material
        properties are evaluated (the latest Newton iterate). ``t_prev`` is the
        *previous time step* field that feeds the transient term ``C@t_prev/dt``;
        it must stay pinned across the Newton iterations of one step. Defaults
        to ``self.T`` for callers assembling outside a Newton loop.

        Three stages:
          1. Boundary terms   -- ``_assemble_boundary`` (independent of self.T)
          2. Element matrices -- ``_element_matrices`` (the nonlinear part)
          3. Backward-Euler combination into A and b

        Corresponds to baseline: "for i in range(LeftEdge.shape[0]):" and "for iel in range(numelem):"
        """
        fT, K_robin, dirichlet_map = self._assemble_boundary(boundary_snapshot)
        K_matrix, C_matrix = self._element_matrices()

        A = K_matrix + K_robin + (C_matrix / float(self.solver.dt_seconds))
        transient_field = self.T if t_prev is None else t_prev
        b = fT + (C_matrix @ transient_field) / float(self.solver.dt_seconds)
        return A, b, dirichlet_map

    def _assemble_boundary(
        self,
        boundary_snapshot: dict,
    ) -> tuple[np.ndarray, sp.csr_matrix, dict[int, float]]:
        """Boundary force vector, Robin stiffness and Dirichlet map.

        Lifted out of ``_assemble_system`` unchanged. These terms depend on the
        boundary snapshot and the geometry only -- never on ``self.T`` -- which
        is what makes the element stage the sole nonlinear part.
        """
        fT = np.zeros(self.numnode)
        dirichlet_map: dict[int, float] = {}

        # ===== Boundary Condition Assembly (robin/neumann on edges) =====
        # Corresponds to baseline:
        #   for i in range(LeftEdge.shape[0]): ... fT_local = -N * landa_e1 * (TT - Te1) * J0 * W1[j]
        #   for i in range(TopEdge.shape[0]):  ... (same pattern)
        #
        # Robin BCs are treated implicitly: h*(T_new - T_env) contributes
        #   K_robin[a,b] += h * N_a * N_b * J0 * w  (stiffness, assembled below)
        #   fT[a]        += h * T_env * N_a * J0 * w (force)
        # This avoids evaluating the flux from the old temperature (explicit), which
        # can introduce inaccuracy at high h_coeff or near the phase-change front.
        rows_r: list[int] = []
        cols_r: list[int] = []
        data_r: list[float] = []

        boundaries = boundary_snapshot.get("boundaries", [])
        for bc in boundaries:
            bc_type = bc.get("bc_type")
            boundary_set = bc.get("boundary_set")
            if not boundary_set or not bc_type:
                continue

            if bc_type == "dirichlet":
                nodes = self._boundary_nodes(boundary_set)
                temps_k = self._bc_nodal_temperature_k(bc, nodes)
                if temps_k is None:
                    continue
                for node_id, temp_k in zip(nodes, temps_k):
                    dirichlet_map[int(node_id)] = float(temp_k)
                continue

            if bc_type not in {"neumann", "robin"}:
                continue

            edges = self.edge_cache.get(boundary_set, {}).get("edges")
            J0s = self.edge_cache.get(boundary_set, {}).get("J0")        #<-- Jacobian determinant scale factor
            if edges is None or J0s is None:
                edges = self._boundary_edges(boundary_set).astype(int)
                coords = self.node[edges, :]
                edge_lengths = np.linalg.norm(coords[:, 1, :] - coords[:, 0, :], axis=1)
                J0s = edge_lengths / 2.0
            # For Robin, precompute h and the per-node ambient temperature (K),
            # which may vary with depth via a temperature_profile.
            robin_h: float | None = None
            robin_amb_k: np.ndarray | None = None
            if bc_type == "robin":
                h_coeff = bc.get("h_coeff")
                if h_coeff is None:
                    continue
                robin_h = float(h_coeff)
                robin_amb_k = self._bc_nodal_temperature_k(bc, np.arange(self.numnode))
                if robin_amb_k is None:
                    continue

            for idx, sctr in enumerate(edges):
                nn = len(sctr)
                J0 = float(J0s[idx])
                amb_nodes = robin_amb_k[sctr] if robin_amb_k is not None else None
                for N, w in self.edge_basis:
                    if bc_type == "robin":
                        # Ambient temperature at the quadrature point from the
                        # (possibly depth-varying) nodal ambient values.
                        temp_k = float(N @ amb_nodes)
                        # Implicit: force term only carries the known T_env part;
                        # T_new part goes into the stiffness matrix K_robin.
                        fT_local = N * robin_h * temp_k * J0 * w
                        for a in range(nn):
                            fT[sctr[a]] += fT_local[a]
                        # K_robin contribution: h * N_a * N_b * J0 * w
                        K_robin_local = robin_h * np.outer(N, N) * J0 * w
                        for a in range(nn):
                            for b_idx in range(nn):
                                rows_r.append(sctr[a])
                                cols_r.append(sctr[b_idx])
                                data_r.append(K_robin_local[a, b_idx])
                    else:
                        heat_flux = bc.get("heat_flux")
                        if heat_flux is None:
                            continue
                        qn = float(heat_flux)
                        fT_local = N * qn * J0 * w
                        for a in range(nn):
                            fT[sctr[a]] += fT_local[a]

        if data_r:
            K_robin = sp.coo_matrix((data_r, (rows_r, cols_r)), shape=(self.numnode, self.numnode)).tocsr()
        else:
            K_robin = sp.csr_matrix((self.numnode, self.numnode))
        return fT, K_robin, dirichlet_map

    # ---------------------------
    # ELEMENT MATRICES
    # ---------------------------
    def _element_matrices(self) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """Stiffness and capacity matrices, batched over the whole mesh.

        This replaced a per-element Python loop that spent its time on
        interpreter overhead around 4x4 operations -- numelem*nqp iterations,
        each issuing a handful of tiny numpy calls and appending 64 Python
        numbers to list accumulators -- rather than on floating-point work. The
        geometry cache already stores everything in ``(numelem, nqp, ...)``
        layout, so batching is a direct rewrite of the same expressions.

        Two further redundancies went with it: the (row, col) triplets depend
        only on the connectivity and are precomputed once by
        ``_build_sparsity_pattern`` instead of rebuilt every step, and the
        duplicate-summing sort inside ``coo_matrix.tocsr()`` is replaced by a
        ``bincount`` into that fixed pattern.

        The arithmetic is unchanged from the loop it replaced, which is kept in
        ``software/tools/golden_reference.py`` and checked against this method to
        machine precision by ``software/tools/verify_assembly_equivalence.py``.
        """
        m = self.material

        # Temperature at every quadrature point of every element: (numelem, nqp)
        T_qp = np.einsum("eqn,en->eq", self.elem_N, self.T[self.elem_sctr])

        # ===== Auxiliary Relations: Phase-Change Logic (elementwise) =====
        exponent = np.clip(m.k_parameter * (T_qp - m.tr_k), -100.0, 100.0)
        X_exp = np.exp(exponent)
        si = 1.0 / (1.0 + X_exp)
        dsidT = -m.k_parameter * X_exp / (1.0 + X_exp) ** 2
        sw = 1.0 - si

        # Same geometric mixing law as _bulk_conductivity, over the whole array.
        landa = (
            self._lambda_solid_factor
            * m.lambda_w ** (m.porosity * sw)
            * m.lambda_i ** (m.porosity * si)
        )
        rhoc_eff = (
            (1.0 - m.porosity) * m.rho_s * m.cap_s
            + (m.porosity * sw) * m.rho_w * m.cap_w
            + (m.porosity * si) * m.rho_i * m.cap_i
        )
        cap_coeff = rhoc_eff - m.latent_heat * m.rho_i * m.porosity * dsidT

        # Fold detJ and the quadrature weight into the per-point coefficients.
        weights = np.asarray(self.W2, dtype=float).reshape(1, -1)
        gk = landa * self.elem_detJ * weights
        gc = cap_coeff * self.elem_detJ * weights

        # localK = landa * (Ge @ Ge.T) * detJ * w, summed over quadrature points
        Ke = np.einsum("eqad,eqbd,eq->eab", self.elem_dNdx, self.elem_dNdx, gk)
        # localC = (Ne @ Ne.T) * cap_coeff * detJ * w, likewise
        Ce = np.einsum("eqa,eqb,eq->eab", self.elem_N, self.elem_N, gc)

        return self._scatter_to_csr(Ke), self._scatter_to_csr(Ce)

    def _scatter_to_csr(self, element_values: np.ndarray) -> sp.csr_matrix:
        """Sum per-element blocks into the precomputed global CSR pattern."""
        data = np.bincount(
            self._scatter_map,
            weights=element_values.ravel(),
            minlength=self._scatter_nnz,
        )
        return sp.csr_matrix(
            (data, self._csr_indices, self._csr_indptr),
            shape=(self.numnode, self.numnode),
        )

    # ---------------------------
    # BOUNDARY CONDITION APPLICATION
    # ---------------------------
    @staticmethod
    def _apply_dirichlet(A_mat: sp.csr_matrix, b_vec: np.ndarray, dofs: dict[int, float]) -> tuple[sp.csr_matrix, np.ndarray]:
        """Apply essential (Dirichlet) boundary conditions by row replacement.

        For each constrained DOF the matrix row is zeroed, a scaled identity is
        placed on the diagonal, and the RHS is set so the solve returns the
        prescribed value: ``A[d,d]*T[d] = scale*value`` -> ``T[d] = value``.

        Only the *row* is replaced; the column is deliberately left intact. The
        constrained value enters each neighbour's equation through the surviving
        ``A[n,d]*T[d]`` coupling, which is correct because ``T[d]`` is pinned.
        Zeroing the column as well (without moving ``A[:,d]*value`` to the RHS)
        drops that coupling from every neighbour and makes the transient solve
        diverge — a -5 C boundary was driving the field to <-150 C over a few
        steps. The matrix is no longer symmetric, but the direct sparse solver
        handles that fine.
        """
        if not dofs:
            return A_mat, b_vec
        diag_scale = float(np.mean(A_mat.diagonal())) if A_mat.nnz else 1.0
        A_lil = A_mat.tolil()
        for dof, value in dofs.items():
            A_lil[dof, :] = 0.0
            A_lil[dof, dof] = diag_scale
            b_vec[dof] = diag_scale * float(value)
        return A_lil.tocsr(), b_vec

    # ---------------------------
    # TIME STEPPING / NEWTON ITERATION
    # ---------------------------
    def _solve_step(self, boundary_snapshot: dict) -> None:
        """Solve a single time step using Newton iteration.

        The system is reassembled at each iteration so that material properties
        (phase-change ice fraction, thermal conductivity, heat capacity) are
        re-evaluated at the latest temperature estimate.  Without reassembly the
        loop was a no-op: spsolve (direct) drives the linear residual to machine
        precision in one step, so additional iterations added nothing.

        The previous-step field ``Xn`` is captured once and held fixed for the
        whole iteration, matching the reference formulation

            residual = Fext + C*Xn/dt - (K + C/dt)*X_new

        Letting the transient term follow the iterate instead (assembling
        ``C@X_new/dt``) collapses the residual to the *steady-state* form
        ``fT - K@X_new``, so every extra iteration marches the step toward
        equilibrium rather than toward the backward-Euler solution. That is
        invisible at ``max_iterations=1`` and progressively over-diffuses the
        field at higher iteration counts.

        Convergence is measured relative to ``||b||`` (the reference's ``rext``).
        An absolute test cannot work here: the residual carries units of watts
        and is O(1e3-1e6), so a fixed 1e-6 threshold never trips.
        """
        Xn = self.T.copy()
        X_new = Xn.copy()

        for _ in range(self.solver.max_iterations):
            # Reassemble with the current temperature estimate so that the
            # nonlinear material properties reflect the latest X_new, while the
            # transient term stays anchored to the previous step (Xn).
            self.T = X_new
            A, b, dirichlet_map = self._assemble_system(boundary_snapshot, t_prev=Xn)
            A_bc, b_bc = self._apply_dirichlet(A, b, dirichlet_map)
            residual = b_bc - A_bc @ X_new
            ref_norm = float(np.linalg.norm(b_bc))
            rel_residual = float(np.linalg.norm(residual)) / (ref_norm if ref_norm > 0.0 else 1.0)
            if rel_residual < self.solver.tol_r:
                break
            dX = spla.spsolve(A_bc, residual)        # replace spsolve with cuSPARSE, a faster GPU based linear solver
            X_new = X_new + dX

        self.T = X_new

    def advance(self, boundary_snapshot: dict, horizon_hours: float) -> None:
        """Advance temperature solution forward in time.
        
        Calls _solve_step() multiple times for the specified horizon."""
        steps = max(1, int(round(float(horizon_hours) * 3600.0 / float(self.solver.dt_seconds))))
        for _ in range(steps):
            self._solve_step(boundary_snapshot)
        self.time_hours = float(boundary_snapshot.get("time_hours", 0.0)) + float(horizon_hours)

    # ---------------------------
    # POST-PROCESSING
    # Extract thaw-front metrics from temperature solution
    # ---------------------------
    def compute_thaw_metrics(self) -> tuple[float | None, float | None, list[dict[str, float]]]:
        """Extract thaw-front metrics from temperature field.

        For each depth level z the front is the point where temperature crosses
        the thaw threshold, located by linear interpolation between the last
        thawed node and the first frozen node outside it. r_max and r_avg are
        computed from these front points only -- not from the entire thawed
        interior -- making them consistent with the contour method in
        thaw_front_reconstruction_server, which averages over isotherm points
        rather than all thawed nodes.

        Why interpolate rather than take the outermost thawed node, which is what
        this did originally: the node radii are the mesh spacing apart (10 mm on
        the shipped mesh), so a node-based radius can only ever report 0, 10, 20
        mm and advances in jumps as the front crosses each ring. Three
        consequences, all of which showed up in practice:

        * the forecast front lagged the measured one by about half an hour purely
          from quantisation, which is why the predictive arm of the safety
          monitor had never once fired before the measured arm;
        * a parameter change that moves the front by less than one cell is
          invisible, so the inversion could not be evaluated against r_max;
        * the safety monitor's own thresholds fell between representable values,
          giving the graded statuses a resolution of one mesh cell.

        The interpolated radius is always within one cell of the node-based one
        and never smaller, so this refines the previous answer rather than
        contradicting it -- a property checked by
        ``software/tools/verify_front_resolution.py``.
        """
        threshold_k = self._thaw_threshold_k()
        if not np.any(self.T >= threshold_k):
            return None, None, []

        r_abs = np.abs(self.node[:, 0])
        z_coords = self.node[:, 1]

        front_points: list[dict[str, float]] = []
        for z in np.unique(z_coords):
            row = np.where(z_coords == z)[0]
            order = np.argsort(r_abs[row])
            row = row[order]
            radii = r_abs[row]
            temps = self.T[row]

            thawed = temps >= threshold_k
            if not np.any(thawed):
                continue

            # Outermost thawed node on this row. Scanning from the outside in
            # rather than taking the first crossing keeps the envelope on the
            # outer front when the row is not monotonic in temperature.
            last = int(np.max(np.where(thawed)[0]))
            if last >= len(radii) - 1:
                # Thawed all the way to the domain edge: there is no crossing to
                # interpolate, and the front is at least the edge radius.
                front_r = float(radii[last])
            else:
                t_in = float(temps[last])
                t_out = float(temps[last + 1])
                span = t_in - t_out
                # span <= 0 would mean the "frozen" neighbour is warmer, so there
                # is no usable crossing; fall back to the node itself.
                fraction = (t_in - threshold_k) / span if span > 0.0 else 0.0
                fraction = min(max(fraction, 0.0), 1.0)
                front_r = float(radii[last] + (radii[last + 1] - radii[last]) * fraction)

            front_points.append({"x_m": front_r, "z_m": float(z)})

        if not front_points:
            return None, None, []

        front_radii = np.array([p["x_m"] for p in front_points])
        return float(np.max(front_radii)), float(np.mean(front_radii)), front_points

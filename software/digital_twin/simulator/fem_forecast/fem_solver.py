"""Reusable 2D FEM solver core for thaw-front forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

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
    """Material and thermal properties"""
    porosity: float = 0.42
    rho_s: float = 2700.0
    rho_w: float = 1000.0
    rho_i: float = 920.0
    cap_s: float = 790.0
    cap_w: float = 4180.0
    cap_i: float = 2090.0
    latent_heat: float = 3.34e5
    k_parameter: float = 5.0
    tr_k: float = 273.15
    lambda_thawed: float = 1.64
    lambda_frozen: float = 2.96


@dataclass(frozen=True)
class SolverConfig:
    """Solver configuration"""
    dt_seconds: float = 36.0
    max_iterations: int = 1
    tol_r: float = 1e-6
    temperature_offset_k: float = 0.0
    initial_temperature_c: float = -1.0
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

        # Initialize temperature field
        initial_temp_k = self._to_kelvin(self.solver.initial_temperature_c)
        self.T = np.full(self.numnode, initial_temp_k, dtype=float)
        self.time_hours = None
        self._build_geometry_cache()

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
    def _assemble_system(self, boundary_snapshot: dict) -> tuple[sp.csr_matrix, np.ndarray, dict[int, float]]:
        """Assemble global stiffness matrix K and force vector f.
        
        This implements the baseline's assembly logic:
          1. Boundary condition assembly (robin/neumann on edges)
          2. Element loop with Jacobian computation
          3. Local stiffness/capacity contributions
          4. Global sparse matrix assembly
          5. Application of Dirichlet boundary conditions
        
        Corresponds to baseline: "for i in range(LeftEdge.shape[0]):" and "for iel in range(numelem):"
        """
        fT = np.zeros(self.numnode)
        dirichlet_map: dict[int, float] = {}

        # ===== Boundary Condition Assembly (robin/neumann on edges) =====
        # Corresponds to baseline:
        #   for i in range(LeftEdge.shape[0]): ... fT_local = -N * landa_e1 * (TT - Te1) * J0 * W1[j]
        #   for i in range(TopEdge.shape[0]):  ... (same pattern)
        boundaries = boundary_snapshot.get("boundaries", [])
        for bc in boundaries:
            bc_type = bc.get("bc_type")
            boundary_set = bc.get("boundary_set")
            if not boundary_set or not bc_type:
                continue

            if bc_type == "dirichlet":
                temperature = bc.get("temperature")
                if temperature is None:
                    continue
                temp_k = self._to_kelvin(float(temperature))
                nodes = self._boundary_nodes(boundary_set)
                for node_id in nodes:
                    dirichlet_map[int(node_id)] = temp_k
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
            for idx, sctr in enumerate(edges):
                nn = len(sctr)
                J0 = float(J0s[idx])
                for N, w in self.edge_basis:
                    if bc_type == "robin":
                        temperature = bc.get("temperature")
                        h_coeff = bc.get("h_coeff")
                        if temperature is None or h_coeff is None:
                            continue
                        temp_k = self._to_kelvin(float(temperature))
                        h_coeff = float(h_coeff)
                        TT = N.T @ self.T[sctr]
                        fT_local = -N * h_coeff * (TT - temp_k) * J0 * w
                    else:
                        heat_flux = bc.get("heat_flux")
                        if heat_flux is None:
                            continue
                        qn = float(heat_flux)
                        fT_local = N * qn * J0 * w
                    for a in range(nn):
                        fT[sctr[a]] += fT_local[a]

        # ===== ELEMENT LOOP: Stiffness and Capacity Assembly =====
        # Assembles K (thermal conductivity) and C (heat capacity) matrices
        rows: list[int] = []
        cols: list[int] = []
        data_k: list[float] = []
        data_c: list[float] = []

        for iel in range(self.numelem):
            sctrT = self.element[iel, :].astype(int)
            nn = len(sctrT)
            for jel, w in enumerate(self.W2):
                # --- Extract precomputed basis and Jacobian (from _build_geometry_cache) ---
                # Baseline computes these fresh each loop; we cached them for performance
                N = self.elem_N[iel, jel, :]
                dNdx = self.elem_dNdx[iel, jel, :, :]
                detJ = self.elem_detJ[iel, jel]
                T_temp = float(N.T @ self.T[sctrT])

                # ===== Auxiliary Relations: Phase-Change Logic =====
                # Corresponds to baseline:
                #   X_exp = np.exp(k_parameter * (T_temp - Tr))
                #   si = 1.0 / (1.0 + X_exp)
                #   dsidT = -k_parameter * X_exp / (1.0 + X_exp)**2
                #   sw = 1.0 - si
                #   landa = (T_temp >= 273.15) ? 1.64 : 2.96
                #   rhoc_eff = (1-poro)*rho_s*cap_s + (poro*sw)*rho_w*cap_w + (poro*si)*rho_i*cap_i
                # With clamping to prevent exponential overflow
                exponent = self.material.k_parameter * (T_temp - self.material.tr_k)
                exponent = np.clip(exponent, -100, 100)
                X_exp = np.exp(exponent)
                si = 1.0 / (1.0 + X_exp)
                dsidT = -self.material.k_parameter * X_exp / (1.0 + X_exp) ** 2
                sw = 1.0 - si

                if T_temp >= self.material.tr_k:
                    landa = self.material.lambda_thawed
                else:
                    landa = self.material.lambda_frozen

                rhoc_eff = (
                    (1.0 - self.material.porosity) * self.material.rho_s * self.material.cap_s
                    + (self.material.porosity * sw) * self.material.rho_w * self.material.cap_w
                    + (self.material.porosity * si) * self.material.rho_i * self.material.cap_i
                )
                cap_coeff = rhoc_eff - self.material.latent_heat * self.material.rho_i * self.material.porosity * dsidT

                # ===== Local Element Contributions =====
                #   localK = landa * (Ge @ Ge.T) * detJ * w
                #   localC = (Ne @ Ne.T) * cap_coeff * detJ * w
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
        # Convert accumulated COO triplets to CSR for efficient solving
        K_matrix = sp.coo_matrix((data_k, (rows, cols)), shape=(self.numnode, self.numnode)).tocsr()
        C_matrix = sp.coo_matrix((data_c, (rows, cols)), shape=(self.numnode, self.numnode)).tocsr()
        A = K_matrix + (C_matrix / float(self.solver.dt_seconds))
        b = fT + (C_matrix @ self.T) / float(self.solver.dt_seconds)
        return A, b, dirichlet_map

    # ---------------------------
    # BOUNDARY CONDITION APPLICATION
    # ---------------------------
    @staticmethod
    def _apply_dirichlet(A_mat: sp.csr_matrix, b_vec: np.ndarray, dofs: dict[int, float]) -> tuple[sp.csr_matrix, np.ndarray]:
        """Apply essential (Dirichlet) boundary conditions by row/column elimination."""
        if not dofs:
            return A_mat, b_vec
        diag_scale = float(np.mean(A_mat.diagonal())) if A_mat.nnz else 1.0
        A_lil = A_mat.tolil()
        for dof, value in dofs.items():
            A_lil[dof, :] = 0.0
            A_lil[:, dof] = 0.0
            A_lil[dof, dof] = diag_scale
            b_vec[dof] = diag_scale * float(value)
        return A_lil.tocsr(), b_vec

    # ---------------------------
    # TIME STEPPING / NEWTON ITERATION
    # ---------------------------
    def _solve_step(self, boundary_snapshot: dict) -> None:
        """Solve a single time step using Newton iteration.
        
        baseline:
          A = K_matrix + (C_matrix / dt)
          b = fT + (C_matrix @ Xn) / dt
          while not converged:
              dX = spla.spsolve(A_bc, residual)
              X_new = X_new + dX
              residual = b - A @ X_new
              if norm(residual) < tol_r: converged=True
        """
        A, b, dirichlet_map = self._assemble_system(boundary_snapshot)
        A_bc, b_bc = self._apply_dirichlet(A, b, dirichlet_map)

        X_new = self.T.copy()
        residual = b_bc - A_bc @ X_new

        for _ in range(self.solver.max_iterations):
            dX = spla.spsolve(A_bc, residual)
            X_new = X_new + dX
            residual = b_bc - A_bc @ X_new
            if np.linalg.norm(residual) < self.solver.tol_r:
                break

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
        """Extract thaw-front radius from temperature field.
        
        Finds all nodes that exceed thaw_threshold_c and computes
        max and mean radial distance from domain center.
        """
        threshold_k = self._thaw_threshold_k()
        thaw_mask = self.T >= threshold_k
        if not np.any(thaw_mask):
            return None, None, []

        thaw_nodes = self.node[thaw_mask, :2]
        points = [{"x_m": float(x), "z_m": float(z)} for x, z in thaw_nodes]
        radii = np.abs(thaw_nodes[:, 0])
        return float(np.max(radii)), float(np.mean(radii)), points

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


@dataclass(frozen=True)
class FEMMeshConfig:
    mesh_dir: Path
    node_file: str
    element_file: str
    edge_sets: Mapping[str, str]
    node_sets: Mapping[str, str]


@dataclass(frozen=True)
class MaterialProps:
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
    dt_seconds: float = 36.0
    max_iterations: int = 25
    tol_r: float = 1e-6
    temperature_offset_k: float = 0.0
    initial_temperature_c: float = -1.0
    thaw_threshold_c: float = 0.0


class FEMSolver2D:
    """Solve transient heat conduction with phase change on a 2D mesh."""

    def __init__(
        self,
        mesh: FEMMeshConfig,
        material: MaterialProps,
        solver: SolverConfig,
    ) -> None:
        self.logger = get_logger("FEMSolver2D")
        self.mesh_cfg = mesh
        self.material = material
        self.solver = solver

        self.node = self._load_array(mesh.mesh_dir / mesh.node_file, dtype=float)
        self.element = self._load_array(mesh.mesh_dir / mesh.element_file, dtype=int)
        self.element = self._ensure_zero_based(self.element)

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

        self.W1, self.Q1 = quadrature(2, "GAUSS", 1)
        self.W2, self.Q2 = quadrature(2, "GAUSS", 2)

        initial_temp_k = self._to_kelvin(self.solver.initial_temperature_c)
        self.T = np.full(self.numnode, initial_temp_k, dtype=float)
        self.time_hours = None

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

    def _assemble_system(self, boundary_snapshot: dict) -> tuple[sp.csr_matrix, np.ndarray, dict[int, float]]:
        Ktt = sp.lil_matrix((self.numnode, self.numnode))
        Ctt = sp.lil_matrix((self.numnode, self.numnode))
        fT = np.zeros(self.numnode)
        dirichlet_map: dict[int, float] = {}

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

            edges = self._boundary_edges(boundary_set)
            for edge in edges:
                sctr = edge.astype(int)
                nn = len(sctr)
                coord = self.node[sctr, :]
                if coord.shape[1] < 2:
                    raise ValueError("Node coordinates must be 2D.")
                edge_length = np.linalg.norm(coord[1] - coord[0])
                J0 = edge_length / 2.0
                for j, w in enumerate(self.W1):
                    xi = self.Q1[j]
                    N, _ = lagrange_basis("L2", xi)
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

        for iel in range(self.numelem):
            sctrT = self.element[iel, :].astype(int)
            nn = len(sctrT)
            for jel, w in enumerate(self.W2):
                pt = self.Q2[jel, :]
                N, dNdxi = lagrange_basis("Q4", pt)
                Xe = self.node[sctrT, :]
                J = Xe.T @ dNdxi
                detJ = float(np.linalg.det(J))
                if detJ <= 0.0:
                    raise ValueError(f"Negative/zero detJ at elem {iel}")
                invJ = np.linalg.inv(J)
                dNdx = dNdxi @ invJ
                T_temp = float(N.T @ self.T[sctrT])

                X_exp = np.exp(self.material.k_parameter * (T_temp - self.material.tr_k))
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

                Ge = dNdx
                localK = landa * (Ge @ Ge.T) * detJ * w
                Ne = N.reshape(nn, 1)
                localC = (Ne @ Ne.T) * cap_coeff * detJ * w

                Ktt[np.ix_(sctrT, sctrT)] += localK
                Ctt[np.ix_(sctrT, sctrT)] += localC

        K_matrix = Ktt.tocsr()
        C_matrix = Ctt.tocsr()
        A = K_matrix + (C_matrix / float(self.solver.dt_seconds))
        b = fT + (C_matrix @ self.T) / float(self.solver.dt_seconds)
        return A, b, dirichlet_map

    @staticmethod
    def _apply_dirichlet(A_mat: sp.csr_matrix, b_vec: np.ndarray, dofs: dict[int, float]) -> tuple[sp.csr_matrix, np.ndarray]:
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

    def _solve_step(self, boundary_snapshot: dict) -> None:
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
        steps = max(1, int(round(float(horizon_hours) * 3600.0 / float(self.solver.dt_seconds))))
        for _ in range(steps):
            self._solve_step(boundary_snapshot)
        self.time_hours = float(boundary_snapshot.get("time_hours", 0.0)) + float(horizon_hours)

    def compute_thaw_metrics(self) -> tuple[float | None, float | None, list[dict[str, float]]]:
        threshold_k = self._thaw_threshold_k()
        thaw_mask = self.T >= threshold_k
        if not np.any(thaw_mask):
            return None, None, []
        thaw_nodes = self.node[thaw_mask]
        points = [{"x_m": float(x), "z_m": float(z)} for x, z in thaw_nodes]
        radii = [abs(float(x)) for x, _ in thaw_nodes]
        return max(radii), sum(radii) / len(radii), points

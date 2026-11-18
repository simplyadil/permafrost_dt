"""Check the interpolated thaw front against the node-based one it replaced.

``compute_thaw_metrics`` used to report the outermost *thawed node* per depth,
which quantised the front radius to the mesh spacing. It now interpolates the
threshold crossing between the last thawed node and the first frozen node
outside it.

This is a deliberate change of answer, so there is no equivalence to assert.
What must hold instead is a refinement property, and that is what is checked:

* **never smaller** -- the crossing lies outside the last thawed node, so the
  interpolated radius is greater than or equal to the node-based one;
* **within one cell** -- the crossing lies inside the next node out, so the
  difference is smaller than the mesh spacing;
* **actually continuous** -- across a marching solve the node-based radius takes
  only a handful of distinct values while the interpolated one takes many. This
  is the property the whole predictive scheme depends on, so it is measured
  rather than assumed.

Run:

    python -m software.tools.verify_front_resolution
"""

from __future__ import annotations

import argparse

import numpy as np

from software.digital_twin.simulator.fem_forecast.fem_solver import FEMMeshConfig, FEMSolver2D
from software.startup.utils.config import load_startup_config
from software.tools.run_fem_case import _material_from_dict, _resolve_mesh_dir, _solver_from_dict


def node_based_metrics(solver: FEMSolver2D):
    """The original implementation: outermost thawed node per depth."""
    threshold_k = solver._thaw_threshold_k()
    thaw_mask = solver.T >= threshold_k
    if not np.any(thaw_mask):
        return None, None, []
    r_abs = np.abs(solver.node[:, 0])
    z_coords = solver.node[:, 1]
    points = []
    for z in np.unique(z_coords[thaw_mask]):
        depth_mask = thaw_mask & (z_coords == z)
        points.append({"x_m": float(np.max(r_abs[depth_mask])), "z_m": float(z)})
    radii = np.array([p["x_m"] for p in points])
    return float(np.max(radii)), float(np.mean(radii)), points


def build(config: dict) -> FEMSolver2D:
    fem_cfg = config["fem_forecast"]
    mesh = FEMMeshConfig(
        mesh_dir=_resolve_mesh_dir(fem_cfg.get("mesh_dir")),
        node_file=str(fem_cfg.get("node_file", "DT_model_nodes.txt")),
        element_file=str(fem_cfg.get("element_file", "DT_model_elements.txt")),
        edge_sets=fem_cfg.get("edge_sets", {}),
        node_sets=fem_cfg.get("node_sets", {}),
    )
    return FEMSolver2D(mesh, _material_from_dict(fem_cfg.get("material", {})),
                       _solver_from_dict(fem_cfg.get("solver", {})))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="startup.conf")
    parser.add_argument("--steps", type=int, default=120, help="Time steps to march.")
    args = parser.parse_args()

    config = load_startup_config(args.config)
    solver = build(config)

    spacing = float(np.min(np.diff(np.unique(np.abs(solver.node[:, 0])))))
    print(f"mesh: {solver.numnode} nodes, radial spacing {spacing * 1000:.1f} mm\n")

    snapshot = {
        "timestamp": "verification",
        "time_hours": 0.0,
        "boundaries": config["boundary_condition_builder"]["boundaries"],
    }
    offset_k = float(solver.solver.temperature_offset_k)
    solver.T = np.full(solver.numnode, offset_k - 4.0)

    node_series: list[float] = []
    interp_series: list[float] = []
    worst_gap = 0.0
    violations: list[str] = []

    for step in range(args.steps):
        solver._solve_step(snapshot)
        r_node, _, _ = node_based_metrics(solver)
        r_interp, _, _ = solver.compute_thaw_metrics()
        if r_node is None or r_interp is None:
            continue
        node_series.append(r_node)
        interp_series.append(r_interp)

        if r_interp < r_node - 1e-12:
            violations.append(f"step {step}: interpolated {r_interp:.6f} < node {r_node:.6f}")
        gap = r_interp - r_node
        worst_gap = max(worst_gap, gap)
        if gap > spacing + 1e-12:
            violations.append(f"step {step}: gap {gap * 1000:.2f} mm exceeds one cell")

    if not node_series:
        raise SystemExit("No thaw front developed; nothing to compare.")

    node_levels = len(np.unique(np.round(node_series, 9)))
    interp_levels = len(np.unique(np.round(interp_series, 9)))
    print(f"over {len(node_series)} steps with a front:")
    print(f"  node-based   : {node_levels:4d} distinct radii   "
          f"({node_series[0] * 1000:.1f} .. {node_series[-1] * 1000:.1f} mm)")
    print(f"  interpolated : {interp_levels:4d} distinct radii   "
          f"({interp_series[0] * 1000:.1f} .. {interp_series[-1] * 1000:.1f} mm)")
    print(f"  worst offset : {worst_gap * 1000:.2f} mm  (one cell = {spacing * 1000:.1f} mm)")

    print()
    if violations:
        for v in violations[:10]:
            print(f"FAIL  {v}")
        raise SystemExit(1)
    print("PASS  interpolated front is never inside the node-based one and never")
    print(f"      more than one cell outside it, and resolves {interp_levels / max(node_levels, 1):.0f}x "
          f"more distinct radii.")


if __name__ == "__main__":
    main()

"""Check that the batched element assembly matches the loop it replaced.

``FEMSolver2D._element_matrices`` was rewritten from a per-element Python loop
into a batched array computation. The rewrite is only defensible if it computes
the same numbers, so the original loop is kept in
``software/tools/golden_reference.py`` and compared against the live solver here.

The services themselves have a single assembly path with no switch; this script
reaches the old one by patching it in for the duration of a call, which is why
the reference lives in a test module rather than in the solver.

Sameness is checked over:

* the assembled system (A, b, Dirichlet map) at several temperature fields,
  including adversarial ones that straddle the phase-change band where the
  material properties vary fastest;
* a full multi-step march, which is where any per-step discrepancy would
  accumulate.

Run:

    python -m software.tools.verify_assembly_equivalence
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from software.digital_twin.simulator.fem_forecast.fem_solver import FEMMeshConfig, FEMSolver2D
from software.startup.utils.config import load_startup_config
from software.tools.golden_reference import reference_assembly
from software.tools.run_fem_case import _material_from_dict, _resolve_mesh_dir, _solver_from_dict

# Round-off floor. Both paths sum the same terms in different orders, so exact
# bit equality is not expected; anything above this is a real divergence.
TOLERANCE = 1e-11


def _build(config: dict) -> FEMSolver2D:
    fem_cfg = config["fem_forecast"]
    mesh = FEMMeshConfig(
        mesh_dir=_resolve_mesh_dir(fem_cfg.get("mesh_dir")),
        node_file=str(fem_cfg.get("node_file", "DT_model_nodes.txt")),
        element_file=str(fem_cfg.get("element_file", "DT_model_elements.txt")),
        edge_sets=fem_cfg.get("edge_sets", {}),
        node_sets=fem_cfg.get("node_sets", {}),
    )
    return FEMSolver2D(
        mesh,
        _material_from_dict(fem_cfg.get("material", {})),
        _solver_from_dict(fem_cfg.get("solver", {})),
    )


def _fields(numnode: int, offset_k: float, seed: int) -> dict[str, np.ndarray]:
    """Temperature fields to assemble at, in solver (Kelvin) units."""
    rng = np.random.default_rng(seed)
    return {
        "uniform frozen (-5 C)": np.full(numnode, offset_k - 5.0),
        "uniform thawed (+20 C)": np.full(numnode, offset_k + 20.0),
        "random -6..25 C": offset_k + rng.uniform(-6.0, 25.0, numnode),
        "astride 0 C (+-0.05 C)": offset_k + rng.uniform(-0.05, 0.05, numnode),
        "wide -40..60 C (clip)": offset_k + rng.uniform(-40.0, 60.0, numnode),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="startup.conf")
    parser.add_argument("--steps", type=int, default=20, help="Time steps for the march check.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_startup_config(args.config)
    solver = _build(config)
    print(f"mesh: {solver.numnode} nodes, {solver.numelem} elements\n")

    snapshot = {
        "timestamp": "verification",
        "time_hours": 0.0,
        "boundaries": config["boundary_condition_builder"]["boundaries"],
    }
    offset_k = float(solver.solver.temperature_offset_k)

    worst = 0.0
    print(f"{'temperature field':<26} {'rel dA':>12} {'rel db':>12} {'nnz':>7}  dirichlet")
    print("-" * 74)
    for label, field in _fields(solver.numnode, offset_k, args.seed).items():
        # Same solver, same state: only the assembly path differs between the two.
        solver.T = field.copy()
        with reference_assembly():
            A0, b0, d0 = solver._assemble_system(snapshot)
        solver.T = field.copy()
        A1, b1, d1 = solver._assemble_system(snapshot)

        scale_a = abs(A0).max() or 1.0
        scale_b = np.abs(b0).max() or 1.0
        rel_a = abs(A0 - A1).max() / scale_a
        rel_b = np.abs(b0 - b1).max() / scale_b
        worst = max(worst, rel_a, rel_b)
        same_nnz = A0.nnz == A1.nnz
        print(
            f"{label:<26} {rel_a:12.3e} {rel_b:12.3e} {A0.nnz:7d}  "
            f"{'same' if d0 == d1 else 'DIFFER'}{'' if same_nnz else '  NNZ DIFFER'}"
        )

    # ---- multi-step: any per-step drift would compound here ----------------
    print()
    ref = _build(config)
    new = _build(config)
    seeded = offset_k + np.linspace(-5.0, 25.0, ref.numnode)
    ref.T = seeded.copy()
    new.T = seeded.copy()

    t0 = time.perf_counter()
    with reference_assembly():
        for _ in range(args.steps):
            ref._solve_step(snapshot)
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(args.steps):
        new._solve_step(snapshot)
    t_new = time.perf_counter() - t0

    drift = np.abs(ref.T - new.T).max()
    span = float(np.ptp(ref.T)) or 1.0
    worst = max(worst, drift / span)
    print(f"after {args.steps} steps: max |dT| = {drift:.3e} K over a {span:.1f} K field range")
    print(f"  original loop  {t_ref:7.3f} s   ({t_ref / args.steps * 1000:6.1f} ms/step)")
    print(f"  current code   {t_new:7.3f} s   ({t_new / args.steps * 1000:6.1f} ms/step)   {t_ref / t_new:.1f}x")

    print()
    if worst < TOLERANCE:
        print(f"PASS  worst relative difference {worst:.3e} < {TOLERANCE:.0e} (round-off)")
    else:
        print(f"FAIL  worst relative difference {worst:.3e} >= {TOLERANCE:.0e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

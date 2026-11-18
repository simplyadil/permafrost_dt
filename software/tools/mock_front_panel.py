"""Single-panel field in the ``frames_current`` style, built from MOCK data.

The temperature field is synthetic: its 0 C contour is placed on the
coupled-model front traced from ``fig_end_fields.pdf`` (panel b), and the
dashed line is the measured front traced from the same figure (panel a).
Nothing here comes from the FEM solver, so the output is titled as mock data.

Usage::

    python -m software.tools.mock_front_panel
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from software.tools.optimization_ab_replay import _draw_field

DOMAIN_DEPTH_MM = 300.0

# Traced from fig_end_fields.pdf as (height cm, distance cm); height 0 is the bottom.
MODEL_FRONT_CM = [(0.0, 4.0), (13.0, 4.1), (26.0, 4.2), (28.9, 4.46), (30.0, 5.0)]
MEASURED_FRONT_CM = [
    (0.0, 2.8), (1.5, 3.3), (3.1, 3.5), (3.9, 3.3), (5.2, 2.8), (5.9, 3.3), (7.0, 3.6),
    (8.1, 3.3), (9.2, 2.95), (10.7, 3.7), (12.4, 4.1), (14.2, 4.2), (15.9, 3.9), (17.0, 3.2),
    (17.9, 3.0), (19.0, 3.8), (20.1, 3.7), (21.1, 3.1), (22.0, 3.9), (23.3, 4.4), (27.0, 4.6),
    (30.0, 3.82),
]

# Thermocouple layout, same as the replay frames (and the PDF's dots).
SENSOR_RADII_MM = [0, 15, 30, 45, 60, 75, 90, 105, 135, 165]
SENSOR_DEPTHS_MM = [10, 50, 90, 130, 170, 210, 250, 290]


def _to_depth_m(points_cm):
    pts = np.array(points_cm, dtype=float)
    depth_m = (DOMAIN_DEPTH_MM - pts[:, 0] * 10.0) / 1000.0
    radius_m = pts[:, 1] / 100.0
    return radius_m, depth_m


def mock_field(R, Z, *, wall_c=7.0):
    """Field whose 0 C isotherm sits exactly on the traced model front."""
    radius_m, depth_m = _to_depth_m(MODEL_FRONT_CM)
    order = np.argsort(depth_m)
    r_front = PchipInterpolator(depth_m[order], radius_m[order])(Z)

    # Warm side: wall temperature decaying to 0 C at the front.
    inside = wall_c * np.clip(1.0 - R / r_front, 0.0, None) ** 1.4
    # Frozen side: colder with depth, like the FEM far field.
    t_far = -2.0 - 2.5 * (Z / (DOMAIN_DEPTH_MM / 1000.0)) ** 1.5
    outside = t_far * (1.0 - np.exp(-(R - r_front) / 0.05))
    return np.where(R < r_front, inside, outside)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="artifacts/optimization_ab_current_panel/mock_current_panel.png")
    parser.add_argument("--vmin", type=float, default=-4.0)
    parser.add_argument("--vmax", type=float, default=7.0)
    parser.add_argument("--grid-resolution", type=int, default=160)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    r = np.linspace(0.0, 0.12, args.grid_resolution)
    z = np.linspace(0.0, DOMAIN_DEPTH_MM / 1000.0, args.grid_resolution)
    R, Z = np.meshgrid(r, z)
    T = mock_field(R, Z)

    measured = _to_depth_m(MEASURED_FRONT_CM)
    sr, sz = np.meshgrid(np.array(SENSOR_RADII_MM) / 1000.0, np.array(SENSOR_DEPTHS_MM) / 1000.0)

    fig = plt.figure(figsize=(5.6, 5.2), dpi=150)
    ax = fig.subplots()
    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad(color=(0.85, 0.85, 0.85, 1.0))
    mesh = _draw_field(ax, R, Z, T, title="Illustrative field (mock data)",
                       vmin=args.vmin, vmax=args.vmax, cmap=cmap,
                       front=measured, markers=(sr.ravel(), sz.ravel()))
    ax.set_ylabel("Depth (mm)")
    ax.set_title("")
    ax.set_xlim(0.0, 120.0)
    ax.set_xticks([0, 30, 60, 90, 120])
    ax.set_ylim(DOMAIN_DEPTH_MM, 0.0)
    fig.colorbar(mesh, ax=ax, pad=0.03).set_label("Temperature (C)")
    handles = [
        Line2D([], [], color="black", linewidth=1.8, label="model front (0 C)"),
        Line2D([], [], color="magenta", linestyle="--", linewidth=1.8, label="measured front"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.85)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"Wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

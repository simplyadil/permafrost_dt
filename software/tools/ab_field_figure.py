"""Publication-quality two-panel (a) measured / (b) FEM temperature field figure.

This is the paper version of a single ``fem_debug_replay`` frame. The replay
frames are diagnostics -- titles, axes, RMSE banner, grey out-of-hull mask --
which is what you want while debugging and not what you want in a manuscript.
Here the same two fields are drawn bare: a black domain frame, the field's own
0 C thaw front, an ``a`` / ``b`` label under each panel, and one shared
colourbar labelled ``Temperature [degC]``.

Differences from the replay frames, all deliberate:

* **Palette** is ``jet``, the same colour map the replay frames use, rather
  than the diverging map of the original figure.
* **Resolution** of the interpolation grid defaults to 600 (replay: 160), so
  the front and the near-pipe gradient are resolved rather than pixelated.
* **Crop**: both panels are cut to the region where *both* fields are defined
  -- the thermocouple footprint -- so neither panel carries a masked band. Pass
  ``--full-domain`` to draw the whole mesh instead and accept the mask.

The physics path is untouched: the boundary condition builder, the pipe
boundary calibration and the FEM forecast server are the real services with
RabbitMQ and InfluxDB stubbed, exactly as in ``fem_debug_replay``, replayed
from t=0 up to ``--time-hours`` because the field is a transient.

Two styles are written by default:

* ``ab_field_bands.png`` -- discrete contour bands (``--band-step``, default
  0.75 C), the banded look of the reference figure
* ``ab_field_smooth.png`` -- continuous shading of the same field

Typical use::

    python -m software.tools.ab_field_figure --time-hours 5.233

The replay is the only expensive step, so its raw output -- the thermocouple
readings and the FEM nodal field at that instant -- is cached to
``ab_field_data.npz``. ``--from-npz`` re-grids and re-renders from it in a
second, which is how ``--interp``, ``--grid-resolution``, ``--band-step`` and
the colour range are meant to be explored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from software.startup.utils.config import load_startup_config
from software.tools.fem_debug_replay import (
    DEFAULT_FLUID_PATH,
    DEFAULT_SENSOR_PATH,
    _build_servers,
    _front_from_grid,
    _grid_from_mesh,
    _interp_to_grid,
    _parse_material_override,
    _sensor_rows,
    _solver_field_c,
)
from software.tools.replay_csv import _load_fluid_temps


# ---------------------------------------------------------------------------
# FIELD COMPUTATION
# ---------------------------------------------------------------------------
def compute_fields(
    *,
    time_hours: float,
    **kwargs,
) -> dict:
    """Replay to ``time_hours`` and return both fields as scattered points.

    Thin wrapper over :func:`compute_snapshots` for the single-instant case.
    """
    return compute_snapshots(times_hours=[float(time_hours)], **kwargs)[0]


def compute_snapshots(
    *,
    path: Path,
    config_path: str | Path,
    fluid_path: Path | None,
    fluid_inlet_col: str,
    fluid_outlet_col: str,
    fluid_time_col: str,
    times_hours: list[float],
    time_column: str | None,
    start_row: int,
    stride: int,
    material_override: dict[str, float] | None,
    use_pipe_boundary: bool,
    horizon_hours: float | None,
    exclude_sensors: set[str] | None,
) -> list[dict]:
    """Replay once and capture both fields as scattered points at each time.

    The replay runs from the first row every time: the FEM field at 5 h is the
    accumulated result of every boundary update before it, so there is no way
    to jump straight to a step. Several capture times therefore cost one pass,
    not one pass each -- which is what makes a five-panel figure affordable.
    Gridding is deliberately left to ``grid_fields`` so that this -- the only
    expensive part -- is cached.
    """
    config = load_startup_config(config_path)
    snapshots = _sensor_rows(path, time_column, start_row, None, stride, exclude_sensors)
    if not snapshots:
        raise ValueError(f"No usable sensor rows parsed from {path}")

    times = np.array([float(s["time_hours"]) for s in snapshots])
    wanted = [int(np.argmin(np.abs(times - float(t)))) for t in times_hours]
    target_index = max(wanted)
    print(
        f"Loaded {len(snapshots)} snapshots from {path.name} "
        f"(t = {times[0]:.3f} .. {times[-1]:.3f} h)"
    )
    for requested, index in zip(times_hours, wanted):
        print(
            f"Target t = {float(requested):.3f} h -> nearest snapshot {index} "
            f"at t = {times[index]:.3f} h"
        )

    fluid = None
    if fluid_path is not None and use_pipe_boundary:
        fluid = _load_fluid_temps(
            fluid_path,
            inlet_col=fluid_inlet_col,
            outlet_col=fluid_outlet_col,
            time_col=fluid_time_col,
        )
        print(f"Loaded {len(fluid[0])} fluid samples from {fluid_path.name}")

    builder, pipe, forecast = _build_servers(
        config, run_baseline=False, horizon_hours=horizon_hours
    )
    if not use_pipe_boundary:
        pipe = None
    print(
        "Pipe boundary stage: "
        + (
            "enabled"
            if pipe is not None and fluid is not None
            else "disabled (configured boundaries used as-is)"
        )
    )

    captured: dict[int, dict] = {}
    for index in range(target_index + 1):
        sensor_msg = snapshots[index]
        t_h = float(sensor_msg["time_hours"])

        boundary_snapshot = builder._build_snapshot(sensor_msg)

        if pipe is not None and fluid is not None:
            t_arr, tin_arr, tout_arr = fluid
            pipe._on_fluid(
                {
                    "time_hours": t_h,
                    "timestamp": sensor_msg["timestamp"],
                    "t_inlet_c": float(np.interp(t_h, t_arr, tin_arr)),
                    "t_outlet_c": float(np.interp(t_h, t_arr, tout_arr)),
                    "site_id": "default",
                }
            )
            pipe._on_boundary(boundary_snapshot)
            enriched = pipe.mq_boundary_out.last
            if isinstance(enriched, dict):
                boundary_snapshot = enriched

        if material_override:
            boundary_snapshot["material_override"] = dict(material_override)

        forecast._process_message(boundary_snapshot)
        if forecast._solver is None:
            raise RuntimeError("FEM solver was never built; check the forecast config.")
        if index % 25 == 0 or index == target_index:
            print(f"  step {index + 1}/{target_index + 1}  t={t_h:7.3f} h")

        if index in wanted:
            node_xy, temps_c = _solver_field_c(forecast._solver)
            R_mesh, Z_mesh = _grid_from_mesh(forecast._solver.node, 2)
            captured[index] = {
                "obs_x": np.array([s["x_m"] for s in sensor_msg["sensors"]], dtype=float),
                "obs_z": np.array([s["z_m"] for s in sensor_msg["sensors"]], dtype=float),
                "obs_t": np.array([s["temperature"] for s in sensor_msg["sensors"]], dtype=float),
                "node_xy": node_xy,
                "node_t": temps_c,
                "footprint": np.array(
                    [R_mesh.min(), R_mesh.max(), Z_mesh.min(), Z_mesh.max()], dtype=float
                ),
                "time_hours": t_h,
            }
            print(f"    captured t={t_h:.3f} h")

    return [captured[index] for index in wanted]


def grid_fields(raw: dict, *, resolution: int, method: str) -> dict:
    """Sample the cached scattered fields onto one plot grid.

    Split out from ``compute_fields`` so that the 600-step replay is paid once
    and every restyle -- a different resolution, ``--interp cubic``, a wider
    crop -- runs off the cached ``.npz`` in a second.

    ``method`` is applied to *both* fields. Cubic (Clough-Tocher) is the
    default because the measured field has only ~80 thermocouples and linear
    interpolation over their Delaunay triangulation makes the isotherms visibly
    polygonal; the FEM field is dense enough that the two methods agree to well
    under the band width, but running them identically keeps the two fronts
    comparable, which is the whole point of the figure.
    """
    r_min, r_max, z_min, z_max = (float(v) for v in raw["footprint"])
    R, Z = np.meshgrid(
        np.linspace(r_min, r_max, resolution), np.linspace(z_min, z_max, resolution)
    )

    obs_x, obs_z, obs_t = raw["obs_x"], raw["obs_z"], raw["obs_t"]
    node_xy, node_t = raw["node_xy"], raw["node_t"]

    T_obs = _interp_to_grid(obs_x, obs_z, obs_t, R, Z, method=method)
    T_fem = _interp_to_grid(node_xy[:, 0], node_xy[:, 1], node_t, R, Z, method=method)

    # Cubic can overshoot near the hull edge; clamp the measured panel to the
    # range the thermocouples actually recorded so the reconstruction never
    # invents a temperature no sensor saw.
    if method == "cubic":
        finite_obs = obs_t[np.isfinite(obs_t)]
        if finite_obs.size:
            T_obs = np.clip(T_obs, finite_obs.min(), finite_obs.max())

    # Error at the sensor positions only (what the dashboard's % error sees).
    fem_at_sensors = _interp_to_grid(
        node_xy[:, 0], node_xy[:, 1], node_t, np.abs(obs_x), obs_z, method="linear"
    )
    residual = fem_at_sensors - obs_t
    finite = np.isfinite(residual)
    rmse_c = float(np.sqrt(np.mean(np.square(residual[finite])))) if np.any(finite) else float("nan")

    obs_rmax, _, _, _ = _front_from_grid(R, Z, T_obs)
    fem_rmax, _, _, _ = _front_from_grid(R, Z, T_fem)

    return {
        "R": R,
        "Z": Z,
        "T_obs": T_obs,
        "T_fem": T_fem,
        "obs_x": obs_x,
        "obs_z": obs_z,
        "time_hours": float(raw["time_hours"]),
        "rmse_c": rmse_c,
        "r_max_measured_m": np.nan if obs_rmax is None else obs_rmax,
        "r_max_fem_m": np.nan if fem_rmax is None else fem_rmax,
    }


def crop_to_common_support(fields: dict) -> dict:
    """Trim the grid to rows/columns where both fields are fully defined.

    The measured field only exists inside the thermocouple hull, so on the full
    mesh footprint panel (a) carries a blank band that panel (b) does not. A
    paper figure wants two panels of the same region, so the grid is cut to the
    largest rectangle in which neither field has a hole.
    """
    R, Z = fields["R"], fields["Z"]
    valid = np.isfinite(fields["T_obs"]) & np.isfinite(fields["T_fem"])
    rows = np.flatnonzero(valid.all(axis=1))
    cols = np.flatnonzero(valid.all(axis=0))
    if rows.size < 2 or cols.size < 2:
        # Nothing rectangular survives; fall back to the extent of the valid
        # region and let whatever holes remain plot as blank.
        rows = np.flatnonzero(valid.any(axis=1))
        cols = np.flatnonzero(valid.any(axis=0))
        if rows.size < 2 or cols.size < 2:
            return fields
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    out = dict(fields)
    out["R"] = R[r0:r1, c0:c1]
    out["Z"] = Z[r0:r1, c0:c1]
    out["T_obs"] = fields["T_obs"][r0:r1, c0:c1]
    out["T_fem"] = fields["T_fem"][r0:r1, c0:c1]
    return out


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------
def _panel(ax, R_mm, Z_mm, T, *, levels, cmap, norm, extend, label, front_lw, markers, marker_size):
    """One bare field panel with its own 0 C front and a letter underneath."""
    masked = np.ma.masked_invalid(T)
    art = ax.contourf(R_mm, Z_mm, masked, levels=levels, cmap=cmap, norm=norm, extend=extend)
    # contourf draws each band as its own polygon; without this the band edges
    # show as pale seams in vector output and in any raster downscale. On
    # matplotlib >= 3.8 the ContourSet is one Collection and ``.collections``
    # is gone, so set it on whichever object this version exposes.
    for artist in getattr(art, "collections", [art]):
        artist.set_edgecolor("face")
        artist.set_linewidth(0.0)

    if np.any(np.isfinite(T)):
        ax.contour(
            R_mm,
            Z_mm,
            np.nan_to_num(T, nan=-999.0),
            levels=[0.0],
            colors="black",
            linewidths=front_lw,
        )
    if markers is not None and markers[0].size:
        ax.scatter(
            np.abs(markers[0]) * 1000.0,
            markers[1] * 1000.0,
            s=marker_size,
            c="black",
            alpha=0.35,
            linewidths=0,
        )

    ax.set_xlim(float(R_mm.min()), float(R_mm.max()))
    ax.set_ylim(float(Z_mm.max()), float(Z_mm.min()))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")
    ax.set_xlabel(label, fontsize=13, labelpad=8)
    return art


def render_figure(
    fields: dict,
    out_path: Path,
    *,
    style: str,
    vmin: float,
    vmax: float,
    band_step: float,
    cmap_name: str,
    show_markers: bool,
    marker_size: float,
    front_lw: float,
    equal_aspect: bool,
    dpi: int,
    also_pdf: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, Normalize

    R_mm = fields["R"] * 1000.0
    Z_mm = fields["Z"] * 1000.0
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="white")

    if style == "bands":
        levels = np.arange(vmin, vmax + 0.5 * band_step, band_step)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)
        ticks = levels
    else:
        levels = np.linspace(vmin, vmax, 256)
        norm = Normalize(vmin=vmin, vmax=vmax)
        ticks = np.arange(vmin, vmax + 0.5 * band_step, band_step)

    # Only wear the colourbar's arrow caps when something actually falls off
    # the scale; with an auto range nothing does, and empty arrows just read as
    # clutter in a paper figure.
    finite = np.concatenate([fields["T_obs"].ravel(), fields["T_fem"].ravel()])
    finite = finite[np.isfinite(finite)]
    below = bool(finite.min() < vmin)
    above = bool(finite.max() > vmax)
    extend = ("both" if above else "min") if below else ("max" if above else "neither")

    height = 5.6
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(height * 1.5, height),
        dpi=dpi,
        sharey=True,
        gridspec_kw={"wspace": 0.10},
    )
    markers = (fields["obs_x"], fields["obs_z"]) if show_markers else None

    art = None
    for ax, key, label in ((axes[0], "T_obs", "a"), (axes[1], "T_fem", "b")):
        art = _panel(
            ax,
            R_mm,
            Z_mm,
            fields[key],
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend=extend,
            label=label,
            front_lw=front_lw,
            markers=markers,
            marker_size=marker_size,
        )
        if equal_aspect:
            ax.set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(art, ax=axes, fraction=0.055, pad=0.04, ticks=ticks, drawedges=False)
    cbar.set_label("Temperature [$^\\circ$C]", fontsize=13, labelpad=10)
    cbar.ax.tick_params(labelsize=10, length=3, width=0.8)
    cbar.outline.set_linewidth(0.8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out_path}")
    if also_pdf:
        pdf_path = out_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        print(f"Wrote {pdf_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", default=DEFAULT_SENSOR_PATH, help="Sensor replay file.")
    parser.add_argument("--fluid-path", default=DEFAULT_FLUID_PATH, help="Fluid inlet/outlet file.")
    parser.add_argument("--fluid-inlet-col", default="Fluid Temperature_Inlet")
    parser.add_argument("--fluid-outlet-col", default="Fluid Temperatue_Outlet")
    parser.add_argument("--fluid-time-col", default="Time (h)")
    parser.add_argument("--config", default="startup.conf", help="Startup config for the FEM/BC setup.")
    parser.add_argument("--out", default="artifacts/paper/ab_field", help="Output directory.")
    parser.add_argument("--time-hours", type=float, default=5.233, help="Snapshot time to draw (nearest row wins).")
    parser.add_argument("--time-column", default=None, help="Override the time column header.")
    parser.add_argument("--start-row", type=int, default=1, help="First data row (1-based).")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth row (thins the replay).")
    parser.add_argument("--grid-resolution", type=int, default=600, help="Interpolation grid resolution.")
    parser.add_argument(
        "--interp",
        choices=("cubic", "linear"),
        default="cubic",
        help="Scattered-data reconstruction for both panels. Cubic (default) keeps the "
        "sparse thermocouple field from going polygonal; linear matches the replay frames.",
    )
    parser.add_argument("--horizon-hours", type=float, default=None, help="Override fem_forecast.horizon_hours.")
    parser.add_argument("--material-override", default=None, help="Inverted parameters, e.g. 'cap_s=1200,lambda_s=5'.")
    parser.add_argument("--no-pipe-boundary", action="store_true", help="Skip the pipe calibration stage.")
    parser.add_argument("--exclude-sensors", default=None, help="Comma-separated channels to drop.")

    figure = parser.add_argument_group("figure")
    figure.add_argument(
        "--style",
        choices=("bands", "smooth", "both"),
        default="both",
        help="Discrete contour bands, continuous shading, or one file of each.",
    )
    figure.add_argument("--cmap", default="jet", help="Colour map (default: jet, as in the replay frames).")
    figure.add_argument("--tmin", type=float, default=None, help="Colour scale minimum (C); default from the data.")
    figure.add_argument("--tmax", type=float, default=None, help="Colour scale maximum (C); default from the data.")
    figure.add_argument("--band-step", type=float, default=0.75, help="Band width / colourbar tick step (C).")
    figure.add_argument(
        "--clip-percentile",
        type=float,
        default=0.5,
        help="Percentile trimmed from each end when the colour range is chosen automatically. "
        "The FEM field spikes at the pipe wall, and on the raw min/max the whole soil block "
        "lands in one or two bands. 0 uses the exact min/max.",
    )
    figure.add_argument("--front-linewidth", type=float, default=1.4, help="0 C thaw front line width.")
    figure.add_argument("--markers", action="store_true", help="Overlay the thermocouple positions.")
    figure.add_argument("--marker-size", type=float, default=5.0, help="Thermocouple marker size.")
    figure.add_argument(
        "--full-domain",
        action="store_true",
        help="Draw the whole mesh footprint instead of cropping to where both fields exist.",
    )
    figure.add_argument(
        "--no-equal-aspect",
        action="store_true",
        help="Let the panels stretch to the axes box instead of holding the true r:z ratio.",
    )
    figure.add_argument("--dpi", type=int, default=400, help="Raster resolution.")
    figure.add_argument("--no-pdf", action="store_true", help="Skip the vector copy.")

    cache = parser.add_argument_group("field cache")
    cache.add_argument(
        "--from-npz",
        default=None,
        help="Re-render from a previous run's ab_field_data.npz instead of replaying.",
    )

    args = parser.parse_args()
    out_dir = Path(args.out)
    npz_path = out_dir / "ab_field_data.npz"

    if args.from_npz:
        with np.load(args.from_npz) as handle:
            raw = {key: handle[key] for key in handle.files}
        print(f"Loaded cached fields from {args.from_npz}")
    else:
        raw = compute_fields(
            path=Path(args.path),
            config_path=args.config,
            fluid_path=Path(args.fluid_path) if args.fluid_path else None,
            fluid_inlet_col=args.fluid_inlet_col,
            fluid_outlet_col=args.fluid_outlet_col,
            fluid_time_col=args.fluid_time_col,
            time_hours=float(args.time_hours),
            time_column=args.time_column,
            start_row=args.start_row,
            stride=max(int(args.stride), 1),
            material_override=_parse_material_override(args.material_override),
            use_pipe_boundary=not args.no_pipe_boundary,
            horizon_hours=args.horizon_hours,
            exclude_sensors=set(args.exclude_sensors.split(",")) if args.exclude_sensors else None,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(npz_path, **raw)
        print(f"Wrote {npz_path}")

    fields = grid_fields(
        raw,
        resolution=max(int(args.grid_resolution), 40),
        method=args.interp,
    )
    print(f"Gridded {args.grid_resolution}x{args.grid_resolution}, {args.interp} reconstruction")

    if not args.full_domain:
        fields = crop_to_common_support(fields)

    both = np.concatenate([fields["T_obs"].ravel(), fields["T_fem"].ravel()])
    both = both[np.isfinite(both)]
    step = float(args.band_step)
    pct = min(max(float(args.clip_percentile), 0.0), 49.0)
    lo, hi = np.percentile(both, [pct, 100.0 - pct])
    vmin = float(args.tmin) if args.tmin is not None else float(np.floor(lo / step) * step)
    vmax = float(args.tmax) if args.tmax is not None else float(np.ceil(hi / step) * step)

    print(
        f"t = {fields['time_hours']:.3f} h | sensor RMSE {fields['rmse_c']:.2f} C | "
        f"r_max measured {float(fields['r_max_measured_m']) * 1000:.0f} mm vs "
        f"FEM {float(fields['r_max_fem_m']) * 1000:.0f} mm"
    )
    print(f"Colour scale: {vmin:.2f} .. {vmax:.2f} C in {step:g} C bands")
    print(
        f"Panel extent: r 0..{fields['R'].max() * 1000:.0f} mm, "
        f"z {fields['Z'].min() * 1000:.0f}..{fields['Z'].max() * 1000:.0f} mm"
    )

    styles = ("bands", "smooth") if args.style == "both" else (args.style,)
    for style in styles:
        render_figure(
            fields,
            out_dir / f"ab_field_{style}.png",
            style=style,
            vmin=vmin,
            vmax=vmax,
            band_step=step,
            cmap_name=args.cmap,
            show_markers=bool(args.markers),
            marker_size=float(args.marker_size),
            front_lw=float(args.front_linewidth),
            equal_aspect=not args.no_equal_aspect,
            dpi=int(args.dpi),
            also_pdf=not args.no_pdf,
        )


if __name__ == "__main__":
    main()

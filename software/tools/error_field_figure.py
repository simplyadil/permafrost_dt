"""Publication-quality panel row of the absolute FEM-vs-measured error field.

The companion to ``ab_field_figure``: where that draws the two temperature
fields side by side at one instant, this draws ``|T_FEM - T_measured|`` over
the thermocouple footprint at several instants, on one shared error scale, so
the spatial *and* temporal growth of the model discrepancy is readable in a
single figure.

Defaults reproduce the reference figure's scale exactly -- 0 to 1.4 C in 0.1 C
bands, five panels at 1, 2, 3, 4 and 5 h, labelled ``a`` .. ``e`` -- but the
greyscale is replaced by ``YlOrRd``: near-white at zero error through orange to
deep red at the top of the scale. Greyscale spends most of its range in
mid-tones the eye cannot rank, so a 0.4 C error and a 0.8 C error look far more
alike than they are. ``YlOrRd`` keeps the same light-is-small reading
direction, is monotonic in lightness (so it survives a monochrome print and
red-green colour blindness), and reads as an error magnitude without a legend.

It is also chosen for *this* data specifically. The error here runs well past
the reference's 1.4 C ceiling, so a large part of each panel sits in the top
band; a map that ends in near-black (``magma_r``, ``inferno_r``) turns that
region into dead ink and hides the structure just below the ceiling, while
``YlOrRd`` ends in a deep red that still reads as data.

The scale is *fixed*, not fitted: if the error runs past ``--error-max`` the
colourbar grows an arrow cap and the tool prints how much of each panel
saturates, rather than silently rescaling and making a worse fit look better.

The physics path is untouched -- the real boundary condition builder, pipe
boundary calibration and FEM forecast server with RabbitMQ and InfluxDB
stubbed, replayed from t=0 -- and all five capture times come out of one pass.

Typical use::

    python -m software.tools.error_field_figure

Outputs, under ``--out`` (default ``artifacts/paper/error_field``):

* ``error_field_bands.png`` / ``.pdf`` -- discrete 0.1 C bands, as the reference
* ``error_field_smooth.png`` / ``.pdf`` -- continuous shading of the same field
* ``error_field_data.npz`` -- the raw fields at every capture time, so
  ``--from-npz`` restyles in a second without replaying
* ``error_field_stats.csv`` -- per-panel mean / median / p95 / max error
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from software.tools.ab_field_figure import compute_snapshots
from software.tools.fem_debug_replay import (
    DEFAULT_FLUID_PATH,
    DEFAULT_SENSOR_PATH,
    _interp_to_grid,
    _parse_material_override,
)

PANEL_LETTERS = "abcdefghijklmnopqrstuvwxyz"


# ---------------------------------------------------------------------------
# FIELD COMPUTATION
# ---------------------------------------------------------------------------
def error_field(raw: dict, *, resolution: int, method: str) -> dict:
    """|FEM - measured| on one plot grid, plus the two fields it came from.

    Both fields are sampled by the same interpolator onto the same grid before
    subtracting. Differencing two reconstructions built by different rules
    would put the reconstruction difference into the error map, which is
    exactly the quantity the figure is supposed to exclude.
    """
    r_min, r_max, z_min, z_max = (float(v) for v in raw["footprint"])
    R, Z = np.meshgrid(
        np.linspace(r_min, r_max, resolution), np.linspace(z_min, z_max, resolution)
    )

    obs_x, obs_z, obs_t = raw["obs_x"], raw["obs_z"], raw["obs_t"]
    node_xy, node_t = raw["node_xy"], raw["node_t"]

    T_obs = _interp_to_grid(obs_x, obs_z, obs_t, R, Z, method=method)
    if method == "cubic":
        finite_obs = obs_t[np.isfinite(obs_t)]
        if finite_obs.size:
            T_obs = np.clip(T_obs, finite_obs.min(), finite_obs.max())
    T_fem = _interp_to_grid(node_xy[:, 0], node_xy[:, 1], node_t, R, Z, method=method)

    return {
        "R": R,
        "Z": Z,
        "error": np.abs(T_fem - T_obs),
        "time_hours": float(raw["time_hours"]),
    }


def crop_panels(panels: list[dict]) -> list[dict]:
    """Cut every panel to the same rectangle of defined error.

    The error only exists where the thermocouples constrain the measured field,
    and the panels must share one window or the eye reads a change of frame as
    a change of field.
    """
    valid = np.ones_like(panels[0]["error"], dtype=bool)
    for panel in panels:
        valid &= np.isfinite(panel["error"])
    rows = np.flatnonzero(valid.all(axis=1))
    cols = np.flatnonzero(valid.all(axis=0))
    if rows.size < 2 or cols.size < 2:
        rows = np.flatnonzero(valid.any(axis=1))
        cols = np.flatnonzero(valid.any(axis=0))
        if rows.size < 2 or cols.size < 2:
            return panels
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    out = []
    for panel in panels:
        cropped = dict(panel)
        cropped["R"] = panel["R"][r0:r1, c0:c1]
        cropped["Z"] = panel["Z"][r0:r1, c0:c1]
        cropped["error"] = panel["error"][r0:r1, c0:c1]
        out.append(cropped)
    return out


def panel_stats(panel: dict, error_max: float) -> dict:
    """What the fixed scale hides: the spread, and how much runs off the top."""
    values = panel["error"]
    values = values[np.isfinite(values)]
    if not values.size:
        return {"time_hours": panel["time_hours"]}
    return {
        "time_hours": round(panel["time_hours"], 3),
        "mean_c": float(values.mean()),
        "median_c": float(np.median(values)),
        "p95_c": float(np.percentile(values, 95)),
        "max_c": float(values.max()),
        "fraction_over_scale": float(np.count_nonzero(values > error_max) / values.size),
    }


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------
def _panel(ax, panel, *, levels, cmap, norm, extend, label, ticks_below):
    masked = np.ma.masked_invalid(panel["error"])
    art = ax.contourf(
        panel["R"] * 1000.0,
        panel["Z"] * 1000.0,
        masked,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend=extend,
    )
    # One polygon per band; without this the band edges show as pale seams.
    for artist in getattr(art, "collections", [art]):
        artist.set_edgecolor("face")
        artist.set_linewidth(0.0)

    ax.set_xlim(float(panel["R"].min()) * 1000.0, float(panel["R"].max()) * 1000.0)
    ax.set_ylim(float(panel["Z"].max()) * 1000.0, float(panel["Z"].min()) * 1000.0)
    ax.set_yticks([])
    if ticks_below:
        # Unlabelled ticks, as in the reference: they give the eye a scale bar
        # without five repeated sets of numbers across the row.
        r_max_mm = float(panel["R"].max()) * 1000.0
        ax.set_xticks(np.linspace(0.0, r_max_mm, 5))
        ax.tick_params(axis="x", length=3, width=0.8, labelbottom=False)
    else:
        ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")
    ax.set_xlabel(label, fontsize=13, labelpad=8)
    return art


def render_figure(
    panels: list[dict],
    out_path: Path,
    *,
    style: str,
    error_max: float,
    band_step: float,
    cmap_name: str,
    equal_aspect: bool,
    ticks_below: bool,
    time_labels: bool,
    dpi: int,
    also_pdf: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, Normalize

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="white")

    if style == "bands":
        levels = np.arange(0.0, error_max + 0.5 * band_step, band_step)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)
        ticks = levels
    else:
        levels = np.linspace(0.0, error_max, 256)
        norm = Normalize(vmin=0.0, vmax=error_max)
        ticks = np.arange(0.0, error_max + 0.5 * band_step, band_step)

    peak = max(
        float(np.nanmax(panel["error"])) for panel in panels if np.any(np.isfinite(panel["error"]))
    )
    extend = "max" if peak > error_max else "neither"

    # Size the figure from the domain's own r:z ratio. Holding a true aspect
    # inside a box of the wrong shape letterboxes every panel, and the shared
    # colourbar -- which is sized from the axes, not from the drawn image --
    # then towers over a row of panels that no longer fill their boxes.
    r_span = float(panels[0]["R"].max() - panels[0]["R"].min())
    z_span = float(panels[0]["Z"].max() - panels[0]["Z"].min())
    aspect = (r_span / z_span) if (equal_aspect and z_span > 0) else 0.55

    panel_h = 4.6
    panel_w = panel_h * aspect
    wspace = 0.12
    row_w = len(panels) * panel_w * (1.0 + wspace) - panel_w * wspace
    cbar_w = 0.85

    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(row_w + cbar_w + 1.5, panel_h + 0.7),
        dpi=dpi,
        sharey=True,
        gridspec_kw={"wspace": wspace},
    )
    if len(panels) == 1:
        axes = [axes]

    art = None
    for position, (ax, panel) in enumerate(zip(axes, panels)):
        label = PANEL_LETTERS[position]
        if time_labels:
            label = f"{label}   ({panel['time_hours']:.0f} h)"
        art = _panel(
            ax,
            panel,
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend=extend,
            label=label,
            ticks_below=ticks_below,
        )
        if equal_aspect:
            ax.set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(
        art,
        ax=axes,
        fraction=cbar_w / row_w,
        pad=0.6 / row_w,
        ticks=ticks,
        drawedges=False,
    )
    cbar.set_label("Absolute error [$^\\circ$C]", fontsize=13, labelpad=10)
    cbar.ax.tick_params(labelsize=9, length=3, width=0.8)
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
# CACHE
# ---------------------------------------------------------------------------
def save_raw(path: Path, raws: list[dict]) -> None:
    """Flatten the per-time raw fields into one npz."""
    flat: dict[str, np.ndarray] = {"count": np.array(len(raws))}
    for index, raw in enumerate(raws):
        for key, value in raw.items():
            flat[f"s{index}__{key}"] = np.asarray(value)
    np.savez(path, **flat)


def load_raw(path: Path) -> list[dict]:
    with np.load(path) as handle:
        count = int(handle["count"])
        return [
            {
                key.split("__", 1)[1]: handle[key]
                for key in handle.files
                if key.startswith(f"s{index}__")
            }
            for index in range(count)
        ]


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
    parser.add_argument("--out", default="artifacts/paper/error_field", help="Output directory.")
    parser.add_argument(
        "--times",
        default="1,2,3,4,5",
        help="Comma-separated capture times in hours, one panel each (default: the reference's 1..5 h).",
    )
    parser.add_argument("--time-column", default=None, help="Override the time column header.")
    parser.add_argument("--start-row", type=int, default=1, help="First data row (1-based).")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth row (thins the replay).")
    parser.add_argument("--grid-resolution", type=int, default=600, help="Interpolation grid resolution.")
    parser.add_argument(
        "--interp",
        choices=("cubic", "linear"),
        default="cubic",
        help="Scattered-data reconstruction, applied to both fields before differencing.",
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
        help="Discrete error bands, continuous shading, or one file of each.",
    )
    figure.add_argument(
        "--cmap",
        default="YlOrRd",
        help="Sequential colour map, light at zero error. Alternatives worth trying: "
        "cividis_r (best colour-blind separation), magma_r or inferno_r (darker top "
        "end), viridis (inverts: high error is the brightest), Greys (the reference's "
        "own scale).",
    )
    figure.add_argument("--error-max", type=float, default=1.4, help="Top of the error scale (C).")
    figure.add_argument("--band-step", type=float, default=0.1, help="Band width / colourbar tick step (C).")
    figure.add_argument(
        "--ticks",
        action="store_true",
        help="Draw the reference's unlabelled radius ticks under each panel.",
    )
    figure.add_argument(
        "--time-labels",
        action="store_true",
        help="Append the capture time to each panel letter instead of leaving it to the caption.",
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
        help="Re-render from a previous run's error_field_data.npz instead of replaying.",
    )

    args = parser.parse_args()
    out_dir = Path(args.out)
    npz_path = out_dir / "error_field_data.npz"
    times = [float(chunk) for chunk in args.times.split(",") if chunk.strip()]
    if not times:
        raise SystemExit("--times listed no usable capture times.")

    if args.from_npz:
        raws = load_raw(Path(args.from_npz))
        print(f"Loaded {len(raws)} cached snapshots from {args.from_npz}")
    else:
        raws = compute_snapshots(
            path=Path(args.path),
            config_path=args.config,
            fluid_path=Path(args.fluid_path) if args.fluid_path else None,
            fluid_inlet_col=args.fluid_inlet_col,
            fluid_outlet_col=args.fluid_outlet_col,
            fluid_time_col=args.fluid_time_col,
            times_hours=times,
            time_column=args.time_column,
            start_row=args.start_row,
            stride=max(int(args.stride), 1),
            material_override=_parse_material_override(args.material_override),
            use_pipe_boundary=not args.no_pipe_boundary,
            horizon_hours=args.horizon_hours,
            exclude_sensors=set(args.exclude_sensors.split(",")) if args.exclude_sensors else None,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        save_raw(npz_path, raws)
        print(f"Wrote {npz_path}")

    resolution = max(int(args.grid_resolution), 40)
    panels = [error_field(raw, resolution=resolution, method=args.interp) for raw in raws]
    panels = crop_panels(panels)
    print(f"Gridded {resolution}x{resolution}, {args.interp} reconstruction")
    print(
        f"Panel extent: r 0..{panels[0]['R'].max() * 1000:.0f} mm, "
        f"z {panels[0]['Z'].min() * 1000:.0f}..{panels[0]['Z'].max() * 1000:.0f} mm"
    )

    stats = [panel_stats(panel, float(args.error_max)) for panel in panels]
    print(f"\nScale fixed at 0 .. {args.error_max:g} C in {args.band_step:g} C bands")
    print(f"{'panel':>5} {'t (h)':>7} {'mean':>7} {'median':>7} {'p95':>7} {'max':>7} {'over scale':>11}")
    for letter, row in zip(PANEL_LETTERS, stats):
        print(
            f"{letter:>5} {row['time_hours']:>7.2f} {row['mean_c']:>7.3f} {row['median_c']:>7.3f} "
            f"{row['p95_c']:>7.3f} {row['max_c']:>7.3f} {row['fraction_over_scale'] * 100:>10.1f}%"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "error_field_stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["panel"] + list(stats[0].keys()))
        writer.writeheader()
        for letter, row in zip(PANEL_LETTERS, stats):
            writer.writerow({"panel": letter, **row})
    print(f"\nWrote {stats_path}")

    styles = ("bands", "smooth") if args.style == "both" else (args.style,)
    for style in styles:
        render_figure(
            panels,
            out_dir / f"error_field_{style}.png",
            style=style,
            error_max=float(args.error_max),
            band_step=float(args.band_step),
            cmap_name=args.cmap,
            equal_aspect=not args.no_equal_aspect,
            ticks_below=bool(args.ticks),
            time_labels=bool(args.time_labels),
            dpi=int(args.dpi),
            also_pdf=not args.no_pdf,
        )


if __name__ == "__main__":
    main()

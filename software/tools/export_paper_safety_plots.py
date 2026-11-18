"""Export the Thaw Front plot and safety gauge for the paper from the cached run.

Uses the dashboard's own plot builders, fed from the predictive run the paper
reports (not from InfluxDB, which does not hold that run). The forecast curve is
the keep-heating branch at the 10 min monitoring window. Thresholds come from
startup.conf. Nothing is written back to the cache or to InfluxDB.

Run:

    python -m software.tools.export_paper_safety_plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from software.digital_twin.visualization.viz_dashboard.viz_dashboard_app import (
    _build_safety_gauge,
    _unified_radius_plot,
)
from software.startup.utils.config import load_startup_config

MONITOR_WINDOW_H = 10.0 / 60.0


def _forecast_frame(traces: list[dict]) -> pd.DataFrame:
    """Keep-heating radius at t + 10 min, read from each update's lookahead samples."""
    records = []
    for trace in traces:
        offsets = [time - trace["t"] for time in trace["times"]]
        idx = min(range(len(offsets)), key=lambda i: abs(offsets[i] - MONITOR_WINDOW_H))
        records.append({"time_hours": trace["t"], "horizon_hours": offsets[idx],
                        "radius_max_m": trace["keep"][idx]})
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--traces", default="artifacts/predictive_control_dtwin/lookahead_traces.json")
    parser.add_argument("--out-dir", default="artifacts/paper")
    args = parser.parse_args()

    cache = json.loads(Path(args.traces).read_text(encoding="utf-8"))
    safety = load_startup_config().get("safety_monitor", {})
    limit_m = float(safety["limit_radius_m"])
    warn, critical = float(safety["warn_fraction"]), float(safety["critical_fraction"])
    release_m = float(safety["release_fraction"]) * limit_m

    thaw = pd.DataFrame(cache["rows"])[["time_hours", "r_max_measured_m", "r_min_measured_m"]]
    thaw = thaw.rename(columns={"r_max_measured_m": "radius_max_m", "r_min_measured_m": "radius_min_m"})
    thaw = thaw.dropna(subset=["radius_max_m"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    radius = _unified_radius_plot(thaw, _forecast_frame(cache["traces"]), pd.DataFrame(),
                                  limit_m, release_m, warn, critical)
    # The dashboard's "current r_max" marker describes a live run, not a finished one.
    radius.data = [trace for trace in radius.data if not trace.name.startswith("Current r_max")]
    radius.update_layout(title=None, template="plotly_white", font=dict(color="#111827"),
                         margin=dict(l=70, r=30, t=20, b=70),
                         paper_bgcolor="white", plot_bgcolor="white", width=900, height=520,
                         legend=dict(font=dict(color="#111827"), bgcolor="rgba(255,255,255,0.9)",
                                     x=0.99, y=0.02, xanchor="right", yanchor="bottom"))

    latest = thaw.iloc[-1]
    gauge = _build_safety_gauge(float(latest["radius_max_m"]), limit_m, warn, critical)
    gauge.update_traces(gauge_bgcolor="white",
                        title_text=f"Safety gauge, t = {float(latest['time_hours']):.2f} h")
    gauge.update_layout(font=dict(color="#111827"), paper_bgcolor="white", width=520, height=320)

    for name, fig in (("fig_thaw_front_paper", radius), ("fig_safety_gauge_paper", gauge)):
        for suffix, scale in ((".pdf", 1), (".png", 3)):
            path = out_dir / f"{name}{suffix}"
            fig.write_image(str(path), scale=scale)
            print(f"Saved {path}")


if __name__ == "__main__":
    main()

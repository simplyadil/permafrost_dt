"""Measurement-only safety classification over the complete test, DT front.

Same figure as the paper's measurement-only safety panel -- measured r_max and
r_min against the warning, critical and alert radii, and the status the monitor
emits -- but the front is the coupled twin's own reconstruction
(``dtwin.front.thaw_front`` on the measured field), the definition the
predictive figures of ``predictive_control_dtwin`` use. The status comes from
the real safety monitor fed the measured stream alone, so no forecast predicate
can contribute to it.

Run:

    python -m software.tools.safety_limits_dtwin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from software.startup.service_runners import build_influx_config, create_safety_monitor_server
from software.startup.utils.config import load_startup_config
from software.tools.fem_debug_replay import _NullInflux
from software.tools.predictive_control_dtwin import DEFAULT_DTWIN, _r_max, _r_min
from software.tools.predictive_control_evidence import _Recorder
from software.tools.safety_limit_calibration import STATUS_COLOUR, STATUS_ORDER


def classify(config: dict, dtwin_root: Path) -> tuple[list[dict], dict]:
    sys.path.insert(0, str(dtwin_root))
    from dtwin.front import thaw_front
    from dtwin.model_setup import dt_setup

    S = dt_setup(dtwin_root / "data")
    front = thaw_front(S, S.T_meas)[0]                            # nz x n_meas, mm

    safety_cfg = config.get("safety_monitor", {})
    monitor = create_safety_monitor_server(
        build_influx_config(config.get("influxdb", {})), config.get("rabbitmq", {}),
        dict(safety_cfg, mode="predictive"))
    monitor.mq_out, monitor.influx = _Recorder(), _NullInflux()

    rows = []
    for k in range(S.T_meas.shape[1]):
        t_h = float(S.t_meas[k]) / 3600.0
        r_max, r_min = _r_max(front[:, k]), _r_min(front[:, k])
        monitor._handle_thaw({"time_hours": t_h, "timestamp": f"{t_h:.6f}",
                              "radius_max_m": r_max, "radius_min_m": r_min})
        alert = monitor.mq_out.last or {}
        rows.append({"time_hours": t_h, "r_max_m": r_max, "r_min_m": r_min,
                     "status": alert.get("status", "ok")})
    meta = {"limit_m": monitor.limit_radius_m, "warn": monitor.warn_fraction,
            "critical": monitor.critical_fraction}
    return rows, meta


def monotone(statuses: list[str]) -> list[str]:
    """Highest rung reached so far.

    The measured front hovers on the warning and critical radii for a while and
    the raw ladder flickers on single-sample dips; the ribbon shows the regime,
    as in the predictive timeline. Alert is latched by the monitor itself.
    """
    out, top = [], 0
    for s in statuses:
        top = max(top, STATUS_ORDER.index(s))
        out.append(STATUS_ORDER[top])
    return out


def plot(rows, meta, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array([r["time_hours"] for r in rows])
    mm = lambda key: np.array([np.nan if r[key] is None else r[key] * 1000.0 for r in rows])
    rmax, rmin = mm("r_max_m"), mm("r_min_m")
    limit_mm = meta["limit_m"] * 1000.0

    fig, (ax, ax_s) = plt.subplots(2, 1, figsize=(9.5, 4.8), sharex=True,
                                   gridspec_kw={"height_ratios": [2.4, 1], "hspace": 0.12})
    ax.plot(t, rmax, color="#111111", lw=1.6, label="r_max measured")
    ax.plot(t, rmin, color="#4A6FA5", lw=1.2, label="r_min measured")
    for frac, name in ((meta["warn"], "warning"), (meta["critical"], "critical"), (1.0, "alert")):
        y = limit_mm * frac
        ax.axhline(y, color=STATUS_COLOUR[name], lw=1.0, ls="--")
        ax.text(t[-1], y, name, color=STATUS_COLOUR[name], va="center", ha="right",
                fontsize=7.5, bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.5))
    ax.set_ylabel("radius (mm)")
    ax.set_title("Measured thaw front against the safety thresholds", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)

    # Ribbon from the first reconstructed front: before it the monitor has
    # nothing to classify.
    seen = ~np.isnan(rmax)
    ts = t[seen]
    statuses = monotone([r["status"] for r, s in zip(rows, seen) if s])
    start = 0
    for i in range(1, len(statuses) + 1):
        if i == len(statuses) or statuses[i] != statuses[start]:
            t0, t1 = ts[start], ts[min(i, len(ts) - 1)]
            ax_s.axvspan(t0, t1, color=STATUS_COLOUR[statuses[start]], alpha=0.9, lw=0)
            ax_s.text((t0 + t1) / 2, 0.5, statuses[start], ha="center", va="center",
                      color="white", fontsize=8, fontweight="bold")
            print(f"  {statuses[start]:<9} {t0:5.3f} .. {t1:5.3f} h")
            start = i
    ax_s.set_yticks([])
    ax_s.set_ylabel("status")
    ax_s.set_xlabel("Experiment time (h)")
    ax_s.set_xlim(t[0], t[-1])

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dtwin-root", default=str(DEFAULT_DTWIN))
    parser.add_argument("--config", default="startup.conf")
    parser.add_argument("--out", default="artifacts/predictive_control_dtwin/safety_limits.png")
    args = parser.parse_args()

    rows, meta = classify(load_startup_config(args.config), Path(args.dtwin_root))
    raw = [r["status"] for r in rows]
    flips = sum(1 for a, b in zip(raw, raw[1:]) if a != b)
    print(f"{len(rows)} samples, raw status changes {flips}")
    plot(rows, meta, Path(args.out))


if __name__ == "__main__":
    main()

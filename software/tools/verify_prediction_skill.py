"""How far ahead can the twin be believed? Scored against what actually happened.

A time-to-limit is a claim about when a future event occurs, so it is falsifiable
the moment the event happens. This scores those claims against the recorded
outcome, reading the per-update record written by ``dt_decision_evidence`` -- no
replay, so it is cheap to re-run while tuning.

Three questions, deliberately separated because they have different answers:

1. **Is the number right?** If a predictor says "limit in N minutes" at time t,
   then ``t + N`` is a prediction of the crossing time and should come out the
   same no matter when it was made. Plotted against t it should be a flat line
   at the true crossing. Slope, drift and scatter are the error, and one known
   event is enough to measure them.

2. **How far ahead does it stay right?** The same errors bucketed by how early
   the prediction was made. This is what sets a defensible lookahead horizon --
   the largest lead at which the error is still inside what the decision can
   tolerate. Choosing the horizon from the thermal lag instead, which is what was
   done first, answers a different question: how much warning you would *like*,
   not how much the predictor can actually deliver.

3. **Is the decision right?** The one that matters, and the one with enough
   samples to be worth trusting. A control rule does not need an accurate TTL, it
   needs to fire when it should and stay quiet when it should not. Scored as a
   warning system over every update -- hits, misses, false alarms -- which turns
   a single event into hundreds of scored forecasts.

Both the median and the 90th percentile of the error are reported throughout.
The mean is not: a predictor that is usually excellent and occasionally claims
the limit is two days away has a mean dominated entirely by the tail, and hiding
that tail behind a robust statistic would be the more dangerous error. The tail
is the safety property.

**On sample size.** One experiment, one crossing. The consistency and skill
scores use every update, but they are all correlated through a single event, so
treat the numbers as characterising *this* run rather than as an estimate of
future skill. Running this across the other recordings in ``data/`` is the only
way to know whether any of it generalises.

Run:

    python -m software.tools.verify_prediction_skill
    python -m software.tools.verify_prediction_skill --tolerance-minutes 5
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from software.startup.utils.config import load_startup_config

PREDICTORS = {
    "ttb_monitored_h": ("monitored time-to-breach", "#1F6FA8"),
    "ttb_fem_h": ("FEM forecast", "#C2531F"),
}
# Lead buckets, in minutes, for the error-vs-lead breakdown.
LEAD_BUCKETS = [(0, 10), (10, 20), (20, 30), (30, 45), (45, 60), (60, 90), (90, 150)]


def _num(row: dict, key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", "None", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# GROUND TRUTH
# ---------------------------------------------------------------------------
def measured_series(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    pairs = [(_num(r, "time_hours"), _num(r, "r_max_measured_m")) for r in rows]
    pairs = [(t, r) for t, r in pairs if t is not None and r is not None]
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def first_crossing(t: np.ndarray, r: np.ndarray, limit_m: float) -> float | None:
    idx = np.where(r >= limit_m)[0]
    return float(t[idx[0]]) if idx.size else None


def breach_within(t: np.ndarray, r: np.ndarray, origin_h: float, lead_h: float, limit_m: float) -> bool:
    """Did the measured front reach the limit in (origin, origin+lead]?

    This is the event the warning is about, and it is defined at every update
    rather than only at the single crossing -- which is what makes hundreds of
    scored forecasts available instead of one.
    """
    window = (t > origin_h) & (t <= origin_h + lead_h)
    return bool(np.any(r[window] >= limit_m))


# ---------------------------------------------------------------------------
# 1 + 2: IS THE NUMBER RIGHT, AND FOR HOW LONG
# ---------------------------------------------------------------------------
def consistency(rows: list[dict], key: str, actual_cross_h: float) -> dict:
    """Predicted crossing time against the real one, bucketed by lead."""
    records = []
    for row in rows:
        t = _num(row, "time_hours")
        ttl = _num(row, key)
        if t is None or ttl is None or t >= actual_cross_h:
            continue
        records.append((t, ttl, (t + ttl - actual_cross_h) * 60.0, (actual_cross_h - t) * 60.0))
    if not records:
        return {"n": 0, "buckets": [], "t": np.array([]), "pred": np.array([])}

    err = np.array([r[2] for r in records])
    lead = np.array([r[3] for r in records])
    # One offset for the whole predictor, not per bucket: a systematic lateness
    # is a single correctable constant, and fitting one per bucket would just be
    # absorbing the scatter.
    overall_bias = float(np.median(err))
    buckets = []
    for lo, hi in LEAD_BUCKETS:
        mask = (lead >= lo) & (lead < hi)
        if not np.any(mask):
            continue
        e = np.abs(err[mask])
        buckets.append({
            "lo": lo, "hi": hi, "n": int(mask.sum()),
            "median_abs": float(np.median(e)),
            "p90_abs": float(np.percentile(e, 90)),
            "bias": float(np.median(err[mask])),
            # What would remain if that one offset were removed. Circular on a
            # single event -- the offset is measured on the same crossing it is
            # then applied to -- so it is reported as a property of this run, not
            # as a correction that can be claimed to generalise.
            "median_abs_debiased": float(np.median(np.abs(err[mask] - overall_bias))),
        })
    medians = [b["median_abs"] for b in buckets]
    return {
        "n": len(records),
        "buckets": buckets,
        "overall_bias": overall_bias,
        # Spread of the median error across lead buckets: how much the accuracy
        # changes as the prediction is made further ahead. A predictor that is
        # consistently wrong by a fixed amount is far more useful than one whose
        # error grows without bound, and that distinction is invisible in any
        # single-bucket number.
        "stability": float(max(medians) - min(medians)) if medians else float("nan"),
        "t": np.array([r[0] for r in records]),
        "pred": np.array([r[0] + r[1] for r in records]),
    }


def usable_horizon(buckets: list[dict], tolerance_min: float) -> float | None:
    """Largest lead whose bucket is inside tolerance on both median and p90.

    Both, because an estimate can have an excellent median and a tail that makes
    it unusable for a safety decision -- which is exactly the failure mode the
    monitored readout shows once the front plateaus and its rate approaches zero.
    """
    best = None
    for b in buckets:
        if b["median_abs"] <= tolerance_min and b["p90_abs"] <= tolerance_min * 3:
            best = b["hi"]
        else:
            break
    return best


# ---------------------------------------------------------------------------
# 3: IS THE DECISION RIGHT
# ---------------------------------------------------------------------------
def score_trigger(rows: list[dict], t: np.ndarray, r: np.ndarray, *, limit_m: float,
                  lead_h: float, keys: list[str], persistence: int, min_rate_m_h: float) -> dict:
    """Contingency table for the rule "some TTL < lead", over every update.

    ``persistence`` requires the rule to hold on that many consecutive updates
    before it counts as a warning, and ``min_rate_m_h`` refuses to extrapolate a
    front that is barely moving. Both exist to suppress the same failure: a
    near-zero measured rate sends the extrapolated time-to-limit to infinity, and
    the update either side of it can swing wildly.
    """
    hits = misses = false_alarms = correct_neg = 0
    lead_times: list[float] = []
    consecutive = 0
    already = np.interp(t, t, r)  # measured radius at each update time

    for row in rows:
        origin = _num(row, "time_hours")
        if origin is None:
            continue
        # Skip updates where the front is already at or past the limit. There is
        # nothing left to warn about, and counting them would fill the table with
        # zero-lead "hits" that merely restate the present -- which is exactly
        # what a warning system must not be credited for.
        if float(np.interp(origin, t, already)) >= limit_m:
            continue

        ttls = [_num(row, k) for k in keys]
        ttls = [x for x in ttls if x is not None]
        rate = _num(row, "front_rate_m_per_h")
        rate_ok = rate is None or rate >= min_rate_m_h
        raw = bool(ttls) and min(ttls) < lead_h and rate_ok

        consecutive = consecutive + 1 if raw else 0
        warned = consecutive >= max(persistence, 1)

        observed = breach_within(t, r, origin, lead_h, limit_m)
        if warned and observed:
            hits += 1
            # How much warning this update actually bought: time from the warning
            # to the first breach inside the window.
            window = (t > origin) & (t <= origin + lead_h)
            first = t[window][np.argmax(r[window] >= limit_m)]
            lead_times.append((first - origin) * 60.0)
        elif warned and not observed:
            false_alarms += 1
        elif not warned and observed:
            misses += 1
        else:
            correct_neg += 1

    total_events = hits + misses
    pod = hits / total_events if total_events else float("nan")
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) else float("nan")
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) else float("nan")
    return {
        "lead_h": lead_h, "hits": hits, "misses": misses,
        "false_alarms": false_alarms, "correct_negatives": correct_neg,
        "pod": pod, "far": far, "csi": csi,
        "median_lead_min": float(np.median(lead_times)) if lead_times else float("nan"),
    }


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
def plot_consistency(results: dict, actual_cross_h: float, out_dir: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.axhline(actual_cross_h, color="#111", lw=2.0, label=f"actual crossing (t = {actual_cross_h:.2f} h)")
    for key, (label, colour) in PREDICTORS.items():
        res = results.get(key)
        if not res or res["n"] == 0:
            continue
        ax.plot(res["t"], res["pred"], "o", ms=3.2, color=colour, alpha=0.75, label=label)
    ax.plot([0, actual_cross_h], [0, actual_cross_h], color="0.6", lw=1.0, ls=":",
            label="prediction = now (zero lead)")
    ax.set_xlabel("time the prediction was made (h)")
    ax.set_ylabel("predicted crossing time (h)")
    ax.set_title("A correct predictor is a flat line at the true crossing time", fontsize=11)
    ax.set_ylim(0, actual_cross_h * 2.2)
    ax.set_xlim(0, actual_cross_h)
    ax.grid(alpha=0.22)
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_dir / "01_crossing_time_consistency.png", dpi=150)
    print(f"  wrote {out_dir / '01_crossing_time_consistency.png'}")
    plt.close(fig)


def plot_error_vs_lead(results: dict, tolerance_min: float, out_dir: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5.0))
    for key, (label, colour) in PREDICTORS.items():
        buckets = results.get(key, {}).get("buckets", [])
        if not buckets:
            continue
        centres = [(b["lo"] + b["hi"]) / 2 for b in buckets]
        ax.plot(centres, [b["median_abs"] for b in buckets], "-o", color=colour, lw=1.8,
                label=f"{label} — median |error|")
        ax.plot(centres, [b["p90_abs"] for b in buckets], "--s", color=colour, lw=1.3,
                alpha=0.65, ms=4, label=f"{label} — p90 |error|")
    ax.axhline(tolerance_min, color="#B3261E", lw=1.4, ls="--")
    ax.text(ax.get_xlim()[1], tolerance_min, f"  tolerance {tolerance_min:g} min",
            color="#B3261E", fontsize=8.5, va="center")
    ax.set_yscale("log")
    ax.set_xlabel("how far ahead the prediction was made (min)")
    ax.set_ylabel("|error| in predicted crossing time (min)")
    ax.set_title("How far ahead the prediction stays usable", fontsize=11)
    ax.grid(alpha=0.22, which="both")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "02_error_vs_lead.png", dpi=150)
    print(f"  wrote {out_dir / '02_error_vs_lead.png'}")
    plt.close(fig)


def plot_skill(sweeps: dict, out_dir: Path):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for name, (scores, colour, style) in sweeps.items():
        leads = [s["lead_h"] * 60 for s in scores]
        ax1.plot(leads, [s["pod"] for s in scores], style, color=colour, lw=1.8, label=f"{name} — hit rate")
        ax1.plot(leads, [s["far"] for s in scores], style, color=colour, lw=1.2, alpha=0.55,
                 label=f"{name} — false alarm rate")
        ax2.plot(leads, [s["csi"] for s in scores], style, color=colour, lw=1.8, label=name)
    ax1.set_xlabel("lead time of the rule (min)")
    ax1.set_ylabel("rate")
    ax1.set_ylim(-0.03, 1.03)
    ax1.set_title("Hit rate and false alarms", fontsize=11)
    ax1.grid(alpha=0.22)
    ax1.legend(fontsize=7.5, loc="best")
    ax2.set_xlabel("lead time of the rule (min)")
    ax2.set_ylabel("critical success index")
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_title("Overall skill (hits / (hits + misses + false alarms))", fontsize=11)
    ax2.grid(alpha=0.22)
    ax2.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "03_trigger_skill.png", dpi=150)
    print(f"  wrote {out_dir / '03_trigger_skill.png'}")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default="artifacts/dt_evidence/dt_updates.csv")
    parser.add_argument("--config", default="startup.conf")
    parser.add_argument("--out", default="artifacts/prediction_skill")
    parser.add_argument("--tolerance-minutes", type=float, default=10.0,
                        help="Error the decision can absorb; sets the usable horizon.")
    parser.add_argument("--persistence", type=int, default=3,
                        help="Consecutive updates the rule must hold before warning.")
    parser.add_argument("--min-rate-mm-per-h", type=float, default=1.0,
                        help="Refuse to extrapolate a front slower than this.")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    config = load_startup_config(args.config)
    limit_m = float(config.get("safety_monitor", {}).get("limit_radius_m", 0.040))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(Path(args.csv))
    t, r = measured_series(rows)
    actual = first_crossing(t, r, limit_m)
    if actual is None:
        raise SystemExit(f"The measured front never reached {limit_m * 1000:.0f} mm; nothing to score.")

    print(f"limit {limit_m * 1000:.0f} mm   actual crossing t = {actual:.3f} h   "
          f"{len(rows)} updates\n")

    # ---- 1 + 2 -------------------------------------------------------------
    print("=" * 74)
    print("IS THE NUMBER RIGHT?  predicted crossing time vs the real one")
    print("=" * 74)
    results = {}
    for key, (label, _colour) in PREDICTORS.items():
        res = consistency(rows, key, actual)
        results[key] = res
        print(f"\n{label}: {res['n']} predictions before the event")
        if not res["buckets"]:
            print("  never produced a time-to-limit before the crossing")
            continue
        print(f"  {'lead made':>14}  {'n':>4}  {'median|e|':>10}  {'p90|e|':>10}  "
              f"{'bias':>8}  {'debiased':>9}")
        for b in res["buckets"]:
            print(f"  {b['lo']:5d}-{b['hi']:3d} min  {b['n']:4d}  "
                  f"{b['median_abs']:9.1f}m  {b['p90_abs']:9.1f}m  {b['bias']:+7.1f}m  "
                  f"{b['median_abs_debiased']:8.1f}m")
        horizon = usable_horizon(res["buckets"], args.tolerance_minutes)
        res["horizon"] = horizon
        print(f"  -> usable horizon at +-{args.tolerance_minutes:g} min tolerance: "
              + (f"{horizon} min" if horizon else "under 10 min"))
        print(f"  -> systematic offset {res['overall_bias']:+.1f} min; median error varies by "
              f"{res['stability']:.1f} min across lead buckets")

    # ---- 3 -----------------------------------------------------------------
    print("\n" + "=" * 74)
    print("IS THE DECISION RIGHT?  the rule scored as a warning system")
    print("=" * 74)
    min_rate = args.min_rate_mm_per_h / 1000.0
    lead_grid = [x / 60.0 for x in (10, 15, 20, 30, 45, 60, 90)]
    keys = list(PREDICTORS)

    variants = {
        "raw (no filter)": (0, 0.0, "#B3261E", "-o"),
        f"filtered (>={args.persistence} updates, >={args.min_rate_mm_per_h:g} mm/h)":
            (args.persistence, min_rate, "#1F6FA8", "-s"),
    }
    sweeps = {}
    for name, (persistence, rate, colour, style) in variants.items():
        scores = [score_trigger(rows, t, r, limit_m=limit_m, lead_h=L, keys=keys,
                                persistence=persistence, min_rate_m_h=rate)
                  for L in lead_grid]
        sweeps[name] = (scores, colour, style)
        print(f"\n{name}")
        print(f"  {'lead':>6}  {'hits':>5} {'miss':>5} {'false':>6}  {'POD':>6} {'FAR':>6} {'CSI':>6}  {'median lead':>12}")
        for s in scores:
            print(f"  {s['lead_h'] * 60:5.0f}m  {s['hits']:5d} {s['misses']:5d} {s['false_alarms']:6d}  "
                  f"{s['pod']:6.2f} {s['far']:6.2f} {s['csi']:6.2f}  {s['median_lead_min']:9.0f} min")

    # ---- verdict -----------------------------------------------------------
    print("\n" + "=" * 74)
    filtered = sweeps[list(variants)[1]][0]
    best = max(filtered, key=lambda s: (s["csi"] if np.isfinite(s["csi"]) else -1))
    kin_h = results.get("ttb_monitored_h", {}).get("horizon")
    fem_h = results.get("ttb_fem_h", {}).get("horizon")
    print("VERDICT")
    print(f"  usable horizon, monitored : {kin_h if kin_h else '<10'} min")
    print(f"  usable horizon, FEM       : {fem_h if fem_h else '<10'} min")
    print(f"  best-scoring rule lead    : {best['lead_h'] * 60:.0f} min "
          f"(CSI {best['csi']:.2f}, hit rate {best['pod']:.2f}, false alarms {best['far']:.2f})")
    print(f"  median warning bought     : {best['median_lead_min']:.0f} min")
    print("\n  Single experiment, single crossing: these characterise this run, not")
    print("  future skill. Re-run across the other recordings before relying on them.")

    print("\nwriting outputs")
    plot_consistency(results, actual, out_dir)
    plot_error_vs_lead(results, args.tolerance_minutes, out_dir)
    plot_skill(sweeps, out_dir)


if __name__ == "__main__":
    main()

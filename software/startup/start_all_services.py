"""Launcher for the permafrost digital twin.

    python -m software.startup.start_all_services --mode sim

Modes:
    sim       no hardware; feed data with ``software.tools.replay_csv``
    sensors   live CR1000 sensing only (no heater)
    hardware  full closed loop: CR1000 + safety rules + PolyScience circulator

RabbitMQ and InfluxDB are (re)started in Docker first unless ``--no-docker``
is given. Each service runs in its own daemon process; Ctrl+C stops them all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from software.startup.docker_services.start_influxdb import start_docker_influxdb
from software.startup.docker_services.start_rabbitmq import start_docker_rabbitmq
from software.startup.service_runners import (
    build_influx_config,
    initialize_service_config,
    _observation_ingestion_2d_runner,
    _boundary_condition_builder_runner,
    _fem_forecast_runner,
    _heater_actuation_runner,
    _fem_inversion_runner,
    _pipe_boundary_runner,
    _safety_monitor_runner,
    _thaw_front_reconstruction_runner,
    _viz_dashboard_runner,
    _cr1000_publisher_runner,
    _polyscience_control_runner,
)
from software.startup.utils.daemon import start_as_daemon
from software.startup.utils.config import configure_logging, load_startup_config


# Default run mode (override with --mode) — selects which services start:
#   "sim"      : no hardware. Feed data with replay_csv; full processing + FEM +
#                the safety/heater decision loop (no real circulator is driven).
#   "sensors"  : live CR1000 sensing, NO heater. Preview the real temperature field.
#   "hardware" : full closed loop — sensing + safety rule engine + PolyScience heater.
# The interactive setpoint pre-flight prompt runs only in "hardware" mode.
MODE = "sim"

_VALID_MODES = {"sim", "sensors", "hardware"}

# Services that hold an exclusive serial (COM) port, mapped to the config
# section and key that names their port. Any new serial-backed service must be
# registered here so the launcher can detect port collisions.
_SERIAL_PORT_SERVICES = {
    "cr1000_publisher": ("cr1000_publisher", "port"),
    "polyscience_control": ("polyscience_control", "port"),
}


def _prompt_float(label: str, current: float | None, *, lo: float, hi: float, unit: str) -> float:
    """Prompt for a float in [lo, hi], defaulting to ``current`` on empty input."""
    default_txt = "unset" if current is None else f"{current}{unit}"
    while True:
        raw = input(f"  {label} [{default_txt}]: ").strip()
        if raw == "":
            if current is None:
                print("    no default configured — you must enter a value")
                continue
            value = float(current)
        else:
            try:
                value = float(raw)
            except ValueError:
                print("    not a number — try again")
                continue
        if not (lo <= value <= hi):
            print(f"    out of range — must be within [{lo}, {hi}]{unit}")
            continue
        return value


def _preflight_operator_inputs(config: dict) -> None:
    """Interactively confirm the two safety-critical setpoints before launch.

    Forces the operator to set/confirm the DT-commanded heat setpoint and the
    STOP radius each run, validated against the controller's setpoint clamp and
    the physical sensor coverage. Skipped when stdin is not a TTY (e.g. an
    automated/replay run) so it never blocks non-interactive use.
    """
    if not sys.stdin.isatty():
        return

    safety = config.setdefault("safety_monitor", {})
    heater = config.setdefault("heater_actuation", {})
    poly = config.get("polyscience_control", {})
    sensors = config.get("cr1000_publisher", {}).get("sensors", [])

    max_r = max((float(s.get("x_m", 0.0)) for s in sensors), default=0.165)
    # Sensor coverage (max_r) is only used for the informational warning below;
    # the STOP radius may legitimately sit beyond it (forecast-only trip), so the
    # prompt is validated against the physical domain extent, not sensor coverage.
    max_domain_r = 0.5
    sp_min = float(poly.get("setpoint_min_c", -30.0))
    sp_max = float(poly.get("setpoint_max_c", 50.0))

    print("\n=== PRE-FLIGHT: confirm safety-critical setpoints (Enter = keep default) ===")
    setpoint = _prompt_float(
        "Heat setpoint (DT-commanded)",
        heater.get("setpoint_on_ok"),
        lo=sp_min,
        hi=sp_max,
        unit=" C",
    )
    limit_max = _prompt_float(
        "STOP thaw radius (limit_radius_m)",
        safety.get("limit_radius_m", 0.1),
        lo=1e-3,
        hi=max_domain_r,
        unit=" m",
    )

    heater["setpoint_on_ok"] = setpoint
    safety["limit_radius_m"] = limit_max

    warn = float(safety.get("warn_fraction", 0.6)) * 100.0
    crit = float(safety.get("critical_fraction", 0.9)) * 100.0
    print("\n  ------------------------------------------------------------")
    print(f"  Heat setpoint (commanded on 'heat') : {setpoint:.2f} C")
    print(f"  STOP when measured r_max >=          : {limit_max:.3f} m")
    print(f"  Sensor radial coverage reaches       : {max_r:.3f} m")
    print(f"  Limit min (informational, no stop)   : {float(safety.get('limit_min_radius_m', 0.0)):.3f} m")
    print(f"  Warn / critical bands                : {warn:.0f}% / {crit:.0f}% of limit")
    print("  ------------------------------------------------------------")
    if limit_max >= max_r:
        print("  WARNING: STOP radius is at/beyond sensor coverage — measured r_max")
        print("           may never reach it; only the FEM forecast could trip the stop.")
    if input("\n  Proceed and start the loop with these values? [y/N]: ").strip().lower() not in ("y", "yes"):
        print("Aborted by operator — no services started.")
        sys.exit(1)


def _check_serial_port_conflicts(config: dict, services_to_start: list[str]) -> None:
    """Raise if two *enabled* serial services are configured on the same COM port.

    pyserial opens ports exclusively, so a shared port would silently break one
    of the services. COM port names are compared case-insensitively.
    """
    used: dict[str, str] = {}
    for service_name in services_to_start:
        spec = _SERIAL_PORT_SERVICES.get(service_name)
        if spec is None:
            continue
        cfg_key, port_key = spec
        raw_port = config.get(cfg_key, {}).get(port_key)
        if not raw_port:
            continue
        port = str(raw_port).strip().upper()
        if port in used:
            raise RuntimeError(
                f"Serial port conflict: '{service_name}' and '{used[port]}' are both "
                f"configured on {raw_port}. pyserial opens COM ports exclusively — "
                f"assign each hardware service its own port in startup.conf."
            )
        used[port] = service_name


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the permafrost digital twin services.")
    parser.add_argument(
        "--mode",
        choices=sorted(_VALID_MODES),
        default=MODE,
        help="sim = replay data, no hardware (default); sensors = live CR1000 only; "
        "hardware = full closed loop with the PolyScience circulator.",
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Do not (re)start the RabbitMQ/InfluxDB containers; use ones already running.",
    )
    parser.add_argument(
        "--config",
        default="startup.conf",
        help="Path to the startup configuration file (default: startup.conf).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    mode = args.mode
    configure_logging()
    fem_fields_dir = Path("artifacts/fem_fields")
    if fem_fields_dir.exists():
        for csv_file in fem_fields_dir.glob("*.csv"):
            csv_file.unlink()

    config = load_startup_config(args.config)

    rabbit_cfg = config.get("rabbitmq", {})
    influx_cfg_values = config.get("influxdb", {})
    obs_2d_cfg = config.get("observation_ingestion_2d", {})
    bc_cfg = config.get("boundary_condition_builder", {})
    thaw_cfg = config.get("thaw_front_reconstruction", {})
    inversion_cfg = config.get("fem_inversion", {})
    fem_cfg = config.get("fem_forecast", {})
    safety_cfg = config.get("safety_monitor", {})
    heater_cfg = config.get("heater_actuation", {})
    cr1000_cfg = config.get("cr1000_publisher", {})
    pipe_boundary_cfg = config.get("pipe_boundary", {})
    polyscience_cfg = config.get("polyscience_control", {})

    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose one of {sorted(_VALID_MODES)}.")

    # Confirm the two safety-critical setpoints before anything starts. Mutates
    # safety_cfg / heater_cfg (references into `config`) so the values below are
    # carried into the daemons via config_dict. Only the full hardware loop heats.
    if mode == "hardware":
        _preflight_operator_inputs(config)

    influx_defaults = build_influx_config(influx_cfg_values)

    # Initialize the shared config dict (works across process boundaries on both Linux and Windows)
    config_dict = initialize_service_config(
        influx_config=influx_defaults,
        rabbit_cfg=rabbit_cfg,
        obs_2d_cfg=obs_2d_cfg,
        bc_cfg=bc_cfg,
        thaw_cfg=thaw_cfg,
        inversion_cfg=inversion_cfg,
        fem_cfg=fem_cfg,
        safety_cfg=safety_cfg,
        heater_cfg=heater_cfg,
        cr1000_cfg=cr1000_cfg,
        pipe_boundary_cfg=pipe_boundary_cfg,
        polyscience_cfg=polyscience_cfg,
    )

    # Full registry of runnable services. Keep every service registered here —
    # what actually starts is selected by MODE below, so this dict never needs
    # hand-editing (that is what previously desynced the two lists).
    available_services = {
        "observation_ingestion_2d": _observation_ingestion_2d_runner,
        "boundary_condition_builder": _boundary_condition_builder_runner,
        "thaw_front_reconstruction": _thaw_front_reconstruction_runner,
        "fem_inversion": _fem_inversion_runner,
        "pipe_boundary": _pipe_boundary_runner,
        "fem_forecast": _fem_forecast_runner,
        "safety_monitor": _safety_monitor_runner,
        "heater_actuation": _heater_actuation_runner,
        "viz_dashboard": _viz_dashboard_runner,
        "cr1000_publisher": _cr1000_publisher_runner,
        "polyscience_control": _polyscience_control_runner,
    }

    # Processing + dashboard — always on, all modes.
    services_to_start = [
        "observation_ingestion_2d",
        "boundary_condition_builder",
        "thaw_front_reconstruction",
        "viz_dashboard",
    ]

    # FEM forecast chain (pipe_boundary + fem_forecast) plus the online material
    # inversion (fem_inversion) that calibrates soil parameters from the observed
    # field. Needed for the forecast plots, the predictive safety stop, and the
    # inverted-parameter evolution figure; not required for the raw temp field.
    if mode in ("sim", "hardware"):
        services_to_start += ["pipe_boundary", "fem_forecast", "fem_inversion"]

    # Live CR1000 sensing bridge (opens the real datalogger serial port).
    if mode in ("sensors", "hardware"):
        services_to_start += ["cr1000_publisher"]

    # Closed-loop control logic: safety rule engine + heater command service.
    # Runs in sim too so the full alert -> actuation decision loop can be shown
    # without hardware (no real circulator is driven — see polyscience below).
    if mode in ("sim", "hardware"):
        services_to_start += ["safety_monitor", "heater_actuation"]

    # PolyScience actuation bridge drives the physical circulator over serial,
    # so it only runs in the full hardware loop.
    if mode == "hardware":
        services_to_start += ["polyscience_control"]

    # Guard against two enabled services grabbing the same serial port — pyserial
    # opens COM ports exclusively, so a shared port would make one of them fail.
    _check_serial_port_conflicts(config, services_to_start)

    # Start local Docker dependencies first (fresh containers each run).
    if not args.no_docker:
        start_docker_rabbitmq()
        start_docker_influxdb()

    print(f"Starting permafrost DT in '{mode}' mode: {', '.join(services_to_start)}")

    processes = []

    for service_name in services_to_start:
        runner = available_services.get(service_name)
        if runner is None:
            raise KeyError(f"Unknown service '{service_name}'. Available: {sorted(available_services)}")
        processes.append(
            start_as_daemon(runner, process_name=service_name, kwargs={"config_dict": config_dict})
        )

    for process in processes:
        process.join()


if __name__ == "__main__":
    main()

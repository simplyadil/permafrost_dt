"""Convenience launcher for the permafrost digital twin.

Uncomment the services you want to start and run:

    python -m software.startup.start_all_services

Each selected service runs in its own daemon process, so stopping the launcher
will shut everything down. Docker helpers for RabbitMQ/InfluxDB are provided
but commented out so you can decide whether to trigger them.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from software.startup.docker_services.start_influxdb import start_docker_influxdb
from software.startup.docker_services.start_rabbitmq import start_docker_rabbitmq
from software.startup.service_runners import (
    build_influx_config,
    make_boundary_condition_builder_runner,
    make_fem_forecast_runner,
    make_heater_actuation_runner,
    make_observation_ingestion_2d_runner,
    make_safety_monitor_runner,
    make_thaw_front_reconstruction_runner,
    make_viz_dashboard_runner,
)
from software.startup.utils.daemon import start_as_daemon
from software.startup.utils.config import configure_logging, load_startup_config


def main() -> None:
    configure_logging()
    config = load_startup_config()

    rabbit_cfg = config.get("rabbitmq", {})
    influx_cfg_values = config.get("influxdb", {})
    obs_2d_cfg = config.get("observation_ingestion_2d", {})
    bc_cfg = config.get("boundary_condition_builder", {})
    thaw_cfg = config.get("thaw_front_reconstruction", {})
    fem_cfg = config.get("fem_forecast", {})
    safety_cfg = config.get("safety_monitor", {})
    heater_cfg = config.get("heater_actuation", {})

    influx_defaults = build_influx_config(influx_cfg_values)

    available_services: Dict[str, Callable[[], Callable[[], None]]] = {
        "observation_ingestion_2d": lambda: make_observation_ingestion_2d_runner(
            influx_defaults,
            rabbit_cfg,
            obs_2d_cfg,
        ),
        "boundary_condition_builder": lambda: make_boundary_condition_builder_runner(
            rabbit_cfg,
            bc_cfg,
        ),
        "thaw_front_reconstruction": lambda: make_thaw_front_reconstruction_runner(
            influx_defaults,
            rabbit_cfg,
            thaw_cfg,
        ),
        "fem_forecast": lambda: make_fem_forecast_runner(
            influx_defaults,
            rabbit_cfg,
            fem_cfg,
        ),
        "safety_monitor": lambda: make_safety_monitor_runner(
            influx_defaults,
            rabbit_cfg,
            safety_cfg,
        ),
        "heater_actuation": lambda: make_heater_actuation_runner(
            influx_defaults,
            rabbit_cfg,
            heater_cfg,
        ),
        "viz_dashboard": lambda: make_viz_dashboard_runner(),
    }

    # observation_ingestion_2d
    # boundary_condition_builder
    # thaw_front_reconstruction
    # pinn_inversion_2d
    # fem_forecast
    # safety_monitor
    # heater_actuation
    # viz_dashboard

    services_to_start: List[str] = [
        "observation_ingestion_2d",
        "boundary_condition_builder",
        "thaw_front_reconstruction",
        "fem_forecast",
        # "safety_monitor",
        # "heater_actuation",
        "viz_dashboard",
    ]

    # Optionally start local Docker dependencies first.
    # start_docker_rabbitmq()
    # start_docker_influxdb()

    processes = []

    for service_name in services_to_start:
        factory = available_services.get(service_name)
        if factory is None:
            raise KeyError(f"Unknown service '{service_name}'. Available: {sorted(available_services)}")
        runner = factory()
        processes.append(start_as_daemon(runner, process_name=service_name))

    for process in processes:
        process.join()


if __name__ == "__main__":
    main()

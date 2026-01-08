"""Convenience launcher for the permafrost digital twin.

Uncomment the services you want to start and run:

    python -m software.startup.start_all_services

Each selected service runs in its own daemon process, so stopping the launcher
will shut everything down. Docker helpers for RabbitMQ/InfluxDB are provided
but commented out so you can decide whether to trigger them.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from software.startup.service_runners import (
    build_influx_config,
    make_boundary_forcing_runner,
    make_fdm_runner,
    make_observation_ingestion_runner,
    make_pinn_forward_runner,
    make_pinn_inversion_runner,
    make_viz_gateway_runner,
    make_viz_dashboard_runner,
)
from software.startup.docker_services.start_influxdb import start_docker_influxdb
from software.startup.docker_services.start_rabbitmq import start_docker_rabbitmq
from software.startup.utils.daemon import start_as_daemon
from software.startup.utils.config import configure_logging, load_startup_config


def main() -> None:
    configure_logging()
    config = load_startup_config()

    rabbit_cfg = config.get("rabbitmq", {})
    influx_cfg_values = config.get("influxdb", {})
    synthetic_obs_cfg = config.get("synthetic_observations", {})
    synthetic_boundary_cfg = config.get("synthetic_boundary", {})
    pinn_forward_cfg = config.get("pinn_forward", {})
    pinn_inversion_cfg = config.get("pinn_inversion", {})
    viz_gateway_cfg = config.get("viz_gateway", {})
    viz_dashboard_cfg = config.get("viz_dashboard", {})

    influx_defaults = build_influx_config(influx_cfg_values)

    available_services: Dict[str, Callable[[], Callable[[], None]]] = {
        "observation_ingestion": lambda: make_observation_ingestion_runner(influx_defaults, rabbit_cfg),
        "boundary_forcing": lambda: make_boundary_forcing_runner(influx_defaults, rabbit_cfg, synthetic_boundary_cfg),
        "fdm": lambda: make_fdm_runner(influx_defaults, rabbit_cfg, synthetic_obs_cfg),
        "pinn_forward": lambda: make_pinn_forward_runner(influx_defaults, rabbit_cfg, pinn_forward_cfg),
        "pinn_inversion": lambda: make_pinn_inversion_runner(influx_defaults, rabbit_cfg, pinn_inversion_cfg),
        "viz_gateway": lambda: make_viz_gateway_runner(influx_defaults, rabbit_cfg, viz_gateway_cfg),
        "viz_dashboard": lambda: make_viz_dashboard_runner(influx_defaults, viz_dashboard_cfg),
    }

    # Uncomment the services you want to run.
    services_to_start: List[str] = [
        # "observation_ingestion",
        "boundary_forcing",
        "fdm",
        # "pinn_forward",
        # "pinn_inversion",
        # "viz_gateway",
         "viz_dashboard",
    ]

    # Optionally start local Docker dependencies first.
    start_docker_rabbitmq()
    start_docker_influxdb()

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

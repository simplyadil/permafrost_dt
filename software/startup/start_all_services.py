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
from software.startup.utils.daemon import start_as_daemon
from software.startup.utils.config import configure_logging, load_startup_config


def main() -> None:
    configure_logging()
    load_startup_config()
    available_services: Dict[str, Callable[[], Callable[[], None]]] = {}

    # Placeholder list for the new 2D DT services:
    # observation_ingestion_2d
    # boundary_and_forcing_builder
    # thaw_front_reconstruction
    # pinn_inversion_2d
    # fem_forecast
    # safety_monitor
    # heater_actuation
    # viz_gateway
    # viz_dashboard
    services_to_start: List[str] = []

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

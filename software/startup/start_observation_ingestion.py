"""Start script for the observation ingestion server."""

from __future__ import annotations

import signal

from software.startup.service_runners import (
    build_influx_config,
    create_observation_ingestion_server,
)
from software.startup.utils.config import configure_logging, load_startup_config


def main() -> None:
    configure_logging()
    config = load_startup_config()

    rabbit_cfg = config.get("rabbitmq", {})
    influx_cfg = config.get("influxdb", {})

    influx_defaults = build_influx_config(influx_cfg)
    server = create_observation_ingestion_server(influx_defaults, rabbit_cfg)

    def shutdown_handler(*_args) -> None:
        server.close()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    server.start()


if __name__ == "__main__":
    main()

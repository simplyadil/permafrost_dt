"""Start script for the visualization dashboard service."""

from __future__ import annotations

from software.startup.service_runners import (
    build_influx_config,
    create_viz_dashboard_service,
)
from software.startup.utils.config import configure_logging, load_startup_config


def main() -> None:
    configure_logging()
    config = load_startup_config()

    influx_cfg_values = config.get("influxdb", {})
    dashboard_cfg = config.get("viz_dashboard", {})

    influx_defaults = build_influx_config(influx_cfg_values)
    service = create_viz_dashboard_service(influx_defaults, dashboard_cfg)

    try:
        service.start()
    finally:
        service.close()


if __name__ == "__main__":
    main()

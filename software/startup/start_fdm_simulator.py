"""Start script for the FDM simulation server."""

from __future__ import annotations

import signal

from software.digital_twin.communication.messaging import RabbitMQConfig
from software.digital_twin.data_access.influx_utils import InfluxConfig
import software.digital_twin.simulator.fdm.fdm_server as fdm_module
from software.startup.utils.config import configure_logging, load_startup_config


def main() -> None:
    configure_logging()
    config = load_startup_config()

    rabbit_cfg = config.get("rabbitmq", {})
    influx_cfg = config.get("influxdb", {})

    server = fdm_module.FDMServer(
        influx_config=InfluxConfig(
            url=influx_cfg.get("url", "http://localhost:8086"),
            token=influx_cfg.get("token", "permafrost"),
            org=influx_cfg.get("org", "permafrost"),
            bucket=influx_cfg.get("bucket", "permafrost_data"),
            username=influx_cfg.get("username", "permafrost"),
            password=influx_cfg.get("password", "permafrost"),
        ),
        boundary_config=RabbitMQConfig(
            host=rabbit_cfg.get("host", "localhost"),
            schema_path=fdm_module.BOUNDARY_SCHEMA,
            username=rabbit_cfg.get("username", "permafrost"),
            password=rabbit_cfg.get("password", "permafrost"),
        ),
        outbound_config=RabbitMQConfig(
            host=rabbit_cfg.get("host", "localhost"),
            schema_path=fdm_module.FDM_SCHEMA,
            username=rabbit_cfg.get("username", "permafrost"),
            password=rabbit_cfg.get("password", "permafrost"),
        ),
    )

    def shutdown_handler(*_args) -> None:
        server.close()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    server.start()


if __name__ == "__main__":
    main()

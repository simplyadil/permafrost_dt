"""Bootstrap all digital twin services with shared startup conventions."""

from __future__ import annotations

import signal
from pathlib import Path
from threading import Thread

# Legacy direct class imports retained for reference:
# from software.digital_twin.monitoring.boundary_forcing.boundary_forcing_server import (
#     BoundaryForcingServer,
# )
# from software.digital_twin.monitoring.observation_ingestion.observation_ingestion_server import (
#     ObservationIngestionServer,
# )
# from software.digital_twin.simulator.fdm.fdm_server import FDMServer
# from software.digital_twin.simulator.pinn_forward.pinn_forward_server import (
#     PINNForwardServer,
# )
# from software.digital_twin.simulator.pinn_inversion.pinn_inversion_server import (
#     PINNInversionServer,
# )
# from software.digital_twin.visualization.viz_gateway.viz_gateway_server import (
#     VizGatewayServer,
# )
import software.digital_twin.monitoring.boundary_forcing.boundary_forcing_server as boundary_module
import software.digital_twin.monitoring.observation_ingestion.observation_ingestion_server as obs_module
import software.digital_twin.simulator.fdm.fdm_server as fdm_module
import software.digital_twin.simulator.pinn_forward.pinn_forward_server as pinn_forward_module
import software.digital_twin.simulator.pinn_inversion.pinn_inversion_server as pinn_inversion_module
import software.digital_twin.visualization.viz_gateway.viz_gateway_server as viz_module
from software.digital_twin.communication.messaging import RabbitMQConfig
from software.digital_twin.data_access.influx_utils import InfluxConfig
from software.startup.docker_services.start_influxdb import start_docker_influxdb
from software.startup.docker_services.start_rabbitmq import start_docker_rabbitmq
from software.startup.utils.config import configure_logging, load_startup_config


SERVERS = [
    obs_module.ObservationIngestionServer,
    boundary_module.BoundaryForcingServer,
    fdm_module.FDMServer,
    pinn_forward_module.PINNForwardServer,
    pinn_inversion_module.PINNInversionServer,
    viz_module.VizGatewayServer,
]


def _start_server(server) -> Thread:
    thread = Thread(target=server.start, daemon=True)
    thread.start()
    return thread


def main() -> None:
    configure_logging()
    config = load_startup_config()

    rabbitmq_config = config.get("rabbitmq", {})
    influx_config_values = config.get("influxdb", {})

    # Start infrastructure dependencies first.
    start_docker_rabbitmq()
    start_docker_influxdb()

    default_influx = InfluxConfig()
    influx_defaults = InfluxConfig(
        url=influx_config_values.get("url", default_influx.url),
        token=influx_config_values.get("token", default_influx.token),
        org=influx_config_values.get("org", default_influx.org),
        bucket=influx_config_values.get("bucket", default_influx.bucket),
        username=influx_config_values.get("username", default_influx.username),
        password=influx_config_values.get("password", default_influx.password),
    )

    def mk_rabbit_config(schema_path: Path) -> RabbitMQConfig:
        return RabbitMQConfig(
            host=rabbitmq_config.get("host", "localhost"),
            schema_path=schema_path,
            username=rabbitmq_config.get("username", "permafrost"),
            password=rabbitmq_config.get("password", "permafrost"),
        )

    instances = [
        obs_module.ObservationIngestionServer(
            rabbitmq_config=mk_rabbit_config(obs_module.SCHEMA_PATH),
            influx_config=influx_defaults,
        ),
        boundary_module.BoundaryForcingServer(
            rabbitmq_config=mk_rabbit_config(boundary_module.SCHEMA_PATH),
            influx_config=influx_defaults,
        ),
        fdm_module.FDMServer(
            influx_config=influx_defaults,
            boundary_config=mk_rabbit_config(fdm_module.BOUNDARY_SCHEMA),
            outbound_config=mk_rabbit_config(fdm_module.FDM_SCHEMA),
        ),
        pinn_forward_module.PINNForwardServer(
            influx_config=influx_defaults,
            fdm_queue_config=mk_rabbit_config(pinn_forward_module.FDM_SCHEMA),
            forward_queue_config=mk_rabbit_config(pinn_forward_module.PINN_FORWARD_SCHEMA),
        ),
        pinn_inversion_module.PINNInversionServer(
            influx_config=influx_defaults,
            forward_queue_config=mk_rabbit_config(pinn_inversion_module.PINN_FORWARD_SCHEMA),
            inversion_queue_config=mk_rabbit_config(pinn_inversion_module.PINN_INVERSION_SCHEMA),
        ),
        viz_module.VizGatewayServer(
            influx_config=influx_defaults,
            inversion_queue_config=mk_rabbit_config(viz_module.PINN_INVERSION_SCHEMA),
            viz_queue_config=mk_rabbit_config(viz_module.VIZ_UPDATE_SCHEMA),
        ),
    ]
    threads = [_start_server(server) for server in instances]

    def _shutdown(*_args) -> None:
        for server in instances:
            if hasattr(server, "close"):
                server.close()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()

"""Factory functions for starting digital-twin services."""

from __future__ import annotations

from typing import Callable, Dict

from software.digital_twin.communication.messaging import RabbitMQConfig
from software.digital_twin.data_access.influx_utils import InfluxConfig
import software.digital_twin.monitoring.boundary_forcing.boundary_forcing_server as boundary_module
import software.digital_twin.monitoring.observation_ingestion.observation_ingestion_server as obs_module
import software.digital_twin.simulator.fdm.fdm_server as fdm_module
import software.digital_twin.simulator.pinn_forward.pinn_forward_server as pinn_forward_module
import software.digital_twin.simulator.pinn_inversion.pinn_inversion_server as pinn_inversion_module
import software.digital_twin.visualization.viz_gateway.viz_gateway_server as viz_module
import software.digital_twin.visualization.viz_gateway.visualization_service as viz_service_module


def build_influx_config(values: Dict[str, str]) -> InfluxConfig:
    """Create an `InfluxConfig` using provided overrides."""
    defaults = InfluxConfig()
    return InfluxConfig(
        url=values.get("url", defaults.url),
        token=values.get("token", defaults.token),
        org=values.get("org", defaults.org),
        bucket=values.get("bucket", defaults.bucket),
        username=values.get("username", defaults.username),
        password=values.get("password", defaults.password),
    )


def _rabbit_config(rabbit_cfg: Dict[str, str], schema_path) -> RabbitMQConfig:
    return RabbitMQConfig(
        host=rabbit_cfg.get("host", "localhost"),
        schema_path=schema_path,
        username=rabbit_cfg.get("username", "permafrost"),
        password=rabbit_cfg.get("password", "permafrost"),
    )


# --------------------------------------------------------------------------- #
# Service creators.
# --------------------------------------------------------------------------- #


def create_observation_ingestion_server(
    influx_config: InfluxConfig,
    rabbit_cfg: Dict[str, str],
):
    return obs_module.ObservationIngestionServer(
        rabbitmq_config=_rabbit_config(rabbit_cfg, obs_module.SCHEMA_PATH),
        influx_config=influx_config,
    )


def create_boundary_forcing_server(
    influx_config: InfluxConfig,
    rabbit_cfg: Dict[str, str],
    synthetic_cfg: Dict[str, object],
):
    return boundary_module.BoundaryForcingServer(
        rabbitmq_config=_rabbit_config(rabbit_cfg, boundary_module.SCHEMA_PATH),
        influx_config=influx_config,
        synthetic_enabled=synthetic_cfg.get("enabled", True),
        synthetic_start_day=float(synthetic_cfg.get("start_day", 0.0)),
        synthetic_step_days=float(synthetic_cfg.get("step_days", 1.0)),
    )


def create_fdm_server(
    influx_config: InfluxConfig,
    rabbit_cfg: Dict[str, str],
    synthetic_obs_cfg: Dict[str, object],
):
    sensor_config = None
    if synthetic_obs_cfg.get("enabled", True):
        sensor_config = _rabbit_config(rabbit_cfg, fdm_module.SENSOR_SCHEMA)

    return fdm_module.FDMServer(
        influx_config=influx_config,
        boundary_config=_rabbit_config(rabbit_cfg, fdm_module.BOUNDARY_SCHEMA),
        outbound_config=_rabbit_config(rabbit_cfg, fdm_module.FDM_SCHEMA),
        sensor_config=sensor_config,
        sensor_depths=tuple(
            int(depth) for depth in synthetic_obs_cfg.get("depths_m", fdm_module.DEFAULT_SENSOR_DEPTHS)
        ),
    )


def create_pinn_forward_server(
    influx_config: InfluxConfig,
    rabbit_cfg: Dict[str, str],
    pinn_cfg: Dict[str, object],
):
    return pinn_forward_module.PINNForwardServer(
        influx_config=influx_config,
        fdm_queue_config=_rabbit_config(rabbit_cfg, pinn_forward_module.FDM_SCHEMA),
        forward_queue_config=_rabbit_config(rabbit_cfg, pinn_forward_module.PINN_FORWARD_SCHEMA),
        enable_training=pinn_cfg.get("enable_training", True),
        model_path=pinn_cfg.get("model_path"),
    )


def create_pinn_inversion_server(
    influx_config: InfluxConfig,
    rabbit_cfg: Dict[str, str],
    pinn_cfg: Dict[str, object],
):
    return pinn_inversion_module.PINNInversionServer(
        influx_config=influx_config,
        fdm_queue_config=_rabbit_config(rabbit_cfg, pinn_inversion_module.FDM_SCHEMA),
        inversion_queue_config=_rabbit_config(rabbit_cfg, pinn_inversion_module.PINN_INVERSION_SCHEMA),
        enable_training=pinn_cfg.get("enable_training", True),
        model_path=pinn_cfg.get("model_path"),
    )


def create_viz_gateway_server(
    influx_config: InfluxConfig,
    rabbit_cfg: Dict[str, str],
):
    return viz_module.VizGatewayServer(
        influx_config=influx_config,
        inversion_queue_config=_rabbit_config(rabbit_cfg, viz_module.PINN_INVERSION_SCHEMA),
        viz_queue_config=_rabbit_config(rabbit_cfg, viz_module.VIZ_UPDATE_SCHEMA),
    )


def create_viz_dashboard_service(
    influx_config: InfluxConfig,
    dashboard_cfg: Dict[str, object],
):
    return viz_service_module.VisualizationService(
        influx_config=influx_config,
        host=str(dashboard_cfg.get("host", "0.0.0.0")),
        port=int(dashboard_cfg.get("port", 8050)),
        refresh_seconds=float(dashboard_cfg.get("refresh_seconds", 60.0)),
        fetch_limit=int(dashboard_cfg.get("fetch_limit", 20000)),
        output_dir=dashboard_cfg.get("snapshot_dir", "software/outputs"),
    )


# --------------------------------------------------------------------------- #
# Runner factories (used by start_all_services).
# --------------------------------------------------------------------------- #


def make_runner(builder: Callable[[], object]) -> Callable[..., None]:
    """Wrap a server builder into a runnable callable.

    The returned function accepts an optional ``ok_queue`` keyword argument
    so it can be launched via ``start_as_daemon`` and signal readiness before
    entering the blocking ``start()`` loop.
    """

    def _runner(*, ok_queue=None) -> None:
        server = builder()
        if ok_queue is not None:
            ok_queue.put("OK")
        server.start()

    return _runner


def make_observation_ingestion_runner(
    influx_config: InfluxConfig,
    rabbit_cfg: Dict[str, str],
) -> Callable[[], None]:
    return make_runner(lambda: create_observation_ingestion_server(influx_config, rabbit_cfg))


def make_boundary_forcing_runner(
    influx_config: InfluxConfig,
    rabbit_cfg: Dict[str, str],
    synthetic_cfg: Dict[str, object],
) -> Callable[[], None]:
    return make_runner(lambda: create_boundary_forcing_server(influx_config, rabbit_cfg, synthetic_cfg))


def make_fdm_runner(
    influx_config: InfluxConfig,
    rabbit_cfg: Dict[str, str],
    synthetic_obs_cfg: Dict[str, object],
) -> Callable[[], None]:
    return make_runner(lambda: create_fdm_server(influx_config, rabbit_cfg, synthetic_obs_cfg))


def make_pinn_forward_runner(
    influx_config: InfluxConfig,
    rabbit_cfg: Dict[str, str],
    pinn_cfg: Dict[str, object],
) -> Callable[[], None]:
    return make_runner(lambda: create_pinn_forward_server(influx_config, rabbit_cfg, pinn_cfg))


def make_pinn_inversion_runner(
    influx_config: InfluxConfig,
    rabbit_cfg: Dict[str, str],
    pinn_cfg: Dict[str, object],
) -> Callable[[], None]:
    return make_runner(lambda: create_pinn_inversion_server(influx_config, rabbit_cfg, pinn_cfg))


def make_viz_gateway_runner(
    influx_config: InfluxConfig,
    rabbit_cfg: Dict[str, str],
) -> Callable[[], None]:
    return make_runner(lambda: create_viz_gateway_server(influx_config, rabbit_cfg))


def make_viz_dashboard_runner(
    influx_config: InfluxConfig,
    dashboard_cfg: Dict[str, object],
) -> Callable[[], None]:
    return make_runner(lambda: create_viz_dashboard_service(influx_config, dashboard_cfg))

"""Factory functions for starting digital-twin services."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

from software.digital_twin.communication.messaging import RabbitMQConfig
from software.digital_twin.data_access.influx_utils import InfluxConfig
import software.digital_twin.monitoring.observation_ingestion.observation_ingestion_server as obs_2d_module
import software.digital_twin.monitoring.boundary_condition_builder.boundary_condition_builder_server as bc_module
import software.digital_twin.monitoring.thaw_front_reconstruction.thaw_front_reconstruction_server as thaw_module
import software.digital_twin.simulator.fem_forecast.fem_forecast_server as fem_module
import software.digital_twin.control.safety_monitor.safety_monitor_server as safety_module
import software.digital_twin.control.heater_actuation.heater_actuation_server as heater_module
from software.startup import start_viz_dashboard as viz_dashboard_module


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


def _rabbit_config(
    rabbit_cfg: Dict[str, str],
    schema_path,
    *,
    queue: str | None = None,
    exchange: str | None = None,
    exchange_type: str | None = None,
    routing_key: str | None = None,
) -> RabbitMQConfig:
    return RabbitMQConfig(
        host=rabbit_cfg.get("host", "localhost"),
        queue=queue or "",
        exchange=exchange,
        exchange_type=exchange_type or "fanout",
        routing_key=routing_key or "",
        schema_path=schema_path,
        username=rabbit_cfg.get("username", "permafrost"),
        password=rabbit_cfg.get("password", "permafrost"),
    )


def create_observation_ingestion_2d_server(
    influx_config: InfluxConfig,
    rabbit_cfg: Dict[str, str],
    obs_cfg: Dict[str, object],
):
    queue_name = str(obs_cfg.get("queue", obs_2d_module.SENSOR_QUEUE))
    exchange = obs_cfg.get("exchange")
    exchange_type = obs_cfg.get("exchange_type")
    routing_key = obs_cfg.get("routing_key")
    return obs_2d_module.ObservationIngestion2DServer(
        rabbitmq_config=_rabbit_config(
            rabbit_cfg,
            obs_2d_module.SCHEMA_PATH,
            queue=queue_name,
            exchange=exchange,
            exchange_type=exchange_type if isinstance(exchange_type, str) else None,
            routing_key=routing_key if isinstance(routing_key, str) else None,
        ),
        influx_config=influx_config,
        queue=queue_name,
        site_id=str(obs_cfg.get("site_id", "default")),
    )


def create_boundary_condition_builder_server(
    rabbit_cfg: Dict[str, str],
    bc_cfg: Dict[str, object],
):
    input_queue = str(bc_cfg.get("input_queue", bc_module.SENSOR_QUEUE))
    input_exchange = bc_cfg.get("input_exchange")
    exchange_type = bc_cfg.get("input_exchange_type")
    routing_key = bc_cfg.get("input_routing_key")
    output_queue = str(bc_cfg.get("output_queue", bc_module.BOUNDARY_QUEUE))
    boundaries = bc_cfg.get("boundaries", [])
    sensor_overrides = bc_cfg.get("sensor_overrides", [])

    return bc_module.BoundaryConditionBuilderServer(
        input_config=_rabbit_config(
            rabbit_cfg,
            bc_module.SENSOR_SCHEMA,
            queue=input_queue,
            exchange=input_exchange if isinstance(input_exchange, str) else None,
            exchange_type=exchange_type if isinstance(exchange_type, str) else None,
            routing_key=routing_key if isinstance(routing_key, str) else None,
        ),
        output_config=_rabbit_config(rabbit_cfg, bc_module.BOUNDARY_SCHEMA, queue=output_queue),
        input_queue=input_queue,
        output_queue=output_queue,
        boundaries=boundaries if isinstance(boundaries, list) else [],
        sensor_overrides=sensor_overrides if isinstance(sensor_overrides, list) else [],
    )


def create_thaw_front_reconstruction_server(
    influx_config: InfluxConfig,
    rabbit_cfg: Dict[str, str],
    thaw_cfg: Dict[str, object],
):
    input_queue = str(thaw_cfg.get("input_queue", thaw_module.SENSOR_QUEUE))
    input_exchange = thaw_cfg.get("input_exchange")
    input_exchange_type = thaw_cfg.get("input_exchange_type")
    input_routing_key = thaw_cfg.get("input_routing_key")
    output_queue = str(thaw_cfg.get("output_queue", thaw_module.THAW_FRONT_QUEUE))
    return thaw_module.ThawFrontReconstructionServer(
        influx_config=influx_config,
        input_config=_rabbit_config(
            rabbit_cfg,
            thaw_module.SENSOR_SCHEMA,
            queue=input_queue,
            exchange=input_exchange if isinstance(input_exchange, str) else None,
            exchange_type=input_exchange_type if isinstance(input_exchange_type, str) else None,
            routing_key=input_routing_key if isinstance(input_routing_key, str) else None,
        ),
        output_config=_rabbit_config(rabbit_cfg, thaw_module.THAW_FRONT_SCHEMA, queue=output_queue),
        input_queue=input_queue,
        output_queue=output_queue,
        thaw_threshold_c=float(thaw_cfg.get("thaw_threshold_c", 0.0)),
        contour_grid_r=int(thaw_cfg.get("contour_grid_r", 120)),
        contour_grid_z=int(thaw_cfg.get("contour_grid_z", 150)),
        primary_metric_source=str(thaw_cfg.get("primary_metric_source", "contour")),
        site_id=str(thaw_cfg.get("site_id", "default")),
    )


def create_fem_forecast_server(
    influx_config: InfluxConfig,
    rabbit_cfg: Dict[str, str],
    fem_cfg: Dict[str, object],
):
    input_queue = str(fem_cfg.get("input_queue", fem_module.BOUNDARY_QUEUE))
    output_queue = str(fem_cfg.get("output_queue", fem_module.FORECAST_QUEUE))
    
    # Resolve mesh_dir to absolute path if relative
    mesh_dir_config = fem_cfg.get("mesh_dir")
    if mesh_dir_config:
        mesh_path = Path(str(mesh_dir_config))
        if not mesh_path.is_absolute():
            # Try relative to the repo root
            repo_root = Path(__file__).resolve().parents[2]
            mesh_path = repo_root / mesh_path
        mesh_dir = str(mesh_path)
    else:
        mesh_dir = None
    
    return fem_module.FEMForecastServer(
        influx_config=influx_config,
        input_config=_rabbit_config(rabbit_cfg, fem_module.BOUNDARY_SCHEMA, queue=input_queue),
        output_config=_rabbit_config(rabbit_cfg, fem_module.FORECAST_SCHEMA, queue=output_queue),
        input_queue=input_queue,
        output_queue=output_queue,
        horizon_hours=float(fem_cfg.get("horizon_hours", 1.0)),
        site_id=str(fem_cfg.get("site_id", "default")),
        mesh_dir=mesh_dir,
        node_file=str(fem_cfg.get("node_file", "DT_model_nodes.txt")),
        element_file=str(fem_cfg.get("element_file", "DT_model_elements.txt")),
        edge_sets=fem_cfg.get("edge_sets", {}) if isinstance(fem_cfg.get("edge_sets"), dict) else {},
        node_sets=fem_cfg.get("node_sets", {}) if isinstance(fem_cfg.get("node_sets"), dict) else {},
        material=fem_cfg.get("material", {}) if isinstance(fem_cfg.get("material"), dict) else {},
        solver=fem_cfg.get("solver", {}) if isinstance(fem_cfg.get("solver"), dict) else {},
    )


def create_safety_monitor_server(
    influx_config: InfluxConfig,
    rabbit_cfg: Dict[str, str],
    safety_cfg: Dict[str, object],
):
    thaw_queue = str(safety_cfg.get("thaw_input_queue", safety_module.THAW_FRONT_QUEUE))
    thaw_exchange = safety_cfg.get("thaw_input_exchange")
    thaw_exchange_type = safety_cfg.get("thaw_input_exchange_type")
    thaw_routing_key = safety_cfg.get("thaw_input_routing_key")

    forecast_queue = str(safety_cfg.get("forecast_input_queue", safety_module.FORECAST_QUEUE))
    forecast_exchange = safety_cfg.get("forecast_input_exchange")
    forecast_exchange_type = safety_cfg.get("forecast_input_exchange_type")
    forecast_routing_key = safety_cfg.get("forecast_input_routing_key")

    output_queue = str(safety_cfg.get("output_queue", safety_module.ALERT_QUEUE))
    output_exchange = safety_cfg.get("output_exchange")
    output_exchange_type = safety_cfg.get("output_exchange_type")
    output_routing_key = safety_cfg.get("output_routing_key")

    return safety_module.SafetyMonitorServer(
        influx_config=influx_config,
        thaw_input_config=_rabbit_config(
            rabbit_cfg,
            safety_module.THAW_FRONT_SCHEMA,
            queue=thaw_queue,
            exchange=thaw_exchange if isinstance(thaw_exchange, str) else None,
            exchange_type=thaw_exchange_type if isinstance(thaw_exchange_type, str) else None,
            routing_key=thaw_routing_key if isinstance(thaw_routing_key, str) else None,
        ),
        forecast_input_config=_rabbit_config(
            rabbit_cfg,
            safety_module.FORECAST_SCHEMA,
            queue=forecast_queue,
            exchange=forecast_exchange if isinstance(forecast_exchange, str) else None,
            exchange_type=forecast_exchange_type if isinstance(forecast_exchange_type, str) else None,
            routing_key=forecast_routing_key if isinstance(forecast_routing_key, str) else None,
        ),
        output_config=_rabbit_config(
            rabbit_cfg,
            safety_module.ALERT_SCHEMA,
            queue=output_queue,
            exchange=output_exchange if isinstance(output_exchange, str) else None,
            exchange_type=output_exchange_type if isinstance(output_exchange_type, str) else None,
            routing_key=output_routing_key if isinstance(output_routing_key, str) else None,
        ),
        thaw_input_queue=thaw_queue,
        forecast_input_queue=forecast_queue,
        output_queue=output_queue,
        limit_radius_m=float(safety_cfg.get("limit_radius_m", 0.1)),
        metric=str(safety_cfg.get("metric", "max")),
        publish_on_change=bool(safety_cfg.get("publish_on_change", False)),
        site_id=str(safety_cfg.get("site_id", "default")),
    )


def make_runner(builder: Callable[[], object]) -> Callable[..., None]:
    """Wrap a server builder into a runnable callable."""

    def _runner(*, ok_queue=None) -> None:
        server = builder()
        if ok_queue is not None:
            ok_queue.put("OK")
        server.start()

    return _runner


def make_observation_ingestion_2d_runner(
    influx_config: InfluxConfig,
    rabbit_cfg: Dict[str, str],
    obs_cfg: Dict[str, object],
) -> Callable[[], None]:
    return make_runner(lambda: create_observation_ingestion_2d_server(influx_config, rabbit_cfg, obs_cfg))


def make_boundary_condition_builder_runner(
    rabbit_cfg: Dict[str, str],
    bc_cfg: Dict[str, object],
) -> Callable[[], None]:
    return make_runner(lambda: create_boundary_condition_builder_server(rabbit_cfg, bc_cfg))


def make_thaw_front_reconstruction_runner(
    influx_config: InfluxConfig,
    rabbit_cfg: Dict[str, str],
    thaw_cfg: Dict[str, object],
) -> Callable[[], None]:
    return make_runner(lambda: create_thaw_front_reconstruction_server(influx_config, rabbit_cfg, thaw_cfg))


def make_fem_forecast_runner(
    influx_config: InfluxConfig,
    rabbit_cfg: Dict[str, str],
    fem_cfg: Dict[str, object],
) -> Callable[[], None]:
    return make_runner(lambda: create_fem_forecast_server(influx_config, rabbit_cfg, fem_cfg))


def make_safety_monitor_runner(
    influx_config: InfluxConfig,
    rabbit_cfg: Dict[str, str],
    safety_cfg: Dict[str, object],
) -> Callable[[], None]:
    return make_runner(lambda: create_safety_monitor_server(influx_config, rabbit_cfg, safety_cfg))


def create_heater_actuation_server(
    influx_config: InfluxConfig,
    rabbit_cfg: Dict[str, str],
    heater_cfg: Dict[str, object],
):
    input_queue = str(heater_cfg.get("input_queue", heater_module.ALERT_QUEUE))
    input_exchange = heater_cfg.get("input_exchange")
    input_exchange_type = heater_cfg.get("input_exchange_type")
    input_routing_key = heater_cfg.get("input_routing_key")

    command_queue = str(heater_cfg.get("command_queue", heater_module.COMMAND_QUEUE))
    command_exchange = heater_cfg.get("command_exchange")
    command_exchange_type = heater_cfg.get("command_exchange_type")
    command_routing_key = heater_cfg.get("command_routing_key")

    action_queue = str(heater_cfg.get("action_queue", heater_module.ACTION_QUEUE))
    action_exchange = heater_cfg.get("action_exchange")
    action_exchange_type = heater_cfg.get("action_exchange_type")
    action_routing_key = heater_cfg.get("action_routing_key")

    return heater_module.HeaterActuationServer(
        influx_config=influx_config,
        input_config=_rabbit_config(
            rabbit_cfg,
            heater_module.ALERT_SCHEMA,
            queue=input_queue,
            exchange=input_exchange if isinstance(input_exchange, str) else None,
            exchange_type=input_exchange_type if isinstance(input_exchange_type, str) else None,
            routing_key=input_routing_key if isinstance(input_routing_key, str) else None,
        ),
        command_config=_rabbit_config(
            rabbit_cfg,
            heater_module.COMMAND_SCHEMA,
            queue=command_queue,
            exchange=command_exchange if isinstance(command_exchange, str) else None,
            exchange_type=command_exchange_type if isinstance(command_exchange_type, str) else None,
            routing_key=command_routing_key if isinstance(command_routing_key, str) else None,
        ),
        action_config=_rabbit_config(
            rabbit_cfg,
            heater_module.ACTION_SCHEMA,
            queue=action_queue,
            exchange=action_exchange if isinstance(action_exchange, str) else None,
            exchange_type=action_exchange_type if isinstance(action_exchange_type, str) else None,
            routing_key=action_routing_key if isinstance(action_routing_key, str) else None,
        ),
        input_queue=input_queue,
        command_queue=command_queue,
        action_queue=action_queue,
        action_on_alert=str(heater_cfg.get("action_on_alert", "stop")),
        action_on_ok=str(heater_cfg.get("action_on_ok", "hold")),
        setpoint_on_alert=heater_cfg.get("setpoint_on_alert"),
        setpoint_on_ok=heater_cfg.get("setpoint_on_ok"),
        publish_on_change=bool(heater_cfg.get("publish_on_change", False)),
        site_id=str(heater_cfg.get("site_id", "default")),
    )


def make_heater_actuation_runner(
    influx_config: InfluxConfig,
    rabbit_cfg: Dict[str, str],
    heater_cfg: Dict[str, object],
) -> Callable[[], None]:
    return make_runner(lambda: create_heater_actuation_server(influx_config, rabbit_cfg, heater_cfg))


def make_viz_dashboard_runner() -> Callable[[], None]:
    def _runner(*, ok_queue=None) -> None:
        if ok_queue is not None:
            ok_queue.put("OK")
        viz_dashboard_module.main()

    return _runner


# Placeholder: additional service factories and runners will be reintroduced
# once the new 2D thaw-front architecture services are implemented.

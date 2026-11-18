"""Factory functions for starting digital-twin services."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional
from multiprocessing import Manager

from software.digital_twin.communication.messaging import RabbitMQConfig
from software.digital_twin.data_access.influx_utils import InfluxConfig
import software.digital_twin.monitoring.observation_ingestion.observation_ingestion_server as obs_2d_module
import software.digital_twin.monitoring.boundary_condition_builder.boundary_condition_builder_server as bc_module
import software.digital_twin.monitoring.thaw_front_reconstruction.thaw_front_reconstruction_server as thaw_module
import software.digital_twin.simulator.fem_forecast.fem_forecast_server as fem_module
import software.digital_twin.simulator.fem_inversion.fem_inversion_server as inversion_module
import software.digital_twin.simulator.pipe_boundary.pipe_boundary_server as pipe_module
import software.digital_twin.control.safety_monitor.safety_monitor_server as safety_module
import software.digital_twin.control.heater_actuation.heater_actuation_server as heater_module
import software.digital_twin.control.polyscience_control.polyscience_control_server as polyscience_module
import software.digital_twin.monitoring.observation_ingestion.cr1000_publisher_server as cr1000_module
from software.startup import start_viz_dashboard as viz_dashboard_module


# Shared config store using multiprocessing.Manager for cross-process access on Windows
_SERVICE_CONFIG: Optional[Any] = None


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
        rmin_depth_bins=int(thaw_cfg.get("rmin_depth_bins", 8)),
        surface_exclusion_m=float(thaw_cfg.get("surface_exclusion_m", 0.03)),
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
        track_observed_time=bool(fem_cfg.get("track_observed_time", True)),
        max_step_hours=float(fem_cfg.get("max_step_hours", 0.5)),
        site_id=str(fem_cfg.get("site_id", "default")),
        mesh_dir=mesh_dir,
        node_file=str(fem_cfg.get("node_file", "DT_model_nodes.txt")),
        element_file=str(fem_cfg.get("element_file", "DT_model_elements.txt")),
        edge_sets=fem_cfg.get("edge_sets", {}) if isinstance(fem_cfg.get("edge_sets"), dict) else {},
        node_sets=fem_cfg.get("node_sets", {}) if isinstance(fem_cfg.get("node_sets"), dict) else {},
        material=fem_cfg.get("material", {}) if isinstance(fem_cfg.get("material"), dict) else {},
        solver=fem_cfg.get("solver", {}) if isinstance(fem_cfg.get("solver"), dict) else {},
        save_field_csv_default=fem_cfg.get("save_field_csv_default", False),
        include_field_default=fem_cfg.get("include_field_default", False),
        sticky_material_override=fem_cfg.get("sticky_material_override", True),
        field_csv_dir=fem_cfg.get("field_csv_dir") if isinstance(fem_cfg.get("field_csv_dir"), str) else None,
        run_baseline=fem_cfg.get("run_baseline", False),
        enable_lookahead=bool(fem_cfg.get("enable_lookahead", True)),
        lookahead_hours=float(fem_cfg.get("lookahead_hours", 1.0)),
        lookahead_samples=int(fem_cfg.get("lookahead_samples", 12)),
        lookahead_stop_setpoint_c=float(fem_cfg.get("lookahead_stop_setpoint_c", 0.0)),
        pipe_boundary_set=str(fem_cfg.get("pipe_boundary_set", "left")),
    )


def create_fem_inversion_server(
    influx_config: InfluxConfig,
    rabbit_cfg: Dict[str, str],
    inversion_cfg: Dict[str, object],
    bc_cfg: Dict[str, object],
    fem_cfg: Dict[str, object],
):
    input_queue = str(inversion_cfg.get("input_queue", inversion_module.SENSOR_QUEUE))
    input_exchange = inversion_cfg.get("input_exchange")
    input_exchange_type = inversion_cfg.get("input_exchange_type")
    input_routing_key = inversion_cfg.get("input_routing_key")

    output_queue = str(inversion_cfg.get("output_queue", inversion_module.INVERSION_QUEUE))
    output_exchange = inversion_cfg.get("output_exchange")
    output_exchange_type = inversion_cfg.get("output_exchange_type")
    output_routing_key = inversion_cfg.get("output_routing_key")

    fem_input_queue = str(inversion_cfg.get("fem_input_queue", fem_cfg.get("input_queue", inversion_module.FEM_BOUNDARY_QUEUE)))

    invert_params = inversion_cfg.get("invert_params", inversion_module.DEFAULT_INVERT_PARAMS)
    if not isinstance(invert_params, list):
        invert_params = list(inversion_module.DEFAULT_INVERT_PARAMS)
    invert_params = [str(name) for name in invert_params]

    relative_bounds = inversion_cfg.get("relative_bounds", {})
    if not isinstance(relative_bounds, dict):
        relative_bounds = {}

    boundaries = inversion_cfg.get("boundaries", bc_cfg.get("boundaries", []))
    if not isinstance(boundaries, list):
        boundaries = []
    sensor_overrides = inversion_cfg.get("sensor_overrides", bc_cfg.get("sensor_overrides", []))
    if not isinstance(sensor_overrides, list):
        sensor_overrides = []

    base_params = inversion_cfg.get("base_params", fem_cfg.get("material", {}))
    if not isinstance(base_params, dict):
        base_params = {}

    return inversion_module.FEMInversionServer(
        influx_config=influx_config,
        input_config=_rabbit_config(
            rabbit_cfg,
            inversion_module.SENSOR_SCHEMA,
            queue=input_queue,
            exchange=input_exchange if isinstance(input_exchange, str) else None,
            exchange_type=input_exchange_type if isinstance(input_exchange_type, str) else None,
            routing_key=input_routing_key if isinstance(input_routing_key, str) else None,
        ),
        output_config=_rabbit_config(
            rabbit_cfg,
            inversion_module.INVERSION_SCHEMA,
            queue=output_queue,
            exchange=output_exchange if isinstance(output_exchange, str) else None,
            exchange_type=output_exchange_type if isinstance(output_exchange_type, str) else None,
            routing_key=output_routing_key if isinstance(output_routing_key, str) else None,
        ),
        fem_input_config=_rabbit_config(
            rabbit_cfg,
            inversion_module.FEM_BOUNDARY_SCHEMA,
            queue=fem_input_queue,
        ),
        input_queue=input_queue,
        output_queue=output_queue,
        fem_input_queue=fem_input_queue,
        site_id=str(inversion_cfg.get("site_id", "default")),
        invert_params=invert_params,
        base_params=base_params,
        relative_bounds=relative_bounds,
        loss=str(inversion_cfg.get("loss", "mse")),
        window_size=int(inversion_cfg.get("window_size", 5)),
        window_stride=int(inversion_cfg.get("window_stride", 1)),
        min_snapshots=int(inversion_cfg.get("min_snapshots", 5)),
        min_sensors=int(inversion_cfg.get("min_sensors", 3)),
        min_sensor_radius_m=float(inversion_cfg.get("min_sensor_radius_m", 0.0)),
        min_update_hours=float(inversion_cfg.get("min_update_hours", 0.0)),
        min_update_seconds=float(inversion_cfg.get("min_update_seconds", 0.0)),
        warmup_hours=float(inversion_cfg.get("warmup_hours", 0.5)),
        relative_time_window=bool(inversion_cfg.get("relative_time_window", True)),
        min_horizon_hours=float(inversion_cfg.get("min_horizon_hours", 1e-3)),
        maxiter=int(inversion_cfg.get("maxiter", 40)),
        maxfun=int(inversion_cfg.get("maxfun", 80)),
        scale_space=str(inversion_cfg.get("scale_space", "linear")),
        prior_params=inversion_cfg.get("prior_params") if isinstance(inversion_cfg.get("prior_params"), dict) else None,
        prior_weight=float(inversion_cfg.get("prior_weight", 0.0)),
        seed=int(inversion_cfg.get("seed", 42)) if inversion_cfg.get("seed") is not None else None,
        publish_material_override_boundary=bool(inversion_cfg.get("publish_material_override_boundary", False)),
        boundaries=boundaries,
        sensor_overrides=sensor_overrides,
        config_path=str(inversion_cfg.get("config_path", "startup.conf")),
    )


def create_pipe_boundary_server(
    influx_config: InfluxConfig,
    rabbit_cfg: Dict[str, str],
    pipe_cfg: Dict[str, object],
):
    boundary_in_queue = str(pipe_cfg.get("input_queue", pipe_module.BOUNDARY_IN_QUEUE))
    boundary_out_queue = str(pipe_cfg.get("output_queue", pipe_module.BOUNDARY_OUT_QUEUE))
    fluid_queue = str(pipe_cfg.get("fluid_queue", pipe_module.FLUID_QUEUE))
    fluid_exchange = str(pipe_cfg.get("fluid_exchange", pipe_module.FLUID_EXCHANGE))
    fluid_exchange_type = str(pipe_cfg.get("fluid_exchange_type", "fanout"))
    calibration_queue = str(pipe_cfg.get("calibration_queue", pipe_module.CALIBRATION_QUEUE))

    initial_params = pipe_cfg.get("initial_params")
    lower_bounds = pipe_cfg.get("lower_bounds")
    upper_bounds = pipe_cfg.get("upper_bounds")
    fixed_params = pipe_cfg.get("fixed_params")

    return pipe_module.PipeBoundaryServer(
        influx_config=influx_config,
        boundary_in_config=_rabbit_config(rabbit_cfg, pipe_module.BOUNDARY_SCHEMA, queue=boundary_in_queue),
        boundary_out_config=_rabbit_config(rabbit_cfg, pipe_module.BOUNDARY_SCHEMA, queue=boundary_out_queue),
        fluid_config=_rabbit_config(
            rabbit_cfg,
            pipe_module.FLUID_SCHEMA,
            queue=fluid_queue,
            exchange=fluid_exchange,
            exchange_type=fluid_exchange_type,
        ),
        calibration_config=_rabbit_config(rabbit_cfg, pipe_module.CALIBRATION_SCHEMA, queue=calibration_queue),
        boundary_in_queue=boundary_in_queue,
        boundary_out_queue=boundary_out_queue,
        fluid_queue=fluid_queue,
        calibration_queue=calibration_queue,
        site_id=str(pipe_cfg.get("site_id", "default")),
        h_sample_m=float(pipe_cfg.get("h_sample_m", 0.30)),
        r_bend_m=float(pipe_cfg.get("r_bend_m", 0.0)),
        n_cells=int(pipe_cfg.get("n_cells", 120)),
        max_dt_s=float(pipe_cfg.get("max_dt_s", 5.0)),
        max_steps=int(pipe_cfg.get("max_steps", 3000)),
        pipe_boundary_set=str(pipe_cfg.get("pipe_boundary_set", "left")),
        pipe_bc_type=str(pipe_cfg.get("pipe_bc_type", "robin")),
        pipe_h_coeff=float(pipe_cfg.get("pipe_h_coeff", 5.0)),
        interface_source=str(pipe_cfg.get("interface_source", "fluid")),
        interface_reduction=str(pipe_cfg.get("interface_reduction", "mean")),
        depth_resolved=bool(pipe_cfg.get("depth_resolved", True)),
        profile_points=int(pipe_cfg.get("profile_points", 17)),
        window_seconds=float(pipe_cfg.get("window_seconds", 7200.0)),
        min_points=int(pipe_cfg.get("min_points", 5)),
        max_points=int(pipe_cfg.get("max_points", 240)),
        recalibrate_seconds=float(pipe_cfg.get("recalibrate_seconds", 300.0)),
        maxiter=int(pipe_cfg.get("maxiter", 40)),
        maxfun=int(pipe_cfg.get("maxfun", 80)),
        initial_params=initial_params if isinstance(initial_params, dict) else None,
        lower_bounds=lower_bounds if isinstance(lower_bounds, dict) else None,
        upper_bounds=upper_bounds if isinstance(upper_bounds, dict) else None,
        fixed_params=fixed_params if isinstance(fixed_params, dict) else None,
        warm_start=bool(pipe_cfg.get("warm_start", True)),
        passthrough_until_ready=bool(pipe_cfg.get("passthrough_until_ready", True)),
        publish_calibration=bool(pipe_cfg.get("publish_calibration", True)),
        publish_outlet_validation=bool(pipe_cfg.get("publish_outlet_validation", True)),
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
        mode=str(safety_cfg.get("mode", "single")),
        limit_min_radius_m=float(safety_cfg.get("limit_min_radius_m", 0.0)),
        warn_fraction=float(safety_cfg.get("warn_fraction", 0.6)),
        critical_fraction=float(safety_cfg.get("critical_fraction", 0.9)),
        publish_on_change=bool(safety_cfg.get("publish_on_change", False)),
        site_id=str(safety_cfg.get("site_id", "default")),
        confirm_updates=int(safety_cfg.get("confirm_updates", 3)),
        release_fraction=float(safety_cfg.get("release_fraction", 0.85)),
        forecast_max_age_hours=float(safety_cfg.get("forecast_max_age_hours", 0.05)),
        measurement_max_age_hours=float(safety_cfg.get("measurement_max_age_hours", 0.05)),
        fail_safe_on_stale_measurement=bool(
            safety_cfg.get("fail_safe_on_stale_measurement", True)),
        latch_alert=bool(safety_cfg.get("latch_alert", True)),
        f2_action=str(safety_cfg.get("f2_action", "stop")),
    )


def create_cr1000_publisher_server(
    rabbit_cfg: Dict[str, str],
    cr1000_cfg: Dict[str, object],
    influx_cfg: Optional[Dict[str, str] | InfluxConfig] = None,
):
    exchange = str(cr1000_cfg.get("exchange", cr1000_module.SENSOR_EXCHANGE))
    exchange_type = str(cr1000_cfg.get("exchange_type", "fanout"))
    mq_config = RabbitMQConfig(
        host=rabbit_cfg.get("host", "localhost"),
        queue="",
        exchange=exchange,
        exchange_type=exchange_type,
        schema_path=cr1000_module.SCHEMA_PATH,
        username=rabbit_cfg.get("username", "permafrost"),
        password=rabbit_cfg.get("password", "permafrost"),
    )
    sensors = cr1000_cfg.get("sensors", [])
    fluid_fields = cr1000_cfg.get("fluid_fields")
    fluid_exchange = cr1000_cfg.get("fluid_exchange", cr1000_module.FLUID_EXCHANGE)
    if not fluid_fields:
        influx_config = None
    elif isinstance(influx_cfg, InfluxConfig):
        influx_config = influx_cfg
    elif influx_cfg:
        influx_config = build_influx_config(influx_cfg)
    else:
        influx_config = None
    return cr1000_module.CR1000PublisherServer(
        port=str(cr1000_cfg.get("port", "COM3")),
        baud_rate=int(cr1000_cfg.get("baud_rate", 115200)),
        table_name=str(cr1000_cfg.get("table_name", "Table1")),
        sensor_map=sensors if isinstance(sensors, list) else [],
        rabbitmq_config=mq_config,
        pakbus_addr=int(cr1000_cfg.get("pakbus_addr", 1)),
        poll_interval_s=float(cr1000_cfg.get("poll_interval_s", 30.0)),
        site_id=str(cr1000_cfg.get("site_id", "default")),
        fluid_fields=fluid_fields if isinstance(fluid_fields, dict) else None,
        fluid_exchange=str(fluid_exchange) if fluid_exchange else None,
        influx_config=influx_config,
        records_per_poll=int(cr1000_cfg.get("records_per_poll", 3)),
        connect_timeout_s=float(cr1000_cfg.get("connect_timeout_s", 60.0)),
        call_timeout_s=float(cr1000_cfg.get("call_timeout_s", 120.0)),
        reconnect_delay_s=float(cr1000_cfg.get("reconnect_delay_s", 15.0)),
    )


def create_polyscience_control_server(
    rabbit_cfg: Dict[str, str],
    polyscience_cfg: Dict[str, object],
    influx_cfg: Optional[Dict[str, str] | InfluxConfig] = None,
):
    """Build the PolyScience circulator actuation service (RS232 hardware bridge).

    Consumes ``heater_command_message`` payloads (the same queue the
    ``heater_actuation`` service publishes to) and drives the physical
    controller. NOTE: its serial ``port`` must differ from the CR1000
    publisher's port — pyserial opens ports exclusively. ``start_all_services``
    enforces this at launch.
    """
    command_queue = str(polyscience_cfg.get("command_queue", polyscience_module.COMMAND_QUEUE))
    command_config = _rabbit_config(rabbit_cfg, polyscience_module.COMMAND_SCHEMA, queue=command_queue)

    if isinstance(influx_cfg, InfluxConfig):
        influx_config = influx_cfg
    elif influx_cfg:
        influx_config = build_influx_config(influx_cfg)
    else:
        influx_config = None

    action_map = polyscience_cfg.get("action_map")

    return polyscience_module.PolyScienceControlServer(
        port=str(polyscience_cfg.get("port", "COM3")),
        baud_rate=int(polyscience_cfg.get("baud_rate", 57600)),
        command_config=command_config,
        influx_config=influx_config,
        command_queue=command_queue,
        action_map=action_map if isinstance(action_map, dict) else None,
        setpoint_min_c=float(polyscience_cfg.get("setpoint_min_c", -30.0)),
        setpoint_max_c=float(polyscience_cfg.get("setpoint_max_c", 50.0)),
        serial_timeout_s=float(polyscience_cfg.get("serial_timeout_s", 3.0)),
        inter_command_delay_s=float(polyscience_cfg.get("inter_command_delay_s", 0.2)),
        open_retries=int(polyscience_cfg.get("open_retries", 3)),
        open_retry_delay_s=float(polyscience_cfg.get("open_retry_delay_s", 2.0)),
        verify_after_apply=bool(polyscience_cfg.get("verify_after_apply", True)),
        stop_on_shutdown=bool(polyscience_cfg.get("stop_on_shutdown", True)),
        site_id=str(polyscience_cfg.get("site_id", "default")),
        dry_run=bool(polyscience_cfg.get("dry_run", False)),
    )


def initialize_service_config(
    influx_config: InfluxConfig,
    rabbit_cfg: Dict[str, str],
    obs_2d_cfg: Dict[str, object],
    bc_cfg: Dict[str, object],
    thaw_cfg: Dict[str, object],
    inversion_cfg: Dict[str, object],
    fem_cfg: Dict[str, object],
    safety_cfg: Dict[str, object],
    heater_cfg: Dict[str, object],
    cr1000_cfg: Optional[Dict[str, object]] = None,
    pipe_boundary_cfg: Optional[Dict[str, object]] = None,
    polyscience_cfg: Optional[Dict[str, object]] = None,
) -> Any:
    """
    Initialize and return a shared config dict using multiprocessing.Manager.
    This dict can be safely shared across processes on both Linux and Windows.
    """
    manager = Manager()
    config_dict = manager.dict()
    config_dict.update({
        "influx_config": influx_config,
        "rabbit_cfg": rabbit_cfg,
        "obs_2d_cfg": obs_2d_cfg,
        "bc_cfg": bc_cfg,
        "thaw_cfg": thaw_cfg,
        "inversion_cfg": inversion_cfg,
        "fem_cfg": fem_cfg,
        "safety_cfg": safety_cfg,
        "heater_cfg": heater_cfg,
        "cr1000_cfg": cr1000_cfg or {},
        "pipe_boundary_cfg": pipe_boundary_cfg or {},
        "polyscience_cfg": polyscience_cfg or {},
    })
    return config_dict


def _get_service_config(config_dict: Any, key: str) -> Any:
    """Retrieve a config value from the shared config dict."""
    if not config_dict:
        raise RuntimeError(
            f"Service config not provided. Tried to access: {key}"
        )
    return config_dict[key]


# Top-level runner functions (pickleable, no closures)
def _observation_ingestion_2d_runner(*, ok_queue=None, config_dict=None) -> None:
    """Observation ingestion 2D service runner."""
    server = create_observation_ingestion_2d_server(
        _get_service_config(config_dict, "influx_config"),
        _get_service_config(config_dict, "rabbit_cfg"),
        _get_service_config(config_dict, "obs_2d_cfg"),
    )
    if ok_queue is not None:
        ok_queue.put("OK")
    server.start()


def _boundary_condition_builder_runner(*, ok_queue=None, config_dict=None) -> None:
    """Boundary condition builder service runner."""
    server = create_boundary_condition_builder_server(
        _get_service_config(config_dict, "rabbit_cfg"),
        _get_service_config(config_dict, "bc_cfg"),
    )
    if ok_queue is not None:
        ok_queue.put("OK")
    server.start()


def _thaw_front_reconstruction_runner(*, ok_queue=None, config_dict=None) -> None:
    """Thaw front reconstruction service runner."""
    server = create_thaw_front_reconstruction_server(
        _get_service_config(config_dict, "influx_config"),
        _get_service_config(config_dict, "rabbit_cfg"),
        _get_service_config(config_dict, "thaw_cfg"),
    )
    if ok_queue is not None:
        ok_queue.put("OK")
    server.start()


def _fem_forecast_runner(*, ok_queue=None, config_dict=None) -> None:
    """FEM forecast service runner."""
    server = create_fem_forecast_server(
        _get_service_config(config_dict, "influx_config"),
        _get_service_config(config_dict, "rabbit_cfg"),
        _get_service_config(config_dict, "fem_cfg"),
    )
    if ok_queue is not None:
        ok_queue.put("OK")
    server.start()


def _fem_inversion_runner(*, ok_queue=None, config_dict=None) -> None:
    """FEM inversion service runner."""
    server = create_fem_inversion_server(
        _get_service_config(config_dict, "influx_config"),
        _get_service_config(config_dict, "rabbit_cfg"),
        _get_service_config(config_dict, "inversion_cfg"),
        _get_service_config(config_dict, "bc_cfg"),
        _get_service_config(config_dict, "fem_cfg"),
    )
    if ok_queue is not None:
        ok_queue.put("OK")
    server.start()


def _safety_monitor_runner(*, ok_queue=None, config_dict=None) -> None:
    """Safety monitor service runner."""
    server = create_safety_monitor_server(
        _get_service_config(config_dict, "influx_config"),
        _get_service_config(config_dict, "rabbit_cfg"),
        _get_service_config(config_dict, "safety_cfg"),
    )
    if ok_queue is not None:
        ok_queue.put("OK")
    server.start()


def _cr1000_publisher_runner(*, ok_queue=None, config_dict=None) -> None:
    """CR1000 hardware publisher service runner."""
    server = create_cr1000_publisher_server(
        _get_service_config(config_dict, "rabbit_cfg"),
        _get_service_config(config_dict, "cr1000_cfg"),
        _get_service_config(config_dict, "influx_config"),
    )
    if ok_queue is not None:
        ok_queue.put("OK")
    server.start()


def _polyscience_control_runner(*, ok_queue=None, config_dict=None) -> None:
    """PolyScience circulator actuation hardware bridge runner."""
    server = create_polyscience_control_server(
        _get_service_config(config_dict, "rabbit_cfg"),
        _get_service_config(config_dict, "polyscience_cfg"),
        _get_service_config(config_dict, "influx_config"),
    )
    if ok_queue is not None:
        ok_queue.put("OK")
    server.start()


def _pipe_boundary_runner(*, ok_queue=None, config_dict=None) -> None:
    """Pipe boundary enrichment service runner."""
    server = create_pipe_boundary_server(
        _get_service_config(config_dict, "influx_config"),
        _get_service_config(config_dict, "rabbit_cfg"),
        _get_service_config(config_dict, "pipe_boundary_cfg"),
    )
    if ok_queue is not None:
        ok_queue.put("OK")
    server.start()


def _heater_actuation_runner(*, ok_queue=None, config_dict=None) -> None:
    """Heater actuation service runner."""
    server = create_heater_actuation_server(
        _get_service_config(config_dict, "influx_config"),
        _get_service_config(config_dict, "rabbit_cfg"),
        _get_service_config(config_dict, "heater_cfg"),
    )
    if ok_queue is not None:
        ok_queue.put("OK")
    server.start()


def _viz_dashboard_runner(*, ok_queue=None, config_dict=None) -> None:
    """Visualization dashboard service runner."""
    if ok_queue is not None:
        ok_queue.put("OK")
    viz_dashboard_module.main()


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
        setpoint_on_reduce=heater_cfg.get("setpoint_on_reduce"),
        publish_on_change=bool(heater_cfg.get("publish_on_change", False)),
        site_id=str(heater_cfg.get("site_id", "default")),
    )


# Placeholder: additional service factories and runners will be reintroduced
# once the new 2D thaw-front architecture services are implemented.

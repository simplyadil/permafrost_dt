"""Configuration helpers for startup scripts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from software.utils.logging_setup import configure_logging as configure_runtime_logging

DEFAULT_CONFIG: Dict[str, Any] = {
    "rabbitmq": {
        "host": "localhost",
        "username": "permafrost",
        "password": "permafrost",
    },
    "influxdb": {
        "url": "http://localhost:8086",
        "token": "permafrost",
        "org": "permafrost",
        "bucket": "permafrost_data",
        "username": "permafrost",
        "password": "permafrost",
    },
    "observation_ingestion_2d": {
        "queue": "permafrost.record.sensors.2d.ingest",
        "exchange": "permafrost.sensors.2d",
        "exchange_type": "fanout",
        "site_id": "default",
    },
    "boundary_condition_builder": {
        "input_queue": "permafrost.record.sensors.2d.bc",
        "input_exchange": "permafrost.sensors.2d",
        "input_exchange_type": "fanout",
        "output_queue": "permafrost.record.boundary.2d",
        "boundaries": [
            {
                "name": "heater_top",
                "boundary_set": "top",
                "bc_type": "robin",
                "temperature": 105.0,
                "h_coeff": 7.0,
            },
            {
                "name": "heater_left",
                "boundary_set": "left",
                "bc_type": "robin",
                "temperature": 105.0,
                "h_coeff": 7.0,
            },
            {
                "name": "insulated_bottom",
                "boundary_set": "bottom",
                "bc_type": "neumann",
                "heat_flux": 0.0,
            },
            {
                "name": "insulated_right",
                "boundary_set": "right",
                "bc_type": "neumann",
                "heat_flux": 0.0,
            },
        ],
        "sensor_overrides": [],
    },
    "thaw_front_reconstruction": {
        "input_queue": "permafrost.record.sensors.2d.thaw",
        "input_exchange": "permafrost.sensors.2d",
        "input_exchange_type": "fanout",
        "output_queue": "permafrost.record.thaw_front.2d",
        "thaw_threshold_c": 0.0,
        "site_id": "default",
    },
    "pinn_inversion_2d": {},
    "fem_forecast": {
        "input_queue": "permafrost.record.boundary.2d",
        "output_queue": "permafrost.record.fem_forecast.2d",
        "horizon_hours": 1.0,
        "site_id": "default",
        "mesh_dir": "software/models/fem_mesh",
        "node_file": "DT_model_nodes.txt",
        "element_file": "DT_model_elements.txt",
        "edge_sets": {
            "top": "DT_model_top_edge.txt",
            "bottom": "DT_model_bottom_edge.txt",
            "left": "DT_model_left_edge.txt",
            "right": "DT_model_right_edge.txt",
        },
        "node_sets": {
            "top": "DT_model_top_nodes.txt",
            "bottom": "DT_model_bottom_nodes.txt",
            "left": "DT_model_left_nodes.txt",
            "right": "DT_model_right_nodes.txt",
        },
        "material": {
            "porosity": 0.42,
            "rho_s": 2700.0,
            "rho_w": 1000.0,
            "rho_i": 920.0,
            "cap_s": 790.0,
            "cap_w": 4180.0,
            "cap_i": 2090.0,
            "latent_heat": 334000.0,
            "k_parameter": 5.0,
            "tr_k": 273.15,
            "lambda_thawed": 1.64,
            "lambda_frozen": 2.96,
        },
        "solver": {
            "dt_seconds": 36.0,
            "max_iterations": 25,
            "tol_r": 1e-6,
            "temperature_offset_k": 273.15,
            "initial_temperature_c": -1.0,
            "thaw_threshold_c": 0.0,
        },
    },
    "safety_monitor": {
        "thaw_input_queue": "permafrost.record.thaw_front.2d",
        "forecast_input_queue": "permafrost.record.fem_forecast.2d",
        "output_queue": "permafrost.record.safety_alerts.2d",
        "limit_radius_m": 0.1,
        "metric": "max",
        "publish_on_change": False,
        "site_id": "default",
    },
    "heater_actuation": {
        "input_queue": "permafrost.record.safety_alerts.2d",
        "command_queue": "permafrost.command.heater.2d",
        "action_queue": "permafrost.record.heater_actions.2d",
        "action_on_alert": "stop",
        "action_on_ok": "hold",
        "setpoint_on_alert": 0.0,
        "setpoint_on_ok": None,
        "publish_on_change": False,
        "site_id": "default",
    },
    "viz_gateway": {},
    "viz_dashboard": {
        "host": "0.0.0.0",
        "port": 8501,
        "refresh_seconds": 20,
        "history_hours": 2.0,
        "max_points": 3000,
        "r_limit_m": 0.1,
        "safe_pct": 0.8,
        "caution_pct": 0.95,
        "site_id": "default",
    },
}


def load_startup_config(path: Path | str = Path("startup.conf")) -> Dict[str, Any]:
    """Load the startup configuration file, falling back to defaults."""

    config_path = Path(path)
    if not config_path.exists():
        return DEFAULT_CONFIG.copy()

    text = config_path.read_text(encoding="utf-8")
    stripped = "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))
    if not stripped.strip():
        return DEFAULT_CONFIG.copy()

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        logging.getLogger("startup").warning("Invalid startup.conf JSON. Using defaults.")
        return DEFAULT_CONFIG.copy()

    merged = DEFAULT_CONFIG.copy()
    merged.update(data)
    return merged


def configure_logging(config_path: Path | str = Path("logging.conf")) -> None:
    """Initialise logging using the shared configuration helper."""

    configure_runtime_logging()

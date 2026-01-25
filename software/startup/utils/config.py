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
    "observation_ingestion_2d": {},
    "boundary_and_forcing_builder": {},
    "thaw_front_reconstruction": {},
    "pinn_inversion_2d": {},
    "fem_forecast": {},
    "safety_monitor": {},
    "heater_actuation": {},
    "viz_gateway": {},
    "viz_dashboard": {},
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

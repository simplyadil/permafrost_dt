"""Configuration helpers for startup scripts."""

from __future__ import annotations

import json
import logging
import logging.config
from pathlib import Path
from typing import Any, Dict

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
    "synthetic_observations": {
        "enabled": True,
        "depths_m": [0, 1, 2, 3, 4, 5],
    },
    "synthetic_boundary": {
        "enabled": True,
        "start_day": 0.0,
        "step_days": 1.0,
    },
    "pinn_forward": {
        "enable_training": False,
        "model_path": "software/models/pinn_forward/freezing_soil_pinn.pt",
    },
    "pinn_inversion": {
        "enable_training": False,
        "model_path": "software/models/pinn_inversion/inversion_pinn.pt",
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
    """Configure standard logging based on the provided config file."""

    path = Path(config_path)
    if path.exists():
        logging.config.fileConfig(path, disable_existing_loggers=False)
    else:
        logging.basicConfig(level=logging.INFO)

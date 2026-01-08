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
    "synthetic_observations": {
        "enabled": True,
        "depths_m": [0, 1, 2, 3, 4, 5],
    },
    "synthetic_boundary": {
        "enabled": True,
        "start_day": 0.0,
        # Increase simulated time advance so we cover a year quickly (365d / 5d ≈ 73 steps).
        "step_days": 5.0,
    },
    "pinn_forward": {
        "enable_training": False,
        "model_path": "software/models/pinn_forward/freezing_soil_pinn.pt",
        "fdm_fetch_limit": 5000,
    },
    "pinn_inversion": {
        "enable_training": False,
        "model_path": "software/models/pinn_inversion/freezing_soil_pinn_inverse.pt",
        "fdm_fetch_limit": 5000,
    },
    "viz_gateway": {
        "fetch_limit": 20000,
    },
    "viz_dashboard": {
        "host": "0.0.0.0",
        "port": 8050,
        "refresh_seconds": 60.0,
        "snapshot_dir": "software/outputs",
        "fetch_limit": 20000,
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

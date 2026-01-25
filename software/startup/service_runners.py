"""Factory functions for starting digital-twin services."""

from __future__ import annotations

from typing import Dict

from software.digital_twin.communication.messaging import RabbitMQConfig
from software.digital_twin.data_access.influx_utils import InfluxConfig


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


# Placeholder: service factories and runners will be reintroduced once the
# new 2D thaw-front architecture services are implemented.

"""Centralised logging configuration for the digital twin services."""

from __future__ import annotations

import json
import logging
import logging.config
import os
from typing import Any, Dict, Optional


_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Emit log records as structured JSON for downstream analysis."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "service": record.name,
            "module": record.module,
            "message": record.getMessage(),
        }
        if hasattr(record, "event"):
            payload["event"] = record.event
        if hasattr(record, "details"):
            payload["details"] = record.details
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _should_enable_json(enable_json: Optional[bool]) -> bool:
    if enable_json is not None:
        return enable_json
    env_value = os.getenv("DT_LOG_JSON", "").strip().lower()
    return env_value in {"1", "true", "yes", "on"}


def _build_config(enable_json: bool) -> Dict[str, Any]:
    datefmt = "%Y-%m-%d %H:%M:%S.%f"
    common_format = "%(asctime)s %(levelname)s %(name)s : %(message)s"

    formatters: Dict[str, Any] = {
        "console": {
            "format": common_format,
            "datefmt": datefmt,
        },
        "standard": {
            "()": "logging.Formatter",
            "format": common_format,
            "datefmt": datefmt,
        },
        "json": {
            "()": "software.utils.logging_setup.JsonFormatter",
            "datefmt": datefmt,
        },
    }

    handlers: Dict[str, Any] = {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "json" if enable_json else "console",
        },
    }

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "root": {
            "level": "INFO",
            "handlers": ["console"],
        },
    }


def configure_logging(
    *,
    enable_json: Optional[bool] = None,
) -> None:
    """Initialise global logging handlers and formatter configuration."""

    global _CONFIGURED
    if _CONFIGURED:
        return

    json_enabled = _should_enable_json(enable_json)

    config_dict = _build_config(json_enabled)
    logging.config.dictConfig(config_dict)

    for noisy_logger in ("pika", "urllib3", "asyncio", "influxdb_client"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(service_name: str, *, level: Optional[int] = None) -> logging.Logger:
    """Return a logger configured for the provided service name."""

    configure_logging()
    logger = logging.getLogger(service_name)
    if level is not None:
        logger.setLevel(level)
    return logger


__all__ = ["configure_logging", "get_logger"]

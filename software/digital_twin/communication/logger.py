"""Backward compatible logger shim."""

from __future__ import annotations

from software.utils.logging_setup import get_logger


def setup_logger(service_name: str):
    """Deprecated entry point – prefer importing :func:`get_logger` directly."""

    return get_logger(service_name)


__all__ = ["setup_logger"]

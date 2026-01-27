"""Start the boundary condition builder service."""

from __future__ import annotations

from software.startup.service_runners import create_boundary_condition_builder_server
from software.startup.utils.config import configure_logging, load_startup_config


def main() -> None:
    configure_logging()
    config = load_startup_config()
    server = create_boundary_condition_builder_server(
        config.get("rabbitmq", {}),
        config.get("boundary_condition_builder", {}),
    )
    server.start()


if __name__ == "__main__":
    main()

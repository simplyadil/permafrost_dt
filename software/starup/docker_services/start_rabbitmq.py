# software/startup/docker_services/start_rabbitmq.py
from __future__ import annotations

import sys
from pathlib import Path

import requests

if __package__ is None:
    package_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(package_root))
    from software.starup.docker_services.common_docker_utils import (  # type: ignore
        DockerServiceConfig,
        kill_container,
        project_root,
        start_service,
    )
else:
    from .common_docker_utils import DockerServiceConfig, kill_container, project_root, start_service


CONTAINER_NAME = "rabbitmq-server"
HEALTHCHECK_URL = "http://localhost:15672/api/overview"
_AUTH = ("permafrost", "permafrost")


def _rabbitmq_ready() -> bool:
    try:
        response = requests.get(HEALTHCHECK_URL, auth=_AUTH, timeout=5)
    except requests.exceptions.ConnectionError:
        print("Waiting for RabbitMQ API to become available...")
        return False
    except requests.exceptions.Timeout:
        print("Request to RabbitMQ timed out")
        return False
    except requests.exceptions.RequestException as exc:
        print(f"Error checking RabbitMQ: {exc}")
        return False

    if response.status_code == 200:
        print("RabbitMQ ready.")
        return True

    print(f"RabbitMQ returned status {response.status_code}")
    return False


def _service_config() -> DockerServiceConfig:
    root = project_root()
    return DockerServiceConfig(
        container_name=CONTAINER_NAME,
        compose_directory=root / "resources" / "docker" / "rabbitmq",
        log_file=root / "logs" / "rabbitmq.log",
        healthcheck=_rabbitmq_ready,
    )


def start_docker_rabbitmq() -> None:
    """Launch RabbitMQ via docker-compose and wait until it's healthy."""

    start_service(_service_config())


def stop_docker_rabbitmq() -> None:
    kill_container(CONTAINER_NAME)


if __name__ == "__main__":
    start_docker_rabbitmq()

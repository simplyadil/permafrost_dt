# software/startup/docker_services/start_influxdb.py
from __future__ import annotations

import requests

from .common_docker_utils import DockerServiceConfig, kill_container, project_root, start_service


CONTAINER_NAME = "influxdb-server"
HEALTHCHECK_URL = "http://localhost:8086/health"


def _influxdb_ready() -> bool:
    try:
        response = requests.get(HEALTHCHECK_URL, timeout=5)
    except requests.exceptions.ConnectionError:
        print("Waiting for InfluxDB API to become available...")
        return False
    except requests.exceptions.Timeout:
        print("Request to InfluxDB timed out")
        return False
    except requests.exceptions.RequestException as exc:
        print(f"⚠️  Error checking InfluxDB: {exc}")
        return False

    if response.status_code == 200:
        print("InfluxDB ready.")
        return True

    print(f"⚠️  InfluxDB returned status {response.status_code}")
    return False


def _service_config() -> DockerServiceConfig:
    root = project_root()
    return DockerServiceConfig(
        container_name=CONTAINER_NAME,
        compose_directory=root / "software" / "digital_twin" / "data_access" / "influxdbserver",
        log_file=root / "logs" / "influxdb.log",
        healthcheck=_influxdb_ready,
    )


def start_docker_influxdb() -> None:
    """Launch InfluxDB via docker-compose and wait until it's healthy."""

    start_service(_service_config())


def stop_docker_influxdb() -> None:
    kill_container(CONTAINER_NAME)


if __name__ == "__main__":
    start_docker_influxdb()

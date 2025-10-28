# software/startup/docker_services/utils_docker_service_starter.py
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


Healthcheck = Callable[[], bool]


@dataclass(frozen=True)
class DockerServiceConfig:
    """Configuration required to launch and validate a Dockerised service."""

    container_name: str
    compose_directory: Path
    log_file: Path
    healthcheck: Healthcheck
    sleep_time: float = 2.0
    max_attempts: int = 15

    @property
    def compose_file(self) -> Path:
        return self.compose_directory / "docker-compose.yml"


def project_root() -> Path:
    """Return the repository root as a :class:`Path`."""

    return Path(__file__).resolve().parents[3]


def kill_container(name: str) -> None:
    """Stop and remove a container if it exists."""

    print(f"--> Stopping existing container: {name}")
    subprocess.run([
        "docker",
        "rm",
        "-f",
        name,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _run_compose(config: DockerServiceConfig) -> None:
    config.log_file.parent.mkdir(parents=True, exist_ok=True)
    compose_path = config.compose_file

    print(f"--> Starting service using compose at {compose_path}")

    with config.log_file.open("w") as log:
        result = subprocess.run(
            ["docker", "compose", "-f", str(compose_path), "up", "-d"],
            cwd=config.compose_directory,
            stdout=log,
            stderr=subprocess.PIPE,
            text=True,
        )

    if result.returncode != 0:
        print(f"--> Failed to start service: {result.stderr}")
        raise RuntimeError(f"Docker Compose failed with exit code {result.returncode}")


def _wait_for_health(config: DockerServiceConfig) -> None:
    print("--> Checking container logs...")

    for attempt in range(int(config.max_attempts)):
        if config.healthcheck():
            return

        logs = subprocess.run(
            ["docker", "logs", "--tail", "5", config.container_name],
            capture_output=True,
            text=True,
        )
        if logs.returncode == 0 and logs.stdout.strip():
            print(f"--> Recent logs:\n{logs.stdout.strip()}")

        print(f"--> Waiting for service... ({attempt + 1}/{config.max_attempts})")
        time.sleep(config.sleep_time)

    print("--> Service failed to start in time.")
    print("--> Try checking the logs in", config.log_file.resolve())
    raise RuntimeError("Service failed to start in time.")


def start_service(config: DockerServiceConfig) -> None:
    """Start a Docker Compose service and wait for its healthcheck to succeed."""

    kill_container(config.container_name)
    _run_compose(config)
    _wait_for_health(config)


def start(log_file: str, compose_dir: str, test_func, sleep_time: int, max_attempts: int):
    """Backward compatible wrapper for legacy callers."""

    config = DockerServiceConfig(
        container_name=f"{Path(compose_dir).name}-server",
        compose_directory=Path(compose_dir),
        log_file=Path(log_file),
        healthcheck=test_func,
        sleep_time=float(sleep_time),
        max_attempts=int(max_attempts),
    )
    start_service(config)

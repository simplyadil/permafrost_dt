# software/startup/docker_services/start_influxdb.py
import requests

from software.starup.docker_services.common_docker_utils import kill_container, start

CONTAINER_NAME = "influxdb-server"

def start_docker_influxdb():
    """
    Launches InfluxDB via docker-compose and waits until it's healthy.
    """
    log_file = "logs/influxdb.log"
    compose_dir = "resources/docker/influxdb"
    sleep_time = 1
    max_attempts = 10

    def test_connection():
        try:
            r = requests.get("http://localhost:8086/health")
            if r.status_code == 200:
                print("✅ InfluxDB ready.")
                return True
        except requests.exceptions.ConnectionError:
            pass
        return False

    kill_container(CONTAINER_NAME)
    start(
        log_file,
        compose_dir,
        test_connection,
        sleep_time,
        max_attempts
    )

def stop_docker_influxdb():
    kill_container(CONTAINER_NAME)

if __name__ == "__main__":
    start_docker_influxdb()

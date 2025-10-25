# software/startup/docker_services/start_influxdb.py
import os
import sys
import requests

# Add repo root to path when running as script
if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, repo_root)
    from software.starup.docker_services.common_docker_utils import kill_container, start
else:
    from .common_docker_utils import kill_container, start

CONTAINER_NAME = "influxdb-server"

def start_docker_influxdb():
    """
    Launches InfluxDB via docker-compose and waits until it's healthy.
    """
    # Get paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    log_file = os.path.join(project_root, "logs/influxdb.log")
    compose_dir = os.path.join(project_root, "resources/docker/influxdb")
    sleep_time = 2  # Increased to give more time between attempts
    max_attempts = 15  # Increased max attempts and add better error handling

    def test_connection():
        try:
            r = requests.get("http://localhost:8086/health", timeout=5)
            if r.status_code == 200:
                print("✅ InfluxDB ready.")
                return True
            else:
                print(f"⚠️  InfluxDB returned status {r.status_code}")
        except requests.exceptions.ConnectionError:
            print("⏳ Waiting for InfluxDB API to become available...")
        except requests.exceptions.Timeout:
            print("⚠️  Request timed out")
        except Exception as e:
            print(f"⚠️  Error checking InfluxDB: {str(e)}")
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

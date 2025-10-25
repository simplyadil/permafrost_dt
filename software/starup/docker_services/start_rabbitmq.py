# software/startup/docker_services/start_rabbitmq.py
import os
import sys
import requests # type: ignore

# Add repo root to path when running as script
if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, repo_root)
    from software.starup.docker_services.common_docker_utils import kill_container, start
else:
    from .common_docker_utils import kill_container, start

CONTAINER_NAME = "rabbitmq-server"

def start_docker_rabbitmq():
    """
    Launches RabbitMQ via docker-compose and waits until it's healthy.
    """
    # Get paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    log_file = os.path.join(project_root, "logs/rabbitmq.log")
    compose_dir = os.path.join(project_root, "resources/docker/rabbitmq")
    sleep_time = 2  # Increased to give more time between attempts
    max_attempts = 15  # Increased max attempts since RabbitMQ can take longer to initialize

    def test_connection():
        try:
            # Use credentials from docker-compose.yml
            r = requests.get("http://localhost:15672/api/overview", 
                           auth=('permafrost', 'permafrost'),
                           timeout=5)
            if r.status_code == 200:
                print("✅ RabbitMQ ready.")
                return True
            else:
                print(f"⚠️  RabbitMQ returned status {r.status_code}")
        except requests.exceptions.ConnectionError:
            print("⏳ Waiting for RabbitMQ API to become available...")
        except requests.exceptions.Timeout:
            print("⚠️  Request timed out")
        except Exception as e:
            print(f"⚠️  Error checking RabbitMQ: {str(e)}")
        return False

    kill_container(CONTAINER_NAME)
    start(
        log_file,
        compose_dir,
        test_connection,
        sleep_time,
        max_attempts
    )

def stop_docker_rabbitmq():
    kill_container(CONTAINER_NAME)

if __name__ == "__main__":
    start_docker_rabbitmq()

# software/startup/docker_services/start_rabbitmq.py
import requests # type: ignore

from software.starup.docker_services.common_docker_utils import kill_container, start

CONTAINER_NAME = "rabbitmq-server"

def start_docker_rabbitmq():
    """
    Launches RabbitMQ via docker-compose and waits until it's healthy.
    """
    log_file = "logs/rabbitmq.log"
    compose_dir = "resources/docker/rabbitmq"
    sleep_time = 1
    max_attempts = 10

    def test_connection():
        try:
            r = requests.get("http://localhost:15672/api/overview", auth=('guest', 'guest'))
            if r.status_code == 200:
                print("✅ RabbitMQ ready.")
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

def stop_docker_rabbitmq():
    kill_container(CONTAINER_NAME)

if __name__ == "__main__":
    start_docker_rabbitmq()

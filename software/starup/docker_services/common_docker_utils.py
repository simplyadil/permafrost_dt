# software/startup/docker_services/utils_docker_service_starter.py
import os
import subprocess
import time

def kill_container(name: str):
    """
    Stops and removes a container if it exists.
    """
    print(f"🧹 Stopping existing container: {name}")
    subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def start(log_file: str, compose_dir: str, test_func, sleep_time: int, max_attempts: int):
    """
    Starts a Docker Compose service and waits until a test function returns True.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    compose_path = os.path.join(compose_dir, "docker-compose.yml")

    print(f"🚀 Starting service using compose at {compose_path}")
    with open(log_file, "w") as log:
        subprocess.Popen(["docker", "compose", "-f", compose_path, "up", "-d"], cwd=compose_dir, stdout=log, stderr=log)

    for i in range(max_attempts):
        if test_func():
            print("✅ Service ready.")
            return
        print(f"⏳ Waiting for service... ({i+1}/{max_attempts})")
        time.sleep(sleep_time)

    raise RuntimeError("❌ Service failed to start in time.")

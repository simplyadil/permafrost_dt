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
    
    # Start the service and wait for process to complete
    with open(log_file, "w") as log:
        result = subprocess.run(
            ["docker", "compose", "-f", compose_path, "up", "-d"],
            cwd=compose_dir,
            stdout=log,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            print(f"❌ Failed to start service: {result.stderr}")
            raise RuntimeError(f"Docker Compose failed with exit code {result.returncode}")
    
    print("📝 Checking container logs...")
    container_name = os.path.basename(os.path.dirname(compose_path))
    
    for i in range(max_attempts):
        if test_func():
            return
        
        # Show recent container logs if available
        logs = subprocess.run(
            ["docker", "logs", "--tail", "5", container_name + "-server"],
            capture_output=True,
            text=True
        )
        if logs.returncode == 0 and logs.stdout.strip():
            print(f"📋 Recent logs:\n{logs.stdout.strip()}")
            
        print(f"⏳ Waiting for service... ({i+1}/{max_attempts})")
        time.sleep(sleep_time)

    print("❌ Service failed to start in time.")
    print("💡 Try checking the logs in", os.path.abspath(log_file))
    raise RuntimeError("Service failed to start in time.")

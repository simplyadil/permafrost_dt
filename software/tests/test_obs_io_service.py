import os
import sys
import time
import datetime
from threading import Thread

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from services.common.messaging import RabbitMQClient
from services.common.influx_utils import InfluxHelper
from services.obs_io.obs_io_service import ObservationIngestionService

def run_service():
    """
    Runs the obs_io_service in a background thread.
    """
    service = ObservationIngestionService()
    service.run()

def test_obs_io_pipeline():
    """
    End-to-end functional test of obs_io_service:
    1. Starts the service in a background thread.
    2. Publishes a test message to RabbitMQ.
    3. Waits for processing.
    4. Queries InfluxDB to verify data ingestion.
    """
    # Ensure services are running
    print("\n 1 - Checking if services are running...")
    try:
        import requests
        # Check RabbitMQ
        r = requests.get("http://localhost:15672/api/overview", 
                        auth=('permafrost', 'permafrost'),
                        timeout=2)
        if r.status_code != 200:
            raise Exception(f"RabbitMQ returned status {r.status_code}")
            
        # Check InfluxDB
        r = requests.get("http://localhost:8086/health", timeout=2)
        if r.status_code != 200:
            raise Exception(f"InfluxDB returned status {r.status_code}")
            
        print("Services are running!")
    except Exception as e:
        print(f"Error checking services: {str(e)}")
        print("-->Try running these commands first:")
        print("   python software/starup/docker_services/start_rabbitmq.py")
        print("   python software/starup/docker_services/start_influxdb.py")
        raise

    # Start service in background
    print("\n 2 - Starting observation ingestion service...")
    thread = Thread(target=run_service, daemon=True)
    thread.start()
    time.sleep(3)

    # Send test message
    mq = RabbitMQClient(queue="permafrost.sensors.temperature")
    msg = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "time_days": 0.001,
        "temperature_0m": -12.0,
        "temperature_1m": -9.5,
        "temperature_2m": -7.0,
        "temperature_3m": -4.5,
        "temperature_4m": -2.0,
        "temperature_5m": 1.0
    }
    mq.publish(msg)

    # Wait for message processing
    time.sleep(5)

    # Query InfluxDB for recently written record
    db = InfluxHelper()
    df = db.query_temperature(limit=10)

    print("\n[TEST RESULT] Influx query results:")
    print(df.head())

    assert not df.empty, "InfluxDB returned no data — service may not have written correctly."
    assert "temperature" in df.columns, "Missing 'temperature' field in InfluxDB results."
    assert any(abs(df["temperature"]) > 0), "Temperature values appear invalid."

    print("\nobs_io_service test passed successfully!\n")

if __name__ == "__main__":
    test_obs_io_pipeline()

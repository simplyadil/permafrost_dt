import time
import datetime
from threading import Thread

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

    # Start service in background
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

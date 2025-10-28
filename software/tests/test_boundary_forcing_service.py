# software/tests/test_boundary_forcing_service.py
import threading
import time
import datetime
import json

from software.services.boundary_forcing.boundary_forcing_service import BoundaryForcingService
from software.services.common.messaging import RabbitMQClient
from software.services.common.influx_utils import InfluxHelper


def run_service():
    service = BoundaryForcingService(polling_interval=3)
    service.run()


def test_boundary_forcing_pipeline():
    print("\n 1 - Checking if services are running...")
    time.sleep(1)
    print("Services are running!\n")

    # --- SETUP PHASE ---
    print(" 2 - Writing test data to InfluxDB...")
    influx = InfluxHelper()

    # Clear a time window and write mock observations
    time_days = 0.123
    depths = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    temperatures = [-10.0, -9.0, -7.5, -6.0, -4.0, -2.0]

    for d, T in zip(depths, temperatures):
        influx.write_temperature(time_days, d, T)

    print("Test data written!\n")

    # --- RUN SERVICE IN BACKGROUND THREAD ---
    print(" 3 -  Starting boundary_forcing_service...")
    service_thread = threading.Thread(target=run_service, daemon=True)
    service_thread.start()
    time.sleep(5)  # let it process a few polling cycles

    # --- VERIFY MESSAGE PUBLICATION ---
    print(" 4 - Listening for published boundary forcing message...")
    mq = RabbitMQClient(
        queue="permafrost.boundary.forcing",
        schema_path="software/services/common/schemas/boundary_forcing_message.json"
    )

    received = {}

    def on_message(msg):
        nonlocal received
        try:
            received = msg
            print(f"Received message from boundary_forcing_service: {msg}")
        except Exception as e:
            print(f"Failed to process message: {e}")

    # Start consumer in background so the test thread can continue
    consumer_thread = threading.Thread(target=lambda: mq.consume(callback=on_message, auto_ack=True), daemon=True)
    consumer_thread.start()
    time.sleep(3)

    assert "Ts" in received, "No Ts field received from boundary_forcing_service"
    assert "time_days" in received, "No time_days field received"
    print("\nboundary_forcing_service test passed successfully!")


if __name__ == "__main__":
    test_boundary_forcing_pipeline()

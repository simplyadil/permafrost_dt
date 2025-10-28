import threading
import time
import json
import os
import sys
from datetime import datetime, UTC

# Ensure repository root is on sys.path so `software` package imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from software.services.common.influx_utils import InfluxHelper
from software.services.common.messaging import RabbitMQClient
from software.services.fdm_simulator.fdm_service import FDMService


def run_service():
    srv = FDMService()
    # Run in the foreground (this call blocks); start in thread from test
    srv.run()


def test_fdm_pipeline():
    print("\n 1 - Checking if services are running...")
    time.sleep(1)
    print("Services are running!\n")

    influx = InfluxHelper()

    # --- Seed an "observed" profile the FDM can start from ---
    print(" 2 - Seeding initial observed temperatures (soil_temperature)...")
    t0 = 0.100
    depths = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    temps  = [-11.0, -9.2, -7.8, -6.1, -4.3, -2.5]
    for d, T in zip(depths, temps):
        influx.write_temperature(time_days=t0, depth=d, temperature=T)
    print("Seeded!\n")

    # --- Start the FDM service in background thread ---
    print(" 3 - Starting FDM service...")
    t = threading.Thread(target=run_service, daemon=True)
    t.start()
    time.sleep(2)  # give the consumer time to connect

    # --- Publish a boundary forcing Ts message ---
    mq_boundary = RabbitMQClient(
        queue="permafrost.boundary.forcing",
        schema_path="software/services/common/schemas/boundary_forcing_message.json"
    )

    ts_msg = {
        "timestamp": datetime.now(UTC).isoformat(),
        "time_days": 0.125,    # later than t0 so dt > 0
        "Ts": -12.5
    }
    mq_boundary.publish(ts_msg)
    print("Published Ts message!\n")

    # --- Allow the FDM service to process and write results ---
    time.sleep(4)

    # --- Verify Influx has fdm_temperature written ---
    print(" 4 - Querying fdm_temperature from Influx...")
    df = influx.query_model_temperature(measurement="fdm_temperature", limit=200)
    print(df.head() if hasattr(df, "head") else df)

    assert df is not None and not df.empty, "No FDM results found in Influx."
    assert "temperature" in df.columns and "depth" in df.columns, "Unexpected schema in FDM results."

    # Check surface was enforced (latest row for depth 0.0 ≈ Ts)
    df0 = df[df["depth"] == "0.0m"].sort_values("_time")
    assert not df0.empty, "No surface rows in FDM results."
    latest_surface_T = float(df0.iloc[-1]["temperature"])
    assert abs(latest_surface_T - ts_msg["Ts"]) < 1e-6, "Surface T not enforced to Ts."

    print("\nfdm_service test passed successfully!")


if __name__ == "__main__":
    test_fdm_pipeline()

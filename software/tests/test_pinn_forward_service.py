import threading
import time
from datetime import datetime, UTC

from services.pinn_forward.pinn_forward_service import PINNForwardService
from services.common.messaging import RabbitMQClient
from services.common.influx_utils import InfluxHelper


def run_service():
    service = PINNForwardService()
    service.run()


def test_pinn_forward_pipeline():
    print("\nℹ️  Checking if services are running...")
    time.sleep(1)
    print("✅ Services are running\n")

    influx = InfluxHelper()

    # Seed FDM temperature data
    print("ℹ️  Seeding FDM data in InfluxDB...")
    t_day = 0.200
    depths = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    temps = [-11.0, -9.5, -8.0, -6.5, -5.0, -3.5]
    for d, T in zip(depths, temps):
        influx.write_model_temperature("fdm_temperature", t_day, d, T, extra_tags={"model": "fdm"})
    print("✅ FDM data seeded.\n")

    # Start the service
    print("ℹ️  Starting PINN forward service...")
    t = threading.Thread(target=run_service, daemon=True)
    t.start()
    time.sleep(2)

    # Publish trigger message
    mq = RabbitMQClient(
        queue="permafrost.sim.fdm.output",
        schema_path="software/services/common/schemas/fdm_output_message.json"
    )
    msg = {
        "timestamp": datetime.now(UTC).isoformat(),
        "time_days": t_day,
        "status": "ready"
    }
    mq.publish(msg)
    print("✅ Published FDM update message.\n")

    # Wait for retraining and writeback
    print("⌛ Waiting for training (approx. 20–40s)...")
    time.sleep(40)

    # Query predictions
    df = influx.query_model_temperature(measurement="pinn_temperature", limit=5000)
    print(df.head())
    assert df is not None and not df.empty, "No PINN predictions found."
    print("\n✅ pinn_forward_service test passed successfully!")


if __name__ == "__main__":
    test_pinn_forward_pipeline()

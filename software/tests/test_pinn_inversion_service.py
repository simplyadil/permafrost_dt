# software/tests/test_pinn_inversion_service.py
import time
import os
import sys

# Ensure repository root is on sys.path so `software` package imports work when running this test directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from software.services.common.logger import setup_logger
from software.services.common.messaging import RabbitMQClient


logger = setup_logger("test_pinn_inversion")

def main():
    logger.info(">>> Starting test for PINN inversion service...")
    mq = RabbitMQClient(
        queue="permafrost.sim.pinn.forward.output",
        schema_path="software/services/common/schemas/pinn_forward_output_message.json"
    )
    msg = {
        "timestamp": "2025-10-25T18:00:00Z",
        "time_days": 365.0,
        "status": "trained",
        "model_path": "software/models/pinn_forward/freezing_soil_pinn.pt"
    }
    mq.publish(msg)
    logger.info("Test message sent. Waiting 10 seconds...")
    time.sleep(10)
    logger.info("Check logs for inversion progress and results.")

if __name__ == "__main__":
    main()

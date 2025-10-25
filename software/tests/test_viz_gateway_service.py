# software/tests/test_viz_gateway_service.py
import os
import sys

# Ensure repository root is on sys.path so `software` package imports work when running this test directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import time
from software.services.common.messaging import RabbitMQClient
from software.services.common.logger import setup_logger

logger = setup_logger("test_viz_gateway")

def main():
    logger.info("Testing viz_gateway_service trigger message...")
    mq = RabbitMQClient(
        queue="permafrost.sim.pinn.inversion.output",
        schema_path="software/services/common/schemas/pinn_inversion_output_message.json"
    )
    msg = {
        "timestamp": "2025-10-25T21:00:00Z",
        "status": "inverted",
        "parameters": {
            "lambda_f": 1.51,
            "C_f": 1.49,
            "eta": 0.41,
            "b": 1.52,
            "T_nabla": -0.98
        }
    }
    mq.publish(msg)
    logger.info("Message sent. Waiting 10 seconds for viz_gateway_service to process...")
    time.sleep(10)
    logger.info("✅ Check logs for viz_ready message and snapshot file in /software/outputs/")

if __name__ == "__main__":
    main()

# software/services/viz_gateway/viz_gateway_service.py
import os
import sys
import json
import pandas as pd
from datetime import datetime
from threading import Thread

# Add repository root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from software.services.common.messaging import RabbitMQClient
from software.services.common.logger import setup_logger
from software.services.common.influx_utils import InfluxHelper


class VizGatewayService:
    """
    Collects results from InfluxDB (FDM, PINN forward, inversion) and publishes visualization-ready updates.
    """

    def __init__(self):
        self.logger = setup_logger("viz_gateway_service")
        self.influx = InfluxHelper()
        self.queue_in = "permafrost.sim.pinn.inversion.output"
        self.queue_out = "permafrost.sim.viz.update"

        self.mq_client = RabbitMQClient(
            queue=self.queue_in,
            schema_path="software/services/common/schemas/pinn_inversion_output_message.json"
        )
        self.out_publisher = RabbitMQClient(
            queue=self.queue_out,
            schema_path="software/services/common/schemas/viz_update_message.json"
        )

        self.output_dir = "software/outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info("viz_gateway_service initialized.")

    # -------------------------------
    # MAIN HANDLER
    # -------------------------------
    def _on_message(self, msg):
        """Triggered when inversion is complete."""
        self.logger.info(f"Received message from inversion service: {msg}")
        if msg.get("status") == "inverted":
            Thread(target=self.aggregate_and_publish, daemon=True).start()
        else:
            self.logger.warning("Message ignored (not 'inverted').")

    # -------------------------------
    # DATA AGGREGATION
    # -------------------------------
    def aggregate_and_publish(self):
        """Fetch latest data from InfluxDB and send a summary message."""
        self.logger.info("Fetching latest FDM, PINN, and inversion results from InfluxDB...")

        fdm_df = self.influx.query_model_temperature("fdm_temperature", limit=5000)
        pinn_df = self.influx.query_model_temperature("pinn_temperature", limit=5000)
        inv_df = self.influx.query_model_temperature("pinn_inversion_results", limit=500)

        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "viz_ready",
            "data_summary": {
                "fdm_points": len(fdm_df) if fdm_df is not None else 0,
                "pinn_points": len(pinn_df) if pinn_df is not None else 0,
                "inversion_points": len(inv_df) if inv_df is not None else 0,
            },
            "parameters": {},
        }

        # Extract last inversion parameters if available
        if inv_df is not None and not inv_df.empty:
            last_row = inv_df.iloc[-1]
            summary["parameters"] = {
                "lambda_f": last_row.get("lambda_f"),
                "C_f": last_row.get("C_f"),
                "eta": last_row.get("eta"),
                "b": last_row.get("b"),
                "T_nabla": last_row.get("T_nabla")
            }

        # Save combined snapshot to file
        snapshot_path = os.path.join(self.output_dir, f"viz_snapshot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
        with open(snapshot_path, "w") as f:
            json.dump(summary, f, indent=2)
        self.logger.info(f"Visualization snapshot saved to {snapshot_path}")

        # Publish to visualization queue
        self.out_publisher.publish(summary)
        self.logger.info(f"Visualization update published to {self.queue_out}.")

    # -------------------------------
    # RUN
    # -------------------------------
    def run(self):
        self.logger.info(f"Listening for inversion messages on queue: {self.queue_in}")
        self.mq_client.consume(callback=self._on_message)


if __name__ == "__main__":
    service = VizGatewayService()
    service.run()

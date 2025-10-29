# pylint: disable=too-many-instance-attributes
"""Visualization gateway server."""

import json
import os
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Optional

from software.digital_twin.communication.logger import setup_logger
from software.digital_twin.communication.messaging import RabbitMQClient, RabbitMQConfig
from software.digital_twin.data_access.influx_utils import InfluxConfig, InfluxHelper

PINN_INVERSION_QUEUE = "permafrost.record.pinn_inversion.state"
VIZ_UPDATE_QUEUE = "permafrost.update.visualization.command"
PINN_INVERSION_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "pinn_inversion_output_message.json"
VIZ_UPDATE_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "viz_update_message.json"


class VizGatewayServer:
    """
    Collects results from InfluxDB (FDM, PINN forward, inversion) and publishes visualization-ready updates.
    """

    def __init__(
        self,
        influx_config: InfluxConfig | None = None,
        inversion_queue_config: RabbitMQConfig | None = None,
        viz_queue_config: RabbitMQConfig | None = None,
        output_dir: str = "software/outputs",
    ) -> None:
        self.logger = setup_logger("VizGatewayServer")
        self.influx_config = influx_config or InfluxConfig()

        if inversion_queue_config is not None and inversion_queue_config.schema_path is None:
            inversion_queue_config = RabbitMQConfig(
                host=inversion_queue_config.host,
                queue=inversion_queue_config.queue,
                schema_path=PINN_INVERSION_SCHEMA,
                username=inversion_queue_config.username,
                password=inversion_queue_config.password,
            )
        if viz_queue_config is not None and viz_queue_config.schema_path is None:
            viz_queue_config = RabbitMQConfig(
                host=viz_queue_config.host,
                queue=viz_queue_config.queue,
                schema_path=VIZ_UPDATE_SCHEMA,
                username=viz_queue_config.username,
                password=viz_queue_config.password,
            )
        inversion_base = inversion_queue_config or RabbitMQConfig(schema_path=PINN_INVERSION_SCHEMA)
        viz_base = viz_queue_config or RabbitMQConfig(schema_path=VIZ_UPDATE_SCHEMA)
        self.inversion_queue_config = inversion_base.with_queue(PINN_INVERSION_QUEUE)
        self.viz_queue_config = viz_base.with_queue(VIZ_UPDATE_QUEUE)
        self.queue_in = self.inversion_queue_config.queue
        self.queue_out = self.viz_queue_config.queue

        self.mq_client: Optional[RabbitMQClient] = None
        self.out_publisher: Optional[RabbitMQClient] = None
        self.influx: Optional[InfluxHelper] = None

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._running = False
        self.logger.info("VizGatewayServer configured.")

    # -------------------------------
    # MAIN HANDLER
    # -------------------------------
    def _on_message(self, msg):
        """Triggered when inversion is complete."""
        self.logger.info(f"Received message from inversion service: {msg}")
        if msg.get("status") == "inverted":
            Thread(target=self.aggregate_and_publish, args=(msg,), daemon=True).start()
        else:
            self.logger.warning("Message ignored (not 'inverted').")

    # -------------------------------
    # DATA AGGREGATION
    # -------------------------------
    def aggregate_and_publish(self, inversion_msg: Optional[dict] = None) -> None:
        """Fetch latest data from InfluxDB and send a summary message."""
        self.logger.info("Fetching latest FDM, PINN, and inversion results from InfluxDB...")

        if self.influx is None:
            raise RuntimeError("Influx helper not initialised. Did you call setup()?")

        fdm_df = self.influx.query_model_temperature("fdm_simulation", limit=5000)
        pinn_df = self.influx.query_model_temperature("pinn_forward", limit=5000)

        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "viz_ready",
            "data_summary": {
                "fdm_points": len(fdm_df) if fdm_df is not None else 0,
                "pinn_points": len(pinn_df) if pinn_df is not None else 0,
                "inversion_points": 1 if inversion_msg else 0,
            },
            "parameters": {},
        }

        if inversion_msg is not None:
            summary["parameters"] = inversion_msg.get("parameters", {})

        # Save combined snapshot to file
        snapshot_path = os.path.join(self.output_dir, f"viz_snapshot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
        with open(snapshot_path, "w") as f:
            json.dump(summary, f, indent=2)
        self.logger.info(f"Visualization snapshot saved to {snapshot_path}")

        # Publish to visualization queue
        if self.out_publisher is None:
            raise RuntimeError("Outbound MQ client not initialised. Did you call setup()?")
        self.out_publisher.publish(summary)
        self.logger.info(f"Visualization update published to {self.queue_out}.")

    # -------------------------------
    # RUN
    # -------------------------------
    def run(self):
        if self.mq_client is None:
            raise RuntimeError("Inbound MQ client not initialised. Did you call setup()?")
        self.logger.info(f"Listening for inversion messages on queue: {self.queue_in}")
        self.mq_client.consume(callback=self._on_message)

    # -------------------------------
    # LIFECYCLE
    # -------------------------------
    def setup(self) -> None:
        """Initialise messaging and data dependencies."""

        if self.influx is None:
            self.influx = InfluxHelper(self.influx_config)
        if self.mq_client is None:
            self.mq_client = RabbitMQClient(self.inversion_queue_config)
        if self.out_publisher is None:
            self.out_publisher = RabbitMQClient(self.viz_queue_config)
        self.queue_in = self.inversion_queue_config.queue
        self.queue_out = self.viz_queue_config.queue
        self._running = True
        self.logger.info("VizGatewayServer setup complete.")

    def start(self) -> None:
        """Start consuming inversion notifications."""

        if self.mq_client is None or self.out_publisher is None or self.influx is None:
            self.setup()

        try:
            self.run()
        except KeyboardInterrupt:  # pragma: no cover
            self.logger.info("VizGatewayServer interrupted. Shutting down...")
        finally:
            self.close()

    def stop(self) -> None:
        """Stop consuming messages."""

        self._running = False
        if self.mq_client and self.mq_client.channel:
            self.mq_client.channel.stop_consuming()

    def close(self) -> None:
        """Release resources."""

        self.stop()
        if self.mq_client is not None:
            self.mq_client.disconnect()
        if self.out_publisher is not None:
            self.out_publisher.disconnect()
        if self.influx is not None:
            self.influx.close()
        self.logger.info("VizGatewayServer shutdown complete.")


if __name__ == "__main__":
    VizGatewayServer().start()

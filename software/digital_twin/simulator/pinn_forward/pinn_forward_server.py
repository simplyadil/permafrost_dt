# pylint: disable=too-many-instance-attributes
"""Physics-informed neural network forward server."""

import os
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Optional

import numpy as np
import torch

from software.digital_twin.communication.logger import setup_logger
from software.digital_twin.communication.messaging import RabbitMQClient, RabbitMQConfig
from software.digital_twin.data_access.influx_utils import InfluxConfig, InfluxHelper
from software.digital_twin.simulator.pinn_forward.freezing_soil_pinn import FreezingSoilPINN

FDM_QUEUE = "permafrost.record.fdm.state"
PINN_FORWARD_QUEUE = "permafrost.record.pinn_forward.state"
FDM_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "fdm_output_message.json"
PINN_FORWARD_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "pinn_forward_output_message.json"


class PINNForwardServer:
    """
    Listens for FDM simulation updates and retrains a phase-change PINN surrogate model.
    """

    def __init__(
        self,
        influx_config: InfluxConfig | None = None,
        fdm_queue_config: RabbitMQConfig | None = None,
        forward_queue_config: RabbitMQConfig | None = None,
        model_dir: str = "software/models/pinn_forward",
    ) -> None:
        self.logger = setup_logger("PINNForwardServer")
        self.influx_config = influx_config or InfluxConfig()
        # Legacy base_config assignment occurred before schema reconciliation.
        # base_config = fdm_queue_config or RabbitMQConfig(schema_path=FDM_SCHEMA)
        if fdm_queue_config is not None and fdm_queue_config.schema_path is None:
            fdm_queue_config = RabbitMQConfig(
                host=fdm_queue_config.host,
                queue=fdm_queue_config.queue,
                schema_path=FDM_SCHEMA,
                username=fdm_queue_config.username,
                password=fdm_queue_config.password,
            )
        base_config = fdm_queue_config or RabbitMQConfig(schema_path=FDM_SCHEMA)
        self.fdm_queue_config = base_config.with_queue(FDM_QUEUE)
        if forward_queue_config is not None and forward_queue_config.schema_path is None:
            forward_queue_config = RabbitMQConfig(
                host=forward_queue_config.host,
                queue=forward_queue_config.queue,
                schema_path=PINN_FORWARD_SCHEMA,
                username=forward_queue_config.username,
                password=forward_queue_config.password,
            )
        output_base = forward_queue_config or RabbitMQConfig(schema_path=PINN_FORWARD_SCHEMA)
        self.forward_queue_config = output_base.with_queue(PINN_FORWARD_QUEUE)

        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.influx: Optional[InfluxHelper] = None
        self.mq_client: Optional[RabbitMQClient] = None
        self.out_publisher: Optional[RabbitMQClient] = None

        # Default PINN physical parameters
        self.params = {
            "L": 3.34e5,       # Latent heat of phase change (KJ/m³)
            "C_i": 1.672,      # Heat capacity of ice (KJ/(m³·K))
            "C_l": 4.18,       # Heat capacity of water
            "C_f": 1.5,        # Heat capacity of soil matrix
            "lambda_i": 2.210, # Thermal conductivity of ice (W/(m·K))
            "lambda_l": 0.465, # Thermal conductivity of water
            "lambda_f": 1.5,   # Thermal conductivity of soil matrix
            "eta": 0.4,        # Porosity
            "b": 1.5,          # Unfrozen water parameter
            "T_nabla": -1.0    # Freezing temperature (°C)
        }

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pinn = FreezingSoilPINN(self.params, device=device, log_callback=self.logger.info)
        self._running = False
        self.logger.info("PINNForwardServer configured.")

    # -------------------------------
    # TRAINING PIPELINE
    # -------------------------------
    def train_from_influx(self):
        """
        Fetches FDM results and retrains the PINN model.
        """
        if self.influx is None:
            raise RuntimeError("Influx helper not initialised. Did you call setup()?")

        self.logger.info("Fetching FDM results from InfluxDB...")
        df = self.influx.query_model_temperature(measurement="fdm_simulation", limit=5000)
        if df is None or df.empty:
            self.logger.error("No FDM data found in InfluxDB. Aborting training.")
            return

        self.logger.info(f"Loaded {len(df)} rows of FDM temperature data for PINN training.")

        # Train model
        try:
            self.pinn.train(x_domain=(0.0, 5.0), t_domain=(0.0, 365.0), epochs=20000, n_samples=5000)
            self.pinn.save(model_dir=self.model_dir)
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return

        # Generate predictions for each time-depth pair
        x_vals = np.linspace(0, 5.0, 6)
        for t_day in sorted(df["time_days"].unique()):
            t_tensor = torch.full((len(x_vals), 1), float(t_day))
            x_tensor = torch.tensor(x_vals).unsqueeze(1)
            T_pred = self.pinn.predict(x_tensor, t_tensor).cpu().numpy()

            for depth, temp in zip(x_vals, T_pred):
                self.influx.write_model_temperature(
                    measurement="pinn_forward",
                    time_days=float(t_day),
                    depth=float(depth),
                    temperature=float(temp),
                    site="default",
                    extra_tags={"model": "pinn"}
                )

        if self.out_publisher is not None:
            message = {
                "timestamp": datetime.utcnow().isoformat(),
                "status": "trained",
                "time_days": float(df["time_days"].max()),
                "model_path": str(Path(self.model_dir) / "freezing_soil_pinn.pt"),
            }
            self.out_publisher.publish(message)

        self.logger.info("✅ PINN retraining and prediction complete.")

    # -------------------------------
    # QUEUE HANDLER
    # -------------------------------
    def _on_message(self, msg):
        """
        Handles messages from RabbitMQ and triggers retraining when FDM output is ready.
        """
        self.logger.info(f"Received message from FDM service: {msg}")
        if msg.get("status") == "ready":
            self.logger.info("New FDM results detected — starting PINN retraining.")
            Thread(target=self.train_from_influx, daemon=True).start()
        else:
            self.logger.warning("Message ignored (no retraining trigger).")

    # -------------------------------
    # RUN
    # -------------------------------
    def run(self):
        if self.mq_client is None:
            raise RuntimeError("RabbitMQ client not initialised. Did you call setup()?")
        self.logger.info("PINNForwardServer listening for FDM output messages...")
        self.mq_client.consume(callback=self._on_message)

    # -------------------------------
    # LIFECYCLE
    # -------------------------------
    def setup(self) -> None:
        """Initialise data and messaging dependencies."""

        if self.influx is None:
            self.influx = InfluxHelper(self.influx_config)
        if self.mq_client is None:
            self.mq_client = RabbitMQClient(self.fdm_queue_config)
        if self.out_publisher is None:
            self.out_publisher = RabbitMQClient(self.forward_queue_config)
        self._running = True
        self.logger.info("PINNForwardServer setup complete.")

    def start(self) -> None:
        """Start consuming FDM notifications."""

        if self.influx is None or self.mq_client is None:
            self.setup()

        try:
            self.run()
        except KeyboardInterrupt:  # pragma: no cover - runtime
            self.logger.info("PINNForwardServer interrupted. Shutting down...")
        finally:
            self.close()

    def stop(self) -> None:
        """Stop consuming further messages."""

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
        self.logger.info("PINNForwardServer shutdown complete.")


if __name__ == "__main__":
    PINNForwardServer().start()

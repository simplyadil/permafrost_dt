# software/services/pinn_forward/pinn_forward_service.py
import os
import time
import torch
import numpy as np
from datetime import datetime
from threading import Thread

from services.pinn_forward.freezing_soil_pinn import FreezingSoilPINN
from services.common.influx_utils import InfluxHelper
from services.common.messaging import RabbitMQClient
from services.common.logger import setup_logger


class PINNForwardService:
    """
    Listens for FDM simulation updates and retrains a phase-change PINN surrogate model.
    """

    def __init__(self):
        self.logger = setup_logger("pinn_forward_service")
        self.influx = InfluxHelper()

        self.queue_name = "permafrost.sim.fdm.output"
        self.model_dir = "software/models/pinn_forward"
        os.makedirs(self.model_dir, exist_ok=True)

        # Attach to queue
        self.mq_client = RabbitMQClient(
            queue=self.queue_name,
            schema_path="software/services/common/schemas/fdm_output_message.json"
        )

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
        self.logger.info("pinn_forward_service initialized.")

    # -------------------------------
    # TRAINING PIPELINE
    # -------------------------------
    def train_from_influx(self):
        """
        Fetches FDM results and retrains the PINN model.
        """
        self.logger.info("Fetching FDM results from InfluxDB...")
        df = self.influx.query_model_temperature(measurement="fdm_temperature", limit=5000)
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
                    measurement="pinn_temperature",
                    time_days=float(t_day),
                    depth=float(depth),
                    temperature=float(temp),
                    site="default",
                    extra_tags={"model": "pinn"}
                )

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
        self.logger.info("pinn_forward_service listening for FDM output messages...")
        self.mq_client.consume(callback=self._on_message)


if __name__ == "__main__":
    service = PINNForwardService()
    service.run()

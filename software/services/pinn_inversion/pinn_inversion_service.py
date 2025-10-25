# software/services/pinn_inversion/pinn_inversion_service.py
import os
import torch
import pandas as pd
from datetime import datetime
from threading import Thread

from software.services.common.influx_utils import InfluxHelper
from software.services.common.logger import setup_logger
from software.services.common.messaging import RabbitMQClient
from software.services.pinn_inversion.inversion_pinn_model import InversionFreezingSoilPINN




class PINNInversionService:
    """
    Performs inversion of soil parameters when forward PINN results are ready.
    """

    def __init__(self):
        self.logger = setup_logger("pinn_inversion_service")
        self.influx = InfluxHelper()
        self.queue_name = "permafrost.sim.pinn.forward.output"
        self.output_queue = "permafrost.sim.pinn.inversion.output"

        self.mq_client = RabbitMQClient(
            queue=self.queue_name,
            schema_path="software/services/common/schemas/pinn_forward_output_message.json"
        )

        self.out_publisher = RabbitMQClient(
            queue=self.output_queue,
            schema_path="software/services/common/schemas/pinn_inversion_output_message.json"
        )

        self.model_dir = "software/models/pinn_inversion"
        os.makedirs(self.model_dir, exist_ok=True)
        self.logger.info("pinn_inversion_service initialized.")

    def _on_message(self, msg):
        self.logger.info(f"Received message from forward service: {msg}")
        if msg.get("status") == "trained":
            Thread(target=self.run_inversion, daemon=True).start()
        else:
            self.logger.warning("Ignored non-training message.")

    def run_inversion(self):
        self.logger.info("Fetching PINN temperature data from InfluxDB...")
        df = self.influx.query_model_temperature(measurement="pinn_temperature", limit=5000)
        if df is None or df.empty:
            self.logger.error("No PINN data found in InfluxDB.")
            return

        # Prepare tensors
        x_obs_list, t_obs_list, T_obs_list = [], [], []
        for _, row in df.iterrows():
            x_obs_list.append([row["depth_m"]])
            t_obs_list.append([row["time_days"]])
            T_obs_list.append([row["temperature_C"] / 20.0])

        x_obs = torch.tensor(x_obs_list, dtype=torch.float32)
        t_obs = torch.tensor(t_obs_list, dtype=torch.float32)
        T_obs = torch.tensor(T_obs_list, dtype=torch.float32)

        # Known constants
        known_params = {
            "L": 3.34e5,
            "C_i": 1.672,
            "C_l": 4.18,
            "lambda_i": 2.210,
            "lambda_l": 0.465
        }
        param_bounds = {
            "lambda_f": [1.3, 1.7],
            "C_f": [1.3, 1.7],
            "eta": [0.3, 0.5],
            "b": [1.0, 2.0],
            "T_nabla": [-1.5, -0.5]
        }

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = InversionFreezingSoilPINN(known_params, param_bounds, device=device, log_callback=self.logger.info)
        model.boundary_data = df
        params_estimated = model.train(
            x_obs=x_obs.to(device),
            t_obs=t_obs.to(device),
            T_obs_true=T_obs.to(device),
            x_domain=(0.0, 5.0),
            t_domain=(0.0, 365.0),
            epochs=5000
        )

        # Log results
        self.logger.info(f"Inversion complete. Parameters: {params_estimated}")

        # Publish result message
        result_msg = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "inverted",
            "parameters": params_estimated
        }
        self.out_publisher.publish(result_msg)
        self.logger.info("Result published to RabbitMQ.")

    def run(self):
        self.logger.info("pinn_inversion_service listening for messages...")
        self.mq_client.consume(callback=self._on_message)


if __name__ == "__main__":
    service = PINNInversionService()
    service.run()

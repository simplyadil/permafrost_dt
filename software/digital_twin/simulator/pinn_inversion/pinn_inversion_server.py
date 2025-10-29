# pylint: disable=too-many-instance-attributes
"""PINN inversion server."""

import os
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Optional

import torch

from software.digital_twin.communication.logger import setup_logger
from software.digital_twin.communication.messaging import RabbitMQClient, RabbitMQConfig
from software.digital_twin.data_access.influx_utils import InfluxConfig, InfluxHelper
from software.digital_twin.simulator.pinn_inversion.inversion_pinn_model import InversionFreezingSoilPINN

PINN_FORWARD_QUEUE = "permafrost.record.pinn_forward.state"
PINN_INVERSION_QUEUE = "permafrost.record.pinn_inversion.state"
PINN_FORWARD_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "pinn_forward_output_message.json"
PINN_INVERSION_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "pinn_inversion_output_message.json"


class PINNInversionServer:
    """
    Performs inversion of soil parameters when forward PINN results are ready.
    """

    def __init__(
        self,
        influx_config: InfluxConfig | None = None,
        forward_queue_config: RabbitMQConfig | None = None,
        inversion_queue_config: RabbitMQConfig | None = None,
        model_dir: str = "software/models/pinn_inversion",
    ) -> None:
        self.logger = setup_logger("PINNInversionServer")
        self.influx_config = influx_config or InfluxConfig()
        if forward_queue_config is not None and forward_queue_config.schema_path is None:
            forward_queue_config = RabbitMQConfig(
                host=forward_queue_config.host,
                queue=forward_queue_config.queue,
                schema_path=PINN_FORWARD_SCHEMA,
                username=forward_queue_config.username,
                password=forward_queue_config.password,
            )
        if inversion_queue_config is not None and inversion_queue_config.schema_path is None:
            inversion_queue_config = RabbitMQConfig(
                host=inversion_queue_config.host,
                queue=inversion_queue_config.queue,
                schema_path=PINN_INVERSION_SCHEMA,
                username=inversion_queue_config.username,
                password=inversion_queue_config.password,
            )
        forward_base = forward_queue_config or RabbitMQConfig(schema_path=PINN_FORWARD_SCHEMA)
        inversion_base = inversion_queue_config or RabbitMQConfig(schema_path=PINN_INVERSION_SCHEMA)
        self.forward_queue_config = forward_base.with_queue(PINN_FORWARD_QUEUE)
        self.inversion_queue_config = inversion_base.with_queue(PINN_INVERSION_QUEUE)

        self.influx: Optional[InfluxHelper] = None
        self.mq_client: Optional[RabbitMQClient] = None
        self.out_publisher: Optional[RabbitMQClient] = None

        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self._running = False
        self.logger.info("PINNInversionServer configured.")

    def _on_message(self, msg):
        self.logger.info(f"Received message from forward service: {msg}")
        if msg.get("status") == "trained":
            Thread(target=self.run_inversion, daemon=True).start()
        else:
            self.logger.warning("Ignored non-training message.")

    def run_inversion(self):
        if self.influx is None:
            raise RuntimeError("Influx helper not initialised. Did you call setup()?")

        self.logger.info("Fetching PINN temperature data from InfluxDB...")
        df = self.influx.query_model_temperature(measurement="pinn_forward", limit=5000)
        if df is None or df.empty:
            self.logger.error("No PINN data found in InfluxDB.")
            return

        # Prepare tensors
        x_obs_list, t_obs_list, T_obs_list = [], [], []
        for _, row in df.iterrows():
            depth_value = row.get("depth", row.get("depth_m"))
            temperature_value = row.get("temperature", row.get("temperature_C"))
            if depth_value is None or temperature_value is None:
                continue
            x_obs_list.append([float(str(depth_value).replace("m", ""))])
            t_obs_list.append([row["time_days"]])
            T_obs_list.append([float(temperature_value) / 20.0])

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
        if self.out_publisher is None:
            raise RuntimeError("Outbound RabbitMQ client not initialised. Did you call setup()?")
        self.out_publisher.publish(result_msg)
        self.logger.info("Result published to RabbitMQ.")

    def run(self):
        if self.mq_client is None:
            raise RuntimeError("RabbitMQ client not initialised. Did you call setup()?")
        self.logger.info("PINNInversionServer listening for messages...")
        self.mq_client.consume(callback=self._on_message)

    # Lifecycle ------------------------------------------------
    def setup(self) -> None:
        """Initialise dependencies."""

        if self.influx is None:
            self.influx = InfluxHelper(self.influx_config)
        if self.mq_client is None:
            self.mq_client = RabbitMQClient(self.forward_queue_config)
        if self.out_publisher is None:
            self.out_publisher = RabbitMQClient(self.inversion_queue_config)
        self._running = True
        self.logger.info("PINNInversionServer setup complete.")

    def start(self) -> None:
        """Start consuming forward PINN notifications."""

        if self.influx is None or self.mq_client is None or self.out_publisher is None:
            self.setup()

        try:
            self.run()
        except KeyboardInterrupt:  # pragma: no cover
            self.logger.info("PINNInversionServer interrupted. Shutting down...")
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
        self.logger.info("PINNInversionServer shutdown complete.")


if __name__ == "__main__":
    PINNInversionServer().start()

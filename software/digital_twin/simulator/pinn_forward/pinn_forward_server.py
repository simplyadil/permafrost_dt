# pylint: disable=too-many-instance-attributes
"""Physics-informed neural network forward server."""

import json
import os
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Optional

import numpy as np
import torch

from software.digital_twin.communication.messaging import RabbitMQClient, RabbitMQConfig, resolve_queue_config
from software.digital_twin.data_access.influx_utils import InfluxConfig, InfluxHelper
from software.digital_twin.simulator.pinn_forward.freezing_soil_pinn import FreezingSoilPINN
from software.utils.logging_setup import get_logger

FDM_QUEUE = "permafrost.record.fdm.state"
PINN_FORWARD_QUEUE = "permafrost.record.pinn_forward.state"
FDM_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "fdm_output_message.json"
PINN_FORWARD_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "pinn_forward_output_message.json"
PINN_HISTORY_FILENAME = "freezing_soil_pinn_history.json"
TRAIN_EPOCHS = 20000


class PINNForwardServer:
    """
    Listens for FDM simulation updates and maintains a phase-change PINN surrogate model.
    """

    def __init__(
        self,
        influx_config: InfluxConfig | None = None,
        fdm_queue_config: RabbitMQConfig | None = None,
        forward_queue_config: RabbitMQConfig | None = None,
        model_dir: str = "software/models/pinn_forward",
        *,
        enable_training: bool = True,
        model_path: str | None = None,
    ) -> None:
        self.logger = get_logger("PINNForwardServer")
        self.influx_config = influx_config or InfluxConfig()
        self.fdm_queue_config = resolve_queue_config(
            fdm_queue_config,
            queue=FDM_QUEUE,
            schema_path=FDM_SCHEMA,
        )
        self.forward_queue_config = resolve_queue_config(
            forward_queue_config,
            queue=PINN_FORWARD_QUEUE,
            schema_path=PINN_FORWARD_SCHEMA,
        )

        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.enable_training = enable_training
        default_checkpoint = Path(self.model_dir) / "freezing_soil_pinn.pt"
        self.model_path = Path(model_path) if model_path is not None else default_checkpoint
        self.history_path = Path(self.model_dir) / PINN_HISTORY_FILENAME

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
        self._load_pretrained_weights()
        self.logger.info(
            "Configured (inbound=%s, outbound=%s, training=%s, device=%s)",
            self.fdm_queue_config.queue,
            self.forward_queue_config.queue,
            "on" if self.enable_training else "off",
            self.pinn.device,
        )

    def _load_pretrained_weights(self) -> None:
        """Load pretrained weights if available."""
        if self.model_path.exists():
            try:
                self.pinn.load(str(self.model_path))
            except Exception as exc:  # pragma: no cover - depends on file IO
                self.logger.error("Failed to load PINN weights from %s: %s", self.model_path, exc)
        else:
            if self.enable_training:
                self.logger.warning(
                    "No pretrained PINN weights found at %s. Training will start from scratch.",
                    self.model_path,
                )
            else:
                self.logger.error(
                    "PINN training disabled and checkpoint missing at %s. "
                    "Inference will fail until a model is provided.",
                    self.model_path,
                )

    # -------------------------------
    # TRAINING / INFERENCE PIPELINE
    # -------------------------------
    def train_from_influx(self):
        """
        Fetches FDM results and either retrains or evaluates the PINN model.
        """
        if self.influx is None:
            raise RuntimeError("Influx helper not initialised. Did you call setup()?")

        self.logger.info("Querying InfluxDB for FDM simulation history")
        df = self.influx.query_model_temperature(measurement="fdm_simulation", limit=5000)
        if df is None or df.empty:
            self.logger.error("FDM history unavailable; skipping PINN update")
            return

        time_steps = sorted(df["time_days"].unique())
        self.logger.info(
            "Fetched FDM history (rows=%d, timesteps=%d)",
            len(df),
            len(time_steps),
        )

        status = "inferred"
        summary = None
        if self.enable_training:
            try:
                self.logger.info("Starting PINN training (epochs=%d)", TRAIN_EPOCHS)
                summary = self.pinn.train(
                    x_domain=(0.0, 5.0), t_domain=(0.0, 365.0), epochs=TRAIN_EPOCHS, n_samples=5000
                )
                self.pinn.save(model_dir=self.model_dir)
                self.model_path = Path(self.model_dir) / "freezing_soil_pinn.pt"
                self.pinn.model.eval()
                status = "trained"
            except Exception as exc:
                self.logger.error("PINN training failed: %s", exc, exc_info=True)
                return
        else:
            if not self.model_path.exists():
                self.logger.error(
                    "Cannot run inference – pretrained PINN missing at %s",
                    self.model_path,
                )
                return
            self.logger.info("Using pretrained weights at %s", self.model_path)
            self._load_pretrained_weights()

        timestep_count = self._write_predictions(df, status=status)
        summary = self._persist_training_summary(status=status, epochs=TRAIN_EPOCHS if status == "trained" else 0, summary=summary)
        self._publish_status(status=status, df=df, summary=summary, steps=timestep_count)

    def _write_predictions(self, df, *, status: str) -> int:
        """Generate predictions and store/publish them."""
        self.pinn.model.eval()
        x_vals = np.linspace(0, 5.0, 50)
        time_points = sorted(df["time_days"].unique())
        for t_day in time_points:
            t_tensor = torch.full(
                (len(x_vals), 1),
                float(t_day),
                device=self.pinn.device,
                dtype=torch.float32,
            )
            x_tensor = torch.tensor(x_vals, device=self.pinn.device, dtype=torch.float32).unsqueeze(1)
            T_pred = self.pinn.predict(x_tensor, t_tensor).cpu().numpy().reshape(-1)

            self.influx.write_depth_series(
                measurement="pinn_forward",
                time_days=float(t_day),
                depths=x_vals.tolist(),
                temperatures=T_pred.tolist(),
                site="default",
                extra_tags={"model": "pinn"},
            )

        self.logger.info(
            "PINN forward %s complete (timesteps=%d, depths=%d)",
            "training" if status == "trained" else "inference",
            len(time_points),
            len(x_vals),
        )
        return len(time_points)

    def _persist_training_summary(
        self,
        *,
        status: str,
        epochs: int,
        summary: dict | None,
    ) -> dict:
        if status == "trained":
            return self.pinn.save_training_summary(self.history_path, status=status, epochs=epochs, summary=summary)

        if self.history_path.exists():
            try:
                content = json.loads(self.history_path.read_text())
                return content
            except Exception as exc:  # pragma: no cover - corrupted file
                self.logger.warning("Failed to load cached PINN history: %s", exc)

        fallback = self.pinn.build_training_summary(status=status, epochs=epochs)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.write_text(json.dumps(fallback, indent=2))
        return fallback

    def _publish_status(self, *, status: str, df, summary: dict, steps: int) -> None:
        if self.out_publisher is None:
            return

        message = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": status,
            "time_days": float(df["time_days"].max()),
            "model_path": str(self.model_path),
            "history_path": str(self.history_path),
            "epochs": summary.get("epochs", 0),
            "timesteps": steps,
        }
        self.out_publisher.publish(message)
        self.logger.info(
            "Published PINN status (status=%s, t=%.2fd) to %s",
            status,
            message["time_days"],
            self.forward_queue_config.queue,
        )

    # -------------------------------
    # QUEUE HANDLER
    # -------------------------------
    def _on_message(self, msg):
        """
        Handles messages from RabbitMQ and triggers retraining when FDM output is ready.
        """
        status_flag = msg.get("status")
        self.logger.info("Received FDM notification (status=%s)", status_flag)
        if status_flag == "ready":
            if self.enable_training:
                self.logger.info("Triggering PINN retraining")
                Thread(target=self.train_from_influx, daemon=True).start()
            else:
                self.logger.info("Training disabled; running PINN inference")
                self.train_from_influx()
        else:
            self.logger.warning("FDM notification ignored (status=%s)", status_flag)

    # -------------------------------
    # RUN
    # -------------------------------
    def run(self):
        if self.mq_client is None:
            raise RuntimeError("RabbitMQ client not initialised. Did you call setup()?")
        self.logger.info("Listening for FDM notifications on %s", self.fdm_queue_config.queue)
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
        self.logger.info(
            "Dependencies ready (inbound=%s, outbound=%s)",
            self.fdm_queue_config.queue,
            self.forward_queue_config.queue,
        )

    def start(self) -> None:
        """Start consuming FDM notifications."""

        if self.influx is None or self.mq_client is None:
            self.setup()

        try:
            self.run()
        except KeyboardInterrupt:  # pragma: no cover - runtime
            self.logger.info("Interrupt received; shutting down")
        finally:
            self.close()

    def stop(self) -> None:
        """Stop consuming further messages."""

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
        self.logger.info("Shutdown complete")


if __name__ == "__main__":
    PINNForwardServer().start()

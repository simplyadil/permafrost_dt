# pylint: disable=too-many-instance-attributes
"""PINN inversion server."""

import json
import os
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Optional

import pandas as pd
import torch

from software.digital_twin.communication.logger import setup_logger
from software.digital_twin.communication.messaging import RabbitMQClient, RabbitMQConfig, resolve_queue_config
from software.digital_twin.data_access.influx_utils import InfluxConfig, InfluxHelper
from software.digital_twin.simulator.pinn_inversion.inversion_pinn_model import InversionFreezingSoilPINN

FDM_QUEUE = "permafrost.record.fdm.state"
PINN_INVERSION_QUEUE = "permafrost.record.pinn_inversion.state"
FDM_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "fdm_output_message.json"
PINN_INVERSION_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "pinn_inversion_output_message.json"
INVERSION_HISTORY_FILENAME = "inversion_history.json"


class PINNInversionServer:
    """
    Performs inversion of soil parameters when forward PINN results are ready.
    """

    def __init__(
        self,
        influx_config: InfluxConfig | None = None,
        fdm_queue_config: RabbitMQConfig | None = None,
        inversion_queue_config: RabbitMQConfig | None = None,
        model_dir: str = "software/models/pinn_inversion",
        *,
        enable_training: bool = True,
        model_path: str | None = None,
    ) -> None:
        self.logger = setup_logger("PINNInversionServer")
        self.influx_config = influx_config or InfluxConfig()
        self.fdm_queue_config = resolve_queue_config(
            fdm_queue_config,
            queue=FDM_QUEUE,
            schema_path=FDM_SCHEMA,
        )
        self.inversion_queue_config = resolve_queue_config(
            inversion_queue_config,
            queue=PINN_INVERSION_QUEUE,
            schema_path=PINN_INVERSION_SCHEMA,
        )

        self.influx: Optional[InfluxHelper] = None
        self.mq_client: Optional[RabbitMQClient] = None
        self.out_publisher: Optional[RabbitMQClient] = None

        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.enable_training = enable_training
        default_checkpoint = Path(self.model_dir) / "inversion_pinn.pt"
        self.model_path = Path(model_path) if model_path is not None else default_checkpoint
        self.history_path = Path(self.model_dir) / INVERSION_HISTORY_FILENAME
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pinn = InversionFreezingSoilPINN(
            known_params={
                "L": 3.34e5,
                "C_i": 1.672,
                "C_l": 4.18,
                "lambda_i": 2.210,
                "lambda_l": 0.465,
            },
            param_bounds={
                "lambda_f": [1.3, 1.7],
                "C_f": [1.3, 1.7],
                "eta": [0.3, 0.5],
                "b": [1.0, 2.0],
                "T_nabla": [-1.5, -0.5],
            },
            device=self.device,
            log_callback=self.logger.info,
        )
        self._load_pretrained_weights()
        self.logger.info("PINNInversionServer configured.")

    def _on_message(self, msg):
        self.logger.info(f"Received message from FDM service: {msg}")
        if msg.get("status") == "ready":
            Thread(target=self.run_inversion, daemon=True).start()
        else:
            self.logger.warning("Ignored message with status '%s'.", msg.get("status"))

    def _load_pretrained_weights(self) -> None:
        if self.model_path.exists():
            try:
                self.pinn.load(str(self.model_path))
            except Exception as exc:  # pragma: no cover - depends on file IO
                self.logger.error("Failed to load inversion model from %s: %s", self.model_path, exc)
        else:
            if self.enable_training:
                self.logger.warning(
                    "No inversion checkpoint found at %s. Training will start from scratch.",
                    self.model_path,
                )
            else:
                self.logger.error(
                    "Inversion training disabled but checkpoint missing at %s. "
                    "Inference will not produce results until a model is provided.",
                    self.model_path,
                )

    def _build_boundary_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract surface temperature time series from FDM output."""
        if df.empty:
            return pd.DataFrame(columns=["time_days", "temperature_0m"])

        boundary_df = df.copy()
        mask = pd.Series(False, index=boundary_df.index)
        if "depth_m" in boundary_df.columns:
            mask |= boundary_df["depth_m"].astype(float).round(6) == 0.0
        if "depth" in boundary_df.columns:
            depth_numeric = (
                boundary_df["depth"]
                .astype(str)
                .str.replace("m", "", regex=False)
                .astype(float)
            )
            mask |= depth_numeric.round(6) == 0.0

        surface_rows = boundary_df.loc[mask].copy()
        if surface_rows.empty:
            self.logger.warning("No surface (0 m) entries found for inversion boundary data.")
            return pd.DataFrame(columns=["time_days", "temperature_0m"])

        surface_rows = surface_rows[["time_days", "temperature"]].dropna()
        surface_rows = surface_rows.sort_values("time_days").drop_duplicates(subset="time_days", keep="last")
        surface_rows = surface_rows.rename(columns={"temperature": "temperature_0m"})
        return surface_rows.reset_index(drop=True)

    def run_inversion(self):
        if self.influx is None:
            raise RuntimeError("Influx helper not initialised. Did you call setup()?") 

        self.logger.info("Fetching FDM temperature data from InfluxDB...")
        df = self.influx.query_model_temperature(measurement="fdm_simulation", limit=5000)
        if df is None or df.empty:
            self.logger.error("No FDM data found in InfluxDB.")
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
        x_obs_device = x_obs.to(self.device)
        t_obs_device = t_obs.to(self.device)
        T_obs_device = T_obs.to(self.device)

        self.pinn.boundary_data = self._build_boundary_dataframe(df)

        summary = None
        validation = None
        if self.enable_training:
            x_domain = (0.0, 5.0)
            t_domain = (0.0, 365.0)
            params_estimated = self.pinn.train(
                x_obs=x_obs_device,
                t_obs=t_obs_device,
                T_obs_true=T_obs_device,
                x_domain=x_domain,
                t_domain=t_domain,
                epochs=5000,
            )
            validation = self.pinn.build_validation_summary(x_obs_device, t_obs_device, T_obs_device)
            summary = self.pinn.save_training_summary(
                self.history_path,
                status="trained",
                epochs=5000,
                validation=validation,
            )
            try:
                self.pinn.save(str(self.model_path))
            except Exception as exc:  # pragma: no cover
                self.logger.error("Failed to save inversion checkpoint: %s", exc)
        else:
            if not self.model_path.exists():
                self.logger.error("Cannot emit inversion results. Checkpoint missing at %s.", self.model_path)
                return
            self._load_pretrained_weights()
            params_estimated = self.pinn.get_params()
            self.logger.info("Using pretrained inversion parameters: %s", params_estimated)
            validation = self.pinn.build_validation_summary(x_obs_device, t_obs_device, T_obs_device)
            summary = self._persist_cached_summary(status="inferred", epochs=0, validation=validation)

        # Log results
        self.logger.info(f"Inversion complete. Parameters: {params_estimated}")

        # Publish result message
        result_msg = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "inverted",
            "parameters": params_estimated,
            "history_path": str(self.history_path),
            "validation_overall": summary.get("validation", {}).get("overall") if summary else None,
        }
        if self.out_publisher is None:
            raise RuntimeError("Outbound RabbitMQ client not initialised. Did you call setup()?")
        self.out_publisher.publish(result_msg)
        self.logger.info("Result published to RabbitMQ.")

    def _persist_cached_summary(self, *, status: str, epochs: int, validation: dict | None) -> dict:
        if self.history_path.exists():
            try:
                return json.loads(self.history_path.read_text())
            except Exception as exc:  # pragma: no cover - corrupted file
                self.logger.warning("Failed to read cached inversion history: %s", exc)

        summary = self.pinn.build_training_summary(status=status, epochs=epochs, validation=validation)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.write_text(json.dumps(summary, indent=2))
        return summary

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
            self.mq_client = RabbitMQClient(self.fdm_queue_config)
        if self.out_publisher is None:
            self.out_publisher = RabbitMQClient(self.inversion_queue_config)
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

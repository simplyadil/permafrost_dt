# software/services/boundary_forcing/boundary_forcing_service.py
import time
import pandas as pd
from datetime import datetime, timedelta

from software.services.common.logger import setup_logger
from software.services.common.messaging import RabbitMQClient
from software.services.common.influx_utils import InfluxHelper


class BoundaryForcingService:
    """
    Reads temperature observations from InfluxDB and extracts
    boundary forcing signals (e.g., surface temperature Ts(t)).
    Publishes them to RabbitMQ for the FDM simulator.
    """

    def __init__(self, polling_interval=5):
        self.logger = setup_logger("boundary_forcing_service")
        self.polling_interval = polling_interval
        self.mq_client = RabbitMQClient(
            queue="permafrost.boundary.forcing",
            schema_path="software/services/common/schemas/boundary_forcing_message.json")

        self.influx = InfluxHelper()
        self.last_timestamp = None
        self.logger.info("boundary_forcing_service initialized.")

    def compute_boundary_forcing(self, df: pd.DataFrame):
        """
        Example logic: extract surface temperature (depth == 0.0)
        and compute its mean or gradient if needed.
        """
        if df.empty:
            return None

        surface = df[df["depth"] == 0.0]
        if surface.empty:
            self.logger.warning("No surface temperature data found.")
            return None

        # Assume Ts(t) = temperature at 0m
        Ts = surface[["time_days", "temperature"]].tail(1).to_dict("records")[0]
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "time_days": Ts["time_days"],
            "Ts": Ts["temperature"]
        }

    def run(self):
        self.logger.info("boundary_forcing_service started polling for new data...")
        while True:
            try:
                df = self.influx.query_temperature(limit=200)
                msg = self.compute_boundary_forcing(df)
                if msg:
                    self.mq_client.publish(msg)
                    self.logger.info(f"Published boundary forcing: Ts={msg['Ts']}°C @ t={msg['time_days']}")
                time.sleep(self.polling_interval)
            except Exception as e:
                self.logger.error(f"Error during boundary forcing computation: {e}")
                time.sleep(self.polling_interval)


if __name__ == "__main__":
    service = BoundaryForcingService(polling_interval=10)
    service.run()

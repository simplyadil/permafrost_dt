# software/services/obs_io_service.py
import datetime
from ..common.messaging import RabbitMQClient
from ..common.influx_utils import InfluxHelper
from ..common.logger import setup_logger

SERVICE_NAME = "obs_io_service"

class ObservationIngestionService:
    """
    Consumes physical twin sensor data from RabbitMQ
    and writes it into InfluxDB for the digital twin pipeline.
    """

    def __init__(self):
        self.logger = setup_logger(SERVICE_NAME)
        self.mq_client = RabbitMQClient(queue="permafrost.sensors.temperature")
        self.db = InfluxHelper()
        self.logger.info("obs_io_service initialized.")

    # -----------------------------------------------------
    # CALLBACK: when new message arrives
    # -----------------------------------------------------
    def process_message(self, msg: dict):
        """
        Processes a validated message:
        - Writes temperature profile by depth into InfluxDB
        """
        try:
            time_days = msg["time_days"]
            timestamp = msg.get("timestamp", datetime.datetime.utcnow().isoformat())

            # Write each depth reading as an individual point
            for depth_key, temp_value in msg.items():
                if depth_key.startswith("temperature_"):
                    depth_m = float(depth_key.replace("temperature_", "").replace("m", ""))
                    self.db.write_temperature(
                        time_days=time_days,
                        depth=depth_m,
                        temperature=temp_value
                    )

            self.logger.info(f"Processed message at t={time_days} days → written 6 depths")
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")

    # -----------------------------------------------------
    # RUN LOOP
    # -----------------------------------------------------
    def run(self):
        """Starts consuming messages from RabbitMQ."""
        self.logger.info("obs_io_service is now listening for sensor data...")
        self.mq_client.consume(callback=self.process_message)


if __name__ == "__main__":
    service = ObservationIngestionService()
    service.run()

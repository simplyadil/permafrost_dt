# software/services/common/messaging.py
import os
import json
import pika
import jsonschema
from jsonschema import validate
from software.services.common.logger import setup_logger


class RabbitMQClient:
    """
    A unified interface for publishing and consuming messages to/from RabbitMQ.
    """

    def __init__(
        self,
        host="localhost",
        queue="permafrost.sensors.temperature",
        schema_path=None
    ):
        self.logger = setup_logger("RabbitMQ")
        self.host = host
        self.queue = queue
        self.connection = None
        self.channel = None

        # ------------------------------------------------------
        # Dynamic schema path resolution
        # ------------------------------------------------------
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if schema_path is None:
            schema_path = os.path.join(base_dir, "schemas", "sensor_message.json")

        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Schema file not found at {schema_path}")

        with open(schema_path, "r") as f:
            self.schema = json.load(f)

    # ------------------------------------------------------
    # CONNECTION MANAGEMENT
    # ------------------------------------------------------
    def connect(self):
        """Connect to RabbitMQ and declare queue if not existing."""
        if self.connection and self.connection.is_open:
            return

        try:
            credentials = pika.PlainCredentials('permafrost', 'permafrost')
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=self.host, credentials=credentials)
            )
            self.channel = self.connection.channel()
            self.channel.queue_declare(queue=self.queue, durable=True)
            self.logger.info(f"Connected to RabbitMQ ({self.host}), queue: {self.queue}")
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            raise

    # ------------------------------------------------------
    # MESSAGE VALIDATION
    # ------------------------------------------------------
    def validate_message(self, message: dict):
        """Validates message content against the JSON schema."""
        try:
            validate(instance=message, schema=self.schema)
        except jsonschema.exceptions.ValidationError as e:
            self.logger.error(f"Message validation failed: {e.message}")
            raise

    # ------------------------------------------------------
    # PUBLISH
    # ------------------------------------------------------
    def publish(self, message: dict):
        """Publishes a validated JSON message to RabbitMQ."""
        self.connect()
        self.validate_message(message)

        try:
            self.channel.basic_publish(
                exchange="",
                routing_key=self.queue,
                body=json.dumps(message),
                properties=pika.BasicProperties(delivery_mode=2)
            )
            self.logger.info(f"Published message to {self.queue}: time_days={message.get('time_days')}")
        except Exception as e:
            self.logger.error(f"Failed to publish message: {e}")
            raise

    # ------------------------------------------------------
    # CONSUME
    # ------------------------------------------------------
    def consume(self, callback, auto_ack=False):
        """
        Consumes messages from the queue and applies a user-defined callback(msg_dict).
        If auto_ack=True, messages are acknowledged automatically.
        """
        self.connect()

        def _callback(ch, method, properties, body):
            try:
                msg = json.loads(body)
                self.validate_message(msg)
                self.logger.info(f"Received message from {self.queue} (t={msg.get('time_days')})")
                callback(msg)
                if not auto_ack:
                    ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
                if not auto_ack:
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=self.queue, on_message_callback=_callback, auto_ack=auto_ack)
        self.logger.info(f"Started consuming from queue: {self.queue}")
        self.channel.start_consuming()

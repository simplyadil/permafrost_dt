"""RabbitMQ helpers aligned with the digital twin blueprint."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import jsonschema
import pika
from pika.adapters.blocking_connection import BlockingChannel

from .logger import setup_logger


Message = Dict[str, Any]
MessageCallback = Callable[[Message], None]


@dataclass(frozen=True)
class RabbitMQConfig:
    """Configuration payload for connecting to RabbitMQ."""

    host: str = "localhost"
    queue: str = ""
    schema_path: Optional[Path | str] = None
    username: str = "permafrost"
    password: str = "permafrost"


def resolve_queue_config(
    config: RabbitMQConfig | None,
    *,
    queue: str,
    schema_path: Path | str,
) -> RabbitMQConfig:
    """Normalise queue configuration with a default schema when missing."""

    if config is None:
        config = RabbitMQConfig(schema_path=schema_path)
    elif config.schema_path is None:
        config = replace(config, schema_path=schema_path)
    return replace(config, queue=queue)


class RabbitMQClient:
    """Unified interface for publishing to and consuming from RabbitMQ queues."""

    def __init__(self, config: RabbitMQConfig) -> None:
        if not config.queue:
            raise ValueError("RabbitMQ queue must be provided.")

        self.logger = setup_logger("RabbitMQ")
        self.config = config
        self.host = config.host
        self.queue = config.queue
        self.connection: Optional[pika.BlockingConnection] = None
        self.channel: Optional[BlockingChannel] = None
        self.schema = self._load_schema(config.schema_path)

    @staticmethod
    def _default_schema_path() -> Path:
        base_dir = Path(__file__).resolve().parent
        return base_dir / "schemas" / "sensor_message.json"

    def _load_schema(self, schema_path: Optional[Path | str]) -> Mapping[str, Any]:
        path = Path(schema_path) if schema_path is not None else self._default_schema_path()
        if not path.exists():
            raise FileNotFoundError(f"Schema file not found at {path}")

        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    # ------------------------------------------------------
    # CONNECTION MANAGEMENT
    # ------------------------------------------------------
    def connect(self) -> None:
        """Connect to RabbitMQ and declare the target queue if needed."""

        if self.connection and self.connection.is_open:
            return

        try:
            credentials = pika.PlainCredentials(self.config.username, self.config.password)
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=self.host, credentials=credentials, heartbeat=600)
            )
            self.channel = self.connection.channel()
            self.channel.queue_declare(queue=self.queue, durable=True)
            self.logger.info("Connected to RabbitMQ (%s), queue: %s", self.host, self.queue)
        except Exception as exc:  # pragma: no cover - requires broker
            self.logger.error("Connection error: %s", exc)
            raise

    def disconnect(self) -> None:
        """Close the connection if it's currently open."""

        if self.connection and self.connection.is_open:
            self.connection.close()
            self.logger.info("Disconnected from RabbitMQ (%s)", self.host)
        self.connection = None
        self.channel = None

    def __enter__(self) -> "RabbitMQClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.disconnect()

    # ------------------------------------------------------
    # MESSAGE VALIDATION
    # ------------------------------------------------------
    def validate_message(self, message: Mapping[str, Any]) -> None:
        """Validate message content against the configured JSON schema."""

        try:
            jsonschema.validate(instance=message, schema=self.schema)
        except jsonschema.exceptions.ValidationError as exc:
            self.logger.error("Message validation failed: %s", exc.message)
            raise

    # ------------------------------------------------------
    # PUBLISH
    # ------------------------------------------------------
    def publish(self, message: Mapping[str, Any]) -> None:
        """Validate and publish a JSON-serialisable message to RabbitMQ."""

        self.connect()
        self.validate_message(message)

        if self.channel is None:  # pragma: no cover - defensive
            raise RuntimeError("Channel unavailable. Did connect() succeed?")

        try:
            self.channel.basic_publish(
                exchange="",
                routing_key=self.queue,
                body=json.dumps(message),
                properties=pika.BasicProperties(delivery_mode=2),
            )
            self.logger.info(
                "Published message to %s: keys=%s",
                self.queue,
                list(message.keys()),
            )
        except Exception as exc:  # pragma: no cover - requires broker
            self.logger.error("Failed to publish message: %s", exc)
            raise

    # ------------------------------------------------------
    # CONSUME
    # ------------------------------------------------------
    def consume(self, callback: MessageCallback, auto_ack: bool = False) -> None:
        """Consume messages from the queue and hand them to ``callback``."""

        self.connect()

        if self.channel is None:  # pragma: no cover - defensive
            raise RuntimeError("Channel unavailable. Did connect() succeed?")

        def _callback(ch: BlockingChannel, method, properties, body: bytes) -> None:
            try:
                msg: Message = json.loads(body)
                self.validate_message(msg)
                self.logger.info(
                    "Received message from %s (keys=%s)",
                    self.queue,
                    list(msg.keys()),
                )
                callback(msg)
                if not auto_ack:
                    ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as exc:  # pragma: no cover - requires broker
                self.logger.error("Error processing message: %s", exc)
                if not auto_ack:
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=self.queue, on_message_callback=_callback, auto_ack=auto_ack)
        self.logger.info("Started consuming from queue: %s", self.queue)
        self.channel.start_consuming()

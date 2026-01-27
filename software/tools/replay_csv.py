"""Replay CSV sensor data into the 2D sensor queue."""

from __future__ import annotations

import argparse
import csv
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from software.digital_twin.communication.messaging import RabbitMQClient, RabbitMQConfig

SENSOR_QUEUE = "permafrost.record.sensors.2d"
SENSOR_SCHEMA = Path(__file__).resolve().parents[1] / "digital_twin" / "communication" / "schemas" / "sensor_message_2d.json"

SENSOR_PATTERN = re.compile(r"D(?P<depth>\d+(?:\.\d+)?)[^0-9]*W(?P<width>\d+(?:\.\d+)?)", re.IGNORECASE)


def _parse_header(header: list[str], time_column: str | None) -> tuple[int, list[dict]]:
    if not header:
        raise ValueError("CSV header is empty.")

    time_idx = None
    if time_column:
        for idx, name in enumerate(header):
            if name.strip() == time_column:
                time_idx = idx
                break
    else:
        for idx, name in enumerate(header):
            cleaned = name.strip()
            if cleaned in {"Time (h)", "Time(h)", "Time"}:
                time_idx = idx
                break
        if time_idx is None:
            for idx, name in enumerate(header):
                cleaned = name.strip().lower()
                if "time" in cleaned and "h" in cleaned:
                    time_idx = idx
                    break

    if time_idx is None:
        raise ValueError("Time column not found. Use --time-column to specify it.")

    sensors = []
    for idx, name in enumerate(header):
        if idx == time_idx:
            continue
        if not name:
            continue
        match = SENSOR_PATTERN.match(name.strip())
        if not match:
            continue
        depth_mm = float(match.group("depth"))
        width_mm = float(match.group("width"))
        sensors.append(
            {
                "index": idx,
                "sensor_id": name.strip(),
                "x_m": width_mm / 1000.0,
                "z_m": depth_mm / 1000.0,
            }
        )

    if not sensors:
        preview = ", ".join(name.strip() for name in header[:10] if name)
        raise ValueError(
            "No sensor columns found. Expected headers like D10-W15. "
            f"Header preview: {preview}"
        )

    return time_idx, sensors


def replay_csv(
    path: Path,
    *,
    queue: str,
    exchange: str | None,
    exchange_type: str,
    routing_key: str,
    host: str,
    username: str,
    password: str,
    time_column: str | None,
    start_row: int,
    limit: int | None,
    sleep_seconds: float,
) -> None:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found at {path}")

    config = RabbitMQConfig(
        host=host,
        queue=queue,
        exchange=exchange,
        exchange_type=exchange_type,
        routing_key=routing_key,
        schema_path=SENSOR_SCHEMA,
        username=username,
        password=password,
    )

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        time_idx, sensors = _parse_header(header, time_column)

        with RabbitMQClient(config) as client:
            for row_index, row in enumerate(reader, start=1):
                if row_index < start_row:
                    continue
                if limit is not None and (row_index - start_row) >= limit:
                    break

                if time_idx >= len(row):
                    continue

                time_value = row[time_idx].strip()
                if not time_value:
                    continue

                try:
                    time_hours = float(time_value)
                except ValueError:
                    continue

                payload_sensors = []
                for sensor in sensors:
                    idx = sensor["index"]
                    if idx >= len(row):
                        continue
                    value = row[idx].strip()
                    if value == "":
                        continue
                    try:
                        temperature = float(value)
                    except ValueError:
                        continue
                    payload_sensors.append(
                        {
                            "sensor_id": sensor["sensor_id"],
                            "x_m": sensor["x_m"],
                            "z_m": sensor["z_m"],
                            "temperature": temperature,
                        }
                    )

                if not payload_sensors:
                    continue

                message = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "time_hours": time_hours,
                    "sensors": payload_sensors,
                }
                client.publish(message)

                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay CSV sensor data to RabbitMQ.")
    parser.add_argument("--path", required=True, help="Path to the CSV file.")
    parser.add_argument("--queue", default=None, help="RabbitMQ queue name.")
    parser.add_argument(
        "--exchange",
        default="permafrost.sensors.2d",
        help="RabbitMQ exchange name for fanout.",
    )
    parser.add_argument("--exchange-type", default="fanout", help="RabbitMQ exchange type.")
    parser.add_argument("--routing-key", default="", help="Routing key for exchange publishing.")
    parser.add_argument("--host", default="localhost", help="RabbitMQ host.")
    parser.add_argument("--username", default="permafrost", help="RabbitMQ username.")
    parser.add_argument("--password", default="permafrost", help="RabbitMQ password.")
    parser.add_argument("--time-column", default=None, help="Override the time column header.")
    parser.add_argument("--start-row", type=int, default=1, help="First data row to replay (1-based).")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of rows to publish.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Sleep between rows.")

    args = parser.parse_args()
    if args.queue is None:
        queue = "" if args.exchange else SENSOR_QUEUE
    else:
        queue = args.queue

    replay_csv(
        Path(args.path),
        queue=queue,
        exchange=args.exchange,
        exchange_type=args.exchange_type,
        routing_key=args.routing_key,
        host=args.host,
        username=args.username,
        password=args.password,
        time_column=args.time_column,
        start_row=args.start_row,
        limit=args.limit,
        sleep_seconds=args.sleep_seconds,
    )


if __name__ == "__main__":
    main()

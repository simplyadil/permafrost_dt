"""Replay CSV/XLSX sensor data into the 2D sensor queue."""

from __future__ import annotations

import argparse
import csv
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from software.digital_twin.communication.messaging import RabbitMQClient, RabbitMQConfig
from software.digital_twin.data_access.influx_utils import InfluxConfig, InfluxHelper

SENSOR_QUEUE = "permafrost.record.sensors.2d"
SENSOR_SCHEMA = Path(__file__).resolve().parents[1] / "digital_twin" / "communication" / "schemas" / "sensor_message_2d.json"
FLUID_EXCHANGE = "permafrost.fluid"
FLUID_SCHEMA = Path(__file__).resolve().parents[1] / "digital_twin" / "communication" / "schemas" / "fluid_message.json"

_INLET_ALIASES = ("Inlet", "Fluid Temperature_Inlet", "Inlet(1)")
_OUTLET_ALIASES = ("Outlet", "Fluid Temperatue_Outlet", "Fluid Temperature_Outlet", "Outlet(1)")

# Matches H{height_mm}D{distance_mm}, e.g. H10D0, H290D165
SENSOR_PATTERN = re.compile(r"H(?P<height>\d+(?:\.\d+)?)D(?P<distance>\d+(?:\.\d+)?)", re.IGNORECASE)


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
        height_mm = float(match.group("height"))
        distance_mm = float(match.group("distance"))
        sensors.append(
            {
                "index": idx,
                "sensor_id": name.strip(),
                "x_m": distance_mm / 1000.0,
                "z_m": height_mm / 1000.0,
            }
        )

    if not sensors:
        preview = ", ".join(name.strip() for name in header[:10] if name)
        raise ValueError(
            "No sensor columns found. Expected headers like H10D0. "
            f"Header preview: {preview}"
        )

    return time_idx, sensors


def _load_fluid_temps(
    path: Path,
    *,
    inlet_col: str,
    outlet_col: str,
    time_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load fluid inlet/outlet temps from CSV or Excel; returns (t_h, t_inlet, t_outlet) sorted by time.

    Column names fall back to known aliases, so both the shipped demo files
    (``Inlet``/``Outlet``) and the original lab spreadsheets
    (``Fluid Temperature_Inlet``/``Fluid Temperatue_Outlet``) load without flags.
    """
    import pandas as pd
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    def _resolve(requested: str, aliases: tuple[str, ...]) -> str:
        for name in (requested, *aliases):
            if name in df.columns:
                return name
        return requested

    time_col = _resolve(time_col, ("Time (h)", "Time(h)", "Time"))
    inlet_col = _resolve(inlet_col, _INLET_ALIASES)
    outlet_col = _resolve(outlet_col, _OUTLET_ALIASES)
    missing = [c for c in [time_col, inlet_col, outlet_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Fluid temp file is missing columns: {missing}\n"
            f"Available: {list(df.columns)}"
        )
    df = df[[time_col, inlet_col, outlet_col]].dropna().sort_values(time_col)
    return (
        df[time_col].to_numpy(dtype=float),
        df[inlet_col].to_numpy(dtype=float),
        df[outlet_col].to_numpy(dtype=float),
    )


def _iter_rows(path: Path) -> Iterator[list[str]]:
    """Yield rows as lists of strings from a CSV or XLSX file."""
    if path.suffix.lower() in {".xlsx", ".xls"}:
        import pandas as pd
        df = pd.read_excel(path, dtype=str, header=None, keep_default_na=False)
        for row in df.itertuples(index=False, name=None):
            yield list(row)
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            yield from csv.reader(handle)


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
    fluid_path: Path | None = None,
    fluid_inlet_col: str = "Inlet",
    fluid_outlet_col: str = "Outlet",
    fluid_time_col: str = "Time (h)",
    fluid_exchange: str = FLUID_EXCHANGE,
    influx_url: str = "http://localhost:8086",
    influx_token: str = "permafrost",
    influx_org: str = "permafrost",
    influx_bucket: str = "permafrost_data",
    site_id: str = "default",
) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found at {path}")

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

    fluid_temps: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    influx: InfluxHelper | None = None
    fluid_client: RabbitMQClient | None = None
    if fluid_path is not None:
        if not fluid_path.exists():
            raise FileNotFoundError(f"Fluid temp file not found at {fluid_path}")
        fluid_temps = _load_fluid_temps(
            fluid_path,
            inlet_col=fluid_inlet_col,
            outlet_col=fluid_outlet_col,
            time_col=fluid_time_col,
        )
        # Publish to the fluid fanout the pipe_boundary service consumes.
        fluid_client = RabbitMQClient(
            RabbitMQConfig(
                host=host,
                queue="",
                exchange=fluid_exchange,
                exchange_type="fanout",
                schema_path=FLUID_SCHEMA,
                username=username,
                password=password,
            )
        )
        fluid_client.connect()
        # Also mirror to Influx (best-effort; no-op if Influx is unreachable).
        influx = InfluxHelper(
            InfluxConfig(url=influx_url, token=influx_token, org=influx_org, bucket=influx_bucket)
        )
        print(
            f"Fluid temps loaded: {len(fluid_temps[0])} rows from {fluid_path.name} "
            f"-> publishing to '{fluid_exchange}' fanout"
        )

    row_iter = _iter_rows(path)
    header = next(row_iter, [])
    time_idx, sensors = _parse_header(header, time_column)

    try:
        with RabbitMQClient(config) as client:
            for row_index, row in enumerate(row_iter, start=1):
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

                if fluid_temps is not None:
                    t_arr, tin_arr, tout_arr = fluid_temps
                    t_in = float(np.interp(time_hours, t_arr, tin_arr))
                    t_out = float(np.interp(time_hours, t_arr, tout_arr))
                    if fluid_client is not None:
                        fluid_client.publish(
                            {
                                "time_hours": time_hours,
                                "timestamp": message["timestamp"],
                                "t_inlet_c": t_in,
                                "t_outlet_c": t_out,
                                "site_id": site_id,
                            }
                        )
                    if influx is not None:
                        influx.write_fluid_temperatures(
                            time_hours=time_hours,
                            t_inlet_c=t_in,
                            t_outlet_c=t_out,
                            site=site_id,
                        )

                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
    finally:
        if fluid_client is not None:
            fluid_client.disconnect()
        if influx is not None:
            influx.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay CSV sensor data to RabbitMQ.")
    parser.add_argument("--path", required=True, help="Path to the CSV or XLSX file.")
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

    fluid_group = parser.add_argument_group("fluid temperatures")
    fluid_group.add_argument("--fluid-path", default=None, help="Path to the fluid inlet/outlet temperature file (CSV or XLSX).")
    fluid_group.add_argument("--fluid-inlet-col", default="Inlet", help="Inlet temperature column name.")
    fluid_group.add_argument("--fluid-outlet-col", default="Outlet", help="Outlet temperature column name.")
    fluid_group.add_argument("--fluid-time-col", default="Time (h)", help="Time column name in the fluid file.")
    fluid_group.add_argument("--fluid-exchange", default=FLUID_EXCHANGE, help="Fanout exchange for fluid messages (consumed by pipe_boundary).")
    fluid_group.add_argument("--site-id", default="default", help="Site ID tag written to InfluxDB.")

    influx_group = parser.add_argument_group("influxdb (required when --fluid-path is set)")
    influx_group.add_argument("--influx-url", default="http://localhost:8086", help="InfluxDB URL.")
    influx_group.add_argument("--influx-token", default="permafrost", help="InfluxDB token.")
    influx_group.add_argument("--influx-org", default="permafrost", help="InfluxDB org.")
    influx_group.add_argument("--influx-bucket", default="permafrost_data", help="InfluxDB bucket.")

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
        fluid_path=Path(args.fluid_path) if args.fluid_path else None,
        fluid_inlet_col=args.fluid_inlet_col,
        fluid_outlet_col=args.fluid_outlet_col,
        fluid_time_col=args.fluid_time_col,
        fluid_exchange=args.fluid_exchange,
        influx_url=args.influx_url,
        influx_token=args.influx_token,
        influx_org=args.influx_org,
        influx_bucket=args.influx_bucket,
        site_id=args.site_id,
    )


if __name__ == "__main__":
    main()

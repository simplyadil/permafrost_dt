"""InfluxDB helpers aligned with the digital twin blueprint."""

import datetime
import os
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision  # type: ignore
from influxdb_client.client.write_api import SYNCHRONOUS  # type: ignore

from ..communication.logger import setup_logger


@dataclass(frozen=True)
class InfluxConfig:
    """Runtime configuration for connecting to InfluxDB."""

    url: str = "http://localhost:8086"
    token: str = "permafrost"
    org: str = "permafrost"
    bucket: str = "permafrost_data"
    username: str = "permafrost"
    password: str = "permafrost"


class InfluxHelper:
    def __init__(self, config: InfluxConfig | None = None) -> None:
        self.config = config or InfluxConfig()
        self.logger = setup_logger("InfluxDB")
        # Allow overriding connection settings from environment variables
        url = os.getenv("INFLUX_URL", self.config.url)
        token = os.getenv("INFLUX_TOKEN", self.config.token)
        org = os.getenv("INFLUX_ORG", self.config.org)
        bucket = os.getenv("INFLUX_BUCKET", self.config.bucket)

        self.bucket = bucket

        # Create client but handle auth/connection failures gracefully so services
        # don't crash on 401/connection errors during startup. In such cases we
        # keep query_api/write_api as None and return empty results or no-op writes.
        try:
            self.client = InfluxDBClient(
                url=url,
                token=token,
                org=org,
                username=self.config.username,
                password=self.config.password,
            )
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.query_api = self.client.query_api()
        except Exception as exc:  # pragma: no cover - depends on runtime infra
            self.logger.error("Failed to create InfluxDB client: %s", exc)
            self.client = None
            self.write_api = None
            self.query_api = None

    def write_temperature(self, time_days, depth, temperature, site="default"):
        point = (
            Point("sensor_temperature")
            .tag("site_id", site)
            .tag("depth", f"{depth:.1f}m")
            .field("temperature", float(temperature))
            .field("time_days", float(time_days))
            .field("depth_m", float(depth))
            .time(datetime.datetime.utcnow(), write_precision=WritePrecision.NS)
        )
        if self.write_api is None:  # pragma: no cover - runtime fallback
            self.logger.warning("Influx write skipped (no client available): T=%s @ %sm, t=%s", temperature, depth, time_days)
            return

        try:
            self.write_api.write(bucket=self.bucket, record=point)
            self.logger.info(f"Written T={temperature}°C @ {depth}m, t={time_days}d")
        except Exception as exc:  # pragma: no cover - runtime infra
            self.logger.error("Error writing to InfluxDB: %s", exc)

    def query_temperature(self, site="default", depth=None, limit=100):
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: -30d)
          |> filter(fn: (r) => r["_measurement"] == "sensor_temperature")
          |> filter(fn: (r) => r["site_id"] == "{site}")
        '''
        if depth is not None:
            query += f'\n  |> filter(fn: (r) => r["depth"] == "{depth:.1f}m")'

        query += f'''
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> limit(n: {limit})
        '''

        tables = self.query_api.query_data_frame(query)
        if isinstance(tables, list) and len(tables) > 0:
            return self._coerce_depth_column(tables[0])
        if isinstance(tables, pd.DataFrame):
            return self._coerce_depth_column(tables)
        return pd.DataFrame()
        

    # Generic writer for model/simulation outputs
    def write_model_temperature(
        self,
        measurement: str,
        time_days: float,
        depth: float,
        temperature: float,
        *,
        site: str = "default",
        extra_tags: Optional[dict] = None,
    ) -> None:
        """
        Write temperature to an arbitrary measurement (e.g., 'fdm_simulation').
        Depth is stored as tag 'depth' (e.g., '1.0m') for fast filtering.
        """
        tags = {"site_id": site, "depth": f"{depth:.1f}m"}
        if extra_tags:
            tags.update(extra_tags)

        point = Point(measurement)
        for k, v in tags.items():
            point = point.tag(k, v)

        point = (point
                 .field("temperature", float(temperature))
                 .field("time_days", float(time_days))
                 .field("depth_m", float(depth))
                 .time(datetime.datetime.utcnow(), write_precision=WritePrecision.NS))  # type: ignore

        if self.write_api is None:  # pragma: no cover - runtime fallback
            self.logger.warning("Influx write skipped (no client available) for measurement %s", measurement)
            return

        try:
            self.write_api.write(bucket=self.bucket, record=point)
            self.logger.info(f"[{measurement}] Written T={temperature}°C @ {depth}m, t={time_days}d")
        except Exception as exc:  # pragma: no cover - runtime infra
            self.logger.error("Error writing to InfluxDB (measurement=%s): %s", measurement, exc)

    def query_model_temperature(
        self,
        measurement: str,
        site: str = "default",
        depth: float | None = None,
        limit: int = 200,
    ):
        """
        Query arbitrary measurement (e.g., 'fdm_simulation'), pivoting fields for DataFrame usability.
        """
        q = f'''
        from(bucket: "{self.bucket}")
          |> range(start: -30d)
          |> filter(fn: (r) => r["_measurement"] == "{measurement}")
          |> filter(fn: (r) => r["site_id"] == "{site}")
        '''
        if depth is not None:
            q += f'\n  |> filter(fn: (r) => r["depth"] == "{depth:.1f}m")'

        q += f'''
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> limit(n: {limit})
        '''
        if self.query_api is None:  # pragma: no cover - runtime fallback
            self.logger.warning("Influx query skipped (no client available) for measurement %s", measurement)
            return pd.DataFrame()

        try:
            tables = self.query_api.query_data_frame(q)
        except Exception as exc:  # pragma: no cover - runtime infra
            self.logger.error("Error querying InfluxDB (measurement=%s): %s", measurement, exc)
            return pd.DataFrame()

        if isinstance(tables, list) and len(tables) > 0:
            return self._coerce_depth_column(tables[0])
        if hasattr(tables, "empty"):
            return self._coerce_depth_column(tables)  # type: ignore[arg-type]
        return pd.DataFrame()

    def close(self) -> None:
        """Release the underlying InfluxDB client."""

        self.logger.info("Closing InfluxDB client.")
        if getattr(self, "client", None) is not None:
            try:
                self.client.close()
            except Exception:
                pass

    @staticmethod
    def _coerce_depth_column(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure a numeric depth column is available for downstream consumers."""

        if df is None or getattr(df, "empty", False):
            return df

        result = df.copy()

        if "depth" in result.columns:
            result["depth_m"] = result["depth"].apply(_parse_depth_value)
        elif "depth_m" in result.columns:
            result["depth_m"] = result["depth_m"].apply(_parse_depth_value)

        return result


def _parse_depth_value(raw: Any) -> float:
    """Convert depth representations such as '0.0m' into float metres."""

    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        value = raw.strip().lower().rstrip("m")
        try:
            return float(value)
        except ValueError:
            return float("nan")
    return float("nan")

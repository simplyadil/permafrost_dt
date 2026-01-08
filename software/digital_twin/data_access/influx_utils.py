"""InfluxDB helpers aligned with the digital twin blueprint."""

import datetime
import os
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Optional, Sequence

import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision  # type: ignore
from influxdb_client.client.write_api import SYNCHRONOUS  # type: ignore

from software.utils.logging_setup import get_logger


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
        self.logger = get_logger("InfluxDB")
        # Allow overriding connection settings from environment variables
        url = os.getenv("INFLUX_URL", self.config.url)
        token = os.getenv("INFLUX_TOKEN", self.config.token)
        org = os.getenv("INFLUX_ORG", self.config.org)
        bucket = os.getenv("INFLUX_BUCKET", self.config.bucket)

        self.bucket = bucket
        self._logged_measurements: set[str] = set()

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
        self.write_depth_series(
            measurement="sensor_temperature",
            time_days=float(time_days),
            depths=[float(depth)],
            temperatures=[float(temperature)],
            site=site,
        )

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
        '''
        if limit is not None and int(limit) > 0:
            query += f'\n  |> limit(n: {int(limit)})'

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
        self.write_depth_series(
            measurement=measurement,
            time_days=time_days,
            depths=[depth],
            temperatures=[temperature],
            site=site,
            extra_tags=extra_tags,
        )

    def write_depth_series(
        self,
        *,
        measurement: str,
        time_days: float,
        depths: Sequence[float],
        temperatures: Sequence[float],
        site: str = "default",
        extra_tags: Optional[dict] = None,
    ) -> None:
        """Write a snapshot of temperatures across depths with a single summary log."""

        if not depths:
            return

        if self.write_api is None:  # pragma: no cover - runtime fallback
            self.logger.warning("Influx write skipped (no client available) for measurement %s", measurement)
            return

        tags = {"site_id": site}
        if extra_tags:
            tags.update(extra_tags)

        timestamp = datetime.datetime.utcnow()
        records = []
        for depth, temperature in zip(depths, temperatures):
            point = Point(measurement)
            for key, value in tags.items():
                point = point.tag(key, value)
            point = (
                point.tag("depth", f"{float(depth):.1f}m")
                .field("temperature", float(temperature))
                .field("time_days", float(time_days))
                .field("depth_m", float(depth))
                .time(timestamp, write_precision=WritePrecision.NS)
            )
            records.append(point)

        try:
            self.write_api.write(bucket=self.bucket, record=records)
        except Exception as exc:  # pragma: no cover - runtime infra
            self.logger.error("Error writing to InfluxDB (measurement=%s): %s", measurement, exc)
            return

        if measurement not in self._logged_measurements:
            values = [float(t) for t in temperatures]
            min_temp = min(values)
            max_temp = max(values)
            mean_temp = fmean(values)
            self.logger.info(
                "%s write complete (t=%.2fd, depths=%d, min=%.2f°C, max=%.2f°C, mean=%.2f°C)",
                measurement,
                float(time_days),
                len(values),
                min_temp,
                max_temp,
                mean_temp,
            )
            self._logged_measurements.add(measurement)

    def write_inversion_parameters(
        self,
        parameters: dict[str, float],
        *,
        status: str = "inverted",
        site: str = "default",
        validation: dict[str, float] | None = None,
        timestamp: datetime.datetime | None = None,
    ) -> None:
        """Persist aggregated PINN inversion parameters for downstream visualization."""

        if not parameters:
            return

        if self.write_api is None:  # pragma: no cover - runtime fallback
            self.logger.warning("Influx write skipped (no client available) for measurement pinn_inversion")
            return

        record = Point("pinn_inversion").tag("site_id", site).tag("status", status)
        timestamp = timestamp or datetime.datetime.utcnow()

        for key, value in parameters.items():
            try:
                record = record.field(key, float(value))
            except (TypeError, ValueError):
                self.logger.warning("Skipping invalid inversion parameter %s=%s", key, value)

        if validation:
            for key, value in validation.items():
                if value is None:
                    continue
                try:
                    record = record.field(f"validation_{key}", float(value))
                except (TypeError, ValueError):
                    self.logger.debug("Skipping invalid validation metric %s=%s", key, value)

        record = record.field("timestamp", timestamp.isoformat()).time(timestamp, write_precision=WritePrecision.NS)

        try:
            self.write_api.write(bucket=self.bucket, record=[record])
        except Exception as exc:  # pragma: no cover - runtime infra
            self.logger.error("Error writing inversion parameters to InfluxDB: %s", exc)

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
        '''
        if limit is not None and int(limit) > 0:
            q += f'\n  |> limit(n: {int(limit)})'
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

    def _query_measurement(
        self,
        measurement: str,
        *,
        site: str = "default",
        limit: int = 1000,
        range_start: str = "-30d",
    ) -> pd.DataFrame:
        """Generic measurement query helper with pivoted fields."""

        if self.query_api is None:  # pragma: no cover - runtime fallback
            self.logger.warning("Influx query skipped (no client available) for measurement %s", measurement)
            return pd.DataFrame()

        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: {range_start})
          |> filter(fn: (r) => r["_measurement"] == "{measurement}")
        '''
        if site:
            query += f'\n  |> filter(fn: (r) => r["site_id"] == "{site}")'
        query += f'''
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> sort(columns: ["_time"])
        '''
        if limit is not None and int(limit) > 0:
            query += f'\n  |> limit(n: {int(limit)})'

        try:
            tables = self.query_api.query_data_frame(query)
        except Exception as exc:  # pragma: no cover - runtime infra
            self.logger.error("Error querying InfluxDB (measurement=%s): %s", measurement, exc)
            return pd.DataFrame()

        if isinstance(tables, list) and tables:
            df = tables[0]
        elif isinstance(tables, pd.DataFrame):
            df = tables
        else:
            return pd.DataFrame()

        if getattr(df, "empty", False):
            return pd.DataFrame()

        df = df.sort_values("_time").reset_index(drop=True)
        return df

    def query_boundary_flux(
        self,
        *,
        site: str = "default",
        limit: int = 1000,
        range_start: str = "-30d",
    ) -> pd.DataFrame:
        """Fetch boundary heat flux diagnostics."""

        return self._query_measurement(
            "boundary_flux",
            site=site,
            limit=limit,
            range_start=range_start,
        )

    def query_pinn_residuals(
        self,
        *,
        site: str = "default",
        limit: int = 5000,
        range_start: str = "-30d",
    ) -> pd.DataFrame:
        """Fetch PINN residual diagnostics."""

        df = self._query_measurement(
            "pinn_residuals",
            site=site,
            limit=limit,
            range_start=range_start,
        )
        return self._coerce_depth_column(df)

    def query_inversion_parameters(
        self,
        *,
        site: str = "default",
        limit: int = 100,
        range_start: str = "-90d",
    ) -> pd.DataFrame:
        """Fetch recorded inversion parameter sets."""

        return self._query_measurement(
            "pinn_inversion",
            site=site,
            limit=limit,
            range_start=range_start,
        )

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

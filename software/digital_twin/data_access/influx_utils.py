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

# The FEM temperature field is stored twice when the forecast service runs its
# baseline twin: once for the calibrated run (material parameters supplied by
# the inversion service) and once for the uncalibrated run (fixed configured
# parameters). Separate measurements rather than a tag, so existing queries and
# dashboards keep seeing only the calibrated stream.
FEM_TEMPERATURE_MEASUREMENT = "fem_temperature_2d"
FEM_TEMPERATURE_BASELINE_MEASUREMENT = "fem_temperature_2d_baseline"


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

    def write_temperature(self, time_hours, depth, temperature, site="default"):
        self.write_depth_series(
            measurement="sensor_temperature",
            time_hours=float(time_hours),
            depths=[float(depth)],
            temperatures=[float(temperature)],
            site=site,
        )

    def write_sensor_snapshot_2d(
        self,
        *,
        time_hours: float,
        sensors: Sequence[dict[str, float | str | None]],
        site: str = "default",
    ) -> None:
        """Write a 2D sensor snapshot with per-sensor coordinates."""

        if not sensors:
            return

        if self.write_api is None:  # pragma: no cover - runtime fallback
            self.logger.warning("Influx write skipped (no client available) for measurement sensor_temperature_2d")
            return

        timestamp = datetime.datetime.utcnow()
        records = []
        for sensor in sensors:
            sensor_id = str(sensor.get("sensor_id", "unknown"))
            temperature = sensor.get("temperature")
            x_m = sensor.get("x_m")
            z_m = sensor.get("z_m")
            y_m = sensor.get("y_m")

            if temperature is None or x_m is None or z_m is None:
                continue

            point = (
                Point("sensor_temperature_2d")
                .tag("site_id", site)
                .tag("sensor_id", sensor_id)
                .field("temperature", float(temperature))
                .field("time_hours", float(time_hours))
                .field("x_m", float(x_m))
                .field("z_m", float(z_m))
            )
            if y_m is not None:
                point = point.field("y_m", float(y_m))
            records.append(point.time(timestamp, write_precision=WritePrecision.NS))

        if not records:
            return

        try:
            self.write_api.write(bucket=self.bucket, record=records)
        except Exception as exc:  # pragma: no cover - runtime infra
            self.logger.error("Error writing sensor snapshot to InfluxDB: %s", exc)
            return

        if "sensor_temperature_2d" not in self._logged_measurements:
            self.logger.info(
                "sensor_temperature_2d write complete (t=%.2fh, sensors=%d)",
                float(time_hours),
                len(records),
            )
            self._logged_measurements.add("sensor_temperature_2d")

    def write_fem_temperature_snapshot_2d(
        self,
        *,
        time_hours: float,
        horizon_hours: float,
        sensors: Sequence[dict[str, float | str | None]],
        site: str = "default",
        measurement: str = FEM_TEMPERATURE_MEASUREMENT,
    ) -> None:
        """Write forecasted 2D FEM temperatures in sensor-like format for comparison.

        ``measurement`` selects the stream: the default calibrated one, or
        ``FEM_TEMPERATURE_BASELINE_MEASUREMENT`` for the uncalibrated twin run.
        """

        if not sensors:
            return

        if self.write_api is None:  # pragma: no cover - runtime fallback
            self.logger.warning("Influx write skipped (no client available) for measurement %s", measurement)
            return

        timestamp = datetime.datetime.utcnow()
        records = []
        for sensor in sensors:
            sensor_id = str(sensor.get("sensor_id", "unknown"))
            temperature = sensor.get("temperature")
            x_m = sensor.get("x_m")
            z_m = sensor.get("z_m")
            if temperature is None or x_m is None or z_m is None:
                continue

            point = (
                Point(measurement)
                .tag("site_id", site)
                .tag("sensor_id", sensor_id)
                .field("temperature", float(temperature))
                .field("time_hours", float(time_hours))
                .field("horizon_hours", float(horizon_hours))
                .field("x_m", float(x_m))
                .field("z_m", float(z_m))
                .time(timestamp, write_precision=WritePrecision.NS)
            )
            records.append(point)

        if not records:
            return

        try:
            self.write_api.write(bucket=self.bucket, record=records)
        except Exception as exc:  # pragma: no cover - runtime infra
            self.logger.error("Error writing FEM temperature snapshot to InfluxDB: %s", exc)
            return

        if measurement not in self._logged_measurements:
            self.logger.info(
                "%s write complete (t=%.2fh, sensors=%d)",
                measurement,
                float(time_hours),
                len(records),
            )
            self._logged_measurements.add(measurement)

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
        time_hours: float,
        depth: float,
        temperature: float,
        *,
        site: str = "default",
        extra_tags: Optional[dict] = None,
    ) -> None:
        self.write_depth_series(
            measurement=measurement,
            time_hours=time_hours,
            depths=[depth],
            temperatures=[temperature],
            site=site,
            extra_tags=extra_tags,
        )

    def write_depth_series(
        self,
        *,
        measurement: str,
        time_hours: float,
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
                .field("time_hours", float(time_hours))
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
                "%s write complete (t=%.2fh, depths=%d, min=%.2f°C, max=%.2f°C, mean=%.2f°C)",
                measurement,
                float(time_hours),
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
        """Persist aggregated FEM inversion parameters for downstream visualization."""

        if not parameters:
            return

        if self.write_api is None:  # pragma: no cover - runtime fallback
            self.logger.warning("Influx write skipped (no client available) for measurement fem_inversion")
            return

        record = Point("fem_inversion").tag("site_id", site).tag("status", status)
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

    def write_thaw_front_metrics(
        self,
        *,
        time_hours: float,
        radius_max_m: float | None,
        radius_avg_m: float | None,
        radius_min_m: float | None = None,
        thaw_volume_m3: float | None = None,
        points: Sequence[dict[str, float]],
        point_count: int | None = None,
        site: str = "default",
    ) -> None:
        """Persist thaw front metrics and representative points."""

        if self.write_api is None:  # pragma: no cover - runtime fallback
            self.logger.warning("Influx write skipped (no client available) for measurement thaw_front")
            return

        timestamp = datetime.datetime.utcnow()

        # NOTE: measurement is "thaw_front_metrics_v2". The original
        # "thaw_front_metrics" name became unwritable in the InfluxDB bucket — a
        # stale field-type conflict got registered in the series index, so every
        # write to that exact measurement name was silently dropped (accepted
        # with HTTP 204 but never persisted), which blanked the safety gauge and
        # all radius/volume/diagnostics plots. Deleting the data and restarting
        # InfluxDB did not clear the index entry, so the measurement was renamed.
        metrics = (
            Point("thaw_front_metrics_v2")
            .tag("site_id", site)
            .field("time_hours", float(time_hours))
            .field("radius_max_m", float(radius_max_m) if radius_max_m is not None else float("nan"))
            .field("radius_avg_m", float(radius_avg_m) if radius_avg_m is not None else float("nan"))
            .field("radius_min_m", float(radius_min_m) if radius_min_m is not None else float("nan"))
            .time(timestamp, write_precision=WritePrecision.NS)
        )
        if thaw_volume_m3 is not None:
            metrics = metrics.field("thaw_volume_m3", float(thaw_volume_m3))
        if point_count is not None:
            metrics = metrics.field("point_count", int(point_count))

        records = [metrics]
        for idx, point in enumerate(points, start=1):
            x_m = point.get("x_m")
            z_m = point.get("z_m")
            if x_m is None or z_m is None:
                continue
            point_timestamp = timestamp + datetime.timedelta(microseconds=idx)
            record = (
                Point("thaw_front_points")
                .tag("site_id", site)
                .field("time_hours", float(time_hours))
                .field("x_m", float(x_m))
                .field("z_m", float(z_m))
                .time(point_timestamp, write_precision=WritePrecision.NS)
            )
            records.append(record)

        try:
            self.write_api.write(bucket=self.bucket, record=records)
        except Exception as exc:  # pragma: no cover - runtime infra
            self.logger.error("Error writing thaw front data to InfluxDB: %s", exc)
            return

        if "thaw_front_metrics" not in self._logged_measurements:
            self.logger.info(
                "thaw_front_metrics write complete (t=%.2fh, points=%d)",
                float(time_hours),
                len(points),
            )
            self._logged_measurements.add("thaw_front_metrics")

    def write_fem_forecast_metrics(
        self,
        *,
        time_hours: float,
        horizon_hours: float,
        radius_max_m: float | None,
        radius_avg_m: float | None,
        points: Sequence[dict[str, float]],
        site: str = "default",
    ) -> None:
        """Persist FEM forecast metrics and representative points."""

        if self.write_api is None:  # pragma: no cover - runtime fallback
            self.logger.warning("Influx write skipped (no client available) for measurement fem_forecast")
            return

        timestamp = datetime.datetime.utcnow()
        metrics = (
            Point("fem_forecast_metrics")
            .tag("site_id", site)
            .field("time_hours", float(time_hours))
            .field("horizon_hours", float(horizon_hours))
            .field("radius_max_m", float(radius_max_m) if radius_max_m is not None else float("nan"))
            .field("radius_avg_m", float(radius_avg_m) if radius_avg_m is not None else float("nan"))
            .time(timestamp, write_precision=WritePrecision.NS)
        )

        records = [metrics]
        for idx, point in enumerate(points, start=1):
            x_m = point.get("x_m")
            z_m = point.get("z_m")
            if x_m is None or z_m is None:
                continue
            point_timestamp = timestamp + datetime.timedelta(microseconds=idx)
            record = (
                Point("fem_forecast_points")
                .tag("site_id", site)
                .field("time_hours", float(time_hours))
                .field("horizon_hours", float(horizon_hours))
                .field("x_m", float(x_m))
                .field("z_m", float(z_m))
                .time(point_timestamp, write_precision=WritePrecision.NS)
            )
            records.append(record)

        try:
            self.write_api.write(bucket=self.bucket, record=records)
        except Exception as exc:  # pragma: no cover - runtime infra
            self.logger.error("Error writing FEM forecast data to InfluxDB: %s", exc)
            return

        if "fem_forecast_metrics" not in self._logged_measurements:
            self.logger.info(
                "fem_forecast_metrics write complete (t=%.2fh, points=%d)",
                float(time_hours),
                len(points),
            )
            self._logged_measurements.add("fem_forecast_metrics")

    def write_safety_alert(
        self,
        *,
        time_hours: float,
        limit_radius_m: float,
        measured_radius_m: float | None,
        forecast_radius_m: float | None,
        triggered: bool,
        reason: str,
        metric: str,
        status: str | None = None,
        r_max_m: float | None = None,
        r_min_m: float | None = None,
        limit_min_radius_m: float | None = None,
        rmin_below_floor: bool | None = None,
        recommended_action: str | None = None,
        site: str = "default",
    ) -> None:
        """Persist safety alert status for downstream monitoring.

        ``status`` may be a graded level (``ok``/``warning``/``critical``/``alert``);
        when omitted it falls back to the binary ``alert``/``ok`` derived from
        ``triggered`` (legacy single-metric mode). The dual-limit rule engine also
        records r_max/r_min, the min-radius floor and the recommended heater action.
        """

        if self.write_api is None:  # pragma: no cover - runtime fallback
            self.logger.warning("Influx write skipped (no client available) for measurement safety_alerts")
            return

        status_tag = status or ("alert" if triggered else "ok")
        timestamp = datetime.datetime.utcnow()
        record = (
            Point("safety_alerts")
            .tag("site_id", site)
            .tag("status", status_tag)
            .tag("reason", reason)
            .tag("metric", metric)
            .field("time_hours", float(time_hours))
            .field("limit_radius_m", float(limit_radius_m))
            .field("measured_radius_m", float(measured_radius_m) if measured_radius_m is not None else float("nan"))
            .field("forecast_radius_m", float(forecast_radius_m) if forecast_radius_m is not None else float("nan"))
            .field("triggered", bool(triggered))
            .time(timestamp, write_precision=WritePrecision.NS)
        )
        if r_max_m is not None:
            record = record.field("r_max_m", float(r_max_m))
        if r_min_m is not None:
            record = record.field("r_min_m", float(r_min_m))
        if limit_min_radius_m is not None:
            record = record.field("limit_min_radius_m", float(limit_min_radius_m))
        if rmin_below_floor is not None:
            record = record.field("rmin_below_floor", bool(rmin_below_floor))
        if recommended_action is not None:
            record = record.tag("recommended_action", recommended_action)

        try:
            self.write_api.write(bucket=self.bucket, record=[record])
        except Exception as exc:  # pragma: no cover - runtime infra
            self.logger.error("Error writing safety alerts to InfluxDB: %s", exc)
            return

        if "safety_alerts" not in self._logged_measurements:
            self.logger.info(
                "safety_alerts write complete (t=%.2fh, status=%s)",
                float(time_hours),
                status_tag,
            )
            self._logged_measurements.add("safety_alerts")

    def write_heater_action(
        self,
        *,
        time_hours: float,
        action: str,
        setpoint: float | None,
        triggered: bool,
        reason: str,
        source: str,
        site: str = "default",
    ) -> None:
        """Persist heater actuation commands/actions."""

        if self.write_api is None:  # pragma: no cover - runtime fallback
            self.logger.warning("Influx write skipped (no client available) for measurement heater_actions")
            return

        timestamp = datetime.datetime.utcnow()
        record = (
            Point("heater_actions")
            .tag("site_id", site)
            .tag("action", action)
            .tag("source", source)
            .tag("reason", reason)
            .field("time_hours", float(time_hours))
            .field("setpoint", float(setpoint) if setpoint is not None else float("nan"))
            .field("triggered", bool(triggered))
            .time(timestamp, write_precision=WritePrecision.NS)
        )

        try:
            self.write_api.write(bucket=self.bucket, record=[record])
        except Exception as exc:  # pragma: no cover - runtime infra
            self.logger.error("Error writing heater actions to InfluxDB: %s", exc)
            return

        if "heater_actions" not in self._logged_measurements:
            self.logger.info(
                "heater_actions write complete (t=%.2fh, action=%s)",
                float(time_hours),
                action,
            )
            self._logged_measurements.add("heater_actions")

    def write_boundary_flux(
        self,
        *,
        time_hours: float,
        q_surface: float,
        q_bottom: float,
        site: str = "default",
    ) -> None:
        """Persist boundary heat flux diagnostics."""

        if self.write_api is None:  # pragma: no cover - runtime fallback
            self.logger.warning("Influx write skipped (no client available) for measurement boundary_flux")
            return

        timestamp = datetime.datetime.utcnow()
        record = (
            Point("boundary_flux")
            .tag("site_id", site)
            .field("time_hours", float(time_hours))
            .field("q_surface", float(q_surface))
            .field("q_bottom", float(q_bottom))
            .time(timestamp, write_precision=WritePrecision.NS)
        )
        try:
            self.write_api.write(bucket=self.bucket, record=[record])
        except Exception as exc:  # pragma: no cover - runtime infra
            self.logger.error("Error writing boundary flux to InfluxDB: %s", exc)

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
        newest: bool = False,
    ) -> pd.DataFrame:
        """Generic measurement query helper with pivoted fields.

        ``limit`` keeps the oldest rows unless ``newest`` is set.
        """

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
            query += f'\n  |> {"tail" if newest else "limit"}(n: {int(limit)})'

        try:
            tables = self.query_api.query_data_frame(query)
        except Exception as exc:  # pragma: no cover - runtime infra
            self.logger.error("Error querying InfluxDB (measurement=%s): %s", measurement, exc)
            return pd.DataFrame()

        if isinstance(tables, list) and tables:
            df = pd.concat(tables, ignore_index=True)
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

    def query_inversion_parameters(
        self,
        *,
        site: str = "default",
        limit: int = 100,
        range_start: str = "-90d",
    ) -> pd.DataFrame:
        """Fetch recorded inversion parameter sets."""

        return self._query_measurement(
            "fem_inversion",
            site=site,
            limit=limit,
            range_start=range_start,
        )

    def query_sensor_temperature_2d(
        self,
        *,
        site: str = "default",
        limit: int = 5000,
        range_start: str = "-6h",
    ) -> pd.DataFrame:
        """Fetch the latest 2D sensor snapshots."""

        return self._query_measurement(
            "sensor_temperature_2d",
            site=site,
            limit=limit,
            range_start=range_start,
        )

    def query_fem_temperature_2d(
        self,
        *,
        site: str = "default",
        limit: int = 5000,
        range_start: str = "-6h",
        measurement: str = FEM_TEMPERATURE_MEASUREMENT,
    ) -> pd.DataFrame:
        """Fetch the latest 2D FEM temperature snapshots at sensor-style points.

        Pass ``measurement=FEM_TEMPERATURE_BASELINE_MEASUREMENT`` for the
        uncalibrated twin run (fixed material parameters, no inversion).
        """

        return self._query_measurement(
            measurement,
            site=site,
            limit=limit,
            range_start=range_start,
        )

    def query_thaw_front_metrics(
        self,
        *,
        site: str = "default",
        limit: int = 1000,
        range_start: str = "-12h",
    ) -> pd.DataFrame:
        """Fetch thaw-front radius metrics."""

        # Renamed from "thaw_front_metrics" — see write_thaw_front_metrics for why.
        return self._query_measurement(
            "thaw_front_metrics_v2",
            site=site,
            limit=limit,
            range_start=range_start,
        )

    def query_thaw_front_points(
        self,
        *,
        site: str = "default",
        limit: int = 5000,
        range_start: str = "-12h",
        newest: bool = False,
    ) -> pd.DataFrame:
        """Fetch thaw-front point geometry."""

        return self._query_measurement(
            "thaw_front_points",
            site=site,
            limit=limit,
            range_start=range_start,
            newest=newest,
        )

    def query_fem_forecast_metrics(
        self,
        *,
        site: str = "default",
        limit: int = 1000,
        range_start: str = "-12h",
    ) -> pd.DataFrame:
        """Fetch FEM forecast metrics."""

        return self._query_measurement(
            "fem_forecast_metrics",
            site=site,
            limit=limit,
            range_start=range_start,
        )

    def query_fem_forecast_points(
        self,
        *,
        site: str = "default",
        limit: int = 5000,
        range_start: str = "-12h",
        newest: bool = False,
    ) -> pd.DataFrame:
        """Fetch FEM forecast point geometry."""

        return self._query_measurement(
            "fem_forecast_points",
            site=site,
            limit=limit,
            range_start=range_start,
            newest=newest,
        )

    def query_safety_alerts(
        self,
        *,
        site: str = "default",
        limit: int = 1000,
        range_start: str = "-12h",
    ) -> pd.DataFrame:
        """Fetch safety alert records."""

        return self._query_measurement(
            "safety_alerts",
            site=site,
            limit=limit,
            range_start=range_start,
        )

    def query_heater_actions(
        self,
        *,
        site: str = "default",
        limit: int = 1000,
        range_start: str = "-12h",
    ) -> pd.DataFrame:
        """Fetch heater action records."""

        return self._query_measurement(
            "heater_actions",
            site=site,
            limit=limit,
            range_start=range_start,
        )

    def write_fluid_temperatures(
        self,
        *,
        time_hours: float,
        t_inlet_c: float,
        t_outlet_c: float,
        site: str = "default",
    ) -> None:
        """Persist fluid inlet/outlet temperatures for energy metric computation."""

        if self.write_api is None:  # pragma: no cover - runtime fallback
            self.logger.warning("Influx write skipped (no client available) for measurement fluid_temperatures")
            return

        timestamp = datetime.datetime.utcnow()
        record = (
            Point("fluid_temperatures")
            .tag("site_id", site)
            .field("time_hours", float(time_hours))
            .field("t_inlet_c", float(t_inlet_c))
            .field("t_outlet_c", float(t_outlet_c))
            .time(timestamp, write_precision=WritePrecision.NS)
        )
        try:
            self.write_api.write(bucket=self.bucket, record=[record])
        except Exception as exc:  # pragma: no cover - runtime infra
            self.logger.error("Error writing fluid temperatures to InfluxDB: %s", exc)

    def write_pipe_outlet_validation(
        self,
        *,
        time_hours: float,
        t_inlet_c: float,
        t_outlet_meas_c: float,
        t_outlet_sim_c: float,
        site: str = "default",
    ) -> None:
        """Persist the pipe model's outlet prediction beside the measurement.

        All three series are written as one point per fluid sample rather than
        joined later from separate measurements: the simulated outlet is only
        meaningful next to the exact inlet/outlet pair the model was driven
        with, so keeping them on one row guarantees the validation plot compares
        like with like.
        """

        if self.write_api is None:  # pragma: no cover - runtime fallback
            self.logger.warning("Influx write skipped (no client available) for measurement pipe_outlet_validation")
            return

        record = (
            Point("pipe_outlet_validation")
            .tag("site_id", site)
            .field("time_hours", float(time_hours))
            .field("t_inlet_c", float(t_inlet_c))
            .field("t_outlet_meas_c", float(t_outlet_meas_c))
            .field("t_outlet_sim_c", float(t_outlet_sim_c))
            .field("error_c", float(t_outlet_sim_c) - float(t_outlet_meas_c))
            .time(datetime.datetime.utcnow(), write_precision=WritePrecision.NS)
        )
        try:
            self.write_api.write(bucket=self.bucket, record=[record])
        except Exception as exc:  # pragma: no cover - runtime infra
            self.logger.error("Error writing pipe outlet validation to InfluxDB: %s", exc)
            return

        if "pipe_outlet_validation" not in self._logged_measurements:
            self.logger.info("pipe_outlet_validation write complete (t=%.3fh)", float(time_hours))
            self._logged_measurements.add("pipe_outlet_validation")

    def query_pipe_outlet_validation(
        self,
        *,
        site: str = "default",
        limit: int = 5000,
        range_start: str = "-12h",
    ) -> pd.DataFrame:
        """Fetch measured inlet/outlet and simulated outlet for the pipe model."""

        return self._query_measurement(
            "pipe_outlet_validation",
            site=site,
            limit=limit,
            range_start=range_start,
        )

    def query_fluid_temperatures(
        self,
        *,
        site: str = "default",
        limit: int = 5000,
        range_start: str = "-12h",
    ) -> pd.DataFrame:
        """Fetch fluid inlet/outlet temperature history."""

        return self._query_measurement(
            "fluid_temperatures",
            site=site,
            limit=limit,
            range_start=range_start,
        )

    def write_pipe_calibration(
        self,
        parameters: dict[str, float],
        *,
        rmse: float | None = None,
        mae: float | None = None,
        interface_temperature_c: float | None = None,
        site: str = "default",
        timestamp: datetime.datetime | None = None,
    ) -> None:
        """Persist calibrated pipe (heat-exchanger loop) parameters + fit metrics."""

        if not parameters:
            return

        if self.write_api is None:  # pragma: no cover - runtime fallback
            self.logger.warning("Influx write skipped (no client available) for measurement pipe_calibration")
            return

        timestamp = timestamp or datetime.datetime.utcnow()
        record = Point("pipe_calibration").tag("site_id", site)
        for key, value in parameters.items():
            try:
                record = record.field(key, float(value))
            except (TypeError, ValueError):
                self.logger.warning("Skipping invalid pipe parameter %s=%s", key, value)
        for key, value in (
            ("rmse", rmse),
            ("mae", mae),
            ("interface_temperature_c", interface_temperature_c),
        ):
            if value is None:
                continue
            try:
                record = record.field(key, float(value))
            except (TypeError, ValueError):
                self.logger.debug("Skipping invalid pipe metric %s=%s", key, value)
        record = record.time(timestamp, write_precision=WritePrecision.NS)

        try:
            self.write_api.write(bucket=self.bucket, record=[record])
        except Exception as exc:  # pragma: no cover - runtime infra
            self.logger.error("Error writing pipe calibration to InfluxDB: %s", exc)

    def query_pipe_calibration(
        self,
        *,
        site: str = "default",
        limit: int = 200,
        range_start: str = "-90d",
    ) -> pd.DataFrame:
        """Fetch recorded pipe calibration parameter sets."""

        return self._query_measurement(
            "pipe_calibration",
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

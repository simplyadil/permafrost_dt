# software/services/common/influx_utils.py
import datetime
import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision  # type: ignore
from influxdb_client.client.write_api import SYNCHRONOUS  # type: ignore
from .logger import setup_logger


class InfluxHelper:
    def __init__(
        self,
        url="http://localhost:8086",
        token="permafrost",
        org="permafrost",
        bucket="permafrost_data"
    ):
        self.logger = setup_logger("InfluxDB")
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.bucket = bucket
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()

    def write_temperature(self, time_days, depth, temperature, site="default"):
        point = (
            Point("soil_temperature")
            .tag("site_id", site)
            .tag("depth", f"{depth:.1f}m")
            .field("temperature", float(temperature))
            .field("time_days", float(time_days))
            .time(datetime.datetime.utcnow(), write_precision=WritePrecision.NS)
        )
        self.write_api.write(bucket=self.bucket, record=point)
        self.logger.info(f"Written T={temperature}°C @ {depth}m, t={time_days}d")

    def query_temperature(self, site="default", depth=None, limit=100):
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: -30d)
          |> filter(fn: (r) => r["_measurement"] == "soil_temperature")
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
            return tables[0]
        elif isinstance(tables, pd.DataFrame):
            return tables
        else:
            return pd.DataFrame()
        

    # Generic writer for model/simulation outputs
    def write_model_temperature(self, measurement: str, time_days: float, depth: float, temperature: float,
                                site: str = "default", extra_tags: dict | None = None):
        """
        Write temperature to an arbitrary measurement (e.g., 'fdm_temperature').
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
                 .time(datetime.datetime.utcnow(), write_precision=WritePrecision.NS))  # type: ignore

        self.write_api.write(bucket=self.bucket, record=point)
        self.logger.info(f"[{measurement}] Written T={temperature}°C @ {depth}m, t={time_days}d")

    def query_model_temperature(self, measurement: str, site: str = "default", depth: float | None = None, limit: int = 200):
        """
        Query arbitrary measurement (e.g., 'fdm_temperature'), pivoting fields for DataFrame usability.
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
        tables = self.query_api.query_data_frame(q)
        if isinstance(tables, list) and len(tables) > 0:
            return tables[0]
        return tables if hasattr(tables, "empty") else pd.DataFrame()


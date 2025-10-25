# software/services/common/influx_utils.py
from influxdb_client import InfluxDBClient, Point, WritePrecision # type: ignore
from influxdb_client.client.write_api import SYNCHRONOUS # type: ignore
import pandas as pd

from .logger import setup_logger

class InfluxHelper:
    def __init__(self, url="http://localhost:8086", token="my-token", org="permafrost", bucket="dt_data"):
        self.logger = setup_logger("InfluxDB")
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.bucket = bucket
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()

    def write_temperature(self, time_days, depth, temperature, site="default"):
        point = Point("soil_temperature") \
            .tag("site_id", site) \
            .field("depth", depth) \
            .field("temperature", temperature) \
            .field("time_days", time_days) \
            .time(write_precision=WritePrecision.NS) # type: ignore
        self.write_api.write(bucket=self.bucket, record=point)
        self.logger.info(f"Written T={temperature}°C @ {depth}m, t={time_days}d")

    def query_temperature(self, site="default", depth=None, limit=100):
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: -30d)
          |> filter(fn: (r) => r["_measurement"] == "soil_temperature")
          |> filter(fn: (r) => r["site_id"] == "{site}")
          |> limit(n: {limit})
        '''
        if depth is not None:
            query += f'|> filter(fn: (r) => r["depth"] == {depth})'

        tables = self.query_api.query_data_frame(query)
        if isinstance(tables, list) and len(tables) > 0:
            return tables[0]
        return pd.DataFrame()

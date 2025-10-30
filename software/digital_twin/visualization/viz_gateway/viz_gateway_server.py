# pylint: disable=too-many-instance-attributes
"""Visualization gateway server."""

import json
import math
import os
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from software.digital_twin.communication.logger import setup_logger
from software.digital_twin.communication.messaging import RabbitMQClient, RabbitMQConfig
from software.digital_twin.data_access.influx_utils import InfluxConfig, InfluxHelper

PINN_INVERSION_QUEUE = "permafrost.record.pinn_inversion.state"
VIZ_UPDATE_QUEUE = "permafrost.update.visualization.command"
PINN_INVERSION_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "pinn_inversion_output_message.json"
VIZ_UPDATE_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "viz_update_message.json"
FORWARD_HISTORY_PATH = Path("software/models/pinn_forward/freezing_soil_pinn_history.json")
INVERSION_HISTORY_PATH = Path("software/models/pinn_inversion/inversion_history.json")
DEFAULT_DEPTHS = (0, 1, 2, 3, 4, 5)
MAX_TIME_POINTS = 500
MAX_SERIES_POINTS = 400


class VizGatewayServer:
    """
    Collects results from InfluxDB (FDM, PINN forward, inversion) and publishes visualization-ready updates.
    """

    def __init__(
        self,
        influx_config: InfluxConfig | None = None,
        inversion_queue_config: RabbitMQConfig | None = None,
        viz_queue_config: RabbitMQConfig | None = None,
        output_dir: str = "software/outputs",
    ) -> None:
        self.logger = setup_logger("VizGatewayServer")
        self.influx_config = influx_config or InfluxConfig()

        if inversion_queue_config is not None and inversion_queue_config.schema_path is None:
            inversion_queue_config = RabbitMQConfig(
                host=inversion_queue_config.host,
                queue=inversion_queue_config.queue,
                schema_path=PINN_INVERSION_SCHEMA,
                username=inversion_queue_config.username,
                password=inversion_queue_config.password,
            )
        if viz_queue_config is not None and viz_queue_config.schema_path is None:
            viz_queue_config = RabbitMQConfig(
                host=viz_queue_config.host,
                queue=viz_queue_config.queue,
                schema_path=VIZ_UPDATE_SCHEMA,
                username=viz_queue_config.username,
                password=viz_queue_config.password,
            )
        inversion_base = inversion_queue_config or RabbitMQConfig(schema_path=PINN_INVERSION_SCHEMA)
        viz_base = viz_queue_config or RabbitMQConfig(schema_path=VIZ_UPDATE_SCHEMA)
        self.inversion_queue_config = inversion_base.with_queue(PINN_INVERSION_QUEUE)
        self.viz_queue_config = viz_base.with_queue(VIZ_UPDATE_QUEUE)
        self.queue_in = self.inversion_queue_config.queue
        self.queue_out = self.viz_queue_config.queue

        self.mq_client: Optional[RabbitMQClient] = None
        self.out_publisher: Optional[RabbitMQClient] = None
        self.influx: Optional[InfluxHelper] = None

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._running = False
        self.logger.info("VizGatewayServer configured.")

    # -------------------------------
    # MAIN HANDLER
    # -------------------------------
    def _on_message(self, msg):
        """Triggered when inversion is complete."""
        self.logger.info(f"Received message from inversion service: {msg}")
        if msg.get("status") == "inverted":
            Thread(target=self.aggregate_and_publish, args=(msg,), daemon=True).start()
        else:
            self.logger.warning("Message ignored (not 'inverted').")

    # -------------------------------
    # DATA AGGREGATION
    # -------------------------------
    def aggregate_and_publish(self, inversion_msg: Optional[dict] = None) -> None:
        """Fetch latest data from InfluxDB and send a dashboard-ready payload."""
        self.logger.info("Collecting FDM, PINN, and inversion artefacts for visualization payload.")

        if self.influx is None:
            raise RuntimeError("Influx helper not initialised. Did you call setup()?")

        fdm_raw = self.influx.query_model_temperature("fdm_simulation", limit=20000)
        pinn_raw = self.influx.query_model_temperature("pinn_forward", limit=20000)

        fdm_df = self._prepare_dataframe(fdm_raw)
        pinn_df = self._prepare_dataframe(pinn_raw)

        payload = self._build_payload(fdm_df, pinn_df, inversion_msg or {})

        snapshot_path = self._write_snapshot(payload)
        self.logger.info("Visualization snapshot saved to %s", snapshot_path)

        if self.out_publisher is None:
            raise RuntimeError("Outbound MQ client not initialised. Did you call setup()?")
        self.out_publisher.publish(payload)
        self.logger.info("Visualization update published to %s.", self.queue_out)

    def _build_payload(self, fdm_df: pd.DataFrame, pinn_df: pd.DataFrame, inversion_msg: Dict[str, object]) -> Dict[str, object]:
        forward_history = self._load_history(FORWARD_HISTORY_PATH)
        inversion_history = self._load_history(INVERSION_HISTORY_PATH)
        comparison_section = self._build_comparison_section(fdm_df, pinn_df)

        payload: Dict[str, object] = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "viz_ready",
            "fdm": self._build_fdm_section(fdm_df),
            "pinn_forward": self._build_pinn_section(pinn_df, forward_history),
            "comparison": comparison_section,
            "inversion": self._build_inversion_section(inversion_msg, inversion_history),
        }
        payload["data_summary"] = {
            "fdm_points": int(len(fdm_df)),
            "pinn_points": int(len(pinn_df)),
            "comparison_pairs": int(comparison_section.get("pair_count", 0)),
            "history_available": {
                "pinn_forward": bool(forward_history),
                "pinn_inversion": bool(inversion_history),
            },
        }
        return payload

    def _prepare_dataframe(self, df) -> pd.DataFrame:
        if df is None or (hasattr(df, "empty") and df.empty):
            return pd.DataFrame(columns=["time_days", "depth_m", "temperature"])
        if isinstance(df, list):
            df = pd.concat(df, ignore_index=True)
        result = df.copy()
        if "depth_m" not in result.columns and "depth" in result.columns:
            result["depth_m"] = result["depth"].apply(self._parse_depth)
        if "temperature" not in result.columns:
            for candidate in ("temperature_C", "value"):
                if candidate in result.columns:
                    result["temperature"] = result[candidate]
                    break
        required = {"time_days", "depth_m", "temperature"}
        if not required.issubset(result.columns):
            return pd.DataFrame(columns=["time_days", "depth_m", "temperature"])
        result = result.dropna(subset=list(required))
        for column in required:
            result[column] = result[column].astype(float)
        return result.sort_values(["time_days", "depth_m"]).reset_index(drop=True)

    def _write_snapshot(self, payload: Dict[str, object]) -> str:
        filename = f"viz_snapshot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        path = Path(self.output_dir) / filename
        path.write_text(json.dumps(payload, indent=2))
        return str(path)

    def _load_history(self, path: Path) -> Dict[str, object]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except Exception as exc:  # pragma: no cover - corrupted file or invalid json
            self.logger.warning("Failed to load history snapshot from %s: %s", path, exc)
            return {}

    def _build_fdm_section(self, df: pd.DataFrame) -> Dict[str, object]:
        return {
            "grid": self._build_grid(df, value_key="temperature"),
            "series": self._build_depth_series(df, depths=DEFAULT_DEPTHS),
            "snapshots": self._build_snapshots(df, value_column="temperature"),
        }

    def _build_pinn_section(self, df: pd.DataFrame, history: Dict[str, object]) -> Dict[str, object]:
        return {
            "grid": self._build_grid(df, value_key="temperature"),
            "series": self._build_depth_series(df, depths=DEFAULT_DEPTHS),
            "snapshots": self._build_snapshots(df, value_column="temperature"),
            "history": history,
            "metadata": {
                "status": history.get("status"),
                "epochs": history.get("epochs"),
                "timestamp": history.get("timestamp"),
                "history_path": str(FORWARD_HISTORY_PATH),
            },
        }

    def _build_comparison_section(self, fdm_df: pd.DataFrame, pinn_df: pd.DataFrame) -> Dict[str, object]:
        result: Dict[str, object] = {
            "abs_error_grid": {"time_days": [], "depth_m": [], "abs_error": []},
            "stats": {"overall": {"mean_abs_error": 0.0, "max_abs_error": 0.0}, "per_depth": []},
            "per_depth_series": [],
            "snapshots": [],
            "pair_count": 0,
        }
        if fdm_df.empty or pinn_df.empty:
            return result

        left = fdm_df.rename(columns={"temperature": "temperature_fdm"})
        right = pinn_df.rename(columns={"temperature": "temperature_pinn"})
        merged = pd.merge(left, right, on=["time_days", "depth_m"], how="inner")
        if merged.empty:
            return result

        merged["abs_error"] = (merged["temperature_pinn"] - merged["temperature_fdm"]).abs()
        result["pair_count"] = int(len(merged))
        result["abs_error_grid"] = self._build_grid(merged, value_column="abs_error", value_key="abs_error")

        overall = {
            "mean_abs_error": float(merged["abs_error"].mean()),
            "max_abs_error": float(merged["abs_error"].max()),
        }
        per_depth_stats = [
            {
                "depth_m": float(depth),
                "mean_abs_error": float(group["abs_error"].mean()),
                "max_abs_error": float(group["abs_error"].max()),
            }
            for depth, group in merged.groupby("depth_m")
        ]
        result["stats"] = {"overall": overall, "per_depth": per_depth_stats}
        result["per_depth_series"] = self._build_comparison_series(merged)
        result["snapshots"] = self._build_comparison_snapshots(merged)
        return result

    def _build_inversion_section(self, inversion_msg: Dict[str, object], history: Dict[str, object]) -> Dict[str, object]:
        parameters = inversion_msg.get("parameters", {}) if inversion_msg else {}
        return {
            "parameters": parameters,
            "history": history,
            "history_path": str(INVERSION_HISTORY_PATH),
            "message": {
                "timestamp": inversion_msg.get("timestamp") if inversion_msg else None,
                "status": inversion_msg.get("status") if inversion_msg else None,
                "validation_overall": inversion_msg.get("validation_overall") if inversion_msg else None,
            },
        }

    def _build_grid(
        self,
        df: pd.DataFrame,
        *,
        value_column: str = "temperature",
        value_key: str = "temperature",
        max_time_points: int = MAX_TIME_POINTS,
    ) -> Dict[str, object]:
        if df.empty or value_column not in df.columns:
            return {"time_days": [], "depth_m": [], value_key: []}
        pivot = df.pivot_table(index="depth_m", columns="time_days", values=value_column, aggfunc="mean")
        pivot = pivot.sort_index().sort_index(axis=1)
        depth_values = pivot.index.to_list()
        time_values = pivot.columns.to_list()
        if not depth_values or not time_values:
            return {"time_days": [], "depth_m": [], value_key: []}
        indices = self._downsample_indices(len(time_values), max_time_points)
        selected_times = [float(time_values[i]) for i in indices]
        reduced = pivot.iloc[:, indices]
        matrix = []
        for row in reduced.to_numpy():
            matrix.append([self._safe_float(val) for val in row])
        return {
            "depth_m": [float(d) for d in depth_values],
            "time_days": selected_times,
            value_key: matrix,
        }

    def _build_depth_series(
        self,
        df: pd.DataFrame,
        *,
        depths: Optional[Sequence[float]] = None,
        max_points: int = MAX_SERIES_POINTS,
    ) -> List[Dict[str, object]]:
        depths = depths or DEFAULT_DEPTHS
        output: List[Dict[str, object]] = []
        if df.empty:
            return output
        for depth in depths:
            mask = np.isclose(df["depth_m"], depth, atol=1e-6)
            depth_df = df.loc[mask, ["time_days", "temperature"]].copy()
            if depth_df.empty:
                continue
            depth_df = depth_df.sort_values("time_days")
            indices = self._downsample_indices(len(depth_df), max_points)
            reduced = depth_df.iloc[indices]
            output.append(
                {
                    "depth_m": float(depth),
                    "time_days": reduced["time_days"].astype(float).tolist(),
                    "temperature": reduced["temperature"].astype(float).tolist(),
                }
            )
        return output

    def _build_snapshots(
        self,
        df: pd.DataFrame,
        *,
        value_column: str,
        max_snapshots: int = 3,
    ) -> List[Dict[str, object]]:
        if df.empty or value_column not in df.columns:
            return []
        unique_times = sorted(df["time_days"].unique())
        snapshots: List[Dict[str, object]] = []
        for time_value in self._select_snapshot_times(unique_times, max_snapshots=max_snapshots):
            slice_df = df[df["time_days"] == time_value].sort_values("depth_m")
            snapshots.append(
                {
                    "time_days": float(time_value),
                    "depth_m": slice_df["depth_m"].astype(float).tolist(),
                    value_column: slice_df[value_column].astype(float).tolist(),
                }
            )
        return snapshots

    def _build_comparison_series(self, df: pd.DataFrame) -> List[Dict[str, object]]:
        series: List[Dict[str, object]] = []
        for depth in DEFAULT_DEPTHS:
            mask = np.isclose(df["depth_m"], depth, atol=1e-6)
            depth_df = df.loc[mask, ["time_days", "temperature_fdm", "temperature_pinn", "abs_error"]].copy()
            if depth_df.empty:
                continue
            depth_df = depth_df.sort_values("time_days")
            indices = self._downsample_indices(len(depth_df), MAX_SERIES_POINTS)
            reduced = depth_df.iloc[indices]
            series.append(
                {
                    "depth_m": float(depth),
                    "time_days": reduced["time_days"].astype(float).tolist(),
                    "fdm": reduced["temperature_fdm"].astype(float).tolist(),
                    "pinn": reduced["temperature_pinn"].astype(float).tolist(),
                    "abs_error": reduced["abs_error"].astype(float).tolist(),
                }
            )
        return series

    def _build_comparison_snapshots(
        self,
        df: pd.DataFrame,
        *,
        max_snapshots: int = 3,
    ) -> List[Dict[str, object]]:
        if df.empty:
            return []
        unique_times = sorted(df["time_days"].unique())
        snapshots: List[Dict[str, object]] = []
        for time_value in self._select_snapshot_times(unique_times, max_snapshots=max_snapshots):
            slice_df = df[df["time_days"] == time_value].sort_values("depth_m")
            snapshots.append(
                {
                    "time_days": float(time_value),
                    "depth_m": slice_df["depth_m"].astype(float).tolist(),
                    "fdm": slice_df["temperature_fdm"].astype(float).tolist(),
                    "pinn": slice_df["temperature_pinn"].astype(float).tolist(),
                    "abs_error": slice_df["abs_error"].astype(float).tolist(),
                }
            )
        return snapshots

    @staticmethod
    def _downsample_indices(length: int, max_points: int) -> List[int]:
        if length <= max_points:
            return list(range(length))
        if max_points <= 1:
            return [length - 1]
        step = (length - 1) / (max_points - 1)
        indices = [int(round(i * step)) for i in range(max_points)]
        return sorted(set(min(max(idx, 0), length - 1) for idx in indices))

    @staticmethod
    def _select_snapshot_times(times: Sequence[float], max_snapshots: int = 3) -> List[float]:
        if not times:
            return []
        if len(times) <= max_snapshots:
            return [float(t) for t in times]
        indices = [0, len(times) // 2, len(times) - 1]
        indices = sorted(set(indices))[:max_snapshots]
        return [float(times[i]) for i in indices]

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) else None

    @staticmethod
    def _parse_depth(value) -> float:
        if value is None:
            return float("nan")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip().lower().rstrip("m")
            try:
                return float(cleaned)
            except ValueError:
                return float("nan")
        return float("nan")

    # -------------------------------
    # RUN
    # -------------------------------
    def run(self):
        if self.mq_client is None:
            raise RuntimeError("Inbound MQ client not initialised. Did you call setup()?")
        self.logger.info(f"Listening for inversion messages on queue: {self.queue_in}")
        self.mq_client.consume(callback=self._on_message)

    # -------------------------------
    # LIFECYCLE
    # -------------------------------
    def setup(self) -> None:
        """Initialise messaging and data dependencies."""

        if self.influx is None:
            self.influx = InfluxHelper(self.influx_config)
        if self.mq_client is None:
            self.mq_client = RabbitMQClient(self.inversion_queue_config)
        if self.out_publisher is None:
            self.out_publisher = RabbitMQClient(self.viz_queue_config)
        self.queue_in = self.inversion_queue_config.queue
        self.queue_out = self.viz_queue_config.queue
        self._running = True
        self.logger.info("VizGatewayServer setup complete.")

    def start(self) -> None:
        """Start consuming inversion notifications."""

        if self.mq_client is None or self.out_publisher is None or self.influx is None:
            self.setup()

        try:
            self.run()
        except KeyboardInterrupt:  # pragma: no cover
            self.logger.info("VizGatewayServer interrupted. Shutting down...")
        finally:
            self.close()

    def stop(self) -> None:
        """Stop consuming messages."""

        self._running = False
        if self.mq_client and self.mq_client.channel:
            self.mq_client.channel.stop_consuming()

    def close(self) -> None:
        """Release resources."""

        self.stop()
        if self.mq_client is not None:
            self.mq_client.disconnect()
        if self.out_publisher is not None:
            self.out_publisher.disconnect()
        if self.influx is not None:
            self.influx.close()
        self.logger.info("VizGatewayServer shutdown complete.")


if __name__ == "__main__":
    VizGatewayServer().start()

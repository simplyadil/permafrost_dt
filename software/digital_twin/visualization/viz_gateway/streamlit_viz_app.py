"""Streamlit dashboard for the permafrost digital twin."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any

import streamlit as st

from software.digital_twin.data_access.influx_utils import InfluxConfig
from software.digital_twin.visualization.viz_gateway.visualization_service import (
    FIGURE_BUILDERS,
    VisualizationConfig,
    VizDataBridge,
    render_summary_lines,
)


def _load_configs() -> Tuple[VisualizationConfig, InfluxConfig]:
    influx_conf = os.environ.get("PERMAFROST_VIZ_INFLUX_CONFIG")
    influx_cfg = InfluxConfig(**json.loads(influx_conf)) if influx_conf else InfluxConfig()
    host = os.environ.get("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")
    port = int(os.environ.get("STREAMLIT_SERVER_PORT", 8501))
    refresh_seconds = float(os.environ.get("PERMAFROST_VIZ_REFRESH_SECONDS", 60.0))
    fetch_limit = int(os.environ.get("PERMAFROST_VIZ_FETCH_LIMIT", 20000))
    output_dir = os.environ.get("PERMAFROST_VIZ_OUTPUT_DIR", "software/outputs")
    viz_cfg = VisualizationConfig(
        host=host,
        port=port,
        refresh_seconds=refresh_seconds,
        fetch_limit=fetch_limit,
        output_dir=Path(output_dir),
    )
    return viz_cfg, influx_cfg


viz_config, influx_cfg = _load_configs()
CACHE_TTL_SECONDS = max(int(viz_config.refresh_seconds), 1)
st.set_page_config(page_title="Permafrost Visualization Dashboard", layout="wide", page_icon="🧊")

auto_refresh_ms = max(int(viz_config.refresh_seconds * 1000), 1000)
st.markdown(
    f"<script>setTimeout(function(){{window.location.reload();}}, {auto_refresh_ms});</script>",
    unsafe_allow_html=True,
)

st.sidebar.title("Visualization Dashboard")
st.sidebar.caption(
    "Explore FDM, PINN, and inversion outputs. Data refreshes automatically every "
    f"{int(viz_config.refresh_seconds)} seconds."
)

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_payload(fetch_limit: int, output_dir_str: str, influx_dict: Dict[str, str]):
    cfg = InfluxConfig(**influx_dict)
    bridge = VizDataBridge(influx_config=cfg, fetch_limit=fetch_limit, output_dir=output_dir_str)
    try:
        return bridge.load_payload()
    finally:
        bridge.close()


def refresh_data() -> Dict[str, Any]:
    payload = load_payload(
        viz_config.fetch_limit,
        str(viz_config.output_dir),
        asdict(influx_cfg),
    )
    return payload




payload = refresh_data()
status = payload.get("status", "unknown")
fetch_timestamp = payload.get("fetch_timestamp") or datetime.utcnow().isoformat()
st.sidebar.markdown(f"**Status:** {status}")
st.sidebar.caption(f"Last refresh at {fetch_timestamp}")

summary_lines = render_summary_lines(payload.get("data_summary") or {})
st.sidebar.subheader("Data Summary")
for line in summary_lines:
    st.sidebar.write(f"- {line}")

if status == "error":
    st.error(payload.get("error", "Unable to load data."))
    st.stop()

plot_keys = list(FIGURE_BUILDERS.keys())
plot_labels = {key: FIGURE_BUILDERS[key][0] for key in plot_keys}
selected_plot = st.sidebar.radio("Panels", plot_keys, format_func=lambda key: plot_labels[key])
figure_builder = FIGURE_BUILDERS[selected_plot][1]

figure, description = figure_builder(payload)
st.plotly_chart(figure, use_container_width=True)
st.caption(description)

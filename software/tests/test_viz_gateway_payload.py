"""Unit tests for VizGatewayServer payload assembly."""

from __future__ import annotations

import pandas as pd

import software.digital_twin.visualization.viz_gateway.viz_gateway_server as viz_module


def _make_df(values):
    return pd.DataFrame(values, columns=["time_days", "depth_m", "temperature"])


def test_build_payload_structure(tmp_path):
    server = viz_module.VizGatewayServer(output_dir=str(tmp_path))

    fdm_df = _make_df([
        (0.0, 0.0, 0.0),
        (0.0, 1.0, -1.0),
        (1.0, 0.0, 0.5),
        (1.0, 1.0, -0.5),
    ])
    pinn_df = _make_df([
        (0.0, 0.0, 0.1),
        (0.0, 1.0, -0.9),
        (1.0, 0.0, 0.6),
        (1.0, 1.0, -0.4),
    ])

    payload = server._build_payload(fdm_df, pinn_df, {"parameters": {"lambda_f": 1.5}})

    assert payload["status"] == "viz_ready"
    assert payload["data_summary"]["fdm_points"] == 4
    assert payload["data_summary"]["pinn_points"] == 4
    assert payload["comparison"]["pair_count"] == 4

    grid = payload["fdm"]["grid"]
    assert grid["time_days"] == [0.0, 1.0]
    assert grid["depth_m"] == [0.0, 1.0]

    comparison_stats = payload["comparison"]["stats"]["overall"]
    assert comparison_stats["mean_abs_error"] is not None
    assert comparison_stats["max_abs_error"] is not None

    inversion_section = payload["inversion"]
    assert inversion_section["parameters"]["lambda_f"] == 1.5
    assert inversion_section["history_path"] == str(viz_module.INVERSION_HISTORY_PATH)

    pinn_section = payload["pinn_forward"]
    assert pinn_section["metadata"]["history_path"] == str(viz_module.FORWARD_HISTORY_PATH)

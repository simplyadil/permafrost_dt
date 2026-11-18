"""Configuration helpers for startup scripts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from software.utils.logging_setup import configure_logging as configure_runtime_logging

DEFAULT_CONFIG: Dict[str, Any] = {
    "rabbitmq": {
        "host": "localhost",
        "username": "permafrost",
        "password": "permafrost",
    },
    "influxdb": {
        "url": "http://localhost:8086",
        "token": "permafrost",
        "org": "permafrost",
        "bucket": "permafrost_data",
        "username": "permafrost",
        "password": "permafrost",
    },
    "observation_ingestion_2d": {
        "queue": "permafrost.record.sensors.2d.ingest",
        "exchange": "permafrost.sensors.2d",
        "exchange_type": "fanout",
        "site_id": "default",
    },
    "boundary_condition_builder": {
        "input_queue": "permafrost.record.sensors.2d.bc",
        "input_exchange": "permafrost.sensors.2d",
        "input_exchange_type": "fanout",
        # Feeds the pipe_boundary enricher, which overrides the pipe boundary
        # set with a calibrated fluid-driven Robin BC and forwards the result to
        # the FEM on permafrost.record.boundary.2d. If pipe_boundary is not run,
        # point fem_forecast.input_queue at this raw queue instead.
        "output_queue": "permafrost.record.boundary.raw.2d",
        # NOTE on orientation: z_m = depth (0 = surface), so mesh edge set
        # "bottom" (z=0) is the physical SURFACE and "top" (z=0.3) is the deep
        # end. The cold Dirichlet anchor therefore belongs on "top" (deep,
        # frozen) and the surface ("bottom") is insulated. Assigning -5 C to
        # "bottom" cooled the surface and pushed heat downward, flipping the
        # forecast field vertically vs the observations.
        "boundaries": [
            {
                "name": "deep_frozen",
                "boundary_set": "top",
                "bc_type": "dirichlet",
                "temperature": -5.0,
            },
            {
                "name": "heater_left",
                "boundary_set": "left",
                "bc_type": "robin",
                "temperature": 27.0,
                "h_coeff": 5.0,
            },
            {
                "name": "soil_surface",
                "boundary_set": "bottom",
                "bc_type": "neumann",
                "heat_flux": 0.0,
            },
            {
                "name": "insulated_right",
                "boundary_set": "right",
                "bc_type": "neumann",
                "heat_flux": 0.0,
            },
        ],
        "sensor_overrides": [],
    },
    "thaw_front_reconstruction": {
        "input_queue": "permafrost.record.sensors.2d.thaw",
        "input_exchange": "permafrost.sensors.2d",
        "input_exchange_type": "fanout",
        "output_queue": "permafrost.record.thaw_front.2d",
        "thaw_threshold_c": 0.0,
        "contour_grid_r": 120,
        "contour_grid_z": 150,
        "rmin_depth_bins": 8,
        # Ignore contour points shallower than this (m) when computing the radius
        # metrics, to skip the near-surface horizontal 0C layer (an experimental
        # artefact) so radius_max reflects only the radial thaw front. The full
        # contour is still plotted; set 0.0 to disable.
        "surface_exclusion_m": 0.03,
        "site_id": "default",
    },
    "fem_inversion": {
        "input_queue": "permafrost.record.sensors.2d.inv",
        "input_exchange": "permafrost.sensors.2d",
        "input_exchange_type": "fanout",
        "output_queue": "permafrost.record.fem_inversion",
        "site_id": "default",
        # Only the soil parameters the FEM actually needs calibrated; the rest
        # of the material dict is held fixed. Bounds are multipliers on the
        # base value, so cap_s spans 395-1580, k_parameter 1.2-12 and lambda_s
        # 1.76-7.02.
        "invert_params": ["cap_s", "k_parameter", "lambda_s"],
        "relative_bounds": {
            "cap_s": [0.5, 2.0],
            "k_parameter": [0.4, 4.0],
            "lambda_s": [0.5, 2.0],
        },
        "loss": "mse",
        "window_size": 5,
        "min_snapshots": 5,
        "min_sensors": 3,
        "min_update_hours": 0.0,
        "min_update_seconds": 30.0,
        "relative_time_window": True,
        "min_horizon_hours": 0.001,
        "maxiter": 40,
        "maxfun": 80,
        "seed": 42,
        "publish_material_override_boundary": False,
        "fem_input_queue": "permafrost.record.boundary.2d",
    },
    "pipe_boundary": {
        # Enricher between the boundary builder and the FEM. Calibrates the
        # U-shaped heat-exchanger loop model against the live fluid stream and
        # rewrites the pipe boundary set with a Robin BC built from the
        # calibrated effective fluid temperature.
        "input_queue": "permafrost.record.boundary.raw.2d",
        "output_queue": "permafrost.record.boundary.2d",
        "fluid_exchange": "permafrost.fluid",
        "fluid_exchange_type": "fanout",
        "fluid_queue": "permafrost.record.fluid.pipe",
        "calibration_queue": "permafrost.record.pipe_calibration",
        "site_id": "default",
        # Pipe geometry / discretisation.
        "h_sample_m": 0.30,
        "r_bend_m": 0.0,
        "n_cells": 120,
        "max_dt_s": 5.0,
        # Cap on pipe-sim time steps; keeps high fluid velocities fast (the
        # scheme is unconditionally stable, so this only trades sub-step
        # transport resolution, which is quasi-steady at high v anyway).
        "max_steps": 3000,
        # Which FEM boundary set is the pipe/heat-exchanger interface, and how
        # the calibrated pipe state is mapped onto it.
        "pipe_boundary_set": "left",
        "pipe_bc_type": "robin",
        # Fluid->soil coupling at the pipe interface. The observed near-axis soil
        # crosses 0 C within minutes of the hot fluid pulse, i.e. very strong
        # coupling; h=5 was a placeholder that badly under-fed the FEM. Tune this
        # against a near-axis sensor's 0 C crossing time (raise for faster warming).
        "pipe_h_coeff": 50.0,
        "interface_source": "fluid",
        "interface_reduction": "mean",
        # Depth-resolved pipe boundary: map the U-tube fluid temperature to a
        # per-depth profile (hot near the surface inlet, coolest at the deep
        # bend) applied node-by-node along the pipe axis, instead of one
        # depth-uniform value. Set false to fall back to the scalar mean.
        "depth_resolved": True,
        "profile_points": 17,
        # Rolling window + calibration cadence.
        "window_seconds": 7200.0,
        "min_points": 5,
        "max_points": 240,
        "recalibrate_seconds": 300.0,
        "maxiter": 40,
        "maxfun": 80,
        # Pipe-model calibration controls for the six effective parameters
        # (v [m/s], kf/kw/kloss [1/s], Tw0/Tenv [C]). initial_params = starting
        # guess p0; lower_bounds/upper_bounds = search box. To FIX a parameter
        # (e.g. impose a known flow rate v), bracket it tightly in the bounds.
        # Omit Tw0 from initial_params to auto-seed it from the first outlet
        # reading. warm_start=true reuses the previous fit as the next start
        # (set false to always start from initial_params).
        "initial_params": {
            "v": 0.002,
            "kf": 0.004,
            "kw": 0.0002,
            "kloss": 0.00002,
            "Tw0": 20.0,
            "Tenv": 20.0,
        },
        "lower_bounds": {
            "v": 1e-5,
            "kf": 1e-5,
            "kw": 1e-7,
            "kloss": 0.0,
            "Tw0": -10.0,
            "Tenv": -10.0,
        },
        "upper_bounds": {
            "v": 0.05,
            "kf": 0.05,
            "kw": 0.01,
            "kloss": 0.005,
            "Tw0": 40.0,
            "Tenv": 40.0,
        },
        # Hold parameters at an exact imposed value (removed from the fit, not
        # just bracketed). E.g. to impose a known flow rate: {"v": 0.004}.
        # Leave empty to calibrate all six.
        # v fixed at the measured pipe velocity 0.338 m/s (bounds no longer apply
        # to v since it is removed from the fit).
        "fixed_params": {"v": 0.338},
        "warm_start": True,
        "passthrough_until_ready": True,
        "publish_calibration": True,
        # Record the model's predicted outlet next to the measured inlet/outlet
        # on every fluid sample, so the pipe model can be validated in the
        # dashboard. One extra pipe simulation per fluid message.
        "publish_outlet_validation": True,
    },
    "fem_forecast": {
        "input_queue": "permafrost.record.boundary.2d",
        "output_queue": "permafrost.record.fem_forecast.2d",
        "horizon_hours": 1.0,
        # Advance the FEM by the real elapsed time between observations so
        # simulated time tracks observed time (thaw-front spread rate matches
        # reality). A fixed horizon per message over-integrates when messages
        # arrive faster than the horizon. max_step_hours caps a single advance
        # after a data gap. Set track_observed_time false for the old behaviour.
        "track_observed_time": True,
        "max_step_hours": 0.5,
        "site_id": "default",
        "mesh_dir": "software/models/fem_mesh",
        "node_file": "DT_model_nodes.txt",
        "element_file": "DT_model_elements.txt",
        "edge_sets": {
            "top": "DT_model_top_edge.txt",
            "bottom": "DT_model_bottom_edge.txt",
            "left": "DT_model_left_edge.txt",
            "right": "DT_model_right_edge.txt",
        },
        "node_sets": {
            "top": "DT_model_top_nodes.txt",
            "bottom": "DT_model_bottom_nodes.txt",
            "left": "DT_model_left_nodes.txt",
            "right": "DT_model_right_nodes.txt",
        },
        "material": {
            "porosity": 0.43,
            "rho_s": 2650.0,
            "rho_w": 1000.0,
            "rho_i": 910.0,
            "cap_s": 790.0,
            "cap_w": 4180.0,
            "cap_i": 1930.0,
            "latent_heat": 334000.0,
            "k_parameter": 3.0,
            "tr_k": 273.15,
            # Bulk conductivity is mixed from the grain/water/ice values and
            # the phase saturations (see FEMSolver2D._bulk_conductivity), so
            # lambda_s replaces the old lambda_thawed/lambda_frozen pair.
            "lambda_s": 3.51,
            "lambda_w": 0.56,
            "lambda_i": 2.24,
        },
        "solver": {
            # Must divide the sensor cadence (CR1000 logs every 30 s). advance()
            # takes round(horizon/dt) whole steps of dt, so a dt that does not
            # divide the message interval integrates the wrong amount of time:
            # at dt=36 s a 30 s interval became one 36 s step (+20% heating) and
            # a 4x-strided 120 s interval became three (-10%).
            "dt_seconds": 30.0,
            "max_iterations": 25,
            "tol_r": 1e-6,
            "temperature_offset_k": 273.15,
            "initial_temperature_c": -1.0,
            "thaw_threshold_c": 0.0,
        },
        "save_field_csv_default": False,
        "include_field_default": False,
        "sticky_material_override": True,
        # Also march an uncalibrated twin (fixed material parameters, inversion
        # overrides ignored) and publish its field to fem_temperature_2d_baseline,
        # so the dashboard can show what the inversion is buying. Costs one extra
        # FEM solve per boundary message; set False to switch it off.
        "run_baseline": True,
    },
    "safety_monitor": {
        "thaw_input_queue": "permafrost.record.thaw_front.2d",
        "forecast_input_queue": "permafrost.record.fem_forecast.2d",
        "output_queue": "permafrost.record.safety_alerts.2d",
        # Closed-loop thaw control: gate the heater on both the outer thaw radius
        # (r_max, upper safety limit) and the least-thawed radius (r_min, lower
        # floor). Rules: r_max>=limit_max -> STOP; else keep HEATING; warning bands
        # on r_max at warn_fraction/critical_fraction of limit_max; r_min below its
        # floor is informational (keep heating). Set mode="single" for the legacy
        # single-metric behaviour.
        "mode": "dual",
        "limit_radius_m": 0.4,
        "limit_min_radius_m": 0.02,
        "warn_fraction": 0.6,
        "critical_fraction": 0.9,
        "metric": "max",
        "publish_on_change": False,
        "site_id": "default",
    },
    "heater_actuation": {
        "input_queue": "permafrost.record.safety_alerts.2d",
        "command_queue": "permafrost.command.heater.2d",
        "action_queue": "permafrost.record.heater_actions.2d",
        # In dual mode the safety monitor sends an explicit recommended_action
        # ("stop"/"heat") which this service honours; these action_on_* values are
        # the fallback for legacy single-metric alerts. setpoint_on_ok is the fluid
        # target commanded while heating (None = leave the controller's setpoint
        # unchanged; set e.g. 40.0 to drive a specific hot-fluid temperature).
        "action_on_alert": "stop",
        "action_on_ok": "heat",
        "setpoint_on_alert": 0.0,
        "setpoint_on_ok": 10.0,
        "publish_on_change": False,
        "site_id": "default",
    },
    "polyscience_control": {
        # Actuation hardware bridge: consumes the heater commands published by
        # heater_actuation and drives the physical PolyScience circulator over
        # RS232. IMPORTANT: 'port' must differ from cr1000_publisher.port —
        # pyserial opens COM ports exclusively (start_all_services enforces this).
        "port": "COM3",
        "baud_rate": 57600,
        "command_queue": "permafrost.command.heater.2d",
        # Map heater_actuation action strings to controller power states.
        # "hold" leaves the ON/OFF state untouched (only the setpoint changes).
        "action_map": {
            "start": "on", "run": "on", "on": "on", "heat": "on",
            "stop": "off", "off": "off", "shutdown": "off",
            "hold": "hold", "maintain": "hold",
        },
        # Safety clamp on any commanded setpoint before it reaches the hardware.
        "setpoint_min_c": -30.0,
        "setpoint_max_c": 50.0,
        "serial_timeout_s": 3.0,
        # Minimum spacing between consecutive RS232 commands. The controller drops
        # a command that lands while it is still processing the previous one, so
        # setpoint+power must not be sent back-to-back. Raise if TIMEOUTs persist.
        "inter_command_delay_s": 0.2,
        "open_retries": 3,
        "open_retry_delay_s": 2.0,
        "verify_after_apply": True,
        # Fail-safe: turn the heater OFF when the actuator process shuts down.
        "stop_on_shutdown": True,
        # dry_run drives the full pipeline without touching hardware (also auto-
        # enabled if pyserial is not installed). Set true to test integration.
        "dry_run": False,
        "site_id": "default",
    },
    "cr1000_publisher": {
        "port": "COM3",
        "baud_rate": 115200,
        "table_name": "Table1",
        "pakbus_addr": 1,
        "poll_interval_s": 2.0,
        "exchange": "permafrost.sensors.2d",
        "exchange_type": "fanout",
        # Fanout for inlet/outlet fluid temperatures consumed by the pipe_boundary
        # service. Requires fluid_fields below (map CR1000 columns to inlet/outlet).
        "fluid_exchange": "permafrost.fluid",
        # "fluid_fields": {"inlet": "Tin_C(1)", "outlet": "Tout_C(1)"},
        "site_id": "default",
        # Each entry maps a CR1000 field name to a physical sensor position.
        # sensor_id, x_m, and z_m are required; y_m is optional.
        # Example:
        #   {"field": "Temp_C_Avg(1)", "sensor_id": "s1", "x_m": 0.0, "z_m": -0.5}
        "sensors": [],
    },
    "viz_dashboard": {
        "host": "0.0.0.0",
        "port": 8501,
        "refresh_seconds": 20,
        "history_hours": 2.0,
        "max_points": 3000,
        "r_limit_m": 0.040,
        "safe_pct": 0.6,
        "caution_pct": 0.9,
        "site_id": "default",
    },
}


def load_startup_config(path: Path | str = Path("startup.conf")) -> Dict[str, Any]:
    """Load the startup configuration file, falling back to defaults."""

    config_path = Path(path)
    if not config_path.exists():
        return DEFAULT_CONFIG.copy()

    text = config_path.read_text(encoding="utf-8")
    stripped = "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))
    if not stripped.strip():
        return DEFAULT_CONFIG.copy()

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        logging.getLogger("startup").warning("Invalid startup.conf JSON. Using defaults.")
        return DEFAULT_CONFIG.copy()

    merged = DEFAULT_CONFIG.copy()
    merged.update(data)
    return merged


def configure_logging(config_path: Path | str = Path("logging.conf")) -> None:
    """Initialise logging using the shared configuration helper."""

    configure_runtime_logging()

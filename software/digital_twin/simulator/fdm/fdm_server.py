# pylint: disable=too-many-instance-attributes
"""Finite difference heat conduction server."""

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from software.digital_twin.communication.logger import setup_logger
from software.digital_twin.communication.messaging import RabbitMQClient, RabbitMQConfig, resolve_queue_config
from software.digital_twin.data_access.influx_utils import InfluxConfig, InfluxHelper, _parse_depth_value


@dataclass
class PhysicsParams:
    # Given parameters
    L: float = 3.34e5       # Latent heat of phase change (KJ/m^3)
    C_i: float = 1.672      # Heat capacity of ice (KJ/(m^3·K))
    C_l: float = 4.18       # Heat capacity of water (KJ/(m^3·K))
    lambda_i: float = 2.210 # Thermal conductivity of ice (W/(m·K))
    lambda_l: float = 0.465 # Thermal conductivity of water (W/(m·K))

    # Unknown/soil parameters (tunable)
    lambda_f: float = 1.5   # Thermal conductivity soil matrix (W/(m·K))
    C_f: float = 1.5        # Heat capacity soil matrix (KJ/(m^3·K))
    eta: float = 0.4        # Porosity
    b: float = 1.5          # Unfrozen water exponent
    T_nabla: float = -1.0   # Freezing temperature (°C)


@dataclass
class GridParams:
    Lx: float = 5.0         # Domain length (m)
    Nx: int = 50            # Spatial points
    # Time will be advanced adaptively between messages. Internal substepping picks dt_days.


BOUNDARY_QUEUE = "permafrost.record.boundary.state"
FDM_QUEUE = "permafrost.record.fdm.state"
BOUNDARY_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "boundary_forcing_message.json"
FDM_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "fdm_output_message.json"
SENSOR_QUEUE = "permafrost.record.sensors.state"
SENSOR_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "sensor_message.json"
DEFAULT_SENSOR_DEPTHS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
ALLOWED_SENSOR_DEPTHS: set[int] = set(DEFAULT_SENSOR_DEPTHS)


class FDMServer:
    """
    Full-physics 1D FDM solver for frozen soil heat conduction with phase change.
    Listens for boundary forcing (Ts, time_days), advances T(x,t) to new time,
    writes fdm_simulation to Influx, and notifies downstream via RabbitMQ.
    """

    def __init__(
        self,
        influx_config: InfluxConfig | None = None,
        boundary_config: RabbitMQConfig | None = None,
        outbound_config: RabbitMQConfig | None = None,
        sensor_config: RabbitMQConfig | None = None,
        sensor_depths: tuple[int, ...] | None = None,
    ) -> None:
        self.logger = setup_logger("FDMServer")
        self.influx_config = influx_config or InfluxConfig()

        self.boundary_config = resolve_queue_config(
            boundary_config,
            queue=BOUNDARY_QUEUE,
            schema_path=BOUNDARY_SCHEMA,
        )
        self.outbound_config = resolve_queue_config(
            outbound_config,
            queue=FDM_QUEUE,
            schema_path=FDM_SCHEMA,
        )
        self.sensor_config = (
            resolve_queue_config(
                sensor_config,
                queue=SENSOR_QUEUE,
                schema_path=SENSOR_SCHEMA,
            )
            if sensor_config is not None
            else None
        )

        self.mq_in: Optional[RabbitMQClient] = None
        self.mq_out: Optional[RabbitMQClient] = None
        self.mq_sensor: Optional[RabbitMQClient] = None
        self.influx: Optional[InfluxHelper] = None
        raw_depths = sensor_depths or DEFAULT_SENSOR_DEPTHS
        cleaned_depths: list[int] = []
        for depth in raw_depths:
            depth_float = float(depth)
            depth_int = int(round(depth_float))
            if not math.isclose(depth_float, depth_int, abs_tol=1e-6):
                raise ValueError(
                    f"Sensor depth {depth} must be specified in whole metres (0-5)."
                )
            if depth_int not in ALLOWED_SENSOR_DEPTHS:
                raise ValueError(
                    f"Unsupported sensor depth {depth}. Allowed discrete depths: {sorted(ALLOWED_SENSOR_DEPTHS)}"
                )
            cleaned_depths.append(depth_int)
        self.sensor_depths = tuple(cleaned_depths)

        # Physics + Grid
        self.phys = PhysicsParams()
        self.grid = GridParams()
        self.x = np.linspace(0.0, self.grid.Lx, self.grid.Nx)
        self.dx = self.x[1] - self.x[0]

        # State
        self.current_time_days: Optional[float] = None
        self.T: Optional[np.ndarray] = None
        self.theta_prev: Optional[np.ndarray] = None  # for dθ/dt
        self.bottom_bc_temp = 1.0  # °C

        # Initialize from an observed snapshot if available; else linear profile
        self.logger.info("FDMServer configured.")

    # ---------------------------
    # Physics helpers (from notebook)
    # ---------------------------
    def pore_water_content(self, T: np.ndarray) -> np.ndarray:
        Tn = self.phys.T_nabla
        b = self.phys.b
        # where(T >= Tn, 1, |Tn|^b * |T|^{-b})
        out = np.where(T >= Tn, 1.0, (abs(Tn) ** b) * (np.abs(T) ** (-b)))
        # clamp absurd values (e.g., very close to 0°C can blow up)
        return np.clip(out, 0.0, 10.0)

    def unfrozen_water_content(self, T: np.ndarray) -> np.ndarray:
        phi = self.pore_water_content(T)
        return self.phys.eta * phi

    def effective_concentration(self, T: np.ndarray) -> np.ndarray:
        # C_eff = phi * (C_f + eta*(C_l - C_i)) + (1 - phi) * C_f
        phi = self.pore_water_content(T)
        C_t = self.phys.C_f + self.phys.eta * (self.phys.C_l - self.phys.C_i)
        return phi * C_t + (1.0 - phi) * self.phys.C_f  # KJ/(m^3 K)

    def effective_lambda(self, T: np.ndarray) -> np.ndarray:
        # lambda_eff = (lambda_f * (lambda_l/lambda_i)^eta)^phi * (lambda_f)^(1-phi)
        phi = self.pore_water_content(T)
        inner = self.phys.lambda_f * ((self.phys.lambda_l / self.phys.lambda_i) ** self.phys.eta)
        return (inner ** phi) * (self.phys.lambda_f ** (1.0 - phi))

    # ---------------------------
    # Initialization
    # ---------------------------
    def _initialize_state(self):
        """
        Try initializing from last observed sensor_temperature (depth-tagged) at latest time.
        If not available, fall back to linear profile between Ts(0) and bottom_bc_temp=1°C.
        """
        try:
            if self.influx is None:
                raise RuntimeError("Influx helper is not initialised. Did you call setup()?")

            # Try to find the newest time_days in sensor_temperature
            df = self.influx.query_temperature(limit=2000)
            if df is not None and not df.empty and "time_days" in df.columns and "temperature" in df.columns:
                latest_t = float(sorted(df["time_days"].unique())[-1])
                depth_column = "depth_m" if "depth_m" in df.columns else "depth"
                if depth_column not in df.columns:
                    raise ValueError("Depth column missing from Influx response.")

                prof = df[df["time_days"] == latest_t].copy()
                prof["_depth_numeric"] = prof[depth_column].apply(_parse_depth_value)
                prof = prof.dropna(subset=["_depth_numeric"]).sort_values("_depth_numeric")

                depth_values = prof["_depth_numeric"].values
                temperature_values = prof["temperature"].values

                if len(depth_values) >= 2:
                    T_interp = np.interp(self.x, depth_values, temperature_values)
                    self.T = T_interp.astype(float)
                    self.current_time_days = latest_t
                    self.theta_prev = self.unfrozen_water_content(self.T)
                    self.logger.info("Initialized T(x) from Influx (sensor_temperature) at t=%s d.", latest_t)
                    return
                else:
                    self.logger.warning("Insufficient depth samples (%s) for interpolation.", len(depth_values))
        except Exception as e:
            self.logger.warning(f"Could not init from Influx, fallback to linear profile. Reason: {e}")

        # Fallback: linear interpolation between Ts(0) and 1°C
        Ts0 = self._sinusoid_air_temp(0.0)
        T_init = Ts0 + (self.bottom_bc_temp - Ts0) * (self.x / self.grid.Lx)
        self.T = T_init.astype(float)
        self.current_time_days = 0.0
        self.theta_prev = self.unfrozen_water_content(self.T)
        profile_debug = dict(zip(np.round(self.x, 1), np.round(self.T, 1)))
        self.logger.info("Initialized fallback linear profile: %s", profile_debug)

    # same sinusoid as notebook (used only if no forcing is available)
    @staticmethod
    def _sinusoid_air_temp(t_days: float) -> float:
        return 4.03 + 16.11 * math.sin((2 * math.pi * t_days / 365.0) - 1.709)

    # ---------------------------
    # Time integration
    # ---------------------------
    def _advance(self, t_from: float, t_to: float, Ts_from: float, Ts_to: float):
        """
        Advance the profile from t_from → t_to using explicit FDM.
        Boundary Ts is linearly interpolated between Ts_from and Ts_to during substeps.
        """
        assert self.T is not None and self.theta_prev is not None

        total_dt_days = float(t_to - t_from)
        if total_dt_days <= 0.0:
            return

        # Choose an internal substep (days). Keep it small for stability.
        # The notebook used huge Nt; here we pick a conservative dt like ~0.005 day (~7.2 min).
        dt_days = min(0.005, total_dt_days)  # adapt to not overshoot the interval
        n_steps = int(math.ceil(total_dt_days / dt_days))
        dt_days = total_dt_days / n_steps  # equalize steps

        for k in range(n_steps):
            t_k0 = t_from + k * dt_days
            t_k1 = t_k0 + dt_days
            alpha = (t_k1 - t_from) / (t_to - t_from) if t_to > t_from else 1.0
            Ts = (1 - alpha) * Ts_from + alpha * Ts_to  # linear interp BC

            # Apply Dirichlet BCs
            self.T[0] = Ts
            self.T[-1] = self.bottom_bc_temp

            # Coefficients at current T
            C_eff = self.effective_concentration(self.T)        # KJ/(m^3 K)
            lam = self.effective_lambda(self.T)                 # W/(m K)
            theta = self.unfrozen_water_content(self.T)

            # Spatial operator (interior nodes)
            rhs = np.zeros_like(self.T)
            # harmonic-like interface conductivities
            lam_plus = 0.5 * (lam[1:-1] + lam[2:])
            lam_minus = 0.5 * (lam[1:-1] + lam[:-2])
            rhs[1:-1] = (lam_plus * (self.T[2:] - self.T[1:-1]) - lam_minus * (self.T[1:-1] - self.T[:-2])) / (self.dx ** 2)

            # Latent term dθ/dt (use previous θ)
            dtheta_dt = (theta - self.theta_prev) / (dt_days + 1e-12)  # per day (consistent with dt_days unit)

            # Explicit Euler update on interior nodes
            self.T[1:-1] += dt_days * (rhs[1:-1] - self.phys.L * dtheta_dt[1:-1]) / (C_eff[1:-1] + 1e-12)

            # Update memory for θ (use latest θ)
            self.theta_prev = theta.copy()

            # Re-apply BCs (to avoid numerical drift)
            self.T[0] = Ts
            self.T[-1] = self.bottom_bc_temp

        # Done: now at t_to
        self.current_time_days = t_to

    # ---------------------------
    # I/O
    # ---------------------------
    def _write_profile(self, t_day: float, measurement: str = "fdm_simulation"):
        """
        Write the current profile to Influx (one point per depth).
        """
        assert self.T is not None
        if self.influx is None:
            raise RuntimeError("Influx helper is not initialised. Did you call setup()?")

        for depth, temp in zip(self.x, self.T):
            self.influx.write_model_temperature(
                measurement=measurement,
                time_days=float(t_day),
                depth=float(depth),
                temperature=float(temp),
                site="default",
                extra_tags={"model": "fdm"}
            )
        self.logger.info(f"Advanced FDM: Ts={self.T[0]:.2f}°C, wrote {len(self.T)} depths at t={t_day:g} d.")

    def _publish_sensor_observation(self, t_day: float) -> None:
        """
        Emit a synthetic sensor message derived from the current temperature profile.
        """
        if self.mq_sensor is None or self.T is None:
            return

        min_depth = float(self.x[0])
        max_depth = float(self.x[-1])
        if any(depth < min_depth or depth > max_depth for depth in self.sensor_depths):
            self.logger.warning(
                "Skipping synthetic sensor publish. Depths %s exceed grid bounds %.2f-%.2f m.",
                self.sensor_depths,
                min_depth,
                max_depth,
            )
            return

        probe_depths = np.array(self.sensor_depths, dtype=float)
        temperatures = np.interp(probe_depths, self.x, self.T)
        msg = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "time_days": float(t_day),
            # Flag downstream subscribers that a fresh profile is ready; tests expect this.
            "status": "ready",
        }
        for depth_int, temp in zip(self.sensor_depths, temperatures):
            msg[f"temperature_{depth_int}m"] = float(temp)

        try:
            self.mq_sensor.publish(msg)
            self.logger.info("Published synthetic sensor message at t=%s d.", t_day)
        except Exception as exc:  # pragma: no cover - depends on broker availability
            self.logger.error("Failed to publish synthetic sensor message: %s", exc)

    def _notify_ready(self, t_day: float):
        msg = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "time_days": float(t_day),
            "status": "ready"
        }
        if self.mq_out is None:
            raise RuntimeError("Outbound MQ client not initialised. Did you call setup()?")
        self.mq_out.publish(msg)

    # ---------------------------
    # Message handling
    # ---------------------------
    def _on_boundary_forcing(self, msg: dict):
        """
        msg schema: { time_days: number, Ts: number, timestamp?: string }
        """
        try:
            t_new: float = float(msg["time_days"])
            Ts_new: float = float(msg["Ts"])
        except Exception as e:
            self.logger.error(f"Malformed message, missing fields: {e}")
            return

        if self.current_time_days is None:
            # Initialize with linear profile between Ts_new and bottom
            T_init = Ts_new + (self.bottom_bc_temp - Ts_new) * (self.x / self.grid.Lx)
            self.T = T_init.astype(float)
            self.theta_prev = self.unfrozen_water_content(self.T)
            self.current_time_days = t_new
            self._write_profile(t_new)
            self._publish_sensor_observation(t_new)
            self._notify_ready(t_new)
            return

        # Advance from current_time_days → t_new
        Ts_prev = float(self.T[0]) if self.T is not None else Ts_new
        self._advance(self.current_time_days, t_new, Ts_prev, Ts_new)
        self._write_profile(t_new)
        self._publish_sensor_observation(t_new)
        self._notify_ready(t_new)

    # ---------------------------
    # Run
    # ---------------------------
    def run(self):
        if self.mq_in is None:
            raise RuntimeError("Inbound MQ client not initialised. Did you call setup()?")
        self.logger.info("FDMServer is now consuming boundary forcing...")
        self.mq_in.consume(callback=self._on_boundary_forcing)

    # ---------------------------
    # Lifecycle
    # ---------------------------
    def setup(self) -> None:
        """Initialise infrastructure dependencies."""

        if self.influx is None:
            self.influx = InfluxHelper(self.influx_config)
        if self.mq_in is None:
            self.mq_in = RabbitMQClient(self.boundary_config)
        if self.mq_out is None:
            self.mq_out = RabbitMQClient(self.outbound_config)
        if self.sensor_config is not None and self.mq_sensor is None:
            self.mq_sensor = RabbitMQClient(self.sensor_config)
        self._initialize_state()
        self.logger.info("FDMServer setup complete.")

    def start(self) -> None:
        """Start consuming boundary forcing messages."""

        if self.mq_in is None or self.mq_out is None or self.influx is None:
            self.setup()

        try:
            self.run()
        except KeyboardInterrupt:  # pragma: no cover - runtime
            self.logger.info("FDMServer interrupted. Shutting down...")
        finally:
            self.close()

    def stop(self) -> None:
        """Signal the run loop to stop."""

        if self.mq_in and self.mq_in.channel:
            self.mq_in.channel.stop_consuming()

    def close(self) -> None:
        """Release external resources."""

        self.stop()
        if self.mq_in is not None:
            self.mq_in.disconnect()
        if self.mq_out is not None:
            self.mq_out.disconnect()
        if self.mq_sensor is not None:
            self.mq_sensor.disconnect()
        if self.influx is not None:
            self.influx.close()
        self.logger.info("FDMServer shutdown complete.")


if __name__ == "__main__":
    FDMServer().start()

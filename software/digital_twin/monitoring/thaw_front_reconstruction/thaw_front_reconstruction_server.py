"""Thaw front reconstruction service for 2D sensor inputs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import griddata

from software.digital_twin.communication.messaging import RabbitMQClient, RabbitMQConfig, resolve_queue_config
from software.digital_twin.data_access.influx_utils import InfluxConfig, InfluxHelper
from software.utils.logging_setup import get_logger

SENSOR_QUEUE = "permafrost.record.sensors.2d"
THAW_FRONT_QUEUE = "permafrost.record.thaw_front.2d"
SENSOR_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "sensor_message_2d.json"
THAW_FRONT_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "thaw_front_message.json"


# =======================================================================================
# THAW FRONT RECONSTRUCTION SERVER
# =======================================================================================
class ThawFrontReconstructionServer:
    """Compute thaw front metrics and points from 2D sensor snapshots.
    
    Workflow:
    1. Receive sensor temperature message (CSV replay or real sensors)
    2. Parse sensor locations (x_m, z_m) and temperatures
    3. Compute two estimates of thaw front:
       a) PROXY method: Direct sensor filtering (simple, no interpolation)
       b) CONTOUR method: Interpolated field + isotherm extraction (smooth, continuous)
    4. Publish both results + primary metric (configured choice)
    5. Store metrics in InfluxDB for analysis
    """

    # =======================================================================================
    # INITIALIZATION AND CONFIGURATION
    # =======================================================================================
    def __init__(
        self,
        influx_config: InfluxConfig | None = None,
        input_config: RabbitMQConfig | None = None,
        output_config: RabbitMQConfig | None = None,
        *,
        input_queue: str | None = None,
        output_queue: str | None = None,
        thaw_threshold_c: float = 0.0,
        contour_grid_r: int = 120,
        contour_grid_z: int = 150,
        primary_metric_source: str = "contour",
        site_id: str = "default",
    ) -> None:
        """Initialize thaw front reconstruction service.
        
        Args:
            thaw_threshold_c: Temperature above which soil is considered thawed (default 0°C)
            contour_grid_r: Grid resolution in radial direction for interpolation
            contour_grid_z: Grid resolution in vertical direction for interpolation
            primary_metric_source: Which method to report as primary ("contour" or "proxy")
            site_id: Site identifier for InfluxDB storage
        
        Configuration flow:
          startup.conf → service_runners.py → this __init__ → configured instance ready to run
        """
        self.logger = get_logger("ThawFrontReconstructionServer")
        self.influx_config = influx_config or InfluxConfig()
        self.input_config = resolve_queue_config(
            input_config,
            queue=input_queue or SENSOR_QUEUE,
            schema_path=SENSOR_SCHEMA,
        )
        self.output_config = resolve_queue_config(
            output_config,
            queue=output_queue or THAW_FRONT_QUEUE,
            schema_path=THAW_FRONT_SCHEMA,
        )
        self.thaw_threshold_c = float(thaw_threshold_c)
        self.contour_grid_r = max(int(contour_grid_r), 10)
        self.contour_grid_z = max(int(contour_grid_z), 10)
        source = str(primary_metric_source).strip().lower()
        if source not in {"contour", "proxy"}:
            raise ValueError("primary_metric_source must be either 'contour' or 'proxy'.")
        self.primary_metric_source = source
        self.site_id = site_id

        self.mq_in: Optional[RabbitMQClient] = None
        self.mq_out: Optional[RabbitMQClient] = None
        self.influx: Optional[InfluxHelper] = None

    # =======================================================================================
    # DATA PARSING AND VALIDATION
    # =======================================================================================
    def _parse_sensor_samples(self, sensors: list[dict]) -> list[dict]:
        """Parse and validate raw sensor data from message.
        
        Input: Raw sensor list from sensor_message_2d.json
          [{"sensor_id": "D10-W15", "x_m": 0.1, "z_m": 0.15, "temperature": 0.25}, ...]
        
        Output: Standardized sensor samples (absolute x, float conversions)
          [{"x_m": 0.1, "z_m": 0.15, "temperature": 0.25}, ...]
        
        Handles missing/invalid data gracefully (skips, logs implicitly).
        """
        parsed: list[dict] = []
        for sensor in sensors:
            try:
                parsed.append(
                    {
                        "x_m": abs(float(sensor["x_m"])),  # <- Make x absolute (radial distance)
                        "z_m": float(sensor["z_m"]),
                        "temperature": float(sensor["temperature"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        return parsed

    # =======================================================================================
    # METHOD 1: PROXY THAW EXTRACTION (Simple, Direct)
    # =======================================================================================
    def _extract_proxy_thaw_points(self, sensors: list[dict]) -> list[dict]:
        """Extract thawed sensors directly (no interpolation).
        
        PROXY METHOD: Filter sensors where T >= threshold
        
        ✓ Advantage: Direct, unambiguous (uses actual measurements)
        ✓ Advantage: Works with sparse sensor networks
        ✗ Disadvantage: Jagged, depends on sensor distribution
        ✗ Disadvantage: Misses thaw between sensors
        
        Use when: You trust the sensor grid coverage and want direct measurements only
        """
        thawed = []
        for sensor in sensors:
            temperature = sensor.get("temperature")
            x_m = sensor.get("x_m")
            z_m = sensor.get("z_m")
            if temperature is None or x_m is None or z_m is None:
                continue
            if float(temperature) >= self.thaw_threshold_c:
                thawed.append({"x_m": float(x_m), "z_m": float(z_m)})
        return thawed

    # =======================================================================================
    # METRIC COMPUTATION
    # =======================================================================================
    def _compute_radius_metrics(self, points: list[dict]) -> tuple[float | None, float | None]:
        """Compute max and mean radial distance from domain center.
        
        Input: List of points with x_m coordinates
        Output: (radius_max_m, radius_avg_m)
        
        Metric definitions:
          - radius_max_m: Maximum radial extent of thaw (captures deepest thaw)
          - radius_avg_m: Average thaw radius (captures bulk thaw progress)
        
        These metrics are used by both proxy and contour methods,
        providing a single scalar measure of thaw-front position.
        """
        if not points:
            return None, None
        radii = [abs(float(point["x_m"])) for point in points]
        return max(radii), sum(radii) / len(radii)

    # =======================================================================================
    # METHOD 2: CONTOUR THAW EXTRACTION (Advanced, Interpolated)
    # =======================================================================================
    # The contour method is more sophisticated and requires several helper functions.
    # 
    # Overall workflow:
    #   1. Parse sensor samples from message
    #   2. Create interpolation grid (120×150 by default)
    #   3. Interpolate temperature field using scipy.griddata
    #   4. Extract isotherm (temperature contour at threshold)
    #   5. Trace contour into polylines
    #   6. Return main (largest) contour as thaw-front boundary
    #
    # Key insight: Converts discrete sensor measurements into a smooth field,
    # then extracts the mathematical isotherm for a continuous thaw boundary.
    # =======================================================================================

    @staticmethod
    def _iso_edge_point(
        p0: tuple[float, float],
        p1: tuple[float, float],
        v0: float,
        v1: float,
        threshold: float,
    ) -> tuple[float, float] | None:
        """Find isotherm crossing point on an edge (linear interpolation).
        
        Given two points p0, p1 with values v0, v1,
        find where the value equals threshold using linear interpolation.
        
        INPUT CASES:
          Case 1: Both values equal threshold → return p0 (endpoint on contour)
          Case 2: Only v0 equals threshold → return p0
          Case 3: Only v1 equals threshold → return p1
          Case 4: Values straddle threshold (v0 < th < v1 or v1 < th < v0)
                  → interpolate to find exact crossing point
          Case 5: Both on same side of threshold → return None (no crossing)
          Case 6: Degenerate cases (v0==v1, infinite values) → return None
        
        Mathematically: p_cross = p0 + t*(p1-p0) where t = (th-v0)/(v1-v0)
        
        OUTPUT: Crossing point as (x, z) tuple, or None if no valid crossing
        """
        if not np.isfinite(v0) or not np.isfinite(v1):
            return None
        d0 = v0 - threshold
        d1 = v1 - threshold
        if d0 == 0.0 and d1 == 0.0:
            return None
        if d0 == 0.0:
            return p0
        if d1 == 0.0:
            return p1
        if d0 * d1 > 0.0:
            return None
        denom = v1 - v0
        if denom == 0.0:
            return None
        ratio = (threshold - v0) / denom
        x = p0[0] + ratio * (p1[0] - p0[0])
        z = p0[1] + ratio * (p1[1] - p0[1])
        return float(x), float(z)

    @staticmethod
    def _point_key(point: tuple[float, float], ndigits: int = 10) -> tuple[float, float]:
        """Round point coordinates for deduplication (avoid floating-point precision issues).
        
        Example: (0.12345678901234, 0.98765432109876) → (0.1234567890, 0.9876543211)
        Prevents duplicate segments from being created due to rounding errors.
        """
        return round(point[0], ndigits), round(point[1], ndigits)

    @staticmethod
    def _classify_state(samples: list[dict], threshold_c: float) -> str:
        """Classify domain thaw state for diagnostic purposes.
        
        Returns:
          "no_data": No valid sensor readings
          "fully_frozen": All sensors T < threshold
          "fully_thawed": All sensors T >= threshold
          "transition_no_contour": Mixed state but contour extraction failed
                                   (usually means contour not captured in grid)
        """
        if not samples:
            return "no_data"
        temps = np.array([float(sample["temperature"]) for sample in samples], dtype=float)
        if np.all(temps < threshold_c):
            return "fully_frozen"
        if np.all(temps >= threshold_c):
            return "fully_thawed"
        return "transition_no_contour"

    def _extract_main_isocontour(
        self,
        T_grid: np.ndarray,
        r_grid: np.ndarray,
        z_grid: np.ndarray,
        threshold_c: float,
    ) -> list[dict]:
        """Extract the 0°C isotherm from interpolated temperature grid.
        
        This is the CORE of the contour method. It:
        1. Loops over all grid cells
        2. Finds edges where temperature crosses threshold
        3. Builds connected line segments
        4. Traces segments into polylines (connected chains)
        5. Returns the longest polyline as the main thaw front
        
        Algorithm: Marching Squares variant (checks each cell's 4 corners + center)
                   Builds segments, then connects them topologically.
        
        INPUT:
          T_grid: 2D array of interpolated temperatures (nz × nr)
          r_grid: Radial coordinates (1D array)
          z_grid: Vertical coordinates (1D array)
          threshold_c: Isotherm level (typically 0°C)
        
        OUTPUT:
          List of points forming the main contour polyline:
          [{"x_m": 0.1, "z_m": 0.2}, {"x_m": 0.12, "z_m": 0.25}, ...]
        """
        nz, nr = T_grid.shape
        segments: list[tuple[tuple[float, float], tuple[float, float]]] = []

        # ===== STEP 1: Find edge crossings in each grid cell =====
        for j in range(nz - 1):
            z0 = float(z_grid[j])
            z1 = float(z_grid[j + 1])
            for i in range(nr - 1):
                r0 = float(r_grid[i])
                r1 = float(r_grid[i + 1])

                # Cell corners: (r0,z0) → (r1,z0) → (r1,z1) → (r0,z1)
                v00 = T_grid[j, i]
                v10 = T_grid[j, i + 1]
                v11 = T_grid[j + 1, i + 1]
                v01 = T_grid[j + 1, i]
                if not np.isfinite(v00) or not np.isfinite(v10) or not np.isfinite(v11) or not np.isfinite(v01):
                    continue

                p00 = (r0, z0)
                p10 = (r1, z0)
                p11 = (r1, z1)
                p01 = (r0, z1)

                # Find contour intersections on each edge of the cell
                edge_points: dict[int, tuple[float, float]] = {}
                top = self._iso_edge_point(p00, p10, float(v00), float(v10), threshold_c)
                right = self._iso_edge_point(p10, p11, float(v10), float(v11), threshold_c)
                bottom = self._iso_edge_point(p11, p01, float(v11), float(v01), threshold_c)
                left = self._iso_edge_point(p01, p00, float(v01), float(v00), threshold_c)
                if top is not None:
                    edge_points[0] = top
                if right is not None:
                    edge_points[1] = right
                if bottom is not None:
                    edge_points[2] = bottom
                if left is not None:
                    edge_points[3] = left

                # Connect crossing points into segments
                # (Depends on configuration: 2, 3, or 4 crossings per cell)
                if len(edge_points) < 2:
                    continue
                if len(edge_points) == 2:
                    ids = list(edge_points.keys())
                    p0 = edge_points[ids[0]]
                    p1 = edge_points[ids[1]]
                    if p0 != p1:
                        segments.append((p0, p1))
                    continue
                if len(edge_points) == 4:
                    # 4-point case: ambiguous, use cell center value to resolve
                    center = float(v00 + v10 + v11 + v01) / 4.0
                    if center >= threshold_c:
                        pairings = [(0, 1), (2, 3)]  # Diagonal pairing A
                    else:
                        pairings = [(0, 3), (1, 2)]  # Diagonal pairing B
                    for e0, e1 in pairings:
                        p0 = edge_points[e0]
                        p1 = edge_points[e1]
                        if p0 != p1:
                            segments.append((p0, p1))
                    continue

                # 3-point case: connect adjacent edges
                ids = list(edge_points.keys())
                for idx in range(len(ids) - 1):
                    p0 = edge_points[ids[idx]]
                    p1 = edge_points[ids[idx + 1]]
                    if p0 != p1:
                        segments.append((p0, p1))

        if not segments:
            return []

        # ===== STEP 2: Build adjacency graph from segments =====
        # Store segments in a graph structure for connectivity queries
        neighbors: dict[tuple[float, float], set[tuple[float, float]]] = {}
        coords: dict[tuple[float, float], tuple[float, float]] = {}
        for p0, p1 in segments:
            k0 = self._point_key(p0)
            k1 = self._point_key(p1)
            if k0 == k1:
                continue
            coords[k0] = p0
            coords[k1] = p1
            neighbors.setdefault(k0, set()).add(k1)
            neighbors.setdefault(k1, set()).add(k0)

        if not neighbors:
            return []

        # ===== STEP 3: Trace connected segments into polylines =====
        def walk_component(start: tuple[float, float], comp: set[tuple[float, float]]) -> list[tuple[float, float]]:
            """Trace a polyline by following segment connectivity.
            
            Strategy: Start from a degree-1 node (line endpoint) if available,
                     otherwise start from any node. Walk the graph, following
                     unvisited edges, until hitting a dead end or returning to start.
            """
            degree1 = [k for k in comp if len(neighbors.get(k, set())) == 1]
            current = degree1[0] if degree1 else start
            prev: tuple[float, float] | None = None
            ordered = [current]
            visited_edges: set[tuple[tuple[float, float], tuple[float, float]]] = set()
            while True:
                next_candidates = [n for n in neighbors.get(current, set()) if n != prev]
                next_key = None
                for cand in next_candidates:
                    edge = (current, cand) if current < cand else (cand, current)
                    if edge not in visited_edges:
                        next_key = cand
                        visited_edges.add(edge)
                        break
                if next_key is None:
                    break
                ordered.append(next_key)
                prev, current = current, next_key
                if not degree1 and current == ordered[0]:
                    break
            return [coords[k] for k in ordered]

        # ===== STEP 4: Find all connected components =====
        # Contour may split into multiple disconnected pieces; find them all
        components: list[set[tuple[float, float]]] = []
        seen: set[tuple[float, float]] = set()
        for key in neighbors:
            if key in seen:
                continue
            stack = [key]
            comp: set[tuple[float, float]] = set()
            while stack:
                k = stack.pop()
                if k in comp:
                    continue
                comp.add(k)
                for n in neighbors.get(k, set()):
                    if n not in comp:
                        stack.append(n)
            seen |= comp
            components.append(comp)

        # ===== STEP 5: Trace each component and select the main one =====
        polylines: list[list[tuple[float, float]]] = []
        for comp in components:
            start = next(iter(comp))
            poly = walk_component(start, comp)
            if len(poly) > 1:
                polylines.append(poly)

        if not polylines:
            return []

        # Select main polyline: longest, with largest radial extent
        main_poly = max(polylines, key=lambda poly: (len(poly), max(abs(p[0]) for p in poly)))
        return [{"x_m": float(p[0]), "z_m": float(p[1])} for p in main_poly]

    def _reconstruct_contour_points(self, samples: list[dict]) -> list[dict]:
        """Reconstruct smooth thaw contour from sensor measurements (CONTOUR METHOD).
        
        Workflow:
        1. Extract sensor locations (r, z) and temperatures
        2. Create regular interpolation grid (120×150 points)
        3. Interpolate temperature field using scipy.griddata (linear method)
        4. Fill NaN values using nearest-neighbor interpolation (edge handling)
        5. Extract isotherm at threshold_c using _extract_main_isocontour()
        6. Return contour points
        
        ✓ Advantages:
          - Smooth, continuous contour (not jagged from discrete sensors)
          - Works with irregular sensor spacing
          - Single contour (main thaw front identified)
        ✗ Disadvantages:
          - Interpolation error (depends on grid resolution & sample distribution)
          - Extrapolation unreliable at domain edges
          - Sensitive to missing data or outliers
        
        Use when: You need a smooth thaw-front boundary for visualization/modeling
        """
        if len(samples) < 4:
            return []

        # Extract coordinate arrays for interpolation
        r_vals = np.array([float(sample["x_m"]) for sample in samples], dtype=float)
        z_vals = np.array([float(sample["z_m"]) for sample in samples], dtype=float)
        temp_vals = np.array([float(sample["temperature"]) for sample in samples], dtype=float)
        if not np.any(np.isfinite(temp_vals)):
            return []

        # Create interpolation grid
        r_min = float(np.min(r_vals))
        r_max = float(np.max(r_vals))
        z_min = float(np.min(z_vals))
        z_max = float(np.max(z_vals))
        r_grid = np.linspace(r_min, r_max, self.contour_grid_r)
        z_grid = np.linspace(z_min, z_max, self.contour_grid_z)
        R, Z = np.meshgrid(r_grid, z_grid)

        # Interpolate temperature field (linear method)
        T_grid = griddata(
            np.column_stack((r_vals, z_vals)),
            temp_vals,
            (R, Z),
            method="linear",
        )
        if np.all(np.isnan(T_grid)):
            return []

        # Fill NaN values with nearest-neighbor for edge robustness
        if np.isnan(T_grid).any():
            T_nearest = griddata(
                np.column_stack((r_vals, z_vals)),
                temp_vals,
                (R, Z),
                method="nearest",
            )
            T_grid = np.where(np.isnan(T_grid), T_nearest, T_grid)

        return self._extract_main_isocontour(T_grid, r_grid, z_grid, self.thaw_threshold_c)

    # =======================================================================================
    # MESSAGE PROCESSING AND OUTPUT
    # =======================================================================================
    def _process_message(self, msg: dict) -> None:
        """Process a single sensor snapshot and publish thaw-front metrics.
        
        MAIN WORKFLOW:
        1. Receive sensor message (from CSV replay or RabbitMQ)
        2. Parse sensor samples (locations + temperatures)
        3. Compute PROXY thaw (direct method)
        4. Compute CONTOUR thaw (interpolated method)
        5. Select primary metric (based on startup.conf configuration)
        6. Publish to RabbitMQ output queue
        7. Write to InfluxDB for archival and analysis
        
        Publishes BOTH methods, so doctor can compare in real-time.
        """
        if self.influx is None or self.mq_out is None:
            raise RuntimeError("Dependencies not initialised. Did you call setup()?")  # pragma: no cover

        time_hours = float(msg["time_hours"])
        samples = self._parse_sensor_samples(msg.get("sensors", []))
        
        # ===== Compute CONTOUR method (smooth interpolated thaw) =====
        contour_points = self._reconstruct_contour_points(samples)
        contour_found = bool(contour_points)
        contour_state = "transition" if contour_found else self._classify_state(samples, self.thaw_threshold_c)

        # ===== Compute PROXY method (direct sensor filtering) =====
        proxy_points = self._extract_proxy_thaw_points(samples)
        proxy_radius_max_m, proxy_radius_avg_m = self._compute_radius_metrics(proxy_points)
        contour_radius_max_m, contour_radius_avg_m = self._compute_radius_metrics(contour_points)

        # ===== Select primary metric (configured choice) =====
        # Configuration: startup.conf → thaw_front_reconstruction.primary_metric_source
        # Can be "proxy" (direct) or "contour" (interpolated)
        if self.primary_metric_source == "proxy":
            radius_max_m, radius_avg_m = proxy_radius_max_m, proxy_radius_avg_m
            output_points = proxy_points
        else:
            radius_max_m, radius_avg_m = contour_radius_max_m, contour_radius_avg_m
            output_points = contour_points

        # ===== Build output message =====
        # Contains BOTH methods for comparison, plus diagnostics
        payload = {
            "timestamp": msg.get("timestamp") or datetime.utcnow().isoformat(),
            "time_hours": time_hours,
            # Primary metrics (what doctor uses for analysis)
            "radius_max_m": radius_max_m,
            "radius_avg_m": radius_avg_m,
            "points": output_points,
            # Contour diagnostics
            "contour_found": contour_found,
            "contour_state": contour_state,
            "contour_point_count": len(contour_points),
            # Proxy diagnostics (for comparison)
            "proxy_radius_max_m": proxy_radius_max_m,
            "proxy_radius_avg_m": proxy_radius_avg_m,
            "proxy_point_count": len(proxy_points),
        }

        # ===== Publish to RabbitMQ and InfluxDB =====
        self.mq_out.publish(payload)
        self.influx.write_thaw_front_metrics(
            time_hours=time_hours,
            radius_max_m=radius_max_m,
            radius_avg_m=radius_avg_m,
            points=output_points,
            contour_found=contour_found,
            contour_state=contour_state,
            contour_point_count=len(contour_points),
            proxy_radius_max_m=proxy_radius_max_m,
            proxy_radius_avg_m=proxy_radius_avg_m,
            proxy_point_count=len(proxy_points),
            site=self.site_id,
        )
        self.logger.info(
            "Thaw front updated (t=%.2fh, source=%s, contour_points=%d, proxy_points=%d, state=%s)",
            time_hours,
            self.primary_metric_source,
            len(contour_points),
            len(proxy_points),
            contour_state,
        )

    # =======================================================================================
    # SERVICE LIFECYCLE
    # =======================================================================================
    def setup(self) -> None:
        """Initialize RabbitMQ and InfluxDB connections.
        
        Called before start() to establish dependencies.
        Separated from __init__ to support lazy connection (only when running).
        """
        if self.mq_in is None:
            self.mq_in = RabbitMQClient(self.input_config)
        if self.mq_out is None:
            self.mq_out = RabbitMQClient(self.output_config)
        if self.influx is None:
            self.influx = InfluxHelper(self.influx_config)
        self.logger.info(
            "Dependencies initialised (in=%s, out=%s)",
            self.input_config.queue,
            self.output_config.queue,
        )

    def start(self) -> None:
        """Start the service: connect to RabbitMQ and listen for sensor messages.
        
        Workflow:
        1. Call setup() to initialize connections
        2. Start consuming messages from sensor queue
        3. For each message, call _process_message()
        4. Catch interrupt (Ctrl+C) and shut down gracefully
        """
        if self.mq_in is None or self.mq_out is None or self.influx is None:
            self.setup()

        self.logger.info("Listening for sensor updates on %s", self.input_config.queue)
        try:
            self.mq_in.consume(callback=self._process_message)
        except KeyboardInterrupt:  # pragma: no cover - runtime behaviour
            self.logger.info("Interrupt received; shutting down")
        finally:
            self.close()

    def stop(self) -> None:
        """Stop listening for messages (non-blocking)."""
        if self.mq_in and self.mq_in.channel:
            self.mq_in.channel.stop_consuming()

    def close(self) -> None:
        """Disconnect all services and clean up resources."""
        self.stop()
        if self.mq_in is not None:
            self.mq_in.disconnect()
        if self.mq_out is not None:
            self.mq_out.disconnect()
        if self.influx is not None:
            self.influx.close()
        self.logger.info("Shutdown complete")


# =======================================================================================
# ENTRY POINT
# =======================================================================================
if __name__ == "__main__":
    ThawFrontReconstructionServer().start()
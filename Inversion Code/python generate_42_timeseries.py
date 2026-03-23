import json
import csv
from software.tools.run_fem_case import FEM

# 1) Material parameters
params = {
    "porosity": 0.42,
    "rho_s": 2700.0,
    "rho_w": 1000.0,
    "rho_i": 920.0,
    "cap_s": 790.0,
    "cap_w": 4180.0,
    "cap_i": 2090.0,
    "latent_heat": 334000.0,
    "k_parameter": 5.0,
    "tr_k": 273.15,
    "lambda_thawed": 1.64,
    "lambda_frozen": 2.96,
}

# 2) 42 sensor positions
depths_mm = (10, 50, 90, 170, 250, 290)
widths_mm = (15, 30, 45, 60, 75, 90, 120)

positions = [
    {
        "sensor_id": f"D{depth_mm}-W{width_mm}",
        "x_m": width_mm / 1000.0,
        "z_m": depth_mm / 1000.0,
    }
    for depth_mm in depths_mm
    for width_mm in widths_mm
]

# 3) Time points: 1 to 5 hours
times_hours = [1, 2, 3, 4, 5]

# 4) Initialize time-series results
series = {"times_hours": times_hours}

for pos in positions:
    sid = pos["sensor_id"]
    series[sid] = {
        "x_m": pos["x_m"],
        "z_m": pos["z_m"],
        "temperature_series": [],
    }

# 5) Run the forward simulation for each time point
for t in times_hours:
    result = FEM(
        params=params,
        positions=positions,
        horizon_hours=float(t),
        return_field=False,
    )

    print(f"Finished forward run for t = {t} h")

    for item in result["temperatures"]:
        sid = item["sensor_id"]
        series[sid]["temperature_series"].append(item["temperature"])

# 6) Save as JSON
json_file = "timeseries_42_positions_1_to_5h.json"
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(series, f, indent=2)

print(f"Saved time series to: {json_file}")

# 7) Save as CSV
csv_file = "timeseries_42_positions_1_to_5h.csv"
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    # Header
    header = ["sensor_id", "x_m", "z_m"] + [f"T_{t}h" for t in times_hours]
    writer.writerow(header)

    # Data rows
    for pos in positions:
        sid = pos["sensor_id"]
        row = [
            sid,
            series[sid]["x_m"],
            series[sid]["z_m"],
            *series[sid]["temperature_series"],
        ]
        writer.writerow(row)

print(f"Saved time series to: {csv_file}")
print(f"Total sensor positions: {len(positions)}")
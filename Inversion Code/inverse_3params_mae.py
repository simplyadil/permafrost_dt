import json
import csv
import random
from copy import deepcopy
from scipy.optimize import minimize
from software.tools.run_fem_case import FEM

# =========================================
# 1) Load observation data
# =========================================
OBS_FILE = "timeseries_42_positions_1_to_5h.json"

with open(OBS_FILE, "r", encoding="utf-8") as f:
    obs_data = json.load(f)

times_hours = obs_data["times_hours"]

positions = []
for key, value in obs_data.items():
    if key == "times_hours":
        continue
    positions.append(
        {
            "sensor_id": key,
            "x_m": value["x_m"],
            "z_m": value["z_m"],
        }
    )

obs_series = {}
for key, value in obs_data.items():
    if key == "times_hours":
        continue
    obs_series[key] = value["temperature_series"]

print(f"Loaded observation data from: {OBS_FILE}")
print(f"Number of sensor positions: {len(positions)}")
print(f"Time points: {times_hours}")

# =========================================
# 2) Fixed parameters
#    Only invert porosity, lambda_thawed, and cap_s
# =========================================
base_params = {
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

inv_param_names = ["porosity", "lambda_thawed", "cap_s"]

# =========================================
# 3) Reference values and bounds for inverted parameters
#    The normalized variables y are optimized on a relative scale
#    Actual parameter x = base_value * y
# =========================================
base_inv_values = {
    "porosity": base_params["porosity"],
    "lambda_thawed": base_params["lambda_thawed"],
    "cap_s": base_params["cap_s"],
}

# Relative scale bounds: 50% ~ 150%
relative_bounds = [
    (0.5, 1.5),  # porosity scale
    (0.5, 1.5),  # lambda_thawed scale
    (0.5, 1.5),  # cap_s scale
]

# Initial values: all start from 1.0, meaning starting from the reference values
# Initial values: randomly assigned within their respective bounds
random.seed(42)  # Keep this line for reproducibility; remove it for different random values each run

y0 = [
    random.uniform(*relative_bounds[0]),
    random.uniform(*relative_bounds[1]),
    random.uniform(*relative_bounds[2]),
]

print("Random initial scales:", y0)

# =========================================
# 4) Forward simulation
# =========================================
def generate_series(params, positions, times_hours):
    series = {pos["sensor_id"]: [] for pos in positions}

    for t in times_hours:
        result = FEM(
            params=params,
            positions=positions,
            horizon_hours=float(t),
            return_field=False,
        )

        for item in result["temperatures"]:
            sid = item["sensor_id"]
            series[sid].append(float(item["temperature"]))

    return series

# =========================================
# 5) MAE
# =========================================
def compute_mae(pred_series, obs_series):
    abs_errors = []

    for sid in obs_series:
        obs_vals = obs_series[sid]
        pred_vals = pred_series[sid]

        for obs, pred in zip(obs_vals, pred_vals):
            abs_errors.append(abs(pred - obs)**2)

    return sum(abs_errors) / len(abs_errors)

# =========================================
# 6) Convert normalized variables to actual parameters
#    y is the relative scaling factor
# =========================================
def normalized_to_params(y):
    params = deepcopy(base_params)

    params["porosity"] = base_inv_values["porosity"] * float(y[0])
    params["lambda_thawed"] = base_inv_values["lambda_thawed"] * float(y[1])
    params["cap_s"] = base_inv_values["cap_s"] * float(y[2])

    return params

# =========================================
# 7) Objective function
# =========================================
eval_count = 0
history_rows = []

def objective(y):
    global eval_count
    eval_count += 1

    params = normalized_to_params(y)
    pred_series = generate_series(params, positions, times_hours)
    mae = compute_mae(pred_series, obs_series)

    row = {
        "eval_id": eval_count,
        "porosity_scale": float(y[0]),
        "lambda_thawed_scale": float(y[1]),
        "cap_s_scale": float(y[2]),
        "porosity": params["porosity"],
        "lambda_thawed": params["lambda_thawed"],
        "cap_s": params["cap_s"],
        "mae": mae,
    }
    history_rows.append(row)

    print(
        f"[Eval {eval_count:03d}] "
        f"porosity={params['porosity']:.6f} ({y[0]:.4f}x), "
        f"lambda_thawed={params['lambda_thawed']:.6f} ({y[1]:.4f}x), "
        f"cap_s={params['cap_s']:.6f} ({y[2]:.4f}x), "
        f"MAE={mae:.6f}"
    )

    return mae

# =========================================
# 8) Optimization, up to 50 iterations
# =========================================
print("\nStarting inversion...\n")

result = minimize(
    objective,
    x0=y0,
    method="L-BFGS-B",
    bounds=relative_bounds,
    options={
        "maxiter": 50,
        "maxfun": 100,
    }
)

print("\nOptimization finished.\n")
print("Success:", result.success)
print("Message:", result.message)
print("Final MAE:", result.fun)

best_params = normalized_to_params(result.x)

print("\nBest inverted parameters:")
for name in inv_param_names:
    print(f"{name} = {best_params[name]}")

# =========================================
# 9) Save the best prediction
# =========================================
best_pred_series = generate_series(best_params, positions, times_hours)

pred_output = {"times_hours": times_hours}
for pos in positions:
    sid = pos["sensor_id"]
    pred_output[sid] = {
        "x_m": pos["x_m"],
        "z_m": pos["z_m"],
        "temperature_series": best_pred_series[sid],
    }

with open("inversion_best_prediction.json", "w", encoding="utf-8") as f:
    json.dump(pred_output, f, indent=2)

print("Saved best prediction to: inversion_best_prediction.json")

# =========================================
# 10) Save the optimization history
# =========================================
with open("inversion_history.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "eval_id",
        "porosity_scale",
        "lambda_thawed_scale",
        "cap_s_scale",
        "porosity",
        "lambda_thawed",
        "cap_s",
        "mae",
    ])
    for row in history_rows:
        writer.writerow([
            row["eval_id"],
            row["porosity_scale"],
            row["lambda_thawed_scale"],
            row["cap_s_scale"],
            row["porosity"],
            row["lambda_thawed"],
            row["cap_s"],
            row["mae"],
        ])

print("Saved inversion history to: inversion_history.csv")

# =========================================
# 11) Save the best parameters
# =========================================
with open("inverted_parameters.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "success": bool(result.success),
            "message": str(result.message),
            "final_mae": float(result.fun),
            "best_parameters": {
                "porosity": best_params["porosity"],
                "lambda_thawed": best_params["lambda_thawed"],
                "cap_s": best_params["cap_s"],
            },
            "best_scales": {
                "porosity_scale": float(result.x[0]),
                "lambda_thawed_scale": float(result.x[1]),
                "cap_s_scale": float(result.x[2]),
            },
            "initial_scales": {
                "porosity_scale": y0[0],
                "lambda_thawed_scale": y0[1],
                "cap_s_scale": y0[2],
            },
            "relative_bounds": {
                "porosity_scale": relative_bounds[0],
                "lambda_thawed_scale": relative_bounds[1],
                "cap_s_scale": relative_bounds[2],
            },
        },
        f,
        indent=2,
    )

print("Saved inverted parameters to: inverted_parameters.json")
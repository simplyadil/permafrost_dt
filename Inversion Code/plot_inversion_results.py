import json
import csv
import matplotlib.pyplot as plt

# =========================
# 1) Load files
# =========================
with open("inverted_parameters.json", "r", encoding="utf-8") as f:
    inv_param_data = json.load(f)

with open("timeseries_42_positions_1_to_5h.json", "r", encoding="utf-8") as f:
    true_data = json.load(f)

with open("inversion_best_prediction.json", "r", encoding="utf-8") as f:
    pred_data = json.load(f)

# Load inversion history from CSV
eval_ids = []
maes = []

with open("inversion_history.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        eval_ids.append(int(row["eval_id"]))
        maes.append(float(row["mae"]))

# =========================
# 2) Define true parameters and load inverted parameters
#    Parameter comparison is normalized so that true = 1
# =========================
true_params = {
    "porosity": 0.42,
    "lambda_thawed": 1.64,
    "cap_s": 790.0,
}

inv_params = inv_param_data["best_parameters"]

param_names = ["porosity", "lambda_thawed", "cap_s"]

true_vals_norm = [1.0 for _ in param_names]
inv_vals_norm = [inv_params[k] / true_params[k] for k in param_names]

# =========================
# 3) Plot normalized parameter comparison
# =========================
plt.figure(figsize=(7, 5))
x = list(range(len(param_names)))
width = 0.35

bars_true = plt.bar(
    [i - width / 2 for i in x],
    true_vals_norm,
    width=width,
    label="True (=1)"
)

bars_inv = plt.bar(
    [i + width / 2 for i in x],
    inv_vals_norm,
    width=width,
    label="Inverted / True"
)

plt.axhline(1.0, linestyle="--")
plt.xticks(x, param_names)
plt.ylabel("Normalized Value")
plt.title("Normalized Parameter Comparison (True = 1)")
plt.legend()

# Annotate inverted bars
for bar in bars_inv:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.6f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()
plt.show()

# =========================
# 4) Plot loss history
# =========================
plt.figure(figsize=(7, 5))
plt.plot(eval_ids, maes, marker="o")
plt.xlabel("Evaluation ID")
plt.ylabel("MAE")
plt.title("Loss History During Inversion")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# 5) Plot time-series comparison at representative sensors
# =========================
times = true_data["times_hours"]

sensor_ids_to_plot = [
    "D10-W15",
    "D90-W60",
    "D290-W120",
]

for sid in sensor_ids_to_plot:
    true_series = true_data[sid]["temperature_series"]
    pred_series = pred_data[sid]["temperature_series"]

    plt.figure(figsize=(7, 5))
    plt.plot(times, true_series, marker="o", label="True")
    plt.plot(times, pred_series, marker="s", label="Inverted")
    plt.xlabel("Time (hours)")
    plt.ylabel("Temperature")
    plt.title(f"Time-Series Comparison at {sid}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# =========================
# 6) Plot scatter: true vs inverted temperatures
# =========================
true_all = []
pred_all = []

for sid in true_data:
    if sid == "times_hours":
        continue
    true_all.extend(true_data[sid]["temperature_series"])
    pred_all.extend(pred_data[sid]["temperature_series"])

plt.figure(figsize=(6, 6))
plt.scatter(true_all, pred_all)

min_v = min(true_all + pred_all)
max_v = max(true_all + pred_all)
plt.plot([min_v, max_v], [min_v, max_v], "--")

plt.xlabel("True Temperature")
plt.ylabel("Inverted Temperature")
plt.title("True vs Inverted Temperatures")
plt.tight_layout()
plt.show()
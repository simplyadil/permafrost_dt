#!/usr/bin/env python3
import torch
from collections import OrderedDict
from pathlib import Path

# Import the forward inversion model class used by the codebase
from software.digital_twin.simulator.pinn_forward.freezing_soil_pinn import FreezingSoilPINN

# --- Configure to match training setup ---
params = {
    "L": 3.34e5,
    "C_i": 1.672,
    "C_l": 4.18,
    "lambda_i": 2.210,
    "lambda_l": 0.465,
    "lambda_f": 1.5,
    "C_f": 1.5,
    "eta": 0.4,
    "b": 1.5,
    "T_nabla": -1.0,
}
device = "cuda" if torch.cuda.is_available() else "cpu"

old_path = Path("software/models/pinn_forward/freezing_soil_pinn.pt")
if not old_path.exists():
    raise SystemExit(f"Checkpoint not found: {old_path.resolve()}")

print("Loading checkpoint:", old_path)
ck = torch.load(old_path, map_location=device)

# If ck is a dict with "model", extract it. Otherwise assume it is a state_dict.
if isinstance(ck, dict) and "model" in ck and isinstance(ck["model"], dict):
    state_dict = ck["model"]
else:
    state_dict = ck

# Show sample keys
print("Sample keys:", list(state_dict.keys())[:8])

# Detect old naming pattern
needs_remap = any(k.startswith("net.linear_") for k in state_dict.keys())
if needs_remap:
    print("Old key pattern detected. Remapping keys...")
    mapped = OrderedDict()
    linear_to_index = {
        "linear_0": "0",
        "linear_1": "2",
        "linear_2": "4",
        "linear_3": "6",
    }
    for k, v in state_dict.items():
        if k.startswith("net."):
            parts = k.split(".")
            if len(parts) == 3 and parts[1] in linear_to_index:
                new_k = f"net.{linear_to_index[parts[1]]}.{parts[2]}"
                mapped[new_k] = v
            else:
                mapped[k] = v
        else:
            mapped[k] = v
    state_dict = mapped
    print("Remapping complete. Example remapped keys:", list(state_dict.keys())[:8])
else:
    print("No remapping needed; keys look compatible.")

# Instantiate forward PINN and load remapped state_dict
print("Instantiating FreezingSoilPINN on device:", device)
pinn = FreezingSoilPINN(params, device=device)

try:
    pinn.model.load_state_dict(state_dict)
    print("State dict loaded into model successfully.")
except Exception as exc:
    print("Failed to load state dict:", exc)
    raise

# Save the full checkpoint expected by the DT loader: save model.state_dict() inside a dict
# The forward pinn class provides a save(model_dir) helper that writes model.state_dict(); we want a
# DT-compatible single file named as before with any additional metadata if needed.
out_path = old_path  # overwrite; change if you prefer new path
# Save raw state_dict (the rest of the code expects a plain state_dict file)
torch.save(pinn.model.state_dict(), out_path)
print("Converted checkpoint saved to:", out_path)


# angelica.portocarrero04@gmail.com
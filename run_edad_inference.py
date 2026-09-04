"""
Run this INSIDE your .venv (where torch is installed).
Usage:
    .\.venv\Scripts\python.exe run_edad_inference.py

Requires in the SAME folder:
    - edad_model.py
    - edad_best.pt
    - synthetic_test_cases_full.csv

Produces:
    - edad_raw_scores.csv   (upload this back to Claude)
"""

import numpy as np
import pandas as pd
import torch

from edad_model import EDADModel

FEATURE_ORDER = [
    "air_temperature_C",
    "relative_humidity_pct",
    "air_pressure_hPa",
    "wind_speed_ms",
    "wind_direction_deg",
    "leaf_wetness_pct",
    "soil_moisture_pct",
    "solar_radiation_Wm2",
]
WINDOW = 12  # t0..t11

df = pd.read_csv("synthetic_test_cases_full.csv")
print("Loaded test cases:", df.shape)

# Build tensor of shape (N, T=12, D=8) in the exact feature order used at training time
N = len(df)
X = np.zeros((N, WINDOW, len(FEATURE_ORDER)), dtype=np.float32)
for d_idx, feat in enumerate(FEATURE_ORDER):
    for t in range(WINDOW):
        col = f"{feat}_t{t}"
        X[:, t, d_idx] = df[col].values

X_tensor = torch.from_numpy(X)

# Load checkpoint
ckpt = torch.load("edad_best.pt", map_location="cpu", weights_only=False)
print("Checkpoint keys:", list(ckpt.keys()))
print("Trained epoch:", ckpt.get("epoch"))

model = EDADModel(input_dim=8, window_size=12, hidden_dim=64, latent_dim=32, temperature=0.1)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Run in batches to be safe on memory
batch_size = 256
scores = []
with torch.no_grad():
    for i in range(0, N, batch_size):
        batch = X_tensor[i:i + batch_size]
        s = model.compute_anomaly_scores(batch)
        scores.append(s)
scores = np.concatenate(scores)

out = pd.DataFrame({
    "test_case_id": df["test_case_id"].values,
    "node_id": df["node_id"].values,
    "true_label": df["label"].values,
    "injection_type": df["injection_type"].values,
    "edad_raw_score": scores,
})
out.to_csv("edad_raw_scores.csv", index=False)
print("Saved edad_raw_scores.csv with", len(out), "rows")
print(out.head())

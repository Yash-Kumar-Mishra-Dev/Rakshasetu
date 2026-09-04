"""
RakshaSetu Condition Prediction & Forecasting Pipeline - Aurora Model Training
SRS Section 8 AI/ML Core: Environmental Condition Forecaster Training Loop
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import set_seed, get_device, load_config, ExecutionTimer
from src.feature_engineering import create_per_node_forecasting_windows, TimeSeriesForecastingDataset
from src.aurora_model import AuroraForecaster


def train_aurora_pipeline(config_path: str = "config/config.yaml"):
    config = load_config(config_path)
    seed = config["pipeline"]["random_seed"]
    set_seed(seed)
    device = get_device()
    print("=" * 80)
    print("  RAKSHASETU AURORA CONDITION FORECASTING MODEL TRAINING")
    print(f"  Device: {device} | Random Seed: {seed}")
    print("=" * 80, flush=True)

    # 1. Load preprocessed splits
    train_csv = config["paths"]["train_data"]
    val_csv = config["paths"]["val_data"]
    checkpoint_path = config["paths"]["aurora_checkpoint"]
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    print(f"[Step 1/4] Loading preprocessed datasets...", flush=True)
    print(f"  Train data: {train_csv}")
    print(f"  Val data:   {val_csv}")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # Features: scaled environmental features
    env_features = config["schema"]["environmental_features"]
    scaled_features = [f"{feat}_scaled" for feat in env_features]

    aurora_cfg = config.get("aurora_model", {})
    input_window = int(aurora_cfg.get("input_window", 12))
    horizon = int(aurora_cfg.get("forecast_horizon", 6))
    stride = int(aurora_cfg.get("stride", 1))
    batch_size = int(aurora_cfg.get("batch_size", 64))
    lr = float(aurora_cfg.get("learning_rate", 0.001))
    weight_decay = float(aurora_cfg.get("weight_decay", 0.0001))
    epochs = int(aurora_cfg.get("epochs", 15))
    patience = int(aurora_cfg.get("early_stopping_patience", 4))
    grad_clip = float(aurora_cfg.get("gradient_clip_val", 5.0))
    hidden_dim = int(aurora_cfg.get("hidden_dim", 64))
    num_layers = int(aurora_cfg.get("num_layers", 2))
    dropout = float(aurora_cfg.get("dropout", 0.1))

    # 2. Windowing strictly PER NODE (no cross-node or future leakage)
    print(f"\n[Step 2/4] Constructing forecasting windows strictly per node...", flush=True)
    print(f"  Input Window (Tin):  {input_window} steps ({input_window * 10} minutes)")
    print(f"  Forecast Horizon (H): {horizon} steps ({horizon * 10} minutes ahead)")
    print(f"  Features:            {len(scaled_features)} channels")

    t0_win = time.perf_counter()
    X_train, Y_train, meta_train = create_per_node_forecasting_windows(
        train_df, scaled_features, input_window=input_window, horizon=horizon, stride=stride
    )
    X_val, Y_val, meta_val = create_per_node_forecasting_windows(
        val_df, scaled_features, input_window=input_window, horizon=horizon, stride=stride
    )
    print(f"  Window construction finished in {time.perf_counter() - t0_win:.2f}s.")
    print(f"  Total Train Forecasting Windows: {len(X_train):,}")
    print(f"  Total Val Forecasting Windows:   {len(X_val):,}")

    train_dataset = TimeSeriesForecastingDataset(X_train, Y_train, meta_train)
    val_dataset = TimeSeriesForecastingDataset(X_val, Y_val, meta_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # 3. Model instantiation
    print(f"\n[Step 3/4] Initializing AuroraForecaster architecture...", flush=True)
    model = AuroraForecaster(
        input_dim=len(scaled_features),
        input_window=input_window,
        forecast_horizon=horizon,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    print(f"\n[Step 4/4] Starting training loop ({epochs} max epochs)...", flush=True)
    start_train_time = time.perf_counter()
    total_batches = len(train_loader)

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_losses = []

        for batch_idx, (x, y, _) in enumerate(train_loader, 1):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = model.compute_loss(y_pred, y)
            loss.backward()

            # Stability guard: gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            train_losses.append(loss.item())

            if batch_idx % 300 == 0 or batch_idx == total_batches:
                print(f"  [Epoch {epoch:02d}/{epochs:02d} | Batch {batch_idx:04d}/{total_batches:04d}] Batch Huber Loss: {loss.item():.5f}", flush=True)

        # Validation loop
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                val_loss = model.compute_loss(y_pred, y)
                val_losses.append(val_loss.item())

        avg_train_loss = float(np.mean(train_losses))
        avg_val_loss = float(np.mean(val_losses))
        epoch_dur = time.perf_counter() - epoch_start

        scheduler.step(avg_val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])

        epoch_stat = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "epoch_seconds": round(epoch_dur, 2),
            "lr": current_lr
        }
        history.append(epoch_stat)

        print(f"--> Epoch {epoch:02d}/{epochs:02d} ({epoch_dur:.1f}s) | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | LR: {current_lr:.1e}", flush=True)

        # Checkpointing & Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": aurora_cfg,
                "total_params": total_params,
                "input_dim": len(scaled_features),
                "input_window": input_window,
                "forecast_horizon": horizon,
                "hidden_dim": hidden_dim
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  --> Saved new best checkpoint to {checkpoint_path} (Val Loss: {best_val_loss:.5f})", flush=True)
        else:
            patience_counter += 1
            print(f"  Early stopping counter: {patience_counter}/{patience}", flush=True)
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.", flush=True)
                break

    total_train_time = time.perf_counter() - start_train_time
    print(f"\nTraining completed in {total_train_time:.2f}s.", flush=True)

    # Save training log
    reports_dir = config["paths"]["reports_dir"]
    os.makedirs(reports_dir, exist_ok=True)
    log_path = os.path.join(reports_dir, "aurora_training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": "Aurora Condition Forecasting Model (Seq2Seq BiGRU)",
            "total_params": total_params,
            "train_time_sec": round(total_train_time, 2),
            "best_val_loss": best_val_loss,
            "epochs_run": len(history),
            "history": history
        }, f, indent=2)
    print(f"Saved training log to: {log_path}", flush=True)

    return model, history


if __name__ == "__main__":
    with ExecutionTimer("Aurora Model Training"):
        train_aurora_pipeline()

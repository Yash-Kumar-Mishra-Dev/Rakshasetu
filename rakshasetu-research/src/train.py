"""
RakshaSetu Anomaly Detection Research Pipeline - EDAD Model Training
Synthetic Anomaly Benchmark — Encode-then-Decompose Training Loop with Early Stopping
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
from src.feature_engineering import create_per_node_windows, TimeSeriesWindowDataset
from src.edad_model import EDADModel

def train_edad_pipeline(config_path: str = "config/config.yaml"):
    config = load_config(config_path)
    seed = config["pipeline"]["random_seed"]
    set_seed(seed)
    device = get_device()
    print(f"Using device: {device}", flush=True)

    # 1. Load preprocessed splits
    train_csv = config["paths"]["train_data"]
    val_csv = config["paths"]["val_data"]
    checkpoint_path = config["paths"]["edad_checkpoint"]
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    print(f"Loading train data: {train_csv}", flush=True)
    print(f"Loading val data: {val_csv}", flush=True)
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # Features: use the scaled environmental features
    env_features = config["schema"]["environmental_features"]
    scaled_features = [f"{feat}_scaled" for feat in env_features]

    window_size = config["edad_model"]["window_size"]
    stride = config["edad_model"]["stride"]
    batch_size = config["edad_model"]["batch_size"]
    lr = float(config["edad_model"]["learning_rate"])
    weight_decay = float(config["edad_model"]["weight_decay"])
    epochs = int(config["edad_model"]["epochs"])
    patience = int(config["edad_model"]["early_stopping_patience"])
    grad_clip = float(config["edad_model"]["gradient_clip_val"])

    # 2. Windowing strictly PER NODE (no cross-node boundary leakage)
    print(f"Constructing sliding windows (eta={window_size}, stride={stride}) strictly per node...", flush=True)
    t0_win = time.perf_counter()
    train_windows, train_meta = create_per_node_windows(
        train_df, scaled_features, window_size=window_size, stride=stride
    )
    val_windows, val_meta = create_per_node_windows(
        val_df, scaled_features, window_size=window_size, stride=stride
    )
    print(f"Windows created in {time.perf_counter() - t0_win:.2f}s.", flush=True)
    print(f"Total train windows: {len(train_windows):,}", flush=True)
    print(f"Total val windows:   {len(val_windows):,}", flush=True)

    train_dataset = TimeSeriesWindowDataset(train_windows, train_meta, return_pairs=True)
    val_dataset = TimeSeriesWindowDataset(val_windows, val_meta, return_pairs=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # 3. Instantiate model
    model = EDADModel(
        input_dim=len(scaled_features),
        window_size=window_size,
        hidden_dim=config["edad_model"]["hidden_dim"],
        latent_dim=config["edad_model"]["latent_dim"],
        temperature=float(config["edad_model"]["contrastive_temperature"])
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized EDAD-inspired model with {total_params:,} trainable parameters.", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    start_train_time = time.perf_counter()
    total_batches = len(train_loader)

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_losses = []
        recon_losses = []
        cont_losses = []
        mi_losses = []

        for batch_idx, (x, x_pos, _) in enumerate(train_loader, 1):
            x = x.to(device)
            x_pos = x_pos.to(device)

            optimizer.zero_grad()
            loss_dict = model.compute_loss(
                x, x_pos,
                lambda_recon=float(config["edad_model"]["lambda_recon"]),
                lambda_contrastive=float(config["edad_model"]["lambda_contrastive"]),
                lambda_mi=float(config["edad_model"]["lambda_mi_penalty"])
            )
            total_loss = loss_dict["total_loss"]
            total_loss.backward()

            # Stability guard: gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            train_losses.append(total_loss.item())
            recon_losses.append(loss_dict["loss_recon"])
            cont_losses.append(loss_dict["loss_contrastive"])
            mi_losses.append(loss_dict["loss_mi_penalty"])

            if batch_idx % 300 == 0 or batch_idx == total_batches:
                print(f"  [Epoch {epoch:02d}/{epochs:02d} | Batch {batch_idx:04d}/{total_batches:04d}] Batch Loss: {total_loss.item():.4f} (Recon: {loss_dict['loss_recon']:.4f}, Cont: {loss_dict['loss_contrastive']:.4f}, MI: {loss_dict['loss_mi_penalty']:.4f})", flush=True)

        # Validation loop
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, x_pos, _ in val_loader:
                x = x.to(device)
                x_pos = x_pos.to(device)
                val_loss_dict = model.compute_loss(
                    x, x_pos,
                    lambda_recon=float(config["edad_model"]["lambda_recon"]),
                    lambda_contrastive=float(config["edad_model"]["lambda_contrastive"]),
                    lambda_mi=float(config["edad_model"]["lambda_mi_penalty"])
                )
                val_losses.append(val_loss_dict["total_loss"].item())

        avg_train_loss = float(np.mean(train_losses))
        avg_recon_loss = float(np.mean(recon_losses))
        avg_cont_loss = float(np.mean(cont_losses))
        avg_mi_loss = float(np.mean(mi_losses))
        avg_val_loss = float(np.mean(val_losses))
        epoch_dur = time.perf_counter() - epoch_start

        scheduler.step(avg_val_loss)

        epoch_stat = {
            "epoch": epoch,
            "train_total_loss": avg_train_loss,
            "train_recon_loss": avg_recon_loss,
            "train_cont_loss": avg_cont_loss,
            "train_mi_penalty": avg_mi_loss,
            "val_total_loss": avg_val_loss,
            "epoch_seconds": round(epoch_dur, 2),
            "lr": float(optimizer.param_groups[0]["lr"])
        }
        history.append(epoch_stat)

        print(f"--> Epoch {epoch:02d}/{epochs:02d} Summary ({epoch_dur:.1f}s) | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {epoch_stat['lr']:.1e}", flush=True)

        # Checkpointing & Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": config["edad_model"],
                "total_params": total_params
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  --> Saved new best checkpoint to {checkpoint_path} (Val Loss: {best_val_loss:.4f})", flush=True)
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}", flush=True)
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.", flush=True)
                break

    total_train_time = time.perf_counter() - start_train_time
    print(f"Training completed in {total_train_time:.2f}s.", flush=True)

    # Save training log history
    log_path = os.path.join(config["paths"]["reports_dir"], "edad_training_log.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": "EDAD-inspired Encode-then-Decompose model",
            "benchmark_classification": "Synthetic Anomaly Benchmark",
            "total_params": total_params,
            "train_time_sec": round(total_train_time, 2),
            "best_val_loss": best_val_loss,
            "epochs_run": len(history),
            "history": history
        }, f, indent=2)

    return model, history

if __name__ == "__main__":
    with ExecutionTimer("Step 3: EDAD Model Training"):
        train_edad_pipeline()

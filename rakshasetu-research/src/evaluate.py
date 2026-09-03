"""
RakshaSetu Anomaly Detection Research Pipeline - Evaluation Module
Synthetic Anomaly Benchmark — Unsupervised Comparative Analysis & Sanity Verification
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import set_seed, get_device, load_config, ExecutionTimer
from src.feature_engineering import create_per_node_windows, TimeSeriesWindowDataset
from src.zscore_baseline import ZScoreBaselineDetector
from src.edad_model import EDADModel

def run_evaluation(config_path: str = "config/config.yaml"):
    config = load_config(config_path)
    seed = config["pipeline"]["random_seed"]
    set_seed(seed)
    device = get_device()

    figures_dir = config["paths"]["figures_dir"]
    reports_dir = config["paths"]["reports_dir"]
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    test_csv = config["paths"]["test_data"]
    val_csv = config["paths"]["val_data"]
    zscore_path = config["paths"]["zscore_model_file"]
    edad_ckpt_path = config["paths"]["edad_checkpoint"]

    print(f"Loading test split from: {test_csv}", flush=True)
    test_df = pd.read_csv(test_csv)
    val_df = pd.read_csv(val_csv)

    env_features = config["schema"]["environmental_features"]
    scaled_features = [f"{feat}_scaled" for feat in env_features]
    window_size = config["edad_model"]["window_size"]
    stride = config["edad_model"]["stride"]

    # 1. Create per-node sliding windows for test set
    test_windows, test_meta = create_per_node_windows(
        test_df, scaled_features, window_size=window_size, stride=stride
    )
    print(f"Constructed {len(test_windows):,} test windows strictly per-node.", flush=True)

    # 2. Evaluate Z-Score Baseline
    print("Evaluating Z-Score Baseline detector on test split...", flush=True)
    zscore_detector = ZScoreBaselineDetector.load(zscore_path, config)
    
    t0_z = time.perf_counter()
    test_df_scored = zscore_detector.predict_dataframe(test_df)
    zscore_runtime = time.perf_counter() - t0_z

    # For window-level Z-Score: compute max abs Z across all time points in each window
    # We map window metadata to DataFrame indices
    zscore_window_max = []
    zscore_window_mean = []
    
    # Pre-index node slices
    node_z_max = {}
    node_z_mean = {}
    for node_id, group in test_df_scored.groupby("node_id"):
        grp_sorted = group.sort_values("timestamp").reset_index(drop=True)
        node_z_max[node_id] = grp_sorted["zscore_max_abs"].values
        node_z_mean[node_id] = grp_sorted["zscore_mean_abs"].values

    for m in test_meta:
        nid = m["node_id"]
        s_idx, e_idx = m["start_idx"], m["end_idx"]
        w_max = np.max(node_z_max[nid][s_idx : e_idx + 1])
        w_mean = np.mean(node_z_mean[nid][s_idx : e_idx + 1])
        zscore_window_max.append(w_max)
        zscore_window_mean.append(w_mean)

    zscore_window_max = np.array(zscore_window_max)
    zscore_window_mean = np.array(zscore_window_mean)

    # 3. Evaluate EDAD-Inspired Model
    print("Evaluating EDAD-inspired model on test split...", flush=True)
    if not os.path.exists(edad_ckpt_path):
        raise FileNotFoundError(f"EDAD checkpoint not found at: {edad_ckpt_path}")

    checkpoint = torch.load(edad_ckpt_path, map_location=device)
    edad_model = EDADModel(
        input_dim=len(scaled_features),
        window_size=window_size,
        hidden_dim=config["edad_model"]["hidden_dim"],
        latent_dim=config["edad_model"]["latent_dim"],
        temperature=float(config["edad_model"]["contrastive_temperature"])
    ).to(device)
    edad_model.load_state_dict(checkpoint["model_state_dict"])
    edad_model.eval()

    # Inference timing & batch scoring
    t0_edad = time.perf_counter()
    edad_scores_list = []
    batch_size = 128
    with torch.no_grad():
        for i in range(0, len(test_windows), batch_size):
            batch_np = test_windows[i : i + batch_size]
            batch_t = torch.tensor(batch_np, dtype=torch.float32, device=device)
            scores = edad_model.compute_anomaly_scores(batch_t)
            edad_scores_list.append(scores)
    edad_runtime = time.perf_counter() - t0_edad
    edad_window_scores = np.concatenate(edad_scores_list)

    # Compute validation calibration for EDAD thresholds
    val_windows, _ = create_per_node_windows(
        val_df, scaled_features, window_size=window_size, stride=stride
    )
    val_scores_list = []
    with torch.no_grad():
        for i in range(0, len(val_windows), batch_size):
            batch_np = val_windows[i : i + batch_size]
            batch_t = torch.tensor(batch_np, dtype=torch.float32, device=device)
            val_scores_list.append(edad_model.compute_anomaly_scores(batch_t))
    val_window_scores = np.concatenate(val_scores_list)

    val_mean = float(np.mean(val_window_scores))
    val_std = float(np.std(val_window_scores))

    # Threshold definitions for EDAD:
    # 1. Standard Gaussian deviation of val scores: mu + 2*sigma, mu + 2.5*sigma, mu + 3*sigma
    # 2. Quantile matches to Z-Score flag rates (e.g. top 5%, top 2.5%, top 1%)
    edad_th_20 = val_mean + 2.0 * val_std
    edad_th_25 = val_mean + 2.5 * val_std
    edad_th_30 = val_mean + 3.0 * val_std

    # 4. Unsupervised Metrics Compilation
    n_windows = len(test_windows)
    
    # Flags for Z-Score
    z_flags = {
        "th_2.0": zscore_window_max > 2.0,
        "th_2.5": zscore_window_max > 2.5,
        "th_3.0": zscore_window_max > 3.0,
    }

    # Flags for EDAD (calibrated thresholds)
    edad_flags = {
        "th_2.0_sigma": edad_window_scores > edad_th_20,
        "th_2.5_sigma": edad_window_scores > edad_th_25,
        "th_3.0_sigma": edad_window_scores > edad_th_30,
    }

    # Model Agreement & Overlap (Jaccard similarity) at 2.5 sigma
    z_default_flag = z_flags["th_2.5"]
    edad_default_flag = edad_flags["th_2.5_sigma"]

    both_flagged = int(np.sum(z_default_flag & edad_default_flag))
    either_flagged = int(np.sum(z_default_flag | edad_default_flag))
    z_only = int(np.sum(z_default_flag & ~edad_default_flag))
    edad_only = int(np.sum(~z_default_flag & edad_default_flag))
    neither_flagged = int(np.sum(~z_default_flag & ~edad_default_flag))

    jaccard_sim = float(both_flagged / either_flagged) if either_flagged > 0 else 0.0
    agreement_rate = float((both_flagged + neither_flagged) / n_windows)

    # Temporal Clustering (Run-Length Analysis)
    def compute_temporal_clusters(flags):
        clusters = []
        curr_len = 0
        for f in flags:
            if f:
                curr_len += 1
            elif curr_len > 0:
                clusters.append(curr_len)
                curr_len = 0
        if curr_len > 0:
            clusters.append(curr_len)
        return {
            "num_bursts": len(clusters),
            "mean_burst_length": float(np.mean(clusters)) if clusters else 0.0,
            "max_burst_length": int(np.max(clusters)) if clusters else 0
        }

    z_clusters = compute_temporal_clusters(z_default_flag)
    edad_clusters = compute_temporal_clusters(edad_default_flag)

    # Top-20 Anomalous Windows per Detector
    top20_z_indices = np.argsort(zscore_window_max)[::-1][:20]
    top20_edad_indices = np.argsort(edad_window_scores)[::-1][:20]

    top20_z_windows = []
    for rank, idx in enumerate(top20_z_indices, 1):
        m = test_meta[idx]
        top20_z_windows.append({
            "rank": rank,
            "window_index": int(idx),
            "node_id": m["node_id"],
            "start_time": m["start_time"],
            "end_time": m["end_time"],
            "zscore_max": float(zscore_window_max[idx]),
            "edad_score": float(edad_window_scores[idx])
        })

    top20_edad_windows = []
    for rank, idx in enumerate(top20_edad_indices, 1):
        m = test_meta[idx]
        top20_edad_windows.append({
            "rank": rank,
            "window_index": int(idx),
            "node_id": m["node_id"],
            "start_time": m["start_time"],
            "end_time": m["end_time"],
            "edad_score": float(edad_window_scores[idx]),
            "zscore_max": float(zscore_window_max[idx])
        })

    # 5. Controlled Sanity Check: Synthetic Perturbation Injection
    print("Executing Controlled Sanity Check: Synthetic Anomaly Injection Benchmark...", flush=True)
    n_injected = int(n_windows * 0.04)  # 4% test windows injected
    rng = np.random.default_rng(seed)
    injected_indices = rng.choice(n_windows, size=n_injected, replace=False)

    synth_windows = test_windows.copy()
    # Inject spike perturbations (+5 standard deviations on random features)
    for idx in injected_indices:
        feat_choice = rng.integers(0, len(scaled_features), size=2)
        synth_windows[idx, rng.integers(3, 9), feat_choice] += 5.0

    # Score injected windows
    synth_edad_scores = []
    with torch.no_grad():
        for i in range(0, len(synth_windows), batch_size):
            batch_np = synth_windows[i : i + batch_size]
            batch_t = torch.tensor(batch_np, dtype=torch.float32, device=device)
            synth_edad_scores.append(edad_model.compute_anomaly_scores(batch_t))
    synth_edad_scores = np.concatenate(synth_edad_scores)

    # Z-score on injected windows
    synth_z_scores = []
    for idx in range(len(synth_windows)):
        # Max absolute value across injected window
        synth_z_scores.append(np.max(np.abs(synth_windows[idx])))
    synth_z_scores = np.array(synth_z_scores)

    # Detection sensitivity on known injected anomalies
    z_detected_synth = np.sum(synth_z_scores[injected_indices] > 2.5) / n_injected
    edad_detected_synth = np.sum(synth_edad_scores[injected_indices] > edad_th_25) / n_injected

    # 6. Plots Generation
    print("Generating evaluation figures...", flush=True)

    # Figure 1: Score Distributions
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Synthetic Anomaly Benchmark — Anomaly Score Distributions (Test Split)", fontsize=13, fontweight="bold")
    
    axes[0].hist(zscore_window_max, bins=60, color="#2563eb", alpha=0.75, edgecolor="black", density=True)
    axes[0].axvline(2.5, color="red", linestyle="--", linewidth=2, label="Threshold 2.5")
    axes[0].set_title("Z-Score Baseline (Max |Z| per window)")
    axes[0].set_xlabel("Max |Z| Score")
    axes[0].set_ylabel("Density")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].hist(edad_window_scores, bins=60, color="#7c3aed", alpha=0.75, edgecolor="black", density=True)
    axes[1].axvline(edad_th_25, color="red", linestyle="--", linewidth=2, label=f"Threshold 2.5σ ({edad_th_25:.2f})")
    axes[1].set_title("EDAD-Inspired Model (Recon + Aux Anomaly Score)")
    axes[1].set_xlabel("Anomaly Score")
    axes[1].set_ylabel("Density")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    dist_fig_path = os.path.join(figures_dir, "score_distributions.png")
    plt.savefig(dist_fig_path, dpi=180)
    plt.close()

    # Figure 2: Model Agreement & Overlap
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["Both Flagged\n(Agreement)", "Z-Score Only", "EDAD Only", "Neither Flagged\n(Normal Agreement)"]
    counts = [both_flagged, z_only, edad_only, neither_flagged]
    colors = ["#16a34a", "#2563eb", "#9333ea", "#64748b"]

    bars = ax.bar(categories, counts, color=colors, edgecolor="black", width=0.55)
    for bar, count in zip(bars, counts):
        pct = count / n_windows * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 150, f"{count:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title(f"Synthetic Anomaly Benchmark — Model Detection Overlap (Test Split)\nJaccard Index: {jaccard_sim:.3f} | Total Windows: {n_windows:,}", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Windows")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    agreement_fig_path = os.path.join(figures_dir, "model_agreement.png")
    plt.savefig(agreement_fig_path, dpi=180)
    plt.close()

    # Figure 3: Per-node time series timeline with flagged anomalies (e.g. SB-001)
    sample_node = "SB-001"
    sample_meta_indices = [i for i, m in enumerate(test_meta) if m["node_id"] == sample_node][:300]
    
    if sample_meta_indices:
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        fig.suptitle(f"Synthetic Anomaly Benchmark — Anomaly Timeline Overlay: Node {sample_node}", fontsize=12, fontweight="bold")

        t_indices = np.arange(len(sample_meta_indices))
        z_sample_scores = zscore_window_max[sample_meta_indices]
        edad_sample_scores = edad_window_scores[sample_meta_indices]

        axes[0].plot(t_indices, z_sample_scores, color="#2563eb", label="Z-Score (Max |Z|)", linewidth=1.2)
        axes[0].axhline(2.5, color="red", linestyle="--", label="Threshold 2.5")
        axes[0].fill_between(t_indices, 0, z_sample_scores, where=(z_sample_scores > 2.5), color="red", alpha=0.3, label="Z-Score Flagged")
        axes[0].set_ylabel("Max |Z|")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right")
        axes[0].set_title("Z-Score Baseline Detections")

        axes[1].plot(t_indices, edad_sample_scores, color="#7c3aed", label="EDAD Anomaly Score", linewidth=1.2)
        axes[1].axhline(edad_th_25, color="red", linestyle="--", label="Threshold 2.5σ")
        axes[1].fill_between(t_indices, 0, edad_sample_scores, where=(edad_sample_scores > edad_th_25), color="purple", alpha=0.3, label="EDAD Flagged")
        axes[1].set_ylabel("EDAD Score")
        axes[1].set_xlabel("Time Step (Sliding Windows)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right")
        axes[1].set_title("EDAD-Inspired Model Detections")

        plt.tight_layout()
        timeline_fig_path = os.path.join(figures_dir, "node_timeline_anomalies.png")
        plt.savefig(timeline_fig_path, dpi=180)
        plt.close()

    # Figure 4: Synthetic Perturbation Injection Response
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Synthetic Perturbation Sanity Benchmark — Controlled Spike Response (Verification Only)", fontsize=11, fontweight="bold")

    axes[0].boxplot([zscore_window_max, synth_z_scores[injected_indices]],
                    tick_labels=["Clean Test Windows", "Injected Spikes (+5σ)"],
                    patch_artist=True, boxprops=dict(facecolor="#93c5fd"))
    axes[0].set_title(f"Z-Score Response\n(Detected Injected: {z_detected_synth*100:.1f}%)")
    axes[0].set_ylabel("Max |Z| Score")
    axes[0].grid(True, alpha=0.3)

    axes[1].boxplot([edad_window_scores, synth_edad_scores[injected_indices]],
                    tick_labels=["Clean Test Windows", "Injected Spikes (+5σ)"],
                    patch_artist=True, boxprops=dict(facecolor="#d8b4fe"))
    axes[1].set_title(f"EDAD Model Response\n(Detected Injected: {edad_detected_synth*100:.1f}%)")
    axes[1].set_ylabel("EDAD Anomaly Score")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    synth_fig_path = os.path.join(figures_dir, "synthetic_injection_response.png")
    plt.savefig(synth_fig_path, dpi=180)
    plt.close()

    # 7. Compile Evaluation Report Dictionary
    eval_results = {
        "benchmark_classification": "Synthetic Anomaly Benchmark",
        "caveat": "Evaluation conducted strictly on synthetic stand-in dataset; metrics reflect algorithmic stability and anomaly sensitivity under unsupervised settings.",
        "test_windows_count": n_windows,
        "computational_cost": {
            "zscore": {
                "inference_time_sec": round(zscore_runtime, 4),
                "windows_per_sec": round(n_windows / max(zscore_runtime, 1e-6), 1),
                "parameter_count": 0,
                "model_size_kb": round(os.path.getsize(zscore_path) / 1024, 2)
            },
            "edad_inspired": {
                "inference_time_sec": round(edad_runtime, 4),
                "windows_per_sec": round(n_windows / max(edad_runtime, 1e-6), 1),
                "parameter_count": checkpoint["total_params"],
                "model_size_kb": round(os.path.getsize(edad_ckpt_path) / 1024, 2)
            }
        },
        "score_distribution_statistics": {
            "zscore_max_abs": {
                "mean": float(np.mean(zscore_window_max)),
                "std": float(np.std(zscore_window_max)),
                "min": float(np.min(zscore_window_max)),
                "p50_median": float(np.median(zscore_window_max)),
                "p95": float(np.percentile(zscore_window_max, 95)),
                "p99": float(np.percentile(zscore_window_max, 99)),
                "max": float(np.max(zscore_window_max))
            },
            "edad_score": {
                "mean": float(np.mean(edad_window_scores)),
                "std": float(np.std(edad_window_scores)),
                "min": float(np.min(edad_window_scores)),
                "p50_median": float(np.median(edad_window_scores)),
                "p95": float(np.percentile(edad_window_scores, 95)),
                "p99": float(np.percentile(edad_window_scores, 99)),
                "max": float(np.max(edad_window_scores))
            }
        },
        "flag_rates_percentage": {
            "zscore": {
                "th_2.0": round(float(np.mean(z_flags["th_2.0"]) * 100), 2),
                "th_2.5": round(float(np.mean(z_flags["th_2.5"]) * 100), 2),
                "th_3.0": round(float(np.mean(z_flags["th_3.0"]) * 100), 2),
            },
            "edad_inspired": {
                "th_2.0_sigma": round(float(np.mean(edad_flags["th_2.0_sigma"]) * 100), 2),
                "th_2.5_sigma": round(float(np.mean(edad_flags["th_2.5_sigma"]) * 100), 2),
                "th_3.0_sigma": round(float(np.mean(edad_flags["th_3.0_sigma"]) * 100), 2),
            }
        },
        "model_agreement": {
            "threshold_compared": "2.5 standard deviations / threshold",
            "both_flagged": both_flagged,
            "zscore_only": z_only,
            "edad_only": edad_only,
            "neither_flagged": neither_flagged,
            "jaccard_similarity": round(jaccard_sim, 4),
            "overall_agreement_rate": round(agreement_rate, 4)
        },
        "temporal_clustering": {
            "zscore": z_clusters,
            "edad_inspired": edad_clusters
        },
        "controlled_synthetic_sanity_benchmark": {
            "injection_description": "Controlled 5-sigma sensor spike injection on 4% isolated test windows (Verification Only)",
            "n_injected": n_injected,
            "zscore_detection_rate": round(float(z_detected_synth * 100), 2),
            "edad_detection_rate": round(float(edad_detected_synth * 100), 2)
        },
        "top_20_anomalous_windows": {
            "zscore": top20_z_windows,
            "edad_inspired": top20_edad_windows
        }
    }

    report_out_path = os.path.join(reports_dir, "unsupervised_evaluation.json")
    with open(report_out_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    print(f"Evaluation completed successfully.")
    print(f"  Report saved to: {report_out_path}")
    print(f"  Figures saved to: {figures_dir}")
    print(f"  Z-Score Flag Rate (th=2.5): {eval_results['flag_rates_percentage']['zscore']['th_2.5']}%")
    print(f"  EDAD Flag Rate (th=2.5s):   {eval_results['flag_rates_percentage']['edad_inspired']['th_2.5_sigma']}%")
    print(f"  Jaccard Agreement:          {eval_results['model_agreement']['jaccard_similarity']}")
    print(f"  Synthetic Spike Detection:  Z-Score={eval_results['controlled_synthetic_sanity_benchmark']['zscore_detection_rate']}%, EDAD={eval_results['controlled_synthetic_sanity_benchmark']['edad_detection_rate']}%")

    return eval_results

if __name__ == "__main__":
    with ExecutionTimer("Step 4: Unsupervised Evaluation"):
        run_evaluation()

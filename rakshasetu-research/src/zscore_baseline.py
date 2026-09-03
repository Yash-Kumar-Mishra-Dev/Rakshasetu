"""
RakshaSetu Anomaly Detection Research Pipeline - Z-Score Statistical Baseline
Synthetic Anomaly Benchmark — SRS §8.1.1 Implementation
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import set_seed, load_config, ExecutionTimer

class ZScoreBaselineDetector:
    """
    Per-node Z-Score Statistical Anomaly Detector.
    Calibrated strictly on training splits per node.
    Supports thresholds: 2.0, 2.5, 3.0
    Supports aggregations: max_|z|, mean_|z|, count_exceeding
    """
    def __init__(self, config: dict):
        self.config = config
        self.env_features = config["schema"]["environmental_features"]
        self.thresholds = config["zscore_baseline"]["thresholds"]
        self.default_threshold = config["zscore_baseline"]["default_threshold"]
        self.aggregation_methods = config["zscore_baseline"]["aggregation_methods"]
        self.node_stats = {}

    def fit(self, train_df: pd.DataFrame, entity_col: str = "node_id"):
        """Computes per-node mean and standard deviation for each environmental feature."""
        self.node_stats = {}
        for node_id, group in train_df.groupby(entity_col):
            node_means = {}
            node_stds = {}
            for feat in self.env_features:
                val = group[feat].values
                m = float(np.mean(val))
                s = float(np.std(val))
                if s < 1e-8:
                    s = 1e-8  # Prevent division by zero
                node_means[feat] = m
                node_stds[feat] = s
            self.node_stats[node_id] = {
                "mean": node_means,
                "std": node_stds
            }
        return self

    def score_sample(self, row: pd.Series) -> dict:
        """Calculates feature-wise and aggregated z-scores for a single reading."""
        node_id = row["node_id"]
        stats = self.node_stats.get(node_id)
        if stats is None:
            raise KeyError(f"Unknown node_id: {node_id}")

        z_scores = {}
        abs_z_list = []
        for feat in self.env_features:
            val = float(row[feat])
            m = stats["mean"][feat]
            s = stats["std"][feat]
            z = (val - m) / s
            z_scores[feat] = z
            abs_z_list.append(abs(z))

        abs_z_arr = np.array(abs_z_list)
        max_abs = float(np.max(abs_z_arr))
        mean_abs = float(np.mean(abs_z_arr))

        scores = {
            "max_abs_z": max_abs,
            "mean_abs_z": mean_abs,
        }

        # Calculate count exceeding for each threshold
        for th in self.thresholds:
            scores[f"count_exceeding_th_{th}"] = int(np.sum(abs_z_arr > th))

        return {
            "feature_z_scores": z_scores,
            "aggregated_scores": scores
        }

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized computation of Z-scores across full dataframe."""
        out_df = df.copy()
        
        # Preallocate score arrays
        max_abs_z_arr = np.zeros(len(df))
        mean_abs_z_arr = np.zeros(len(df))
        flag_arrays = {th: np.zeros(len(df), dtype=bool) for th in self.thresholds}
        count_arrays = {th: np.zeros(len(df), dtype=int) for th in self.thresholds}

        for node_id, idxs in df.groupby("node_id").groups.items():
            stats = self.node_stats[node_id]
            node_subset = df.loc[idxs, self.env_features].values
            means = np.array([stats["mean"][f] for f in self.env_features])
            stds = np.array([stats["std"][f] for f in self.env_features])

            z_mat = (node_subset - means) / stds
            abs_z_mat = np.abs(z_mat)

            max_abs_z = np.max(abs_z_mat, axis=1)
            mean_abs_z = np.mean(abs_z_mat, axis=1)

            max_abs_z_arr[idxs] = max_abs_z
            mean_abs_z_arr[idxs] = mean_abs_z

            for th in self.thresholds:
                flag_arrays[th][idxs] = max_abs_z > th
                count_arrays[th][idxs] = np.sum(abs_z_mat > th, axis=1)

        out_df["zscore_max_abs"] = max_abs_z_arr
        out_df["zscore_mean_abs"] = mean_abs_z_arr
        for th in self.thresholds:
            out_df[f"zscore_flag_th_{th}"] = flag_arrays[th]
            out_df[f"zscore_count_th_{th}"] = count_arrays[th]

        return out_df

    def save(self, file_path: str):
        """Saves model parameters and configuration to JSON."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        payload = {
            "model_type": "ZScoreBaselineDetector",
            "benchmark_classification": "Synthetic Anomaly Benchmark",
            "environmental_features": self.env_features,
            "thresholds": self.thresholds,
            "default_threshold": self.default_threshold,
            "aggregation_methods": self.aggregation_methods,
            "node_stats": self.node_stats
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Z-Score baseline model saved to: {file_path}")

    @classmethod
    def load(cls, file_path: str, config: dict):
        """Loads model parameters from JSON."""
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        detector = cls(config)
        detector.node_stats = payload["node_stats"]
        return detector

def train_zscore_baseline(config_path: str = "config/config.yaml"):
    config = load_config(config_path)
    set_seed(config["pipeline"]["random_seed"])

    train_path = config["paths"]["train_data"]
    model_save_path = config["paths"]["zscore_model_file"]

    print(f"Loading train split for Z-Score fitting: {train_path}")
    train_df = pd.read_csv(train_path)

    detector = ZScoreBaselineDetector(config)
    detector.fit(train_df, entity_col=config["schema"]["entity_column"])
    detector.save(model_save_path)

    # Validate calibration on validation set
    val_path = config["paths"]["val_data"]
    val_df = pd.read_csv(val_path)
    val_scored = detector.predict_dataframe(val_df)
    
    print("Z-Score Calibration Validation Summary:")
    for th in config["zscore_baseline"]["thresholds"]:
        pct_flagged = float(val_scored[f"zscore_flag_th_{th}"].mean() * 100)
        print(f"  Threshold {th:4.1f}: {pct_flagged:6.2f}% flagged in validation split")

    return detector

if __name__ == "__main__":
    with ExecutionTimer("Step 2: Z-Score Statistical Baseline (train + save)"):
        train_zscore_baseline()

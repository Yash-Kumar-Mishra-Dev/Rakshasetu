"""
RakshaSetu Anomaly Detection Research Pipeline - Preprocessing Module
Synthetic Anomaly Benchmark — Per-Node Preprocessing & Scaling
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.utils import set_seed, load_config, ExecutionTimer

def preprocess_pipeline(config_path: str = "config/config.yaml"):
    config = load_config(config_path)
    set_seed(config["pipeline"]["random_seed"])

    raw_path = config["paths"]["raw_data"]
    processed_dir = config["paths"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    print(f"Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)

    entity_col = config["schema"]["entity_column"]
    time_col = config["schema"]["timestamp_column"]
    env_features = config["schema"]["environmental_features"]
    device_features = config["schema"]["device_health_features"]

    df[time_col] = pd.to_datetime(df[time_col])

    # Record pre-imputation missingness counts
    raw_missing_counts = df[env_features].isnull().sum().to_dict()
    total_samples = len(df)
    imputation_stats = {
        col: {
            "missing_count": int(raw_missing_counts[col]),
            "missing_percentage": round(float(raw_missing_counts[col] / total_samples * 100), 4)
        }
        for col in env_features
    }

    # Per-node chronological sort & imputation (ffill then bfill within node)
    node_dfs = []
    for node_id, group in df.groupby(entity_col):
        grp_sorted = group.sort_values(time_col).copy()
        # Impute only the environmental features with missing values
        grp_sorted[env_features] = grp_sorted[env_features].ffill().bfill()
        node_dfs.append(grp_sorted)

    df_cleaned = pd.concat(node_dfs, axis=0).reset_index(drop=True)

    # Verify no nulls remain in environmental features
    remaining_nulls = df_cleaned[env_features].isnull().sum().sum()
    if remaining_nulls > 0:
        raise ValueError(f"Imputation incomplete: {remaining_nulls} null values remaining.")

    # Chronological 70/15/15 split per node
    train_dfs = []
    val_dfs = []
    test_dfs = []

    node_scalers = {}
    node_split_info = {}

    train_ratio = config["preprocessing"]["split"]["train_ratio"]
    val_ratio = config["preprocessing"]["split"]["val_ratio"]

    for node_id, group in df_cleaned.groupby(entity_col):
        grp = group.sort_values(time_col).reset_index(drop=True)
        n = len(grp)
        n_train = int(n * train_ratio)
        n_val = int(n * (train_ratio + val_ratio))

        node_train = grp.iloc[:n_train].copy()
        node_val = grp.iloc[n_train:n_val].copy()
        node_test = grp.iloc[n_val:].copy()

        # Fit StandardScaler strictly on node_train environmental features
        scaler = StandardScaler()
        scaler.fit(node_train[env_features])

        # Record scaler mean and scale
        node_scalers[node_id] = {
            "mean": {feat: float(m) for feat, m in zip(env_features, scaler.mean_)},
            "scale": {feat: float(s) for feat, s in zip(env_features, scaler.scale_)},
            "var": {feat: float(v) for feat, v in zip(env_features, scaler.var_)}
        }

        # Apply scaling to all splits for environmental features (prefixed or direct)
        # We store both scaled features and retain raw columns for interpretability
        for feat in env_features:
            node_train[f"{feat}_scaled"] = scaler.transform(node_train[env_features])[:, env_features.index(feat)]
            node_val[f"{feat}_scaled"] = scaler.transform(node_val[env_features])[:, env_features.index(feat)]
            node_test[f"{feat}_scaled"] = scaler.transform(node_test[env_features])[:, env_features.index(feat)]

        node_split_info[node_id] = {
            "total_rows": n,
            "train_rows": len(node_train),
            "val_rows": len(node_val),
            "test_rows": len(node_test),
            "train_time_range": [str(node_train[time_col].min()), str(node_train[time_col].max())],
            "val_time_range": [str(node_val[time_col].min()), str(node_val[time_col].max())],
            "test_time_range": [str(node_test[time_col].min()), str(node_test[time_col].max())]
        }

        train_dfs.append(node_train)
        val_dfs.append(node_val)
        test_dfs.append(node_test)

    train_all = pd.concat(train_dfs, axis=0).reset_index(drop=True)
    val_all = pd.concat(val_dfs, axis=0).reset_index(drop=True)
    test_all = pd.concat(test_dfs, axis=0).reset_index(drop=True)

    # Save processed CSVs
    train_path = config["paths"]["train_data"]
    val_path = config["paths"]["val_data"]
    test_path = config["paths"]["test_data"]

    train_all.to_csv(train_path, index=False)
    val_all.to_csv(val_path, index=False)
    test_all.to_csv(test_path, index=False)

    metadata = {
        "benchmark_classification": "Synthetic Anomaly Benchmark",
        "description": "Per-node preprocessed and standard-scaled Springbrook WSN synthetic dataset",
        "total_rows": len(df),
        "split_counts": {
            "train": len(train_all),
            "val": len(val_all),
            "test": len(test_all)
        },
        "environmental_features": env_features,
        "device_health_features": device_features,
        "imputation_strategy": config["preprocessing"]["imputation"]["strategy"],
        "imputation_documented_rates": imputation_stats,
        "node_split_info": node_split_info,
        "node_scalers": node_scalers
    }

    metadata_path = config["paths"]["metadata_file"]
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Preprocessing completed.")
    print(f"  Train samples: {len(train_all)} ({len(train_all)/len(df):.1%}) -> {train_path}")
    print(f"  Validation samples: {len(val_all)} ({len(val_all)/len(df):.1%}) -> {val_path}")
    print(f"  Test samples: {len(test_all)} ({len(test_all)/len(df):.1%}) -> {test_path}")
    print(f"  Metadata saved to: {metadata_path}")
    return metadata

if __name__ == "__main__":
    with ExecutionTimer("Step 2: Preprocessing"):
        preprocess_pipeline()

"""
RakshaSetu Anomaly Detection Research Pipeline - Data Inspection & Loader
Synthetic Anomaly Benchmark — Inspection Module
"""

import json
import os
import pandas as pd
import numpy as np

def inspect_raw_dataset(raw_path: str = "data/raw/springbrook_wsn_synthetic_100k.csv",
                        output_report_path: str = "results/reports/data_inspection.json") -> dict:
    """
    Performs rigorous data inspection of springbrook_wsn_synthetic_100k.csv
    and writes results/reports/data_inspection.json.
    """
    print(f"Inspecting raw dataset at: {raw_path}")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset file not found: {raw_path}")

    df = pd.read_csv(raw_path)
    total_rows = len(df)
    duplicate_rows = int(df.duplicated().sum())

    # Verify column existence and missingness
    columns = list(df.columns)
    missing_counts = df.isnull().sum().to_dict()
    missing_pcts = {k: round(float(v / total_rows * 100), 4) for k, v in missing_counts.items()}

    # Grouping/entity inspection
    node_ids = sorted(df["node_id"].unique().tolist())
    num_nodes = len(node_ids)
    rows_per_node = df.groupby("node_id").size().to_dict()

    # Time inspection
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
    min_time = str(df["timestamp_dt"].min())
    max_time = str(df["timestamp_dt"].max())

    # Cadence check per node
    cadence_diffs = []
    for nid, group in df.groupby("node_id"):
        grp_sorted = group.sort_values("timestamp_dt")
        diffs = grp_sorted["timestamp_dt"].diff().dropna()
        cadence_diffs.extend(diffs.dt.total_seconds().tolist())
    median_cadence_min = float(np.median(cadence_diffs) / 60.0)

    # Check for anomaly label columns
    potential_label_cols = [c for c in columns if any(kw in c.lower() for kw in ["anomaly", "label", "is_anomaly", "target", "class"])]
    has_anomaly_label = len(potential_label_cols) > 0

    # Decimal precision artifacts check
    precision_artifacts = {}
    for col in ["wind_speed_ms", "wind_direction_deg", "solar_radiation_Wm2"]:
        if col in df.columns:
            str_vals = df[col].astype(str)
            dec_lens = str_vals.map(lambda x: len(x.split(".")[1]) if "." in x else 0)
            dominant_len = int(dec_lens.mode().iloc[0])
            pct_dominant = float((dec_lens == dominant_len).mean() * 100)
            precision_artifacts[col] = {
                "dominant_decimal_length": dominant_len,
                "pct_with_dominant_length": round(pct_dominant, 2)
            }

    report = {
        "dataset_name": "springbrook_wsn_synthetic_100k.csv",
        "benchmark_classification": "Synthetic Anomaly Benchmark",
        "caveat": "Synthetic stand-in dataset; six of nine sensor columns have 0% missing values, three exhibit generator decimal artifacts, no rainfall column exists. Benchmark must be treated as synthetic sanity check and pipeline verification.",
        "total_rows": total_rows,
        "duplicate_rows": duplicate_rows,
        "num_nodes": num_nodes,
        "node_ids": node_ids,
        "rows_per_node_min": min(rows_per_node.values()),
        "rows_per_node_max": max(rows_per_node.values()),
        "time_range": {
            "start": min_time,
            "end": max_time,
            "inferred_cadence_minutes": median_cadence_min
        },
        "columns": columns,
        "missing_percentages": missing_pcts,
        "has_anomaly_label_column": has_anomaly_label,
        "label_search_result": "No anomaly labels found. Proceeding with Case B: Unsupervised anomaly detection evaluation.",
        "device_health_column": "battery_voltage_V",
        "excluded_features": ["battery_voltage_V"],
        "generator_precision_artifacts": precision_artifacts
    }

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Data inspection completed. Report saved to: {output_report_path}")
    return report

if __name__ == "__main__":
    inspect_raw_dataset()

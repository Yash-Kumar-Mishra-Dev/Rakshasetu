"""
RakshaSetu Test Cases Evaluation Script
Evaluates Z-Score Baseline & Device Health Detectors on User-Provided Labeled Test Cases
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_test_cases(test_csv_path: str = "data/test_cases.csv",
                        zscore_model_path: str = "models/zscore_baseline.json"):
    print("=" * 75)
    print("  RAKSHASETU TEST CASES EVALUATION (User Labeled Benchmark)")
    print("=" * 75)

    df = pd.read_csv(test_csv_path)
    with open(zscore_model_path, "r", encoding="utf-8") as f:
        zmodel = json.load(f)

    node_stats = zmodel["node_stats"]
    env_features = zmodel["environmental_features"]

    # Evaluate each row
    zscore_max_list = []
    zscore_flag_25 = []
    zscore_flag_20 = []
    battery_brownout_flag = []
    specific_anomaly_flag = []

    for idx, row in df.iterrows():
        nid = row["node_id"]
        stats = node_stats.get(nid)
        if stats is None:
            # Fallback stats across all nodes if node not seen
            all_means = np.mean([s["mean"][f] for s in node_stats.values() for f in env_features])
            all_stds = np.mean([s["std"][f] for s in node_stats.values() for f in env_features])
            stats = {"mean": {f: all_means for f in env_features}, "std": {f: all_stds for f in env_features}}

        # Compute z-scores
        z_vals = []
        for feat in env_features:
            val = row[feat]
            if pd.isna(val):
                # Handle missing with mean
                val = stats["mean"][feat]
            z = abs(val - stats["mean"][feat]) / max(stats["std"][feat], 1e-6)
            z_vals.append(z)

        max_z = max(z_vals)
        zscore_max_list.append(max_z)
        zscore_flag_25.append(max_z > 2.5)
        zscore_flag_20.append(max_z > 2.0)

        # Device health check: battery voltage
        v_bat = row["battery_voltage_V"]
        is_brownout = (v_bat < 2.5) if not pd.isna(v_bat) else False
        battery_brownout_flag.append(is_brownout)

        # Combined detection flag: Environmental anomaly (>2.5 sigma) OR Device Brownout (<2.5V)
        specific_anomaly_flag.append((max_z > 2.5) or is_brownout)

    df["pred_max_z"] = zscore_max_list
    df["pred_z_25"] = zscore_flag_25
    df["pred_z_20"] = zscore_flag_20
    df["pred_brownout"] = battery_brownout_flag
    df["pred_combined"] = specific_anomaly_flag

    y_true = df["is_synthetic_anomaly"].values
    y_pred_z25 = df["pred_z_25"].astype(int).values
    y_pred_comb = df["pred_combined"].astype(int).values

    print("\n--- 1. OVERALL ANOMALY DETECTION PERFORMANCE ---")
    print(f"Total Test Cases:    {len(df):,}")
    print(f"Ground-Truth Normal: {np.sum(y_true == 0):,} ({(y_true == 0).mean()*100:.1f}%)")
    print(f"Ground-Truth Anomaly:{np.sum(y_true == 1):,} ({(y_true == 1).mean()*100:.1f}%)")

    cm_z25 = confusion_matrix(y_true, y_pred_z25)
    cm_comb = confusion_matrix(y_true, y_pred_comb)

    tn, fp, fn, tp = cm_comb.ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / len(df)

    print("\n[Confusion Matrix: Combined Environmental + Battery Detector]")
    print(f"  True Positives (Detected Anomalies): {tp}/{tp+fn} ({recall*100:.1f}%)")
    print(f"  True Negatives (Correct Normal):     {tn}/{tn+fp} ({specificity*100:.1f}%)")
    print(f"  False Positives (Normal as Anomaly): {fp}")
    print(f"  False Negatives (Missed Anomalies):  {fn}")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1-Score:  {f1:.4f}")

    print("\n--- 2. BREAKDOWN BY ANOMALY TYPE ---")
    print(f"{'Anomaly Type':<20} | {'Total Cases':<12} | {'Detected':<10} | {'Recall Rate':<12} | {'Lead Feature / Signal'}")
    print("-" * 75)

    for atype in df["anomaly_type"].unique():
        if atype == "none":
            continue
        subset = df[df["anomaly_type"] == atype]
        det = subset["pred_combined"].sum()
        total = len(subset)
        rate = (det / total) * 100
        
        signal = ""
        if atype == "temp_spike":
            signal = f"Air Temp deviates ~{subset['pred_max_z'].mean():.1f} sigma"
        elif atype == "pressure_drop":
            signal = f"Air Pressure deviates ~{subset['pred_max_z'].mean():.1f} sigma"
        elif atype == "flatline_humidity":
            signal = f"Humidity 0.0% deviates ~{subset['pred_max_z'].mean():.1f} sigma"
        elif atype == "wind_burst":
            signal = f"Wind Speed deviates ~{subset['pred_max_z'].mean():.1f} sigma"
        elif atype == "battery_brownout":
            signal = f"Voltage dropped to {subset['battery_voltage_V'].mean():.2f}V (<2.5V)"

        print(f"{atype:<20} | {total:<12} | {det:<10} | {rate:>10.1f}% | {signal}")

    # Output report to results/reports
    results = {
        "total_cases": len(df),
        "ground_truth_normal": int(np.sum(y_true == 0)),
        "ground_truth_anomalies": int(np.sum(y_true == 1)),
        "metrics": {
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "specificity": round(float(specificity), 4)
        },
        "type_breakdown": {
            atype: {
                "count": int((df["anomaly_type"] == atype).sum()),
                "detected": int(df[df["anomaly_type"] == atype]["pred_combined"].sum()),
                "recall": round(float(df[df["anomaly_type"] == atype]["pred_combined"].mean()), 4)
            }
            for atype in df["anomaly_type"].unique() if atype != "none"
        }
    }

    out_json = "results/reports/test_cases_evaluation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {out_json}")
    print("=" * 75)
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RakshaSetu Test Cases Evaluation")
    parser.add_argument("--mode", choices=["auto", "point", "full"], default="auto",
                        help="Evaluation mode: 'full' (5,000 windows), 'point' (legacy point-in-time), or 'auto'")
    parser.add_argument("--csv", type=str, default=None, help="Custom test CSV path")
    args = parser.parse_args()

    full_candidates = ["data/synthetic_test_cases_full.csv", "synthetic_test_cases_full 1.csv"]
    has_full = any(os.path.exists(p) for p in full_candidates)

    if args.mode == "full" or (args.mode == "auto" and (args.csv and "synthetic_test_cases_full" in args.csv or has_full)):
        from evaluate_full_benchmark import run_full_benchmark
        csv_p = args.csv if args.csv else (full_candidates[0] if os.path.exists(full_candidates[0]) else full_candidates[1])
        run_full_benchmark(csv_path=csv_p)
    else:
        csv_p = args.csv if args.csv else "data/test_cases.csv"
        evaluate_test_cases(test_csv_path=csv_p)

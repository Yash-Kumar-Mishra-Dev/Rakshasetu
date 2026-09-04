"""
RakshaSetu Test Data Manager & Verifier
Ensures test cases are verified, cataloged, and ready for benchmark evaluation.
Supports:
  1. Full 5,000-window benchmark dataset (synthetic_test_cases_full.csv)
  2. Single-step point benchmark dataset (test_cases.csv)
"""

import os
import shutil
import pandas as pd

def verify_test_data():
    print("=" * 70)
    print("  RAKSHASETU TEST DATA MANAGER & VERIFIER")
    print("=" * 70)

    # 1. Check Full Windowed Test Cases
    full_candidate_paths = [
        "data/synthetic_test_cases_full.csv",
        "synthetic_test_cases_full 1.csv",
        "data/synthetic_test_cases_full 1.csv"
    ]
    
    found_full = False
    target_full = "data/synthetic_test_cases_full.csv"

    for p in full_candidate_paths:
        if os.path.exists(p):
            if p != target_full and not os.path.exists(target_full):
                shutil.copy2(p, target_full)
                print(f"[Copied] {p} -> {target_full}")
            found_full = True
            active_full = target_full if os.path.exists(target_full) else p
            df_full = pd.read_csv(active_full)
            print(f"\n[Verified] Full Windowed Benchmark Dataset: {active_full}")
            print(f"  Total Windows:       {len(df_full):,}")
            print(f"  Feature Dimensions:  {df_full.shape[1]} columns (12 timesteps x 8 features)")
            print(f"  Normal Cases:        {(df_full['label'] == 0).sum():,} ({(df_full['label'] == 0).mean()*100:.1f}%)")
            print(f"  Anomaly Cases:       {(df_full['label'] == 1).sum():,} ({(df_full['label'] == 1).mean()*100:.1f}%)")
            print(f"  Anomaly Types:       {dict(df_full['injection_type'].value_counts())}")
            print(f"  Unique Nodes:        {df_full['node_id'].nunique()} nodes ({list(df_full['node_id'].unique()[:5])}...)")
            break

    if not found_full:
        print("\n[Notice] Full windowed benchmark dataset not found.")

    # 2. Check Legacy Point-in-Time Test Cases
    point_path = "data/test_cases.csv"
    if os.path.exists(point_path):
        df_pt = pd.read_csv(point_path)
        print(f"\n[Verified] Point-in-Time Test Cases: {point_path}")
        print(f"  Total Records:       {len(df_pt):,}")
        print(f"  Normal Cases:        {(df_pt['is_synthetic_anomaly'] == 0).sum():,}")
        print(f"  Anomaly Cases:       {(df_pt['is_synthetic_anomaly'] == 1).sum():,}")
        print(f"  Anomaly Types:       {dict(df_pt['anomaly_type'].value_counts())}")
    else:
        print(f"\n[Notice] Point-in-time dataset ({point_path}) not found.")

    print("\n" + "=" * 70)
    print("  Verification complete. Ready for evaluation via evaluate_full_benchmark.py")
    print("=" * 70)

if __name__ == "__main__":
    verify_test_data()

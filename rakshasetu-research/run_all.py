"""
RakshaSetu Anomaly Detection Research Pipeline - Master Execution Runner
Synthetic Anomaly Benchmark — End-to-End Orchestrator
"""

import os
import sys
import time

# Ensure rakshasetu-research root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils import ExecutionTimer
from src.data_loader import inspect_raw_dataset
from src.preprocessing import preprocess_pipeline
from src.zscore_baseline import train_zscore_baseline
from src.train import train_edad_pipeline
from src.train_aurora import train_aurora_pipeline
from src.evaluate import run_evaluation
from src.evaluate_aurora import evaluate_aurora_pipeline
from src.generate_report import generate_comparison_report

def main():
    print("=" * 80)
    print("  RAKSHA-SETU AI/ML RESEARCH PIPELINE (ANOMALY DETECTION + AURORA FORECASTING)")
    print("  Scope: springbrook_wsn_synthetic_100k.csv (Synthetic Anomaly & Forecasting Benchmark)")
    print("=" * 80)

    t_master_start = time.perf_counter()

    with ExecutionTimer("Step 0: Raw Dataset Inspection & Schema Verification"):
        inspect_raw_dataset()

    with ExecutionTimer("Step 1: Per-Node Preprocessing & Chronological Splitting"):
        preprocess_pipeline()

    with ExecutionTimer("Step 2: Z-Score Statistical Baseline Fitting & Calibration"):
        train_zscore_baseline()

    with ExecutionTimer("Step 3: EDAD Anomaly Detection Model Training & Checkpointing"):
        train_edad_pipeline()

    with ExecutionTimer("Step 4: Aurora Condition Forecasting Model Training & Checkpointing"):
        train_aurora_pipeline()

    with ExecutionTimer("Step 5: EDAD Anomaly Detection Evaluation & Sanity Check"):
        run_evaluation()

    with ExecutionTimer("Step 6: Aurora Forecasting Evaluation & Baselines"):
        evaluate_aurora_pipeline()

    with ExecutionTimer("Step 7: Final Anomaly Research Report Generation"):
        generate_comparison_report()

    total_time = time.perf_counter() - t_master_start
    print("=" * 80)
    print(f"  ALL PIPELINE STAGES COMPLETED IN {total_time:.2f}s")
    print("  Anomaly Report:   results/reports/final_model_comparison.md")
    print("  Aurora Report:    results/reports/aurora_forecast_report.md")
    print("  Figures:          results/figures/")
    print("=" * 80)

if __name__ == "__main__":
    main()

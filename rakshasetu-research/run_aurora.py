"""
RakshaSetu Condition Prediction & Forecasting Pipeline - Master Aurora Runner
Executes Step 1 (Train Aurora) and Step 2 (Evaluate & Benchmark Aurora)
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train_aurora import train_aurora_pipeline
from src.evaluate_aurora import evaluate_aurora_pipeline
from src.utils import ExecutionTimer

def main():
    print("*" * 80)
    print("  RAKSHASETU AURORA CONDITION FORECASTING MODEL PIPELINE")
    print("  SRS Section 8 AI/ML Core: Environmental Condition Forecaster")
    print("*" * 80)

    # 1. Train Aurora
    with ExecutionTimer("Step 1: Train Aurora Forecaster"):
        train_aurora_pipeline()

    # 2. Evaluate Aurora
    with ExecutionTimer("Step 2: Evaluate Aurora Benchmarks"):
        evaluate_aurora_pipeline()

    print("\n" + "=" * 80)
    print("  AURORA PIPELINE EXECUTION COMPLETE!")
    print("  Model Checkpoint: models/aurora_best.pt")
    print("  Report:           results/reports/aurora_forecast_report.md")
    print("  Figures:          results/figures/aurora_forecast_trajectories.png")
    print("                    results/figures/aurora_horizon_error.png")
    print("=" * 80)

if __name__ == "__main__":
    main()

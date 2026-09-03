# RakshaSetu Anomaly Detection Research Pipeline
### Synthetic Anomaly Benchmark (`springbrook_wsn_synthetic_100k.csv`)
**Implementation of SRS v2.0 Section 8.1 (Anomaly Detection: V1 Statistical Baseline + V2 Deep Learning Model)**

> **IMPORTANT BENCHMARK CAVEAT**  
> This dataset is a synthetic stand-in (`springbrook_wsn_synthetic_100k.csv`), confirmed by inspection: 6 of 9 sensor columns have 0% missing values across a 2-year multi-node deployment, 3 columns show generator decimal artifacts, and no rainfall column is present. **All results are labeled as Synthetic Anomaly Benchmarks** to establish pipeline mechanics and baseline comparisons. This run does **not** validate real-world detection performance on actual field sensors.

---

## Project Structure

```
rakshasetu-research/
├── config/
│   └── config.yaml                     # Pipeline hyperparameters, paths, and feature specs
├── data/
│   ├── raw/
│   │   └── springbrook_wsn_synthetic_100k.csv  # 100k row synthetic benchmark dataset
│   └── processed/
│       ├── train.csv                   # Per-node chronological 70% split
│       ├── validation.csv              # Per-node chronological 15% split
│       ├── test.csv                    # Per-node chronological 15% split
│       └── preprocessing_metadata.json # Imputation logs and per-node scaler parameters
├── models/
│   ├── zscore_baseline.json            # SRS §8.1.1 per-node statistical baseline parameters
│   └── edad_best.pt                    # Best checkpoint for EDAD-inspired PyTorch model
├── results/
│   ├── figures/
│   │   ├── score_distributions.png     # Anomaly score distribution histograms
│   │   ├── model_agreement.png         # Venn/overlap detection agreement
│   │   ├── node_timeline_anomalies.png # Time-series overlay with flagged anomalies
│   │   └── synthetic_injection_response.png # Controlled perturbation benchmark
│   └── reports/
│       ├── data_inspection.json        # Schema verification & artifact inspection
│       ├── edad_training_log.json      # Training loss curves & early stopping log
│       ├── unsupervised_evaluation.json# Full unsupervised metrics & top-20 windows
│       └── final_model_comparison.md   # Final comprehensive research report
├── src/
│   ├── data_loader.py                  # Dataset inspection & schema verification
│   ├── preprocessing.py                # Per-node imputation, 70/15/15 chronological split
│   ├── feature_engineering.py          # Per-node sliding windowing (eta=12, stride=1)
│   ├── zscore_baseline.py              # SRS §8.1.1 Statistical baseline detector
│   ├── edad_model.py                   # EDAD-inspired Encode-then-Decompose architecture
│   ├── train.py                        # PyTorch training loop with stability guards
│   ├── train_edad.py                   # Step 3 runner entrypoint
│   ├── evaluate.py                     # Unsupervised evaluation & sanity check
│   ├── generate_report.py              # Step 5 report generator
│   └── utils.py                        # Reproducibility, seed=42, timing, device detection
├── requirements.txt
├── run_all.py                          # End-to-end master orchestrator
└── README.md
```

---

## Execution Order

You can run the entire pipeline with one command:
```bash
python run_all.py
```

Or execute stage by stage:
```bash
# 1. Inspect raw dataset
python src/data_loader.py

# 2. Preprocess raw data -> train/val/test splits
python src/preprocess.py

# 3. Train & save Z-Score statistical baseline
python src/zscore_baseline.py

# 4. Train EDAD-inspired model with checkpointing
python src/train_edad.py

# 5. Run unsupervised evaluation & sanity check
python src/evaluate.py

# 6. Generate final markdown comparison report
python src/generate_report.py
```

---

## Key Technical Guards
- **No Cross-Node Leakage**: Windowing and train/val/test splits are conducted strictly *per node*. A node's future is never leaked into its training split.
- **Mutual Information Guard**: Replaced raw unbounded mutual-information loss with a squared-magnitude penalty `(mi**2).mean()` to guarantee numerical stability and prevent $-\infty$ divergence.
- **Unsupervised Evaluation (Case B)**: No fabricated labels; metrics focus on score distributions, flag rates across thresholds, model agreement (Jaccard similarity), and run-length burst analysis.

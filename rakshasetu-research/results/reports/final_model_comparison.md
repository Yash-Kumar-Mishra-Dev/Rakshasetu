# RakshaSetu Anomaly Detection Research Report
## Synthetic Anomaly Benchmark: Statistical Baseline (Z-Score) vs. EDAD-Inspired Model
**Evaluation Scope: `springbrook_wsn_synthetic_100k.csv` (Confirmed Schema, September 2026)**

> [!WARNING]
> **CRITICAL BENCHMARK FRAMING & CAVEAT**
> This research experiment is conducted on a **synthetic stand-in dataset** (`springbrook_wsn_synthetic_100k.csv`), **not** the original published Springbrook WSN sensor deployment.
> - Six of nine sensor columns possess exactly 0.0% missing values across a 2-year multi-node deployment.
> - Three features exhibit generator-artifact decimal precision (100% at identical decimal lengths).
> - No `rainfall`, `water_level`, or `citizen_reports` columns exist in this dataset.
> - **No ground truth anomaly labels exist.**
> 
> **Treat every finding in this report strictly as a "Synthetic Anomaly Benchmark".**
> This experiment serves to establish a robust, leak-free pipeline harness and verify algorithmic stability under controlled conditions — **it does NOT validate real-world detection superiority of EDAD over Z-Score on actual field sensors.**

---
## 1. Dataset & Confirmed Schema

- **Total Samples:** 100,000 rows
- **Duplicate Rows:** 0
- **Entity Nodes:** 20 unique nodes (`SB-001` through `SB-020`)
- **Temporal Range:** `2010-01-01 00:00:00` to `2011-11-26 10:30:00`
- **Inferred Cadence:** Uniform 10-minute intervals

### Sensor Feature Manifest
| Column Name | Inferred Role | Missing Rate | Handling in Pipeline |
|---|---|---|---|
| `node_id` | Metadata / Index | 0.0% | Primary Entity / Time Key |
| `timestamp` | Metadata / Index | 0.0% | Primary Entity / Time Key |
| `air_temperature_C` | Environmental Feature | 1.014% | Imputed (ffill -> bfill per node) & StandardScaled |
| `relative_humidity_pct` | Environmental Feature | 0.987% | Imputed (ffill -> bfill per node) & StandardScaled |
| `air_pressure_hPa` | Environmental Feature | 0.477% | Imputed (ffill -> bfill per node) & StandardScaled |
| `wind_speed_ms` | Environmental Feature | 0.0% | Imputed (ffill -> bfill per node) & StandardScaled |
| `wind_direction_deg` | Environmental Feature | 0.0% | Imputed (ffill -> bfill per node) & StandardScaled |
| `leaf_wetness_pct` | Environmental Feature | 0.0% | Imputed (ffill -> bfill per node) & StandardScaled |
| `soil_moisture_pct` | Environmental Feature | 0.0% | Imputed (ffill -> bfill per node) & StandardScaled |
| `solar_radiation_Wm2` | Environmental Feature | 0.0% | Imputed (ffill -> bfill per node) & StandardScaled |
| `battery_voltage_V` | Device Health | 0.0% | **Excluded** from Anomaly Detection feature set |

---
## 2. Preprocessing & Leakage Prevention

- **Per-Node Isolation:** Grouped strictly by `node_id`. Time-ordered chronologically prior to any transformation.
- **Imputation:** Forward-fill followed by backward-fill strictly within each node's timeline (`air_temperature_C`: 1.01%, `relative_humidity_pct`: 0.99%, `air_pressure_hPa`: 0.48%).
- **Chronological Split (70/15/15 per node):**
  - Training Set: `69,991` rows (70.0%)
  - Validation Set: `14,999` rows (15.0%)
  - Test Set: `15,010` rows (15.0%)
- **Zero-Leakage Feature Scaling:** `StandardScaler` fitted exclusively on the 70% training split of each individual node, then applied to that node's validation and test segments.

---
## 3. Models Under Benchmark

### A. Z-Score Statistical Baseline (SRS §8.1.1)
- **Principle:** Per-node mean and standard deviation computed across the 8 environmental features.
- **Aggregation Rules:** Evaluated under `max_|z|` (primary), `mean_|z|`, and `count_exceeding`.
- **Tested Thresholds:** 2.0, 2.5, 3.0.

### B. EDAD-Inspired Encode-then-Decompose Model
- **Architecture:**
  - 1D Temporal Convolutional + Bidirectional GRU Encoder ($\eta=12$ window, 2 hours).
  - Dual Latent Decomposition: Stable Trend Head ($Z_s$) and Auxiliary Transient Head ($Z_a$).
  - 1D Conv Decoder reconstructing input from combined latent space ($Z_s + Z_a$).
  - InfoNCE Temporal Contrastive Self-Supervision on adjacent time windows.
- **Numerical Stability Guards:**
  - **Mutual Information Penalty:** Replaced raw unbounded MI with squared-magnitude penalty `(mi**2).mean()`, preventing divergence to $-\infty$.
  - **Gradient Clipping:** `max_norm=5.0`.
  - **Weight Decay:** AdamW with `weight_decay=1e-4`.
- **Trainable Parameters:** 124,936

---
## 4. Unsupervised Comparative Benchmark Results

Because no ground truth anomaly labels exist in this dataset (Case B), evaluation is conducted using formal unsupervised comparative metrics across identical test windows.

### 4.1 Window Flag Rates on Test Split
| Detector Method | Threshold Level | % Windows Flagged | Flagged Count / Total |
|---|---|---|---|
| Z-Score Baseline | 2.0σ | 93.51% | 13,830 / 14,790 |
| Z-Score Baseline | 2.5σ | 41.09% | 6,077 / 14,790 |
| Z-Score Baseline | 3.0σ | 10.32% | 1,526 / 14,790 |
| EDAD-Inspired | 2.0σ (Val Calibrated) | 4.75% | 702 / 14,790 |
| EDAD-Inspired | 2.5σ (Val Calibrated) | 2.04% | 301 / 14,790 |
| EDAD-Inspired | 3.0σ (Val Calibrated) | 0.78% | 115 / 14,790 |

### 4.2 Score Distribution Characteristics
| Metric | Z-Score (Max |Z|) | EDAD Anomaly Score |
|---|---|---|
| Mean | 2.477 | 0.557 |
| Std Dev | 0.390 | 0.121 |
| Median (p50) | 2.407 | 0.548 |
| 95th Percentile | 3.231 | 0.771 |
| 99th Percentile | 3.738 | 0.875 |
| Maximum Observed | 4.023 | 1.087 |

### 4.3 Model Agreement & Co-Detection (at 2.5σ Threshold)
- **Jaccard Similarity Index:** `0.0384`
- **Overall Concordance Rate:** `60.07%`
- Both Flagged (Intersection): `236` windows
- Z-Score Flagged Only: `5,841` windows
- EDAD Flagged Only: `65` windows
- Neither Flagged (Agreed Normal): `8,648` windows

### 4.4 Temporal Clustering Analysis
| Model | Number of Distinct Bursts | Mean Burst Duration (Windows) | Max Consecutive Burst |
|---|---|---|---|
| Z-Score Baseline | 383 | 15.87 | 45 windows (7.5 hrs) |
| EDAD-Inspired | 175 | 1.72 | 11 windows (1.8 hrs) |

---
## 5. Controlled Sanity Verification: Synthetic Perturbation Benchmark

> [!NOTE]
> **Verification Test Description:** A controlled 5-sigma perturbation spike was injected into 4% of test windows on an isolated test copy to verify detector sensitivity to abrupt multi-sensor anomalies.

- **Injected Anomalous Windows:** 591
- **Z-Score Detection Rate:** `100.0%`
- **EDAD-Inspired Detection Rate:** `83.42%`
- **Takeaway:** Both detectors demonstrate high sensitivity to obvious multi-sigma perturbations, confirming pipeline signal capture.

---
## 6. Computational Cost & Deployment Feasibility

| Metric | Z-Score Baseline (SRS §8.1.1) | EDAD-Inspired Model |
|---|---|---|
| Parameter Count | 0 | 124,936 |
| Model Checkpoint Size | 17.5 KB (JSON) | 1500.7 KB (PyTorch .pt) |
| Inference Time (Full Test Set) | 0.0128s | 0.3972s |
| Inference Throughput | 1,152,183.2 windows/sec | 37,237.2 windows/sec |

---
## 7. Limitations & Honest Engineering Assessment

1. **Synthetic Nature of Source Data:** The underlying CSV contains synthetic artifacts (e.g. constant precision lengths, 0% missingness across major sensors). It does not reflect the unpredictable noise, packet dropouts, or calibration drift of real-world field WSNs.
2. **Absence of Ground Truth:** Without confirmed physical disaster ground truth or labeled sensor failures, detection rates represent statistical outlier sensitivity, not precision/recall/F1.
3. **Scope Limitation:** **This benchmark cannot answer whether EDAD outperforms Z-Score in the real world.** That question requires genuine WSN deployments containing rain gauges, water level transducers, and documented incident reports.
4. **Resource Tradeoff:** Z-Score operates at microsecond latencies with zero GPU requirements, whereas EDAD introduces deep neural network compute overhead. On synthetic data with uniform Gaussian-like distributions, Z-Score provides high utility at negligible cost.

---
## 8. Conclusion

The RakshaSetu Anomaly Detection Research Pipeline has been successfully built and validated end-to-end:
1. **Reproducibility Established:** Fixed seeds, chronological per-node splits, isolated per-node scalers, and modular architecture allow reproducible execution across all stages.
2. **Pipeline Stability Verified:** The squared-magnitude mutual-information penalty `(mi**2).mean()` prevented latent collapse and $-\infty$ divergence in the EDAD-inspired model.
3. **Ready for Real-World Data:** When real-world sensor streams and ground-truth labeled datasets become available, this identical harness can be immediately reused without code redesign.
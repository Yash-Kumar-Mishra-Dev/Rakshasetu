"""
RakshaSetu Anomaly Detection Research Pipeline - Feature Engineering & Windowing
Synthetic Anomaly Benchmark — Per-Node Sliding Windows (never crosses node boundaries)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class TimeSeriesWindowDataset(Dataset):
    """
    PyTorch Dataset for time series sliding windows.
    Constructed strictly PER NODE: never windows across node boundaries.
    Window size eta=12 (2 hours at 10-minute cadence).
    """
    def __init__(self, windows: np.ndarray, meta: list, return_pairs: bool = True):
        """
        windows: (N, window_size, num_features) float32 numpy array
        meta: list of dicts with window metadata (node_id, start_time, end_time)
        return_pairs: if True, provides temporal contrastive positive pairs
        """
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.meta = meta
        self.return_pairs = return_pairs

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx]
        if not self.return_pairs:
            return x, idx

        # Temporal contrastive positive pair: adjacent window from same node if available
        meta_curr = self.meta[idx]
        node_id = meta_curr["node_id"]

        # If next window belongs to the same node, use it as temporal adjacent view (positive pair)
        if idx + 1 < len(self.windows) and self.meta[idx + 1]["node_id"] == node_id:
            x_pos = self.windows[idx + 1]
        elif idx - 1 >= 0 and self.meta[idx - 1]["node_id"] == node_id:
            x_pos = self.windows[idx - 1]
        else:
            # Fallback: self with slight jitter
            x_pos = x + torch.randn_like(x) * 0.01

        return x, x_pos, idx

def create_per_node_windows(df: pd.DataFrame,
                            feature_cols: list,
                            window_size: int = 12,
                            stride: int = 1,
                            entity_col: str = "node_id",
                            time_col: str = "timestamp"):
    """
    Slices dataframe into windows strictly per node.
    Returns:
      windows: np.ndarray of shape (total_windows, window_size, num_features)
      metadata: list of dicts with entity, time range, and original indices
    """
    all_windows = []
    all_meta = []

    for node_id, group in df.groupby(entity_col):
        grp_sorted = group.sort_values(time_col).reset_index(drop=True)
        feat_matrix = grp_sorted[feature_cols].values
        times = grp_sorted[time_col].values
        n_samples = len(grp_sorted)

        if n_samples < window_size:
            continue

        for i in range(0, n_samples - window_size + 1, stride):
            window_slice = feat_matrix[i : i + window_size]
            all_windows.append(window_slice)
            all_meta.append({
                "node_id": node_id,
                "start_idx": i,
                "end_idx": i + window_size - 1,
                "start_time": str(times[i]),
                "end_time": str(times[i + window_size - 1])
            })

    if len(all_windows) == 0:
        windows_arr = np.empty((0, window_size, len(feature_cols)), dtype=np.float32)
    else:
        windows_arr = np.array(all_windows, dtype=np.float32)

    return windows_arr, all_meta


class TimeSeriesForecastingDataset(Dataset):
    """
    PyTorch Dataset for multi-step time series forecasting.
    Yields:
      X: (input_window, num_features) past telemetry
      Y: (horizon, num_features) future ground-truth conditions
      idx: sample index
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, meta: list):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.meta = meta

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], idx


def create_per_node_forecasting_windows(df: pd.DataFrame,
                                        feature_cols: list,
                                        input_window: int = 12,
                                        horizon: int = 6,
                                        stride: int = 1,
                                        entity_col: str = "node_id",
                                        time_col: str = "timestamp"):
    """
    Slices dataframe into past sequence (X) and future forecast target (Y) strictly per node.
    Guarantees no window crosses node boundaries or leaks future steps.
    
    Returns:
      X: np.ndarray of shape (N, input_window, num_features)
      Y: np.ndarray of shape (N, horizon, num_features)
      meta: list of metadata dicts
    """
    all_x = []
    all_y = []
    all_meta = []

    for node_id, group in df.groupby(entity_col):
        grp_sorted = group.sort_values(time_col).reset_index(drop=True)
        feat_matrix = grp_sorted[feature_cols].values
        times = grp_sorted[time_col].values
        n_samples = len(grp_sorted)

        total_req = input_window + horizon
        if n_samples < total_req:
            continue

        for i in range(0, n_samples - total_req + 1, stride):
            x_slice = feat_matrix[i : i + input_window]
            y_slice = feat_matrix[i + input_window : i + total_req]
            all_x.append(x_slice)
            all_y.append(y_slice)
            all_meta.append({
                "node_id": node_id,
                "input_start": str(times[i]),
                "input_end": str(times[i + input_window - 1]),
                "forecast_start": str(times[i + input_window]),
                "forecast_end": str(times[i + total_req - 1])
            })

    if len(all_x) == 0:
        X_arr = np.empty((0, input_window, len(feature_cols)), dtype=np.float32)
        Y_arr = np.empty((0, horizon, len(feature_cols)), dtype=np.float32)
    else:
        X_arr = np.array(all_x, dtype=np.float32)
        Y_arr = np.array(all_y, dtype=np.float32)

    return X_arr, Y_arr, all_meta


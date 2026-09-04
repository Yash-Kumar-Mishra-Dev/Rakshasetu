"""
RakshaSetu Condition Prediction & Forecasting Pipeline - Aurora Model
SRS Section 8 AI/ML Core: Environmental Condition Forecaster

Includes:
  1. PersistenceForecaster: Meteorological persistence baseline (last known value)
  2. RidgeAutoregressiveForecaster: Multi-output linear autoregressive baseline
  3. AuroraForecaster: Deep Seq2Seq Temporal 1D-CNN + Bidirectional GRU with Temporal Attention
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge


class PersistenceForecaster:
    """
    Standard meteorological persistence baseline.
    Assumes future values remain equal to the most recent observation:
    Y_hat[t+k] = X[t] for all k in 1..horizon.
    """
    def __init__(self, horizon: int = 6):
        self.horizon = horizon

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        X: (N, input_window, num_features)
        Returns: (N, horizon, num_features)
        """
        last_val = X[:, -1:, :]  # (N, 1, D)
        return np.repeat(last_val, self.horizon, axis=1)


class RidgeAutoregressiveForecaster:
    """
    Multi-output linear autoregressive baseline.
    Flattens past window into feature vector and predicts all future horizon steps simultaneously.
    """
    def __init__(self, alpha: float = 1.0, horizon: int = 6, input_window: int = 12, num_features: int = 8):
        self.alpha = alpha
        self.horizon = horizon
        self.input_window = input_window
        self.num_features = num_features
        self.model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        X: (N, input_window, num_features)
        Y: (N, horizon, num_features)
        """
        N = X.shape[0]
        X_flat = X.reshape(N, -1)
        Y_flat = Y.reshape(N, -1)
        self.model.fit(X_flat, Y_flat)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        X_flat = X.reshape(N, -1)
        Y_pred_flat = self.model.predict(X_flat)
        return Y_pred_flat.reshape(N, self.horizon, self.num_features)


class TemporalAttention(nn.Module):
    """
    Computes attention weights over encoder hidden states.
    Produces a context vector focusing on the most informative time steps.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, encoder_outputs: torch.Tensor):
        # encoder_outputs: (B, T, hidden_dim)
        scores = self.attn(encoder_outputs)  # (B, T, 1)
        weights = F.softmax(scores, dim=1)   # (B, T, 1)
        context = torch.sum(weights * encoder_outputs, dim=1)  # (B, hidden_dim)
        return context, weights


class AuroraForecaster(nn.Module):
    """
    Aurora Condition Forecasting Neural Network (Seq2Seq Temporal 1D-CNN + BiGRU + Attention).
    
    Architecture:
      1. 1D Temporal Convolution: Extracts multi-channel micro-trends and gradient dynamics.
      2. Bidirectional GRU: Captures forward and reverse temporal context across the input window.
      3. Temporal Attention: Learns dynamic weighting over historical timesteps.
      4. Multi-Horizon Projection Decoder: Maps combined contextual representations to future
         condition trajectories across all environmental features simultaneously.
    """
    def __init__(self,
                 input_dim: int = 8,
                 input_window: int = 12,
                 forecast_horizon: int = 6,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.hidden_dim = hidden_dim

        # 1. 1D Temporal Feature Extractor
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.gelu = nn.GELU()

        # 2. Bidirectional GRU Encoder
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        gru_out_dim = hidden_dim * 2

        # 3. Temporal Attention Pooling
        self.attention = TemporalAttention(gru_out_dim)

        # 4. Multi-Step Condition Projection Head
        total_repr_dim = gru_out_dim * 2  # context + last hidden state
        self.decoder = nn.Sequential(
            nn.Linear(total_repr_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, forecast_horizon * input_dim)
        )

        # 5. Residual Shortcut from last observed step
        self.residual_proj = nn.Linear(input_dim, forecast_horizon * input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, input_window, input_dim)
        Returns:
          y_pred: (B, forecast_horizon, input_dim)
        """
        B, T, D = x.shape

        # 1D Conv feature extraction: expects (B, D, T)
        x_perm = x.permute(0, 2, 1)
        h_conv = self.gelu(self.conv1(x_perm))
        h_conv = self.norm1(self.gelu(self.conv2(h_conv)))
        h_conv = h_conv.permute(0, 2, 1)  # (B, T, hidden_dim)

        # BiGRU encoding
        gru_out, _ = self.gru(h_conv)     # (B, T, hidden_dim * 2)

        # Attention pooling
        context, _ = self.attention(gru_out)  # (B, hidden_dim * 2)
        last_step = gru_out[:, -1, :]         # (B, hidden_dim * 2)

        # Combine global context and recency state
        combined = torch.cat([context, last_step], dim=1)  # (B, hidden_dim * 4)

        # Forecast projection
        delta = self.decoder(combined)  # (B, forecast_horizon * input_dim)

        # Residual shortcut from current observation
        last_obs = x[:, -1, :]  # (B, D)
        shortcut = self.residual_proj(last_obs)

        out = (delta + shortcut).view(B, self.forecast_horizon, self.input_dim)
        return out

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Smooth L1 (Huber) loss between forecasted trajectory and ground-truth trajectory.
        """
        return F.smooth_l1_loss(y_pred, y_true, beta=1.0)

    @torch.no_grad()
    def predict_numpy(self, X: np.ndarray, device: str = "cpu", batch_size: int = 256) -> np.ndarray:
        """
        Batched prediction for numpy inputs.
        """
        self.eval()
        N = len(X)
        preds = []
        for i in range(0, N, batch_size):
            batch = torch.from_numpy(X[i:i + batch_size]).float().to(device)
            p = self(batch)
            preds.append(p.cpu().numpy())
        return np.concatenate(preds, axis=0)

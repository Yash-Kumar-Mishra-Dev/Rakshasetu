"""
RakshaSetu Anomaly Detection Research Pipeline - EDAD-Inspired Model
Synthetic Anomaly Benchmark — Encode-then-Decompose Architecture with Contrastive Supervision
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class EDADModel(nn.Module):
    """
    EDAD-inspired Encode-then-Decompose model for multi-sensor time series anomaly detection.
    
    Architecture:
      1. Temporal 1D Convolutional / GRU Encoder: X (B, T, D) -> Latent representation H
      2. Dual Decomposition Heads:
         - Stable/Trend component Z_s
         - Auxiliary/Transient component Z_a
      3. Reconstruction Decoder: (Z_s + Z_a) -> X_hat (B, T, D)
      4. Contrastive Projection Head: Z_s -> normalized projection for InfoNCE temporal loss
      5. Decomposition Regularization: Squared mutual-information / cosine penalty (mi**2).mean()
         to prevent divergence to -inf.
    """
    def __init__(self,
                 input_dim: int = 8,
                 window_size: int = 12,
                 hidden_dim: int = 64,
                 latent_dim: int = 32,
                 temperature: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.temperature = temperature

        # 1. Temporal Encoder (1D Convolutions + LayerNorm + GRU)
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.enc_fc = nn.Linear(hidden_dim * 2, hidden_dim)

        # 2. Decomposition Heads
        # Stable / Regular pattern representation
        self.head_stable = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # Auxiliary / Transient perturbation representation
        self.head_auxiliary = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # 3. Temporal Contrastive Projector (for InfoNCE)
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        # 4. Reconstruction Decoder
        self.dec_fc = nn.Linear(latent_dim, hidden_dim * window_size)
        self.dec_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=1)
        )

    def encode(self, x: torch.Tensor):
        """
        x: (B, T, D)
        returns: h (B, hidden_dim)
        """
        # (B, T, D) -> (B, D, T) for Conv1d
        x_trans = x.transpose(1, 2)
        h_conv = F.gelu(self.conv1(x_trans))
        h_conv = F.gelu(self.conv2(h_conv))

        # (B, hidden_dim, T) -> (B, T, hidden_dim) for GRU
        h_seq, _ = self.gru(h_conv.transpose(1, 2))
        
        # Mean pool across time window
        h_pooled = h_seq.mean(dim=1)
        h = F.gelu(self.enc_fc(h_pooled))
        return h

    def decompose(self, h: torch.Tensor):
        """
        Decomposes latent representation into stable (trend) and auxiliary (transient) components.
        """
        z_s = self.head_stable(h)
        z_a = self.head_auxiliary(h)
        return z_s, z_a

    def decode(self, z: torch.Tensor):
        """
        Reconstructs window from latent components.
        z: (B, latent_dim)
        returns: x_hat (B, T, D)
        """
        B = z.shape[0]
        h_dec = self.dec_fc(z).view(B, -1, self.window_size)  # (B, hidden_dim, T)
        x_hat_trans = self.dec_conv(h_dec)                   # (B, D, T)
        x_hat = x_hat_trans.transpose(1, 2)                  # (B, T, D)
        return x_hat

    def forward(self, x: torch.Tensor):
        """Full forward pass."""
        h = self.encode(x)
        z_s, z_a = self.decompose(h)
        z_comb = z_s + z_a
        x_hat = self.decode(z_comb)
        proj_s = F.normalize(self.projector(z_s), dim=-1)
        return {
            "x_hat": x_hat,
            "z_s": z_s,
            "z_a": z_a,
            "proj_s": proj_s
        }

    def compute_loss(self, x: torch.Tensor, x_pos: torch.Tensor,
                     lambda_recon: float = 1.0,
                     lambda_contrastive: float = 0.5,
                     lambda_mi: float = 0.1):
        """
        Computes composite loss with stability guards against -inf divergence:
        1. Reconstruction Loss: MSE(x, x_hat)
        2. Contrastive InfoNCE Loss on stable projections
        3. Mutual Information / Orthogonality Guard: (cos_sim(z_s, z_a)**2).mean()
        """
        out_curr = self.forward(x)
        out_pos = self.forward(x_pos)

        x_hat = out_curr["x_hat"]
        z_s = out_curr["z_s"]
        z_a = out_curr["z_a"]
        proj_curr = out_curr["proj_s"]
        proj_pos = out_pos["proj_s"]

        # 1. Reconstruction Loss
        loss_recon = F.mse_loss(x_hat, x)

        # 2. InfoNCE Contrastive Loss
        # Cosine similarities
        B = x.shape[0]
        sim_matrix = torch.matmul(proj_curr, proj_pos.T) / self.temperature
        labels = torch.arange(B, device=x.device)
        loss_contrastive = F.cross_entropy(sim_matrix, labels)

        # 3. Guarded Mutual Information Penalty (squared magnitude penalty)
        # Prevents divergence to -inf by penalizing deviation from orthogonality (0)
        cos_sim = F.cosine_similarity(z_s, z_a, dim=-1)
        loss_mi_penalty = (cos_sim ** 2).mean()

        total_loss = (lambda_recon * loss_recon +
                      lambda_contrastive * loss_contrastive +
                      lambda_mi * loss_mi_penalty)

        return {
            "total_loss": total_loss,
            "loss_recon": loss_recon.item(),
            "loss_contrastive": loss_contrastive.item(),
            "loss_mi_penalty": loss_mi_penalty.item()
        }

    @torch.no_grad()
    def compute_anomaly_scores(self, x: torch.Tensor) -> np.ndarray:
        """
        Computes anomaly score per window:
        Combines reconstruction error with auxiliary component energy ratio.
        Score = Recon_MSE * (1 + ||z_a|| / (||z_s|| + ||z_a|| + eps))
        """
        self.eval()
        out = self.forward(x)
        x_hat = out["x_hat"]
        z_s = out["z_s"]
        z_a = out["z_a"]

        # Reconstruction error per window (mean squared error across T and D)
        recon_err = ((x - x_hat) ** 2).mean(dim=[1, 2])  # (B,)

        # Auxiliary energy ratio
        norm_s = torch.norm(z_s, p=2, dim=-1)
        norm_a = torch.norm(z_a, p=2, dim=-1)
        aux_ratio = norm_a / (norm_s + norm_a + 1e-6)

        anomaly_score = recon_err * (1.0 + aux_ratio)
        return anomaly_score.cpu().numpy()

"""
Visual-Text Anomaly Mapper (VTAM).

Implements an Anomaly Mixture-of-Experts (AnomalyMoE) paradigm with a
dual-gating mechanism:
  • Global gate — selects which experts operate on each token based on
                  its global semantic context.
  • Local gate  — spatially filters expert outputs to suppress background
                  noise and highlight anomaly-sensitive regions.

The module then calibrates the global anomaly score produced by CLIP
similarity with local spatial evidence from AnomalyMoE, resolving the
coarse global-local disconnection.

Reference: SSVP (arXiv 2601.09147), Section 3.4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """Single MLP expert network."""

    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class AnomalyMoE(nn.Module):
    """
    Anomaly Mixture-of-Experts with dual-gating.

    Args:
        dim:          Feature dimensionality.
        num_experts:  Total number of expert networks.
        top_k:        Number of experts activated per token (sparse routing).
        dropout:      Dropout probability.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k

        # Expert pool
        self.experts = nn.ModuleList(
            [Expert(dim, expansion=2, dropout=dropout) for _ in range(num_experts)]
        )

        # Gate 1: global scale selection — predicts expert routing weights
        self.global_gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_experts),
        )

        # Gate 2: local spatial filter — scalar mask to highlight anomaly regions
        self.local_gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor   # (B, N, dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            out          : (B, N, dim)  — spatially filtered expert output
            local_weight : (B, N, 1)   — per-token anomaly saliency mask
        """
        B, N, D = x.shape

        # --- Global gate: top-k sparse expert selection ---
        gate_logits = self.global_gate(x)          # (B, N, num_experts)
        topk_vals, topk_idx = gate_logits.topk(self.top_k, dim=-1)
        topk_weights = torch.softmax(topk_vals, dim=-1)  # (B, N, top_k)

        # Run all experts, then selectively accumulate
        expert_outs = torch.stack(
            [e(x) for e in self.experts], dim=2
        )  # (B, N, num_experts, D)

        # Gather top-k expert outputs
        idx_expanded = topk_idx.unsqueeze(-1).expand(B, N, self.top_k, D)
        selected = expert_outs.gather(2, idx_expanded)  # (B, N, top_k, D)

        # Weighted sum over selected experts
        global_out = (selected * topk_weights.unsqueeze(-1)).sum(dim=2)  # (B, N, D)

        # --- Local gate: spatial anomaly saliency ---
        local_weight = self.local_gate(x)          # (B, N, 1)

        out = global_out * local_weight
        return out, local_weight


class VTAM(nn.Module):
    """
    Visual-Text Anomaly Mapper.

    Calibrates a global image-level anomaly score (computed from CLIP
    cosine similarity) with fine-grained local spatial evidence produced
    by AnomalyMoE, bridging the global-local detection gap.

    Args:
        visual_dim:    Dimensionality of visual tokens (HSVS output).
        text_dim:      Dimensionality of conditioned text tokens (VCPG output).
        num_experts:   Number of experts in AnomalyMoE.
        top_k:         Sparse routing top-k.
        dropout:       Dropout probability.
    """

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.anomaly_moe = AnomalyMoE(visual_dim, num_experts, top_k, dropout)

        # Project text to visual space for joint calibration
        self.text_proj = nn.Linear(text_dim, visual_dim, bias=False)

        # Calibration head: fuses MoE output with text evidence → score delta
        self.calibration = nn.Sequential(
            nn.Linear(visual_dim * 2, visual_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(visual_dim, 1),
        )

        self.norm = nn.LayerNorm(visual_dim)

    def forward(
        self,
        visual_feat: torch.Tensor,   # (B, N, visual_dim)
        text_feat: torch.Tensor,     # (B, T, text_dim) — conditioned prompts from VCPG
        global_score: torch.Tensor,  # (B,) or (B, N) — coarse CLIP anomaly score
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            calibrated_score : (B, N) — refined per-token anomaly score
            local_weight     : (B, N) — spatial anomaly saliency from MoE
        """
        # AnomalyMoE: route features through sparse experts
        local_evidence, local_weight = self.anomaly_moe(visual_feat)
        local_evidence = self.norm(visual_feat + local_evidence)  # residual

        # Aggregate text embedding to a single vector per sample
        text_global = self.text_proj(text_feat.mean(dim=1))       # (B, visual_dim)
        text_global = text_global.unsqueeze(1).expand_as(local_evidence)

        # Calibrate: concatenate local MoE evidence with text context
        combined = torch.cat([local_evidence, text_global], dim=-1)  # (B, N, 2*V)
        score_delta = self.calibration(combined).squeeze(-1)          # (B, N)

        # Broadcast global score to per-token shape if needed
        if global_score.dim() == 1:
            global_score = global_score.unsqueeze(1).expand_as(score_delta)

        calibrated_score = global_score + score_delta
        return calibrated_score, local_weight.squeeze(-1)

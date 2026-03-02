"""
Hierarchical Semantic-Visual Synergy (HSVS) module.

Injects DINOv2 multi-scale structural priors into the CLIP semantic space
via an Adaptive Token Features Fusion (ATF) block that uses dual-path
cross-modal attention and learnable projection matrices.

Reference: SSVP (arXiv 2601.09147), Section 3.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ATFBlock(nn.Module):
    """
    Adaptive Token Features Fusion Block.

    Uses dual-path cross-modal attention to align CLIP and DINOv2 token
    features, then fuses them back into the CLIP semantic manifold.

    Args:
        clip_dim:   Dimensionality of CLIP token embeddings.
        dino_dim:   Dimensionality of DINOv2 token embeddings.
        num_heads:  Number of attention heads.
        dropout:    Dropout probability applied after attention.
    """

    def __init__(
        self,
        clip_dim: int,
        dino_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Learnable projection: DINOv2 space → CLIP space
        self.proj_dino = nn.Linear(dino_dim, clip_dim, bias=False)

        # Path 1: CLIP tokens (Q) attend to projected DINOv2 tokens (K, V)
        self.cross_attn_c2d = nn.MultiheadAttention(
            clip_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Path 2: projected DINOv2 tokens (Q) attend to CLIP tokens (K, V)
        self.cross_attn_d2c = nn.MultiheadAttention(
            clip_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Fuse dual-path outputs back to clip_dim
        self.fusion = nn.Sequential(
            nn.Linear(clip_dim * 2, clip_dim),
            nn.GELU(),
        )

        self.norm1 = nn.LayerNorm(clip_dim)
        self.norm2 = nn.LayerNorm(clip_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(clip_dim, clip_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(clip_dim * 4, clip_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        clip_feat: torch.Tensor,   # (B, N_clip, clip_dim)
        dino_feat: torch.Tensor,   # (B, N_dino, dino_dim)
    ) -> torch.Tensor:             # (B, N_clip, clip_dim)
        # Project DINOv2 features to CLIP dimensionality
        dino_proj = self.proj_dino(dino_feat)           # (B, N_dino, clip_dim)

        # Path 1: CLIP queries DINOv2
        attn_c2d, _ = self.cross_attn_c2d(clip_feat, dino_proj, dino_proj)

        # Path 2: DINOv2 queries CLIP
        attn_d2c, _ = self.cross_attn_d2c(dino_proj, clip_feat, clip_feat)
        # Align sequence length back to N_clip via pooling if needed
        if attn_d2c.size(1) != clip_feat.size(1):
            attn_d2c = F.adaptive_avg_pool1d(
                attn_d2c.transpose(1, 2), clip_feat.size(1)
            ).transpose(1, 2)

        # Fuse both paths, then residual + LayerNorm
        fused = self.fusion(torch.cat([attn_c2d, attn_d2c], dim=-1))
        out = self.norm1(clip_feat + fused)

        # FFN with residual + LayerNorm
        out = self.norm2(out + self.ffn(out))
        return out


class HSVS(nn.Module):
    """
    Hierarchical Semantic-Visual Synergy.

    Stacks multiple ATF blocks, one per DINOv2 scale, and combines the
    outputs with learnable per-scale weights.

    Args:
        clip_dim:    Dimensionality of CLIP token embeddings.
        dino_dim:    Dimensionality of DINOv2 token embeddings.
        num_scales:  Number of DINOv2 feature scales to fuse.
        num_heads:   Attention heads for each ATF block.
        dropout:     Dropout probability.
    """

    def __init__(
        self,
        clip_dim: int,
        dino_dim: int,
        num_scales: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_scales = num_scales

        self.atf_blocks = nn.ModuleList(
            [ATFBlock(clip_dim, dino_dim, num_heads, dropout) for _ in range(num_scales)]
        )

        # Learnable log-scale weights (softmax-normalised at forward time)
        self.scale_logits = nn.Parameter(torch.zeros(num_scales))

    def forward(
        self,
        clip_feat: torch.Tensor,           # (B, N, clip_dim)
        dino_feats: list[torch.Tensor],    # list of (B, N_i, dino_dim), len == num_scales
    ) -> torch.Tensor:                     # (B, N, clip_dim)
        if len(dino_feats) != self.num_scales:
            raise ValueError(
                f"Expected {self.num_scales} DINOv2 scales, got {len(dino_feats)}"
            )

        weights = torch.softmax(self.scale_logits, dim=0)   # (num_scales,)

        fused_scales = [
            weights[i] * self.atf_blocks[i](clip_feat, dino_feats[i])
            for i in range(self.num_scales)
        ]
        return torch.stack(fused_scales, dim=0).sum(dim=0)  # (B, N, clip_dim)

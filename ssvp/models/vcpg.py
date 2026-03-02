"""
Vision-Conditioned Prompt Generator (VCPG).

Uses a Variational Autoencoder (VAE) to model the latent distribution of
visual anomaly features, then employs text-latent cross-modal attention so
that text embeddings dynamically retrieve and integrate generative visual
biases — enabling linguistic queries to precisely anchor to unseen defect
patterns via the reparameterisation trick.

Reference: SSVP (arXiv 2601.09147), Section 3.3
"""

import torch
import torch.nn as nn


class VAEEncoder(nn.Module):
    """
    Encodes visual features into a Gaussian latent distribution (μ, log σ²).
    """

    def __init__(self, visual_dim: int, latent_dim: int) -> None:
        super().__init__()
        hidden = (visual_dim + latent_dim) // 2
        self.shared = nn.Sequential(
            nn.Linear(visual_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.mu_head = nn.Linear(hidden, latent_dim)
        self.logvar_head = nn.Linear(hidden, latent_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:   # μ, log σ²
        h = self.shared(x)
        return self.mu_head(h), self.logvar_head(h)


class VCPG(nn.Module):
    """
    Vision-Conditioned Prompt Generator.

    Args:
        visual_dim:  Dimensionality of enhanced visual tokens from HSVS.
        text_dim:    Dimensionality of CLIP text embeddings.
        latent_dim:  VAE latent space dimensionality.
        num_heads:   Attention heads for text-latent cross-attention.
        dropout:     Dropout probability.
    """

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        latent_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # VAE encoder: visual features → (μ, log σ²)
        self.encoder = VAEEncoder(visual_dim, latent_dim)

        # Project latent samples to text embedding space (used as K, V)
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, text_dim),
            nn.LayerNorm(text_dim),
        )

        # Cross-attention: text embeddings (Q) attend to visual latent (K, V)
        self.cross_attn = nn.MultiheadAttention(
            text_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm = nn.LayerNorm(text_dim)
        self.dropout = nn.Dropout(dropout)

        # Lightweight refinement MLP
        self.refine = nn.Sequential(
            nn.Linear(text_dim, text_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(text_dim * 2, text_dim),
        )
        self.norm2 = nn.LayerNorm(text_dim)

    @staticmethod
    def reparameterise(
        mu: torch.Tensor, logvar: torch.Tensor, training: bool
    ) -> torch.Tensor:
        """Sample z = μ + ε · σ during training; return μ at inference."""
        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def kl_loss(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """KL divergence loss D_KL[ q(z|x) || N(0,I) ]."""
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(
        self,
        visual_feat: torch.Tensor,   # (B, N_v, visual_dim)  — HSVS output
        text_feat: torch.Tensor,     # (B, N_t, text_dim)    — CLIP text tokens
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            conditioned_text : (B, N_t, text_dim)  — visually conditioned prompts
            mu               : (B, N_v, latent_dim)
            logvar           : (B, N_v, latent_dim)
        """
        # Encode visual features into latent distribution
        mu, logvar = self.encoder(visual_feat)          # (B, N_v, latent_dim)

        # Sample latent vector via reparameterisation trick
        z = self.reparameterise(mu, logvar, self.training)

        # Project latent samples to text space to serve as keys/values
        visual_kv = self.latent_proj(z)                 # (B, N_v, text_dim)

        # Text embeddings (Q) attend to visual latent priors (K, V)
        attended, _ = self.cross_attn(text_feat, visual_kv, visual_kv)
        attended = self.dropout(attended)

        # Residual + LayerNorm
        out = self.norm(text_feat + attended)

        # Refinement MLP with residual
        out = self.norm2(out + self.refine(out))

        return out, mu, logvar

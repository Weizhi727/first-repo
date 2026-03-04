"""
SSVP: Synergistic Semantic-Visual Prompting for Industrial Zero-Shot
Anomaly Detection.

Main model class that wires HSVS, VCPG, and VTAM into an end-to-end
zero-shot anomaly detection pipeline backed by frozen CLIP and DINOv2
encoders.

Reference: SSVP (arXiv 2601.09147)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hsvs import HSVS
from .vcpg import VCPG
from .vtam import VTAM


@dataclass
class SSVPConfig:
    # ---- backbone dimensionalities ----
    clip_dim: int = 512          # CLIP ViT-B/16 → 512; ViT-L/14 → 768
    dino_dim: int = 768          # DINOv3 ViT-B/16 → 768; ViT-L/16 → 1024
    patch_size: int = 16         # DINOv3 patch size (DINOv2 was 14)

    # ---- HSVS ----
    hsvs_num_scales: int = 3     # how many DINOv2 intermediate layers to fuse
    hsvs_num_heads: int = 8

    # ---- VCPG ----
    vcpg_latent_dim: int = 256
    vcpg_num_heads: int = 8
    vcpg_kl_weight: float = 1e-4  # weight for VAE KL-divergence loss

    # ---- VTAM ----
    vtam_num_experts: int = 4
    vtam_top_k: int = 2

    # ---- shared ----
    dropout: float = 0.1

    # ---- text prompts ----
    normal_prompts: list[str] = field(default_factory=lambda: [
        "a photo of a {} without any defects",
        "a flawless photo of a {}",
        "a photo of a normal {}",
    ])
    anomaly_prompts: list[str] = field(default_factory=lambda: [
        "a photo of a {} with defects",
        "a damaged photo of a {}",
        "a photo of an anomalous {}",
    ])


class SSVP(nn.Module):
    """
    Synergistic Semantic-Visual Prompting model.

    Args:
        clip_model:  A CLIP model instance (e.g. from openai/clip-vit-base-patch16)
                     with .encode_image() and .encode_text() methods.
                     The model is kept frozen; only the SSVP heads are trained.
        dino_model:  A DINOv3 model instance from torch.hub or transformers.
                     The model is kept frozen.
        config:      SSVPConfig dataclass with all hyperparameters.
    """

    def __init__(
        self,
        clip_model: nn.Module,
        dino_model: nn.Module,
        config: SSVPConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or SSVPConfig()
        cfg = self.config

        # Frozen backbone encoders
        self.clip = clip_model
        self.dino = dino_model
        self._freeze_backbones()

        # Trainable SSVP heads
        self.hsvs = HSVS(
            clip_dim=cfg.clip_dim,
            dino_dim=cfg.dino_dim,
            num_scales=cfg.hsvs_num_scales,
            num_heads=cfg.hsvs_num_heads,
            dropout=cfg.dropout,
        )
        self.vcpg = VCPG(
            visual_dim=cfg.clip_dim,
            text_dim=cfg.clip_dim,
            latent_dim=cfg.vcpg_latent_dim,
            num_heads=cfg.vcpg_num_heads,
            dropout=cfg.dropout,
        )
        self.vtam = VTAM(
            visual_dim=cfg.clip_dim,
            text_dim=cfg.clip_dim,
            num_experts=cfg.vtam_num_experts,
            top_k=cfg.vtam_top_k,
            dropout=cfg.dropout,
        )

    # ------------------------------------------------------------------
    # Backbone helpers
    # ------------------------------------------------------------------

    def _freeze_backbones(self) -> None:
        for param in self.clip.parameters():
            param.requires_grad_(False)
        for param in self.dino.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def _encode_text(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        """
        Tokenise and encode a list of text prompts with CLIP.
        Returns averaged embeddings: (1, clip_dim).
        """
        import clip as openai_clip  # lazy import; avoids hard dependency at module level

        tokens = openai_clip.tokenize(prompts).to(device)
        embeddings = self.clip.encode_text(tokens)           # (P, clip_dim)
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings.mean(dim=0, keepdim=True)          # (1, clip_dim)

    @torch.no_grad()
    def _extract_dino_multiscale(
        self, images: torch.Tensor
    ) -> list[torch.Tensor]:
        """
        Extract intermediate patch tokens from DINOv3 at multiple depths.

        Returns a list of tensors (B, N_patches, dino_dim), one per scale,
        corresponding to the last `hsvs_num_scales` transformer blocks.

        DINOv3 has 4 register tokens prepended after the CLS token.
        get_intermediate_layers() with default return_class_token=False already
        strips CLS and register tokens, but we enforce the patch count explicitly
        to be safe across different DINOv3 releases.
        """
        _, _, H, W = images.shape
        num_patches = (H // self.config.patch_size) * (W // self.config.patch_size)

        outputs = self.dino.get_intermediate_layers(
            images, n=self.config.hsvs_num_scales
        )
        # Keep exactly the last num_patches tokens (pure patch tokens).
        # This strips any leading CLS / register tokens that some DINOv3
        # versions may include in the output.
        return [o[:, -num_patches:, :] for o in outputs]

    @staticmethod
    def _interp_pos_embed(
        pos_embed: torch.Tensor, num_patches: int
    ) -> torch.Tensor:
        """
        Bicubic-interpolate CLIP positional embeddings to a different grid size.

        CLIP ViT-B/16 is pretrained at 224×224 (196 patch positions).  When the
        input resolution differs (e.g. 512×512 → 1024 patches) the stored
        positional embeddings must be resized before they can be added to the
        patch sequence.

        Args:
            pos_embed:   (N_orig+1, C) — index 0 is the CLS position, the rest
                         are patch positions from the pretrained model.
            num_patches: Target number of patch positions.

        Returns:
            (num_patches+1, C) — CLS position preserved, patch grid rescaled.
        """
        N_orig = pos_embed.shape[0] - 1
        if N_orig == num_patches:
            return pos_embed

        cls_pos   = pos_embed[:1]                            # (1, C)
        patch_pos = pos_embed[1:].float()                    # (N_orig, C)

        h_o = w_o = int(math.isqrt(N_orig))
        h_n = w_n = int(math.isqrt(num_patches))

        # Reshape to 2D spatial grid, interpolate, then flatten back
        grid = patch_pos.reshape(1, h_o, w_o, -1).permute(0, 3, 1, 2)  # (1,C,h,w)
        grid = F.interpolate(grid, size=(h_n, w_n),
                             mode="bicubic", align_corners=False)
        grid = grid.permute(0, 2, 3, 1).reshape(num_patches, -1)        # (num_patches,C)

        return torch.cat([cls_pos, grid.to(pos_embed.dtype)], dim=0)    # (num_patches+1,C)

    @torch.no_grad()
    def _extract_clip_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract patch-level token embeddings from CLIP's visual transformer.
        Returns (B, N_patches, clip_dim) — excludes the [CLS] token.

        Supports any input resolution that is a multiple of the CLIP patch size
        by bicubic-interpolating the stored positional embeddings when needed.
        """
        # openai/CLIP stores the visual transformer as self.clip.visual
        # CLIPHFAdapter exposes the same interface via _HFVisualNamespace
        visual = self.clip.visual
        x = visual.conv1(images)                              # (B, C, H, W)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)   # (B, N, C)
        cls = visual.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)                       # (B, N+1, C)

        # Interpolate positional embeddings if image resolution != training res
        pos = self._interp_pos_embed(
            visual.positional_embedding, num_patches=x.size(1) - 1
        )
        x = x + pos.to(x.dtype)

        x = visual.ln_pre(x)
        x = visual.transformer(x)
        x = visual.ln_post(x[:, 1:, :])                      # drop [CLS], keep patches
        if visual.proj is not None:
            x = x @ visual.proj
        return x                                              # (B, N, clip_dim)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,            # (B, 3, H, W)
        class_name: str = "object",
        return_maps: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images:      Batch of pre-processed images.
            class_name:  Category name inserted into text prompt templates.
            return_maps: If True, also return the spatial anomaly map.

        Returns a dict with keys:
            "score"        : (B,)   — image-level anomaly score (higher → more anomalous)
            "kl_loss"      : scalar — VAE KL loss (0 during inference)
            "anomaly_map"  : (B, H, W) only when return_maps=True
            "saliency"     : (B, N)   only when return_maps=True
        """
        B, _, H, W = images.shape
        device = images.device
        cfg = self.config

        # ---- 1. Backbone feature extraction (frozen) ----
        clip_patch = self._extract_clip_patch_tokens(images)   # (B, N, clip_dim)
        dino_scales = self._extract_dino_multiscale(images)    # list[(B, N_i, dino_dim)]

        normal_prompts = [t.format(class_name) for t in cfg.normal_prompts]
        anomaly_prompts = [t.format(class_name) for t in cfg.anomaly_prompts]

        text_normal = self._encode_text(normal_prompts, device)   # (1, clip_dim)
        text_anomaly = self._encode_text(anomaly_prompts, device)  # (1, clip_dim)

        # Stack as sequence: (B, 2, clip_dim)
        text_feat = torch.cat(
            [text_normal.expand(B, -1, -1), text_anomaly.expand(B, -1, -1)], dim=1
        )

        # ---- 2. HSVS: inject DINOv2 structure into CLIP space ----
        enhanced_visual = self.hsvs(clip_patch, dino_scales)   # (B, N, clip_dim)

        # ---- 3. VCPG: generate vision-conditioned prompts ----
        cond_text, mu, logvar = self.vcpg(enhanced_visual, text_feat)

        # ---- 4. Global anomaly score via cosine similarity ----
        v_norm = F.normalize(enhanced_visual, dim=-1)           # (B, N, clip_dim)
        t_norm = F.normalize(cond_text, dim=-1)                 # (B, 2, clip_dim)

        # Compare each patch to [normal, anomaly] text
        sim = torch.einsum("bnc,btc->bnt", v_norm, t_norm)     # (B, N, 2)
        # Anomaly score: anomaly_sim − normal_sim (higher → more anomalous)
        global_patch_score = sim[:, :, 1] - sim[:, :, 0]       # (B, N)
        global_score = global_patch_score.mean(dim=1)           # (B,)

        # ---- 5. VTAM: calibrate with local MoE evidence ----
        calibrated, saliency = self.vtam(enhanced_visual, cond_text, global_patch_score)
        final_score = calibrated.mean(dim=1)                    # (B,)

        kl = self.vcpg.kl_loss(mu, logvar) if self.training else torch.tensor(0.0, device=device)

        out: dict[str, torch.Tensor] = {
            "score": final_score,
            "global_score": global_score,
            "kl_loss": kl,
        }

        if return_maps:
            out["saliency"] = saliency                         # (B, N)
            out["anomaly_map"] = self._to_spatial_map(calibrated, H, W)

        return out

    def _to_spatial_map(
        self, patch_scores: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        """Reshape flat patch scores to a spatial map and upsample to (H, W)."""
        N = patch_scores.size(1)
        side = int(math.isqrt(N))
        map_2d = patch_scores.reshape(-1, 1, side, side)       # (B, 1, s, s)
        map_up = F.interpolate(map_2d, size=(H, W), mode="bilinear", align_corners=False)
        return map_up.squeeze(1)                                # (B, H, W)

    # ------------------------------------------------------------------
    # Convenience: trainable parameters only
    # ------------------------------------------------------------------

    def trainable_parameters(self):
        """Yields only parameters that are not part of frozen backbones."""
        for name, param in self.named_parameters():
            if param.requires_grad:
                yield name, param

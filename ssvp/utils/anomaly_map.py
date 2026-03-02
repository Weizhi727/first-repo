"""
Anomaly map post-processing utilities.
"""

import numpy as np
import torch
import torch.nn.functional as F


def normalize_map(anomaly_map: torch.Tensor) -> torch.Tensor:
    """
    Min-max normalize an anomaly map to [0, 1].

    Args:
        anomaly_map: (B, H, W) or (H, W) tensor of raw anomaly scores.
    Returns:
        Normalized tensor with the same shape.
    """
    flat = anomaly_map.flatten(start_dim=-2 if anomaly_map.dim() == 3 else 0)
    mn = flat.min(dim=-1, keepdim=True).values
    mx = flat.max(dim=-1, keepdim=True).values
    flat_norm = (flat - mn) / (mx - mn + 1e-8)
    return flat_norm.reshape(anomaly_map.shape)


def postprocess_map(
    anomaly_map: torch.Tensor,
    target_size: tuple[int, int] | None = None,
    gaussian_sigma: float = 4.0,
) -> torch.Tensor:
    """
    Post-process a raw anomaly map: optional upsampling + Gaussian smoothing
    + normalization.

    Args:
        anomaly_map:   (B, H, W) raw anomaly map.
        target_size:   (H_out, W_out) to upsample to; None keeps original size.
        gaussian_sigma: σ for Gaussian blur (kernel size = 2*ceil(3σ)+1).

    Returns:
        Processed map (B, H_out, W_out) in [0, 1].
    """
    if target_size is not None:
        anomaly_map = F.interpolate(
            anomaly_map.unsqueeze(1),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    # Gaussian smoothing
    if gaussian_sigma > 0:
        anomaly_map = _gaussian_blur(anomaly_map, gaussian_sigma)

    return normalize_map(anomaly_map)


def apply_colormap(
    anomaly_map: torch.Tensor,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Convert a normalized (B, H, W) or (H, W) anomaly map to an RGB image
    using a matplotlib colormap.

    Args:
        anomaly_map: Tensor in [0, 1], (H, W) or (B, H, W).
        colormap:    Any matplotlib colormap name.

    Returns:
        uint8 numpy array of shape (H, W, 3) or (B, H, W, 3).
    """
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(colormap)
    arr = anomaly_map.detach().cpu().numpy()

    def _apply(x2d: np.ndarray) -> np.ndarray:
        rgba = cmap(x2d)                    # (H, W, 4)
        return (rgba[..., :3] * 255).astype(np.uint8)

    if arr.ndim == 2:
        return _apply(arr)
    return np.stack([_apply(arr[i]) for i in range(arr.shape[0])], axis=0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply separable Gaussian blur to (B, H, W) tensor."""
    import math

    radius = math.ceil(3 * sigma)
    ksize = 2 * radius + 1

    # 1-D Gaussian kernel
    coords = torch.arange(ksize, dtype=x.dtype, device=x.device) - radius
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Outer product → 2-D kernel
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]     # (k, k)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)          # (1, 1, k, k)

    # Convolve (treat batch as channels)
    x_4d = x.unsqueeze(1)                                    # (B, 1, H, W)
    blurred = F.conv2d(x_4d, kernel_2d, padding=radius)
    return blurred.squeeze(1)                                 # (B, H, W)

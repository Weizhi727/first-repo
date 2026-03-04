"""
SSVP Training Script
====================
Trains the SSVP head modules (HSVS + VCPG + VTAM) on top of frozen
CLIP and DINOv2/v3 backbones.

Training strategy
-----------------
Only the three learnable SSVP modules are updated; the backbone encoders
stay frozen throughout. Normal images are loaded from the training split
and Perlin-noise synthetic anomalies are generated on-the-fly to produce
paired (normal, anomaly) batches.

Loss components (paper § 3)
----------------------------
  L_total = L_align + λ_pixel * L_pixel + λ_kl * L_kl + λ_margin * L_margin

  L_align  : temperature-τ cross-entropy on image-level [normal, anomaly] logits
  L_pixel  : focal-BCE between predicted anomaly map and (synthetic) GT mask
  L_kl     : VAE KL divergence from VCPG (prevents representation collapse)
  L_margin : cosine-margin penalty when normal similarities drop below threshold

Typical hyper-parameters (closest related work: VCP-CLIP / AnomalyCLIP)
---------------------------------------------------------------------------
  optimizer : AdamW,  lr=4e-5,  weight_decay=1e-4
  scheduler : cosine annealing
  epochs    : 10
  batch     : 16
  τ (temp.) : 0.07

Dataset layout expected (MVTec-AD style)
-----------------------------------------
  root/
    <category>/
      train/good/*.png
      test/good/*.png
      test/<defect_type>/*.png
      ground_truth/<defect_type>/*.png   (binary masks)

Usage
-----
  python ssvp/train.py \\
      --data-root /path/to/mvtec \\
      --category  bottle \\
      --output-dir checkpoints/ \\
      [--clip-ckpt  /path/to/clip.pt] \\
      [--dino-ckpt  /path/to/dinov3.pth] \\
      [--clip-model ViT-B/16] \\
      [--dino-model dinov2_vitb14] \\
      [--epochs 10] [--batch-size 16] [--lr 4e-5]

References
----------
  SSVP:        arXiv 2601.09147
  VCP-CLIP:    arXiv 2407.12276
  AnomalyCLIP: arXiv 2310.18961
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# Lazy imports resolved inside functions to keep top-level import fast
# ---------------------------------------------------------------------------
# from models import SSVP, SSVPConfig  (imported below inside load helpers)


# ===========================================================================
# Synthetic anomaly generation (Perlin-noise based, DRAEM-style)
# ===========================================================================

def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def _fade(t: np.ndarray) -> np.ndarray:
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def _perlin_2d(shape: tuple[int, int], scale: int = 8) -> np.ndarray:
    """Return a Perlin-noise map in [0, 1] of the given shape."""
    H, W = shape
    # Grid
    gx = np.linspace(0, scale, W, endpoint=False)
    gy = np.linspace(0, scale, H, endpoint=False)
    gx0, gy0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
    gx1, gy1 = gx0 + 1, gy0 + 1
    tx = _fade(gx - np.floor(gx))
    ty = _fade(gy - np.floor(gy))

    rng = np.random.default_rng()

    def _rand_grad(ix, iy):
        angle = rng.uniform(0, 2 * math.pi, size=(len(iy), len(ix)))
        return np.cos(angle), np.sin(angle)

    gx0 %= scale; gx1 %= scale; gy0 %= scale; gy1 %= scale

    g00c, g00s = _rand_grad(gx0, gy0)
    g10c, g10s = _rand_grad(gx1, gy0)
    g01c, g01s = _rand_grad(gx0, gy1)
    g11c, g11s = _rand_grad(gx1, gy1)

    ex0 = gx - np.floor(gx)
    ex1 = ex0 - 1.0
    ey0 = (gy - np.floor(gy))[:, None]
    ey1 = ey0 - 1.0
    ex0, ex1 = ex0[None, :], ex1[None, :]

    n00 = g00c * ex0 + g00s * ey0
    n10 = g10c * ex1 + g10s * ey0
    n01 = g01c * ex0 + g01s * ey1
    n11 = g11c * ex1 + g11s * ey1

    tx_r = tx[None, :]
    ty_r = ty[:, None]
    n_x0 = n00 + tx_r * (n10 - n00)
    n_x1 = n01 + tx_r * (n11 - n01)
    noise = n_x0 + ty_r * (n_x1 - n_x0)
    return (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)


class PerlinAnomalyAugmentor:
    """
    Generates synthetic anomaly images + binary masks via Perlin noise.

    A random texture patch (sampled from DTD or from image itself) is
    blended into the input image at positions dictated by a Perlin-noise
    threshold mask.
    """

    def __init__(
        self,
        image_size: int = 518,
        min_coverage: float = 0.02,
        max_coverage: float = 0.25,
        perlin_scale: int = 8,
        blend_alpha: float = 0.4,
    ) -> None:
        self.image_size = image_size
        self.min_coverage = min_coverage
        self.max_coverage = max_coverage
        self.perlin_scale = perlin_scale
        self.blend_alpha = blend_alpha

    def __call__(
        self, image: torch.Tensor   # (3, H, W) float32 in [0,1]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (augmented_image, mask) where mask ∈ {0,1}, shape (H, W)."""
        _, H, W = image.shape

        for _ in range(20):   # retry until coverage constraint is met
            noise = _perlin_2d((H, W), scale=self.perlin_scale)
            threshold = random.uniform(0.4, 0.8)
            mask = (noise > threshold).astype(np.float32)
            coverage = mask.mean()
            if self.min_coverage <= coverage <= self.max_coverage:
                break

        mask_t = torch.from_numpy(mask)                            # (H, W)

        # Texture: randomly shuffle pixels of the image (simple noise source)
        flat_img = image.view(3, -1)
        perm = torch.randperm(flat_img.size(1))
        texture = flat_img[:, perm].view(image.shape)

        # Blend texture into image at masked positions
        mask_3 = mask_t.unsqueeze(0)                               # (1, H, W)
        augmented = image * (1 - self.blend_alpha * mask_3) + texture * (self.blend_alpha * mask_3)
        augmented = augmented.clamp(0, 1)

        return augmented, mask_t


# ===========================================================================
# Dataset
# ===========================================================================

class MVTecDataset(Dataset):
    """
    MVTec-AD compatible dataset.

    In *train* mode: returns only normal images; the anomaly augmentor is
    applied on-the-fly so every batch contains a synthetic (normal, anomaly)
    pair.

    In *val* mode: returns all test images with their GT binary masks and
    image-level labels (0=normal, 1=anomaly).

    Args:
        root:        Path to the dataset root (contains category folders).
        category:    Category subfolder name (e.g. "bottle").
        split:       "train" or "val".
        image_size:  Spatial resolution for resizing.
        augmentor:   PerlinAnomalyAugmentor instance (train only).
    """

    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        root: str | Path,
        category: str,
        split: str = "train",
        image_size: int = 518,
        augmentor: PerlinAnomalyAugmentor | None = None,
    ) -> None:
        self.split = split
        self.augmentor = augmentor

        base = Path(root) / category

        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        )
        self.normalize = transforms.Normalize(mean=self.CLIP_MEAN, std=self.CLIP_STD)

        if split == "train":
            self.image_paths = sorted((base / "train" / "good").glob("*.*"))
            self.labels: list[int] = [0] * len(self.image_paths)
            self.mask_paths: list[Path | None] = [None] * len(self.image_paths)
        else:
            self.image_paths, self.labels, self.mask_paths = [], [], []
            test_dir = base / "test"
            gt_dir   = base / "ground_truth"
            for defect_dir in sorted(test_dir.iterdir()):
                if not defect_dir.is_dir():
                    continue
                is_normal = (defect_dir.name == "good")
                for img_path in sorted(defect_dir.glob("*.*")):
                    self.image_paths.append(img_path)
                    self.labels.append(0 if is_normal else 1)
                    if is_normal:
                        self.mask_paths.append(None)
                    else:
                        mask_path = gt_dir / defect_dir.name / (img_path.stem + "_mask.png")
                        self.mask_paths.append(mask_path if mask_path.exists() else None)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found for category '{category}' split '{split}' in {root}")

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = self.resize(img)
        t = self.to_tensor(img)           # (3, H, W) in [0,1]
        return self.normalize(t)

    def _load_mask(self, path: Path | None, hw: tuple[int, int]) -> torch.Tensor:
        if path is None:
            return torch.zeros(hw, dtype=torch.float32)
        mask = Image.open(path).convert("L")
        mask = self.resize(mask)
        t = self.to_tensor(mask)          # (1, H, W)
        return (t.squeeze(0) > 0.5).float()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        img = self._load_image(self.image_paths[idx])
        H, W = img.shape[-2], img.shape[-1]

        if self.split == "train":
            # Always produce a (normal, synthetic_anomaly) pair
            aug_img, syn_mask = self.augmentor(
                img.clone()
                # un-normalize to [0,1] before augmentation
                * torch.tensor(self.CLIP_STD).view(3, 1, 1)
                + torch.tensor(self.CLIP_MEAN).view(3, 1, 1)
            )
            # Re-normalize augmented image
            aug_img = self.normalize(aug_img)
            return {
                "normal_image":  img,       # (3, H, W) normalised
                "anomaly_image": aug_img,   # (3, H, W) normalised
                "syn_mask":      syn_mask,  # (H, W) binary
            }
        else:
            mask = self._load_mask(self.mask_paths[idx], (H, W))
            return {
                "image":       img,
                "label":       torch.tensor(self.labels[idx], dtype=torch.long),
                "mask":        mask,
                "image_path":  str(self.image_paths[idx]),
            }


# ===========================================================================
# Loss functions
# ===========================================================================

def alignment_loss(
    score: torch.Tensor,   # (B,) — raw anomaly score from SSVP
    labels: torch.Tensor,  # (B,) — 0=normal, 1=anomaly
    tau: float = 0.07,
) -> torch.Tensor:
    """
    Temperature-scaled binary cross-entropy on image-level anomaly scores.

    Converts the raw score into a 2-class logit pair [−score/τ, score/τ]
    and applies cross-entropy.  Higher score → more anomalous.
    """
    logits = torch.stack([-score / tau, score / tau], dim=-1)   # (B, 2)
    return F.cross_entropy(logits, labels)


def focal_bce_loss(
    pred: torch.Tensor,    # (B, H, W) — predicted anomaly map in (−∞, +∞)
    target: torch.Tensor,  # (B, H, W) — binary GT mask in {0, 1}
    gamma: float = 2.0,
    pos_weight_factor: float = 5.0,
) -> torch.Tensor:
    """
    Focal-BCE loss for pixel-level anomaly map supervision.
    Up-weights hard examples (focal) and corrects class imbalance (pos_weight).
    """
    p = torch.sigmoid(pred)
    bce = F.binary_cross_entropy_with_logits(
        pred, target,
        pos_weight=torch.tensor(pos_weight_factor, device=pred.device),
        reduction="none",
    )
    focal_w = (1 - torch.where(target > 0.5, p, 1 - p)) ** gamma
    return (focal_w * bce).mean()


def margin_loss(
    score: torch.Tensor,   # (B,) — anomaly score
    labels: torch.Tensor,  # (B,) — 0=normal, 1=anomaly
    margin: float = 0.3,
) -> torch.Tensor:
    """
    Cosine-margin penalty (paper § 3):
    penalise normal images whose anomaly score is too high (≥ −margin)
    and anomaly images whose score is too low (≤ margin).
    """
    normal_mask  = (labels == 0).float()
    anomaly_mask = (labels == 1).float()
    loss_normal  = (F.relu( score + margin) * normal_mask).mean()
    loss_anomaly = (F.relu(-score + margin) * anomaly_mask).mean()
    return loss_normal + loss_anomaly


# ===========================================================================
# Metrics
# ===========================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    class_name: str,
    device: torch.device,
) -> dict[str, float]:
    """
    Compute Image-AUROC and Pixel-AUROC on the validation split.

    Returns:
        {"image_auroc": float, "pixel_auroc": float}
    """
    model.eval()
    img_scores, img_labels = [], []
    px_scores,  px_labels  = [], []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].numpy()
        masks  = batch["mask"].numpy()       # (B, H, W) binary

        out = model(images, class_name=class_name, return_maps=True)
        scores = out["score"].cpu().numpy()
        amap   = out["anomaly_map"].cpu().numpy()   # (B, H, W)

        img_scores.append(scores)
        img_labels.append(labels)
        px_scores.append(amap.ravel())
        px_labels.append(masks.ravel())

    img_scores = np.concatenate(img_scores)
    img_labels = np.concatenate(img_labels)
    px_scores  = np.concatenate(px_scores)
    px_labels  = np.concatenate(px_labels)

    image_auroc = roc_auc_score(img_labels, img_scores) if img_labels.max() > 0 else 0.0
    pixel_auroc = roc_auc_score(px_labels,  px_scores)  if px_labels.max()  > 0 else 0.0

    return {"image_auroc": image_auroc, "pixel_auroc": pixel_auroc}


# ===========================================================================
# Training loop
# ===========================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    class_name: str,
    device: torch.device,
    cfg: dict,
    epoch: int,
) -> dict[str, float]:
    model.train()
    stats: dict[str, list[float]] = {
        "loss_total": [], "loss_align": [], "loss_pixel": [],
        "loss_kl": [], "loss_margin": [],
    }

    for step, batch in enumerate(loader):
        normal_img  = batch["normal_image"].to(device)   # (B, 3, H, W)
        anomaly_img = batch["anomaly_image"].to(device)  # (B, 3, H, W)
        syn_mask    = batch["syn_mask"].to(device)       # (B, H, W)

        B = normal_img.size(0)

        # Stack normal + anomaly → single forward pass
        images = torch.cat([normal_img, anomaly_img], dim=0)    # (2B, 3, H, W)
        labels = torch.cat([
            torch.zeros(B, dtype=torch.long),
            torch.ones(B,  dtype=torch.long),
        ]).to(device)

        out = model(images, class_name=class_name, return_maps=True)

        score = out["score"]                  # (2B,)
        kl    = out["kl_loss"]                # scalar
        amap  = out["anomaly_map"]            # (2B, H, W)

        # ---- Loss 1: Image-level alignment ----
        l_align = alignment_loss(score, labels, tau=cfg["tau"])

        # ---- Loss 2: Pixel-level focal-BCE (anomaly half only) ----
        amap_anomaly = amap[B:]               # (B, H, W)
        l_pixel = focal_bce_loss(amap_anomaly, syn_mask)

        # ---- Loss 3: VAE KL divergence (from VCPG) ----
        l_kl = kl

        # ---- Loss 4: Cosine margin ----
        l_margin = margin_loss(score, labels, margin=cfg["margin"])

        total = (
            l_align
            + cfg["lambda_pixel"]  * l_pixel
            + cfg["lambda_kl"]     * l_kl
            + cfg["lambda_margin"] * l_margin
        )

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        stats["loss_total"].append(total.item())
        stats["loss_align"].append(l_align.item())
        stats["loss_pixel"].append(l_pixel.item())
        stats["loss_kl"].append(l_kl.item())
        stats["loss_margin"].append(l_margin.item())

        if (step + 1) % cfg["log_interval"] == 0:
            avg = {k: np.mean(v[-cfg["log_interval"]:]) for k, v in stats.items()}
            print(
                f"  epoch {epoch+1} step {step+1}/{len(loader)} | "
                f"total={avg['loss_total']:.4f}  "
                f"align={avg['loss_align']:.4f}  "
                f"pixel={avg['loss_pixel']:.4f}  "
                f"kl={avg['loss_kl']:.5f}  "
                f"margin={avg['loss_margin']:.4f}"
            )

    return {k: float(np.mean(v)) for k, v in stats.items()}


# ===========================================================================
# Entry point
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train SSVP on an anomaly detection dataset")

    # --- data ---
    p.add_argument("--data-root",   required=True, help="Dataset root (MVTec-AD style)")
    p.add_argument("--category",    required=True, help="Category name (e.g. 'bottle')")
    p.add_argument("--image-size",  type=int, default=512,
                   help="Input resolution (must be a multiple of 16 for DINOv3)")
    p.add_argument("--num-workers", type=int, default=4)

    # --- backbone ---
    p.add_argument("--clip-model", default="ViT-B/16")
    p.add_argument("--dino-model", default="dinov3_vitb16",
                   help="DINOv3 hub model name (e.g. dinov3_vitb16, dinov3_vitl16)")
    p.add_argument("--clip-ckpt",  default=None, help="Local CLIP checkpoint (.pt)")
    p.add_argument("--dino-ckpt",  default=None, help="Local DINOv3 checkpoint (.pth)")

    # --- training ---
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--batch-size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=4e-5)
    p.add_argument("--weight-decay",type=float, default=1e-4)
    p.add_argument("--tau",         type=float, default=0.07,  help="Temperature τ for alignment loss")
    p.add_argument("--margin",      type=float, default=0.30,  help="Cosine margin threshold")
    p.add_argument("--lambda-pixel",type=float, default=0.50)
    p.add_argument("--lambda-kl",   type=float, default=1e-4)
    p.add_argument("--lambda-margin",type=float, default=0.10)
    p.add_argument("--log-interval",type=int,   default=10)

    # --- output ---
    p.add_argument("--output-dir",  default="checkpoints")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Build model ---
    # Import here so the script works when run from the ssvp/ directory
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from models import SSVP, SSVPConfig
    from inference import load_clip, load_dino

    clip_model = load_clip(args.clip_model, device, local_ckpt=args.clip_ckpt)
    dino_model = load_dino(args.dino_model, device, local_ckpt=args.dino_ckpt)

    clip_dim = 768 if "L" in args.clip_model else 512
    dino_dim = 1024 if "vitl" in args.dino_model else 768

    config = SSVPConfig(clip_dim=clip_dim, dino_dim=dino_dim)
    model  = SSVP(clip_model, dino_model, config).to(device)

    print(f"[info] Trainable parameters: "
          f"{sum(p.numel() for _, p in model.trainable_parameters()):,}")

    # --- Datasets & loaders ---
    augmentor = PerlinAnomalyAugmentor(image_size=args.image_size)

    train_ds = MVTecDataset(args.data_root, args.category, "train",
                            args.image_size, augmentor=augmentor)
    val_ds   = MVTecDataset(args.data_root, args.category, "val",
                            args.image_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    print(f"[info] Train: {len(train_ds)} normal images | "
          f"Val: {len(val_ds)} images")

    # --- Optimizer & scheduler ---
    trainable = [p for _, p in model.trainable_parameters()]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    cfg = {
        "tau":           args.tau,
        "margin":        args.margin,
        "lambda_pixel":  args.lambda_pixel,
        "lambda_kl":     args.lambda_kl,
        "lambda_margin": args.lambda_margin,
        "log_interval":  args.log_interval,
    }

    best_pixel_auroc = 0.0
    best_ckpt_path   = out_dir / f"ssvp_{args.category}_best.pt"

    print(f"\n{'='*60}")
    print(f"  Training SSVP  |  category: {args.category}  |  device: {device}")
    print(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}  τ={args.tau}")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, args.category, device, cfg, epoch
        )
        scheduler.step()

        metrics = evaluate(model, val_loader, args.category, device)

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"loss={train_stats['loss_total']:.4f} | "
            f"Image-AUROC={metrics['image_auroc']*100:.1f}%  "
            f"Pixel-AUROC={metrics['pixel_auroc']*100:.1f}%"
        )

        if metrics["pixel_auroc"] > best_pixel_auroc:
            best_pixel_auroc = metrics["pixel_auroc"]
            # Save only the trainable SSVP heads (not the frozen backbones)
            ssvp_state = {
                k: v for k, v in model.state_dict().items()
                if not k.startswith("clip.") and not k.startswith("dino.")
            }
            torch.save(
                {
                    "epoch":        epoch + 1,
                    "model":        ssvp_state,
                    "metrics":      metrics,
                    "config":       config,
                    "args":         vars(args),
                },
                best_ckpt_path,
            )
            print(f"  -> Saved best checkpoint: {best_ckpt_path} "
                  f"(Pixel-AUROC={best_pixel_auroc*100:.1f}%)")

    # Also save last epoch
    last_ckpt_path = out_dir / f"ssvp_{args.category}_last.pt"
    ssvp_state = {
        k: v for k, v in model.state_dict().items()
        if not k.startswith("clip.") and not k.startswith("dino.")
    }
    torch.save({"epoch": args.epochs, "model": ssvp_state,
                "metrics": metrics, "config": config, "args": vars(args)},
               last_ckpt_path)

    print(f"\nDone.  Best Pixel-AUROC: {best_pixel_auroc*100:.2f}%")
    print(f"Best checkpoint : {best_ckpt_path}")
    print(f"Last checkpoint : {last_ckpt_path}")


if __name__ == "__main__":
    main()

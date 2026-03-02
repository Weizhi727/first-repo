# SSVP: Synergistic Semantic-Visual Prompting for Industrial Zero-Shot Anomaly Detection

> Implementation of **SSVP** (arXiv 2601.09147) — a zero-shot anomaly detection framework that fuses CLIP semantic representations with DINOv2 fine-grained structural features via three tightly coupled modules.

---

## Overview

Standard ZSAD (Zero-Shot Anomaly Detection) methods are constrained by a single visual backbone, forcing a trade-off between global semantic generalisation and fine-grained structural discriminability. SSVP resolves this by efficiently fusing two complementary encoders:

| Module | Role |
|--------|------|
| **HSVS** — Hierarchical Semantic-Visual Synergy | Injects DINOv2 multi-scale structural priors into the CLIP semantic manifold via dual-path cross-modal attention (ATF blocks) |
| **VCPG** — Vision-Conditioned Prompt Generator | Uses a VAE-style encoder + cross-modal attention to anchor text embeddings on defect regions |
| **VTAM** — Visual-Text Anomaly Mapper | Mixture-of-Experts with differentiable soft-gating that dynamically calibrates global scores with local patch evidence |

CLIP and DINOv2 backbones remain **frozen**; only the three SSVP heads are trained.

### Results on MVTec-AD (zero-shot)

| Method | Image-AUROC | Pixel-AUROC |
|--------|:-----------:|:-----------:|
| WinCLIP | 85.1 | 85.1 |
| AnomalyCLIP | 91.5 | 91.1 |
| VCP-CLIP | 91.8 | 91.7 |
| Bayes-PFL | 92.4 | 91.9 |
| **SSVP (ours)** | **93.0** | **92.2** |

---

## Repository Structure

```
first-repo/
└── ssvp/
    ├── models/
    │   ├── __init__.py
    │   ├── hsvs.py       # Hierarchical Semantic-Visual Synergy
    │   ├── vcpg.py       # Vision-Conditioned Prompt Generator
    │   ├── vtam.py       # Visual-Text Anomaly Mapper + MoE
    │   └── ssvp.py       # Top-level SSVP model & SSVPConfig
    ├── utils/
    │   ├── __init__.py
    │   └── anomaly_map.py
    ├── train.py          # Full training pipeline
    ├── inference.py      # CLI & Python API for inference
    └── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/<your-username>/first-repo.git
cd first-repo

pip install -r ssvp/requirements.txt

# CLIP (OpenAI)
pip install git+https://github.com/openai/CLIP.git

# DINOv2 is loaded via torch.hub — timm is required
pip install timm>=0.9.0
```

**Requirements snapshot:**

| Package | Version |
|---------|---------|
| torch | ≥ 2.1.0 |
| torchvision | ≥ 0.16.0 |
| openai-clip | ≥ 1.0.1 |
| timm | ≥ 0.9.0 |
| Pillow | ≥ 10.0.0 |
| scikit-learn | ≥ 1.3.0 |
| numpy | ≥ 1.24.0 |
| scipy | ≥ 1.11.0 |

---

## Dataset Preparation

SSVP follows the standard **MVTec-AD** folder layout. Only normal images are required in the training split — synthetic anomalies are generated on-the-fly via Perlin-noise augmentation.

```
dataset_root/
└── <category>/               # e.g. "bottle", "transistor"
    ├── train/
    │   └── good/             # normal training images only
    │       ├── 000.png
    │       └── ...
    ├── test/
    │   ├── good/             # normal test images
    │   ├── broken_large/     # one folder per defect type
    │   └── contamination/
    └── ground_truth/         # binary masks (for Pixel-AUROC)
        ├── broken_large/
        │   └── 000_mask.png
        └── contamination/
```

**Custom industrial dataset:** create the same structure for your own category. Collect normal-product images in `train/good/` (≥ 200 recommended) and place test images with their masks in `test/` and `ground_truth/` respectively.

---

## Training

```bash
python ssvp/train.py \
    --data-root  /path/to/mvtec \
    --category   bottle \
    --clip-ckpt  /path/to/ViT-B-16.pt \
    --dino-ckpt  /path/to/dinov2_vitb14.pth \
    --output-dir checkpoints/
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-root` | — | Dataset root directory |
| `--category` | — | Category subfolder name |
| `--image-size` | `518` | Input resolution (square) |
| `--epochs` | `10` | Number of training epochs |
| `--batch-size` | `16` | Batch size |
| `--lr` | `4e-5` | AdamW learning rate |
| `--tau` | `0.07` | Temperature τ for alignment loss |
| `--margin` | `0.30` | Cosine margin threshold |
| `--lambda-pixel` | `0.50` | Weight for pixel-level focal-BCE loss |
| `--lambda-kl` | `1e-4` | Weight for VAE KL-divergence |
| `--lambda-margin` | `0.10` | Weight for margin loss |
| `--clip-model` | `ViT-B/16` | CLIP variant (used for dim inference) |
| `--dino-model` | `dinov2_vitb14` | DINOv2 architecture variant |
| `--output-dir` | `checkpoints/` | Where to save checkpoints |

### Loss Function

```
L_total = L_align + λ_pixel · L_pixel + λ_kl · L_kl + λ_margin · L_margin
```

| Loss | Description |
|------|-------------|
| `L_align` | Temperature-τ cross-entropy on image-level [normal, anomaly] logits |
| `L_pixel` | Focal-BCE between predicted anomaly map and synthetic GT mask |
| `L_kl` | VAE KL divergence from VCPG (prevents representation collapse) |
| `L_margin` | Cosine-margin penalty when normal/anomaly cosine similarities violate the margin |

Checkpoints are saved to `--output-dir`:
- `ssvp_<category>_best.pt` — best Pixel-AUROC
- `ssvp_<category>_last.pt` — final epoch

Only the SSVP head weights (HSVS + VCPG + VTAM) are saved; frozen backbone weights are not included.

---

## Inference

### Command Line

```bash
# Single image
python ssvp/inference.py \
    --images     path/to/image.jpg \
    --class-name bottle \
    --checkpoint checkpoints/ssvp_bottle_best.pt \
    --clip-ckpt  models/ViT-B-16.pt \
    --dino-ckpt  models/dinov2_vitb14.pth \
    --output-dir results/

# Batch of images with custom threshold
python ssvp/inference.py \
    --images     test/images/*.jpg \
    --class-name transistor \
    --threshold  0.5 \
    --checkpoint checkpoints/ssvp_transistor_best.pt \
    --output-dir results/
```

Each processed image produces a `<stem>_anomaly_map.png` heat map in `--output-dir`.

### Python API

```python
import torch
from PIL import Image
from ssvp.inference import build_model, build_transforms, infer_image
from ssvp.utils import apply_colormap

device = torch.device("cuda")

# Load model once
model = build_model(
    clip_name="ViT-B/16",
    dino_name="dinov2_vitb14",
    checkpoint="checkpoints/ssvp_bottle_best.pt",
    device=device,
    clip_ckpt="models/ViT-B-16.pt",
    dino_ckpt="models/dinov2_vitb14.pth",
)
clip_tf, dino_tf = build_transforms(image_size=518)

# Run on one image
pil_img = Image.open("test.jpg").convert("RGB")
result = infer_image(model, pil_img, class_name="bottle",
                     clip_tf=clip_tf, dino_tf=dino_tf,
                     device=device, image_size=518)

print(f"Anomaly score : {result['score']:.4f}")
# result["anomaly_map"] → torch.Tensor (518, 518), higher = more anomalous

colored = apply_colormap(result["anomaly_map"], colormap="jet")  # (H, W, 3) uint8
Image.fromarray(colored).save("heatmap.png")
```

### Output Reference

| Key | Type | Description |
|-----|------|-------------|
| `score` | `float` | Image-level anomaly score (higher → more anomalous) |
| `anomaly_map` | `Tensor (H, W)` | Pixel-level anomaly heat map, values in [0, 1] |

---

## Model Architecture

```
Input Image (B, 3, H, W)
        │
        ├──[frozen]──► CLIP ViT ──► patch tokens  (B, N, 512)
        │
        └──[frozen]──► DINOv2  ──► multi-scale tokens  3 × (B, N, 768)
                                          │
                               ┌──────────▼──────────┐
                               │        HSVS          │
                               │  (3 × ATF blocks)    │
                               │  dual-path cross-attn│
                               └──────────┬──────────┘
                                          │ enhanced visual (B, N, 512)
                               ┌──────────▼──────────┐
                               │        VCPG          │
                               │  VAE encoder         │
                               │  cross-modal attn    │
                               └──────┬───────┬───────┘
                           cond text  │       │ KL loss
                                      │
                               ┌──────▼──────────────┐
                               │        VTAM          │
                               │  MoE soft-gating     │
                               │  local-global fusion │
                               └──────────┬──────────┘
                                          │
                              score (B,)  +  anomaly_map (B, H, W)
```

---

## Citation

```bibtex
@article{ssvp2026,
  title   = {SSVP: Synergistic Semantic-Visual Prompting for Industrial Zero-Shot Anomaly Detection},
  journal = {arXiv preprint arXiv:2601.09147},
  year    = {2026},
}
```

---

## Acknowledgements

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [DINOv2 — facebookresearch](https://github.com/facebookresearch/dinov2)
- [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP) and [VCP-CLIP](https://github.com/xiaozhen228/VCP-CLIP) for methodology reference
- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

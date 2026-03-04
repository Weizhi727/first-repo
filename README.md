# SSVP: Synergistic Semantic-Visual Prompting for Industrial Zero-Shot Anomaly Detection

> Implementation of **SSVP** (arXiv 2601.09147) — a zero-shot anomaly detection framework that fuses CLIP semantic representations with **DINOv3** fine-grained structural features via three tightly coupled modules.

---

## Overview

Standard ZSAD (Zero-Shot Anomaly Detection) methods are constrained by a single visual backbone, forcing a trade-off between global semantic generalisation and fine-grained structural discriminability. SSVP resolves this by efficiently fusing two complementary encoders:

| Module | Role |
|--------|------|
| **HSVS** — Hierarchical Semantic-Visual Synergy | Injects DINOv3 multi-scale structural priors into the CLIP semantic manifold via dual-path cross-modal attention (ATF blocks) |
| **VCPG** — Vision-Conditioned Prompt Generator | Uses a VAE-style encoder + cross-modal attention to anchor text embeddings on defect regions |
| **VTAM** — Visual-Text Anomaly Mapper | Mixture-of-Experts with differentiable soft-gating that dynamically calibrates global scores with local patch evidence |

CLIP and DINOv3 backbones remain **frozen**; only the three SSVP heads are trained.

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
├── README.md
├── README.zh.md       # Chinese documentation
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

# DINOv3 via torch.hub; HuggingFace CLIP format needs transformers
pip install timm>=0.9.0 transformers>=4.40.0
```

**Requirements snapshot:**

| Package | Version |
|---------|---------|
| torch | ≥ 2.7.1 |
| torchvision | ≥ 0.16.0 |
| openai-clip | ≥ 1.0.1 |
| timm | ≥ 0.9.0 |
| transformers | ≥ 4.40.0 |
| Pillow | ≥ 10.0.0 |
| scikit-learn | ≥ 1.3.0 |
| numpy | ≥ 1.24.0 |
| scipy | ≥ 1.11.0 |

---

## Model Weights

### CLIP

Download all files from `openai/clip-vit-base-patch16` on HuggingFace and place them in a local folder:

```
clip-weight/           ← pass this directory to --clip-ckpt
├── config.json
├── model.safetensors
├── preprocessor_config.json
├── tokenizer.json
└── vocab.json
```

> Both a **HuggingFace directory** and a single OpenAI **`.pt` file** are supported — the loader auto-detects the format.

### DINOv3

```
dino-weight/
└── dinov3_vitb16_pretrain_lvd1689m.pth   ← pass to --dino-ckpt
```

---

## Dataset Preparation

SSVP uses the standard **MVTec-AD-compatible** folder layout. Only normal images are needed in the training split — synthetic anomalies are generated on-the-fly via Perlin-noise augmentation.

### Option A: MVTec-AD (use directly)

Download from the [MVTec website](https://www.mvtec.com/company/research/datasets/mvtec-ad). The archive already follows the required structure (15 categories):

```
mvtec/
└── bottle/
    ├── train/good/        ← normal images only
    ├── test/
    │   ├── good/
    │   ├── broken_large/
    │   └── contamination/
    └── ground_truth/
        └── broken_large/
            └── 000_mask.png
```

### Option B: VisA dataset — convert with spot-diff-main `prepare_data`

VisA contains **12 categories** (10,821 images) covering PCBs, food items, and industrial parts.

**Step 1 — Download raw VisA:**
```bash
# Follow the official download instructions for VisA_highres
```

**Step 2 — Convert to MVTec-AD format:**
```bash
git clone https://github.com/<spot-diff-repo>/spot-diff-main.git
cd spot-diff-main

python prepare_data.py \
    --data-root  /path/to/VisA_highres \
    --output-dir /path/to/VisA_converted \
    --dataset    visa
```

After conversion the structure is identical to MVTec-AD:

```
VisA_converted/
├── candle/   ├── capsules/  ├── cashew/   ├── chewinggum/
├── fryum/    ├── macaroni1/ ├── macaroni2/├── pcb1/
├── pcb2/     ├── pcb3/      ├── pcb4/     └── pipe_fryum/
    each:
        train/good/
        test/good/ + test/<defect_class>/
        ground_truth/<defect_class>/
```

---

## Training

### Zero-Shot Cross-Dataset Protocol

| Train on | Evaluate on | Notes |
|----------|-------------|-------|
| **VisA** | **MVTec-AD** | No category overlap — true zero-shot |
| **MVTec-AD** | **VisA** | Reverse direction |

### Train on VisA → Evaluate on MVTec-AD (recommended)

```bash
# Single category
python ssvp/train.py \
    --data-root  /path/to/VisA_converted \
    --category   pcb1 \
    --clip-ckpt  clip-weight/ \
    --dino-ckpt  dino-weight/dinov3_vitb16_pretrain_lvd1689m.pth \
    --output-dir checkpoints/visa/

# All 12 VisA categories
for cat in candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum; do
    python ssvp/train.py \
        --data-root  /path/to/VisA_converted \
        --category   $cat \
        --clip-ckpt  clip-weight/ \
        --dino-ckpt  dino-weight/dinov3_vitb16_pretrain_lvd1689m.pth \
        --output-dir checkpoints/visa/$cat \
        --epochs 10
done
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-root` | — | Dataset root directory |
| `--category` | — | Category subfolder name |
| `--image-size` | `512` | Input resolution (must be multiple of 16 for DINOv3) |
| `--epochs` | `10` | Number of training epochs |
| `--batch-size` | `16` | Batch size |
| `--lr` | `4e-5` | AdamW learning rate |
| `--tau` | `0.07` | Temperature τ for alignment loss |
| `--margin` | `0.30` | Cosine margin threshold |
| `--lambda-pixel` | `0.50` | Weight for pixel-level focal-BCE loss |
| `--lambda-kl` | `1e-4` | Weight for VAE KL-divergence |
| `--lambda-margin` | `0.10` | Weight for margin loss |
| `--clip-model` | `ViT-B/16` | CLIP variant (used for dim inference) |
| `--dino-model` | `dinov3_vitb16` | DINOv3 hub model name |
| `--output-dir` | `checkpoints/` | Where to save checkpoints |

### Loss Function

```
L_total = L_align + λ_pixel · L_pixel + λ_kl · L_kl + λ_margin · L_margin
```

| Loss | Description |
|------|-------------|
| `L_align` | Temperature-τ cross-entropy on image-level [normal, anomaly] logits |
| `L_pixel` | Focal-BCE between predicted anomaly map and Perlin-noise GT mask |
| `L_kl` | VAE KL divergence from VCPG (prevents representation collapse) |
| `L_margin` | Cosine-margin penalty when normal/anomaly similarities violate the margin |

Checkpoints saved to `--output-dir`:
- `ssvp_<category>_best.pt` — best Pixel-AUROC (auto-saved)
- `ssvp_<category>_last.pt` — final epoch

Only the SSVP head weights (HSVS + VCPG + VTAM) are saved; frozen backbone weights are excluded.

---

## Inference

### Command Line

```bash
# Single image
python ssvp/inference.py \
    --images     path/to/image.jpg \
    --class-name bottle \
    --checkpoint checkpoints/visa/pcb1/ssvp_pcb1_best.pt \
    --clip-ckpt  clip-weight/ \
    --dino-ckpt  dino-weight/dinov3_vitb16_pretrain_lvd1689m.pth \
    --output-dir results/

# Batch — all test images of a MVTec category
python ssvp/inference.py \
    --images     /path/to/mvtec/bottle/test/**/*.png \
    --class-name bottle \
    --checkpoint checkpoints/visa/pcb1/ssvp_pcb1_best.pt \
    --clip-ckpt  clip-weight/ \
    --dino-ckpt  dino-weight/dinov3_vitb16_pretrain_lvd1689m.pth \
    --threshold  0.5 \
    --output-dir results/bottle/

# Full cross-dataset evaluation (VisA-trained → MVTec-AD)
CKPT="checkpoints/visa/pcb1/ssvp_pcb1_best.pt"
for cat in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper; do
    python ssvp/inference.py \
        --images     /path/to/mvtec/$cat/test/**/*.png \
        --class-name $cat \
        --checkpoint $CKPT \
        --clip-ckpt  clip-weight/ \
        --dino-ckpt  dino-weight/dinov3_vitb16_pretrain_lvd1689m.pth \
        --output-dir results/mvtec/$cat/
done
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
    dino_name="dinov3_vitb16",
    checkpoint="checkpoints/visa/pcb1/ssvp_pcb1_best.pt",
    device=device,
    clip_ckpt="clip-weight/",
    dino_ckpt="dino-weight/dinov3_vitb16_pretrain_lvd1689m.pth",
)
clip_tf, dino_tf = build_transforms(image_size=512)

# Run inference
pil_img = Image.open("test.jpg").convert("RGB")
result = infer_image(model, pil_img, class_name="bottle",
                     clip_tf=clip_tf, dino_tf=dino_tf,
                     device=device, image_size=512)

print(f"Anomaly score: {result['score']:.4f}")
# result["anomaly_map"] → torch.Tensor (512, 512), higher = more anomalous

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
Input Image (B, 3, 512, 512)
        │
        ├──[frozen]──► CLIP ViT-B/16 ──► patch tokens  (B, 1024, 512)
        │               patch_size=16
        │               pos_embed bicubic-interpolated to 512×512
        │
        └──[frozen]──► DINOv3 ViT-B/16 ──► multi-scale tokens  3×(B, 1024, 768)
                        patch_size=16, 4 register tokens stripped automatically
                                          │
                               ┌──────────▼──────────┐
                               │        HSVS          │
                               │  (3 × ATF blocks)    │
                               │  dual-path cross-attn│
                               └──────────┬──────────┘
                                          │ enhanced visual (B, 1024, 512)
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
                              score (B,)  +  anomaly_map (B, 512, 512)
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
- [DINOv3 — facebookresearch](https://github.com/facebookresearch/dinov3)
- [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP) and [VCP-CLIP](https://github.com/xiaozhen228/VCP-CLIP) for methodology reference
- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [VisA Dataset](https://github.com/amazon-science/spot-diff)

---

> Chinese documentation: [README.zh.md](./README.zh.md)

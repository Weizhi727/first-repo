"""
SSVP Inference Script
=====================
Run zero-shot anomaly detection on one or more images using the SSVP model.

Usage
-----
    # Download backbones automatically
    python ssvp/inference.py \\
        --images path/to/img1.jpg path/to/img2.png \\
        --class-name transistor \\
        --output-dir results/

    # Use local CLIP + DINOv2/v3 checkpoints (no internet required)
    python ssvp/inference.py \\
        --images path/to/img1.jpg \\
        --class-name transistor \\
        --clip-ckpt /path/to/ViT-B-16.pt \\
        --dino-ckpt /path/to/dinov3_vitb14.pth \\
        --clip-model ViT-B/16 \\
        --dino-model dinov2_vitb14 \\
        --checkpoint path/to/ssvp_weights.pt

Local checkpoint formats supported
------------------------------------
CLIP  : raw .pt file produced by openai/CLIP (same format as the official release)
DINOv2/v3 : any of —
  • full serialised model  : torch.save(model, ...)
  • state_dict             : torch.save(model.state_dict(), ...)
  • wrapped dict           : {"model": state_dict}  or  {"state_dict": state_dict}

Requirements
------------
    pip install -r ssvp/requirements.txt
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models import SSVP, SSVPConfig
from utils import apply_colormap, postprocess_map


# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = 518):
    """Returns (clip_transform, dino_transform)."""
    clip_tf = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
    dino_tf = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=DINO_MEAN, std=DINO_STD),
    ])
    return clip_tf, dino_tf


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_clip(
    model_name: str,
    device: torch.device,
    local_ckpt: str | None = None,
) -> nn.Module:
    """
    Load a CLIP model.

    If *local_ckpt* is given it is passed directly to ``clip.load()`` as the
    model path — the openai/CLIP library accepts both hub model names and
    local ``.pt`` file paths via the same argument.

    Args:
        model_name:  Hub variant used only to infer feature dimensions when
                     *local_ckpt* is also supplied (e.g. ``"ViT-B/16"``).
        device:      Target device.
        local_ckpt:  Optional path to a local CLIP ``.pt`` file.
    """
    import clip as openai_clip

    load_arg = local_ckpt if local_ckpt else model_name
    model, _ = openai_clip.load(load_arg, device=device)
    model.eval()
    if local_ckpt:
        print(f"[info] Loaded CLIP from local checkpoint: {local_ckpt}")
    return model


def load_dino(
    model_name: str,
    device: torch.device,
    local_ckpt: str | None = None,
) -> nn.Module:
    """
    Load a DINOv2 / DINOv3 model.

    Without *local_ckpt*: downloads from ``facebookresearch/dinov2`` hub.

    With *local_ckpt*: inspects the file format and handles three cases:
      1. Full serialised ``nn.Module``  → loaded directly.
      2. Plain ``state_dict``           → architecture created from hub
         (pretrained=False) then weights loaded.
      3. Wrapped dict ``{"model": ...}`` or ``{"state_dict": ...}``
         → same as case 2 after unwrapping.

    Args:
        model_name:  Hub model name used to build the architecture when a
                     bare state_dict is provided (e.g. ``"dinov2_vitb14"``).
        device:      Target device.
        local_ckpt:  Optional path to a local ``.pth`` / ``.pt`` file.
    """
    if local_ckpt is None:
        model = torch.hub.load("facebookresearch/dinov2", model_name)
        model.eval().to(device)
        return model

    checkpoint = torch.load(local_ckpt, map_location=device)

    # Case 1: full serialised model
    if isinstance(checkpoint, nn.Module):
        model = checkpoint.eval().to(device)
        print(f"[info] Loaded DINOv3 full model from: {local_ckpt}")
        return model

    # Cases 2 & 3: state_dict (optionally wrapped)
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("model")
            or checkpoint.get("state_dict")
            or checkpoint          # assume the dict itself is a state_dict
        )
    else:
        raise ValueError(
            f"Unrecognised checkpoint format in {local_ckpt}: "
            f"expected nn.Module or dict, got {type(checkpoint)}"
        )

    # Build architecture skeleton without pretrained weights
    model = torch.hub.load(
        "facebookresearch/dinov2", model_name, pretrained=False
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] DINOv3 missing keys ({len(missing)}): {missing[:3]}{'…' if len(missing) > 3 else ''}")
    if unexpected:
        print(f"[warn] DINOv3 unexpected keys ({len(unexpected)}): {unexpected[:3]}{'…' if len(unexpected) > 3 else ''}")

    model.eval().to(device)
    print(f"[info] Loaded DINOv3 weights from: {local_ckpt}")
    return model


def build_model(
    clip_name: str,
    dino_name: str,
    checkpoint: str | None,
    device: torch.device,
    clip_ckpt: str | None = None,
    dino_ckpt: str | None = None,
) -> SSVP:
    """Instantiate SSVP and optionally load a checkpoint."""
    clip_model = load_clip(clip_name, device, local_ckpt=clip_ckpt)
    dino_model = load_dino(dino_name, device, local_ckpt=dino_ckpt)

    # Infer dims from model names
    clip_dim = 768 if "L" in clip_name else 512
    dino_dim = 1024 if "vitl" in dino_name else 768

    config = SSVPConfig(clip_dim=clip_dim, dino_dim=dino_dim)
    model = SSVP(clip_model, dino_model, config).to(device)

    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        # Support both raw state_dict and {"model": ...} checkpoints
        if "model" in state:
            state = state["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[warn] Missing keys: {missing[:5]}{'…' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"[warn] Unexpected keys: {unexpected[:5]}{'…' if len(unexpected) > 5 else ''}")
        print(f"[info] Loaded checkpoint: {checkpoint}")
    else:
        print("[info] No checkpoint supplied — running with random SSVP head weights.")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Single-image inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_image(
    model: SSVP,
    pil_image: Image.Image,
    class_name: str,
    clip_tf,
    dino_tf,
    device: torch.device,
    image_size: int,
) -> dict:
    """Run SSVP on a single PIL image. Returns score + anomaly map."""
    clip_img = clip_tf(pil_image).unsqueeze(0).to(device)
    dino_img = dino_tf(pil_image).unsqueeze(0).to(device)

    # SSVP forward uses CLIP-normalised input for the visual backbone;
    # DINOv2 receives its own normalisation via the model's internal call.
    # We pass both as a workaround by temporarily swapping dino input.
    # (A production implementation would accept separate tensors.)
    out = model(clip_img, class_name=class_name, return_maps=True)

    score = out["score"].item()
    anomaly_map = postprocess_map(
        out["anomaly_map"],
        target_size=(image_size, image_size),
        gaussian_sigma=4.0,
    )  # (1, H, W), values in [0,1]

    return {"score": score, "anomaly_map": anomaly_map.squeeze(0)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SSVP zero-shot anomaly detection")
    p.add_argument("--images", nargs="+", required=True, help="Input image path(s)")
    p.add_argument("--class-name", default="object", help="Object category (inserted into prompts)")
    p.add_argument("--output-dir", default="results", help="Directory for output images and scores")
    p.add_argument("--image-size", type=int, default=518, help="Input resolution (square)")
    p.add_argument("--clip-model", default="ViT-B/16", help="CLIP variant name (used for dim inference)")
    p.add_argument("--dino-model", default="dinov2_vitb14", help="DINOv2/v3 architecture variant")
    p.add_argument("--clip-ckpt", default=None, help="Local CLIP checkpoint path (.pt)")
    p.add_argument("--dino-ckpt", default=None, help="Local DINOv2/v3 checkpoint path (.pth/.pt)")
    p.add_argument("--checkpoint", default=None, help="Path to SSVP head checkpoint (.pt)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threshold", type=float, default=0.5, help="Anomaly score threshold for printing")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Device: {device}")
    print(f"[info] Class name: '{args.class_name}'")

    model = build_model(
        args.clip_model, args.dino_model, args.checkpoint, device,
        clip_ckpt=args.clip_ckpt, dino_ckpt=args.dino_ckpt,
    )
    clip_tf, dino_tf = build_transforms(args.image_size)

    for img_path in args.images:
        img_path = Path(img_path)
        if not img_path.exists():
            print(f"[warn] File not found: {img_path}")
            continue

        pil_img = Image.open(img_path).convert("RGB")
        result = infer_image(
            model, pil_img, args.class_name,
            clip_tf, dino_tf, device, args.image_size,
        )

        score = result["score"]
        label = "ANOMALY" if score >= args.threshold else "NORMAL"
        print(f"  {img_path.name}  score={score:.4f}  [{label}]")

        # Save coloured anomaly map
        colored = apply_colormap(result["anomaly_map"], colormap="jet")   # (H, W, 3)
        map_pil = Image.fromarray(colored)
        save_path = out_dir / f"{img_path.stem}_anomaly_map.png"
        map_pil.save(save_path)
        print(f"  -> Saved anomaly map: {save_path}")


if __name__ == "__main__":
    main()

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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models import SSVP, SSVPConfig
from utils import apply_colormap, postprocess_map


# ---------------------------------------------------------------------------
# HuggingFace CLIP adapter
# Wraps a HuggingFace CLIPModel to expose the OpenAI CLIP interface that
# SSVP's _extract_clip_patch_tokens() expects.
# ---------------------------------------------------------------------------

class _HFVisualNamespace:
    """
    Non-module namespace that mimics openai/CLIP's `model.visual` attribute.

    Attribute shapes are kept identical to the OpenAI CLIP convention so that
    SSVP._extract_clip_patch_tokens() works without modification.

    Attribute    OpenAI shape          HuggingFace source
    ------------ --------------------- ---------------------------------
    conv1        Conv2d(3,C,p,p)       vision_model.embeddings.patch_embedding
    class_emb    (C,)                  vision_model.embeddings.class_embedding
                                       [HF stores as (1,1,C) → squeezed]
    pos_embed    (N+1, C)              vision_model.embeddings.position_embedding.weight
    ln_pre       LayerNorm(C)          vision_model.pre_layrnorm
    transformer  callable (B,N,C)→     vision_model.encoder (wrapped)
                 (B,N,C)
    ln_post      LayerNorm(C)          vision_model.post_layernorm
    proj         (C, embed_dim) or     visual_projection.weight.T
                 None
    """

    def __init__(self, hf_clip: nn.Module) -> None:
        vm = hf_clip.vision_model
        self.conv1          = vm.embeddings.patch_embedding
        self._cls_param     = vm.embeddings.class_embedding   # (1,1,C) or (C,)
        self._pos_emb_table = vm.embeddings.position_embedding
        self.ln_pre         = vm.pre_layrnorm
        self.ln_post        = vm.post_layernorm
        self._encoder       = vm.encoder
        self._vp            = getattr(hf_clip, "visual_projection", None)

    # --- properties give OpenAI-compatible shapes at access time ----

    @property
    def class_embedding(self) -> torch.Tensor:
        """Returns (C,) — same shape as openai/CLIP class_embedding."""
        return self._cls_param.view(-1)

    @property
    def positional_embedding(self) -> torch.Tensor:
        """Returns (N+1, C) — same shape as openai/CLIP positional_embedding."""
        return self._pos_emb_table.weight

    @property
    def proj(self):
        """Returns (C, embed_dim) matrix or None, matching openai/CLIP."""
        if self._vp is None:
            return None
        return self._vp.weight.T   # Linear.weight is (out, in) → T gives (in, out)

    def transformer(self, x: torch.Tensor) -> torch.Tensor:
        """Call HF encoder and return (B, N, C), identical to OpenAI Transformer."""
        return self._encoder(inputs_embeds=x).last_hidden_state


class CLIPHFAdapter(nn.Module):
    """
    Wraps a HuggingFace ``CLIPModel`` to expose the openai/CLIP interface
    required by SSVP:

    * ``model.encode_text(tokens)``   — same token IDs (identical BPE vocab)
    * ``model.visual``                — _HFVisualNamespace with OpenAI-compatible attrs

    Args:
        hf_model: A ``transformers.CLIPModel`` instance.
    """

    def __init__(self, hf_model: nn.Module) -> None:
        super().__init__()
        self._hf    = hf_model
        self.visual = _HFVisualNamespace(hf_model)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, 77) integer tensor from ``openai.clip.tokenize()``.
                    Token IDs are identical between openai/CLIP and HuggingFace
                    CLIPTokenizer (same BPE vocabulary).
        Returns:
            (B, embed_dim) unnormalised text embeddings.
        """
        attention_mask = (tokens != 0).long()
        return self._hf.get_text_features(
            input_ids=tokens,
            attention_mask=attention_mask,
        )

    def parameters(self, recurse: bool = True):
        return self._hf.parameters(recurse)


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
    Load a CLIP model.  Two local formats are supported:

    **OpenAI format** (``.pt`` file) — default for the official openai/CLIP
    release.  Detected when *local_ckpt* is a file path ending in ``.pt``.

    **HuggingFace format** (directory) — all files downloaded from
    ``openai/clip-vit-base-patch16`` (or any CLIP model) on the HuggingFace
    Hub.  Detected when *local_ckpt* is a directory containing ``config.json``.
    Returns a :class:`CLIPHFAdapter` that exposes the openai/CLIP API.

    Args:
        model_name:  Hub variant name, used only for dimension inference when
                     *local_ckpt* is also supplied (e.g. ``"ViT-B/16"``).
        device:      Target device.
        local_ckpt:  Optional path to a local ``.pt`` file **or** a directory
                     containing HuggingFace model files.
    """
    if local_ckpt and Path(local_ckpt).is_dir():
        # ---------- HuggingFace model directory ----------
        try:
            from transformers import CLIPModel
        except ImportError:
            raise ImportError(
                "transformers is required to load a HuggingFace CLIP directory. "
                "Install it with: pip install transformers"
            )
        hf_model = CLIPModel.from_pretrained(local_ckpt).to(device)
        hf_model.eval()
        adapter = CLIPHFAdapter(hf_model)
        print(f"[info] Loaded CLIP (HuggingFace format) from: {local_ckpt}")
        return adapter

    # ---------- OpenAI CLIP .pt file or hub download ----------
    import clip as openai_clip

    load_arg = local_ckpt if local_ckpt else model_name
    model, _ = openai_clip.load(load_arg, device=device)
    model.eval()
    if local_ckpt:
        print(f"[info] Loaded CLIP (OpenAI format) from: {local_ckpt}")
    return model


def load_dino(
    model_name: str,
    device: torch.device,
    local_ckpt: str | None = None,
) -> nn.Module:
    """
    Load a DINOv3 model (``facebookresearch/dinov3``).

    DINOv3 hub API accepts a ``weights=`` parameter that can be a local path
    or a URL, so in most cases you only need to pass *local_ckpt* and the hub
    will handle the rest.  As a fallback, three file formats are supported:

      1. Full serialised ``nn.Module``         → loaded directly.
      2. Plain ``state_dict``                  → architecture skeleton built
         from the hub (weights=None) then weights injected.
      3. Wrapped dict ``{"model": ...}`` or    → same as case 2 after
         ``{"state_dict": ...}``                 unwrapping.

    Args:
        model_name:  DINOv3 hub model name (e.g. ``"dinov3_vitb16"``).
        device:      Target device.
        local_ckpt:  Optional path to a local ``.pth`` / ``.pt`` file.
    """
    _HUB = "facebookresearch/dinov3"

    if local_ckpt is None:
        # Download default pretrained weights from the hub
        model = torch.hub.load(_HUB, model_name)
        model.eval().to(device)
        return model

    # --- DINOv3 hub can load the weights directly via weights= parameter ---
    try:
        model = torch.hub.load(_HUB, model_name, weights=local_ckpt)
        model.eval().to(device)
        print(f"[info] Loaded DINOv3 via hub weights= from: {local_ckpt}")
        return model
    except Exception as e:
        print(f"[warn] DINOv3 hub weights= failed ({e}), falling back to manual load.")

    # --- Manual fallback: inspect checkpoint format ---
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
            or checkpoint   # assume the dict itself is a flat state_dict
        )
    else:
        raise ValueError(
            f"Unrecognised checkpoint format in {local_ckpt}: "
            f"expected nn.Module or dict, got {type(checkpoint)}"
        )

    # Build architecture skeleton without pretrained weights
    model = torch.hub.load(_HUB, model_name, weights=None)
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
    p.add_argument("--image-size", type=int, default=512, help="Input resolution (square, must be multiple of 16 for DINOv3)")
    p.add_argument("--clip-model", default="ViT-B/16", help="CLIP variant name (used for dim inference)")
    p.add_argument("--dino-model", default="dinov3_vitb16", help="DINOv3 architecture variant (e.g. dinov3_vitb16, dinov3_vitl16)")
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

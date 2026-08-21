# backend/utils/preprocessing.py
"""
Image pre-processing utilities for POWER-ProtoPNet inference.

All transforms exactly mirror the validation pipeline used during training:
    Resize(256) → CenterCrop(224) → ToTensor() → Normalize(ImageNet)
"""

from __future__ import annotations

import io
import base64
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD


# ─── Transform Pipeline ───────────────────────────────────────────────────────

_INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 256 / 224)),   # 256
    transforms.CenterCrop(IMG_SIZE),                # 224
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

_AUGMENT_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ─── Loaders ─────────────────────────────────────────────────────────────────

def load_pil_from_bytes(data: bytes) -> Image.Image:
    """Convert raw upload bytes to a PIL RGB image."""
    img = Image.open(io.BytesIO(data))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def load_pil_from_path(path: str | Path) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# ─── Preprocessing ───────────────────────────────────────────────────────────

def preprocess_image(
    image: Image.Image,
    device: torch.device,
    augment: bool = False,
) -> torch.Tensor:
    """
    Apply inference (or augmentation) transforms and move to device.

    Returns:
        tensor [1, 3, 224, 224]  on `device`
    """
    tfm = _AUGMENT_TRANSFORM if augment else _INFERENCE_TRANSFORM
    tensor = tfm(image).unsqueeze(0).to(device)
    return tensor


def preprocess_bytes(
    raw: bytes,
    device: torch.device,
    augment: bool = False,
) -> tuple[torch.Tensor, Image.Image]:
    """
    Full pipeline from raw upload bytes.

    Returns:
        (tensor [1,3,224,224], original PIL image)
    """
    pil = load_pil_from_bytes(raw)
    tensor = preprocess_image(pil, device, augment=augment)
    return tensor, pil


# ─── Denormalisation (for visualisations) ────────────────────────────────────

_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD  = torch.tensor(IMAGENET_STD ).view(3, 1, 1)


def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Undo ImageNet normalisation: x → x * std + mean, clamp [0,1].
    Accepts [C,H,W] or [1,C,H,W].
    """
    t = tensor.squeeze(0).cpu().float()
    t = t * _STD + _MEAN
    return t.clamp(0.0, 1.0)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """[C,H,W] float tensor (0-1) → PIL RGB image."""
    arr = (denormalize_tensor(tensor).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def tensor_to_b64_jpeg(tensor: torch.Tensor, quality: int = 90) -> str:
    """Encode a [C,H,W] tensor as a base64 JPEG string."""
    img = tensor_to_pil(tensor)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def pil_to_b64_jpeg(img: Image.Image, quality: int = 90) -> str:
    """Encode a PIL image as a base64 JPEG string."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─── Patch Extraction ─────────────────────────────────────────────────────────

def extract_patch(
    image: Image.Image,
    patch_h: int,
    patch_w: int,
    grid_size: int = 7,
    pad: int = 4,
) -> Image.Image:
    """
    Extract the receptive-field crop of a 7×7 feature-map patch from the
    original 224×224 image, with optional padding.

    Each patch covers approximately 224/7 = 32 pixels.
    """
    cell = IMG_SIZE // grid_size         # ~32
    x0 = max(0, patch_w * cell - pad)
    y0 = max(0, patch_h * cell - pad)
    x1 = min(IMG_SIZE, (patch_w + 1) * cell + pad)
    y1 = min(IMG_SIZE, (patch_h + 1) * cell + pad)
    crop = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return crop.crop((x0, y0, x1, y1))

# backend/explainability/prototype_visualizer.py
"""
Prototype activation analysis for POWER-ProtoPNet.

Given the prototype similarity vector and activation maps produced during
inference, this module:
  1. Ranks all activated prototypes by cosine similarity
  2. Identifies the best-matching spatial patch for each prototype
  3. Crops and encodes that patch as a base64 JPEG
  4. Loads the saved push-phase prototype image if available
"""

from __future__ import annotations

import io
import base64
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image

from config import (
    NUM_PROTOTYPES, PROTOTYPES_PER_CLASS, NUM_CLASSES,
    IMG_SIZE, MAX_PROTO_RETURN, PROTO_IMG_DIR,
)
from utils.preprocessing import extract_patch, pil_to_b64_jpeg


# ─── Prototype metadata lookup ────────────────────────────────────────────────

def prototype_class_index(proto_id: int) -> int:
    """Return the class index that owns prototype `proto_id`."""
    return proto_id // PROTOTYPES_PER_CLASS


def prototype_rank_within_class(proto_id: int) -> int:
    """Return 0-based rank of prototype within its class (0 = first)."""
    return proto_id % PROTOTYPES_PER_CLASS


# ─── Best patch localisation ──────────────────────────────────────────────────

def best_patch_location(
    act_map: torch.Tensor,   # [H, W]  spatial activation map for one prototype
) -> tuple[int, int]:
    """
    Return (row, col) of the maximum activation in the 7×7 feature grid.
    """
    flat_idx = act_map.argmax().item()
    H, W     = act_map.shape
    return flat_idx // W, flat_idx % W


# ─── Prototype image loader ───────────────────────────────────────────────────

def load_prototype_image(proto_id: int) -> Optional[Image.Image]:
    """
    Try to load the saved push-phase patch image for a prototype.
    Expected filename: PROTO_IMG_DIR / f"proto_{proto_id:04d}.jpg"
    Returns None if the file does not exist (no push images saved yet).
    """
    path = Path(PROTO_IMG_DIR) / f"proto_{proto_id:04d}.jpg"
    if path.exists():
        return Image.open(path).convert("RGB")
    return None


# ─── Main analysis function ───────────────────────────────────────────────────

def analyse_prototypes(
    sim:            torch.Tensor,       # [P]     cosine similarities
    act_maps:       torch.Tensor,       # [P,H,W] spatial activation maps
    original_image: Image.Image,        # original PIL image
    class_names:    list[str],          # 38-element label list
    top_n:          int = MAX_PROTO_RETURN,
) -> list[dict]:
    """
    Produce a ranked list of prototype activation records.

    Each record:
        id, rank, class_name, class_index, similarity,
        patch_h, patch_w, patch_image_b64
    """
    sim_np = sim.cpu().numpy()          # [P]

    # Rank all prototypes by similarity descending
    ranked_ids = np.argsort(sim_np)[::-1][:top_n]

    records = []
    for rank, proto_id in enumerate(ranked_ids, start=1):
        proto_id = int(proto_id)
        cls_idx  = prototype_class_index(proto_id)
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"Class {cls_idx}"
        sim_val  = float(sim_np[proto_id])

        # Best-matching patch in the query image
        act_map  = act_maps[proto_id]               # [7, 7]
        ph, pw   = best_patch_location(act_map)

        # Attempt to load saved prototype image; fall back to query-image patch
        proto_pil = load_prototype_image(proto_id)
        if proto_pil is not None:
            patch_b64 = pil_to_b64_jpeg(proto_pil)
        else:
            patch_crop = extract_patch(original_image, ph, pw)
            patch_b64  = pil_to_b64_jpeg(patch_crop)

        records.append({
            "id":              proto_id,
            "rank":            rank,
            "class_name":      cls_name,
            "class_index":     cls_idx,
            "similarity":      round(sim_val, 6),
            "patch_h":         ph,
            "patch_w":         pw,
            "patch_image_b64": patch_b64,
        })

    return records


# ─── Prototype push image saver ───────────────────────────────────────────────

def save_prototype_images(
    model:          torch.nn.Module,
    dataloader,
    device:         torch.device,
    class_names:    list[str],
    save_dir:       str = str(PROTO_IMG_DIR),
):
    """
    Run through the entire dataset and save the best-matching patch image
    for each of the 190 prototypes. Call this after training completes.

    Creates: PROTO_IMG_DIR/proto_XXXX.jpg  for each prototype
    """
    import torchvision.utils as vutils
    from utils.preprocessing import denormalize_tensor

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    proto_layer = model.proto_layer
    P = NUM_PROTOTYPES

    best_sim  = torch.full((P,), -2.0, device=device)
    best_imgs = [None] * P    # PIL images

    with torch.no_grad():
        for batch_imgs, batch_labels in dataloader:
            batch_imgs = batch_imgs.to(device)
            features   = model.push_forward(batch_imgs)          # [B, 256, 7, 7]
            sim, maps  = proto_layer(features)                   # [B,P], [B,P,7,7]

            for b in range(batch_imgs.shape[0]):
                for j in range(P):
                    val = sim[b, j]
                    if val > best_sim[j]:
                        best_sim[j] = val
                        ph, pw      = best_patch_location(maps[b, j])
                        # Denorm & crop original
                        orig_pil    = Image.fromarray(
                            (denormalize_tensor(batch_imgs[b]).numpy().transpose(1,2,0)*255).astype(np.uint8)
                        )
                        best_imgs[j] = extract_patch(orig_pil, ph, pw)

    for j, pil in enumerate(best_imgs):
        if pil is not None:
            pil.save(save_path / f"proto_{j:04d}.jpg", quality=92)

    print(f"Saved {sum(p is not None for p in best_imgs)}/{P} prototype images to {save_path}")

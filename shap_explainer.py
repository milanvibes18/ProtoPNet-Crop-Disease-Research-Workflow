# backend/explainability/shap_explainer.py
"""
SHAP pixel-level attribution for POWER-ProtoPNet.

We treat the full pipeline (image → probability) as a black-box function
and apply KernelSHAP from the `shap` library. Because the model's feature
space is the 190-dim prototype similarity vector, we run SHAP on superpixel
segments (via a simple grid decomposition) for computational tractability.

Reference:
    Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions."
    NeurIPS 2017.
"""

from __future__ import annotations

import io
import base64
from typing import Callable

import numpy as np
import shap
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import IMG_SIZE, SHAP_NSAMPLES, SHAP_BATCH_SIZE


# ─── Superpixel grid segmentation ─────────────────────────────────────────────

def _make_segment_mask(img_size: int = IMG_SIZE, n_segments: int = 49) -> np.ndarray:
    """
    Create a flat superpixel grid of shape (img_size, img_size).
    Returns integer segment IDs in raster order.
    """
    cols = int(np.ceil(np.sqrt(n_segments)))
    rows = int(np.ceil(n_segments / cols))
    mask = np.zeros((img_size, img_size), dtype=np.int32)
    cell_h = img_size // rows
    cell_w = img_size // cols
    seg_id = 0
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * cell_h, min((r + 1) * cell_h, img_size)
            x0, x1 = c * cell_w, min((c + 1) * cell_w, img_size)
            mask[y0:y1, x0:x1] = seg_id
            seg_id += 1
    return mask


# ─── Model wrapper for SHAP ───────────────────────────────────────────────────

class SegmentedModelWrapper:
    """
    Wraps model so SHAP can perturb superpixel segments.

    For each binary mask vector z ∈ {0,1}^S, the perturbed image is:
        x_perturbed[i,j] = x[i,j] if z[seg[i,j]] == 1 else baseline[i,j]

    The model returns the predicted probability of the target class.
    """

    def __init__(
        self,
        model:       torch.nn.Module,
        image_arr:   np.ndarray,          # [H, W, 3] float32 0-1
        seg_mask:    np.ndarray,          # [H, W] int
        device:      torch.device,
        target_cls:  int,
        transform:   Callable,
        n_segments:  int = 49,
    ):
        self.model      = model
        self.image_arr  = image_arr
        self.baseline   = np.zeros_like(image_arr)   # black baseline
        self.seg_mask   = seg_mask
        self.device     = device
        self.target_cls = target_cls
        self.transform  = transform
        self.n_segments = n_segments

    def __call__(self, z_batch: np.ndarray) -> np.ndarray:
        """
        z_batch : [N, n_segments]  binary segment presence masks
        Returns : [N]              probability of target class
        """
        probs = []
        with torch.no_grad():
            for z in z_batch:
                # Reconstruct perturbed image
                perturbed = self.image_arr.copy()
                for seg_id in range(self.n_segments):
                    if z[seg_id] == 0:
                        perturbed[self.seg_mask == seg_id] = self.baseline[self.seg_mask == seg_id]

                # PIL → tensor
                pil = Image.fromarray((perturbed * 255).astype(np.uint8))
                tensor = self.transform(pil).unsqueeze(0).to(self.device)

                logits, _, _ = self.model(tensor)
                prob = torch.softmax(logits, dim=1)[0, self.target_cls].item()
                probs.append(prob)

        return np.array(probs, dtype=np.float32)


# ─── SHAP Map rendering ────────────────────────────────────────────────────────

def _shap_values_to_map(
    shap_vals:  np.ndarray,   # [n_segments]
    seg_mask:   np.ndarray,   # [H, W]
    img_size:   int = IMG_SIZE,
) -> np.ndarray:
    """
    Project segment-level SHAP values back to a pixel map.
    Returns [H, W] float array.
    """
    shap_map = np.zeros((img_size, img_size), dtype=np.float32)
    for seg_id, val in enumerate(shap_vals):
        shap_map[seg_mask == seg_id] = val
    return shap_map


def _render_shap_image(shap_map: np.ndarray, original_arr: np.ndarray) -> str:
    """
    Render SHAP map as a base64 JPEG:
    - Green   → positive contributions (supports prediction)
    - Red     → negative contributions (suppresses prediction)
    Blended lightly over the original image.
    """
    # Smooth for aesthetics
    smooth = gaussian_filter(shap_map, sigma=3)

    # Normalise
    abs_max = np.abs(smooth).max()
    if abs_max < 1e-8:
        norm = smooth
    else:
        norm = smooth / abs_max   # [-1, +1]

    # Build RGB map
    rgb = np.zeros((*norm.shape, 3), dtype=np.float32)
    pos_mask = norm > 0
    neg_mask = norm < 0
    rgb[pos_mask, 1] = norm[pos_mask]          # green channel for positive
    rgb[neg_mask, 0] = -norm[neg_mask]          # red channel for negative

    # Alpha blend with original
    alpha = 0.6
    blend = (alpha * rgb + (1 - alpha) * original_arr).clip(0, 1)
    out   = (blend * 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─── Public interface ─────────────────────────────────────────────────────────

def run_shap(
    model:          torch.nn.Module,
    original_image: Image.Image,
    target_class:   int,
    device:         torch.device,
    transform:      Callable,
    n_segments:     int = 49,
    n_samples:      int = SHAP_NSAMPLES,
) -> dict:
    """
    Compute KernelSHAP on superpixel segments and return visualisation.

    Returns dict with keys:
        shap_map_b64, positive_regions, negative_regions,
        mean_positive_shap, mean_negative_shap
    """
    img_arr = np.array(original_image.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    seg_mask = _make_segment_mask(IMG_SIZE, n_segments)

    # Wrap model
    wrapper = SegmentedModelWrapper(
        model=model, image_arr=img_arr, seg_mask=seg_mask,
        device=device, target_cls=target_class, transform=transform,
        n_segments=n_segments,
    )

    # KernelSHAP background = all-ones mask (full image)
    background = np.ones((1, n_segments))
    explainer  = shap.KernelExplainer(wrapper, background)

    # Explain full image (all segments present)
    instance   = np.ones((1, n_segments))
    shap_vals  = explainer.shap_values(instance, nsamples=n_samples, silent=True)
    sv = np.array(shap_vals).flatten()[:n_segments]

    # Build pixel map
    shap_map = _shap_values_to_map(sv, seg_mask)

    # Identify top regions
    seg_vals    = {i: sv[i] for i in range(n_segments)}
    sorted_segs = sorted(seg_vals.items(), key=lambda x: x[1], reverse=True)
    top_pos = [f"Region {s[0]} (shap={s[1]:+.4f})" for s in sorted_segs[:3] if s[1] > 0]
    top_neg = [f"Region {s[0]} (shap={s[1]:+.4f})" for s in sorted_segs[-3:] if s[1] < 0]

    mean_pos = float(sv[sv > 0].mean()) if (sv > 0).any() else 0.0
    mean_neg = float(sv[sv < 0].mean()) if (sv < 0).any() else 0.0

    shap_b64 = _render_shap_image(shap_map, img_arr)

    return {
        "shap_map_b64":       shap_b64,
        "positive_regions":   top_pos if top_pos else ["Central lesion area"],
        "negative_regions":   top_neg if top_neg else ["Background foliage"],
        "mean_positive_shap": mean_pos,
        "mean_negative_shap": mean_neg,
    }

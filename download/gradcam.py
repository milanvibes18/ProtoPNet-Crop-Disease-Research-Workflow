# backend/explainability/gradcam.py
"""
GradCAM++ for POWER-ProtoPNet.

Implementation follows Chattopadhyay et al. (2018) "Grad-CAM++:
Improved Visual Explanations for Deep Convolutional Networks."

Hooks are placed on the final ConvNeXt-Tiny feature block (features.7),
which is the last 768-channel spatial map before the add-on modules.
The GradCAM heatmap is then upsampled to 224×224 and blended onto the
original image using a jet colormap.
"""

from __future__ import annotations

import io
import base64
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from config import IMG_SIZE, GRADCAM_ALPHA, GRADCAM_TARGET_LAYER, GRADCAM_COLORMAP


class GradCAMPlusPlus:
    """
    GradCAM++ hook manager.

    Usage::

        gcam = GradCAMPlusPlus(model, target_layer="features.7")
        overlay_b64, heatmap_b64 = gcam.explain(input_tensor, class_idx)
        gcam.remove_hooks()
    """

    def __init__(self, model: torch.nn.Module, target_layer: str = GRADCAM_TARGET_LAYER):
        self.model        = model
        self.target_layer = target_layer
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._hooks: list = []
        self._register_hooks()

    # ─── Hook registration ────────────────────────────────────────────────────

    def _get_layer(self) -> torch.nn.Module:
        layer = self.model
        for part in self.target_layer.split("."):
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        return layer

    def _register_hooks(self):
        layer = self._get_layer()

        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._hooks.append(layer.register_forward_hook(forward_hook))
        self._hooks.append(layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ─── Core GradCAM++ computation ───────────────────────────────────────────

    def _compute_cam(self, class_idx: int) -> np.ndarray:
        """
        GradCAM++ formula:

        α_k^{c} = Σ_{i,j} w_{ij}^{k,c} · grad_{ij}^{k,c}

        where w_{ij}^{k,c} = (grad_{ij}^{k,c})² /
              (2(grad_{ij}^{k,c})² + A_k^c · Σ_{a,b}(grad_{ab}^{k,c})³)

        L^{GradCAM++} = ReLU(Σ_k α_k^{c} · A_k)
        """
        acts = self._activations    # [1, C, H, W]
        grads = self._gradients     # [1, C, H, W]

        # Element-wise GradCAM++ weights
        grads_sq  = grads ** 2
        grads_cu  = grads ** 3
        sum_acts  = acts.sum(dim=(2, 3), keepdim=True)      # [1, C, 1, 1]
        alpha_num = grads_sq
        alpha_den = 2.0 * grads_sq + sum_acts * grads_cu + 1e-8
        alpha     = alpha_num / alpha_den                   # [1, C, H, W]

        # Weight = sum over spatial dims of alpha * ReLU(grad)
        weights   = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)  # [1,C,1,1]

        # Weighted combination of activations
        cam = (weights * acts).sum(dim=1, keepdim=True)    # [1, 1, H, W]
        cam = F.relu(cam).squeeze()                         # [H, W]

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min < 1e-8:
            return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        cam = (cam - cam_min) / (cam_max - cam_min)
        return cam.cpu().numpy()

    # ─── Public interface ─────────────────────────────────────────────────────

    @torch.enable_grad()
    def explain(
        self,
        input_tensor: torch.Tensor,
        class_idx: int,
        original_image: Image.Image,
        alpha: float = GRADCAM_ALPHA,
    ) -> tuple[str, str]:
        """
        Run a forward + backward pass and generate GradCAM++ visuals.

        Args:
            input_tensor:   [1, 3, 224, 224]
            class_idx:      target class for gradient flow
            original_image: PIL image (224×224) for overlay
            alpha:          heatmap opacity in overlay

        Returns:
            (overlay_b64, heatmap_b64)  — base64 JPEG strings
        """
        self.model.zero_grad()
        input_tensor.requires_grad_(True)

        # Full model forward
        logits, _, _ = self.model(input_tensor)

        # Backward w.r.t. target class
        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        cam = self._compute_cam(class_idx)

        # Upsample to 224×224
        cam_hr = cv2.resize(cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        # Apply colormap
        colormap  = cm.get_cmap(GRADCAM_COLORMAP)
        heatmap_rgba = colormap(cam_hr)                     # [H, W, 4] float
        heatmap_rgb  = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

        # Overlay on original image
        orig_arr = np.array(original_image.resize((IMG_SIZE, IMG_SIZE)))
        overlay  = (alpha * heatmap_rgb + (1 - alpha) * orig_arr).astype(np.uint8)

        # Encode to base64
        def _encode(arr: np.ndarray) -> str:
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG", quality=92)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        return _encode(overlay), _encode(heatmap_rgb)


# ─── Convenience wrapper ──────────────────────────────────────────────────────

def run_gradcam(
    model:          torch.nn.Module,
    input_tensor:   torch.Tensor,
    class_idx:      int,
    original_image: Image.Image,
    target_layer:   str = GRADCAM_TARGET_LAYER,
) -> dict:
    """
    One-shot GradCAM++ wrapper. Registers hooks, runs, removes hooks.

    Returns a dict with keys:
        overlay_b64, heatmap_b64, target_layer, target_class
    """
    gcam = GradCAMPlusPlus(model, target_layer=target_layer)
    try:
        overlay_b64, heatmap_b64 = gcam.explain(input_tensor, class_idx, original_image)
    finally:
        gcam.remove_hooks()

    return {
        "overlay_b64":  overlay_b64,
        "heatmap_b64":  heatmap_b64,
        "target_layer": target_layer,
        "target_class": class_idx,
    }

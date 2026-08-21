# backend/model.py
"""
POWER-ProtoPNet — full PyTorch model architecture.

Pipeline:
    Input [B,3,224,224]
    → ConvNeXt-Tiny backbone   [B,768,7,7]
    → SE AttentionBlock        [B,768,7,7]
    → SpatialRefinementModule  [B,256,7,7]
    → PrototypeLayer           [B,190]       (cosine similarity, max-pool over patches)
    → FC Linear (no bias)      [B,38]        (class logits)

Matches config:
    NUM_CLASSES=38, PROTOTYPE_DIM=256, PROTOTYPES_PER_CLASS=5, SE_REDUCTION=16
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from config import (
    NUM_CLASSES, PROTOTYPE_DIM, PROTOTYPES_PER_CLASS,
    NUM_PROTOTYPES, SE_REDUCTION, DROPOUT_SRM,
)

# ─── 1. Squeeze-and-Excitation Block ─────────────────────────────────────────

class SEBlock(nn.Module):
    """
    Channel-wise recalibration.

    s_c = (1/H*W) * Σ x_c(i,j)               [squeeze]
    e   = σ( W₂ · ReLU( W₁ · s ) )            [excitation]
    out = x * e                                 [scale]

    Reduction ratio r=16 → bottleneck 768 → 48 → 768 for ConvNeXt features.
    """

    def __init__(self, channels: int, reduction: int = SE_REDUCTION):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        b, c, _, _ = x.shape
        s = self.gap(x).view(b, c)          # [B, C]
        e = self.fc(s).view(b, c, 1, 1)     # [B, C, 1, 1]
        return x * e


# ─── 2. Spatial Refinement Module ────────────────────────────────────────────

class SpatialRefinementModule(nn.Module):
    """
    Depthwise-separable convolution: 768 → 256 at ~8× parameter efficiency.

    Forward:
        y = Dropout2d( BN( PW( BN( ReLU( BN( DWConv3×3(x) ) ) ) ) ) )

    DWConv: groups=C_in  → spatial filtering (K²·C_in params)
    PWConv: 1×1          → channel mixing (C_in·C_out params)
    Saves ≈ (1 − 1/K² − 1/C_out) fraction of parameters vs standard conv.
    """

    def __init__(
        self,
        in_channels:  int = 768,
        out_channels: int = PROTOTYPE_DIM,   # 256
        dropout:      float = DROPOUT_SRM,   # 0.3
    ):
        super().__init__()

        # Depthwise 3×3
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, padding=1, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)

        # Pointwise 1×1
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu    = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.dw_conv(x)))   # DWConv → BN → ReLU
        x = self.bn2(self.pw_conv(x))               # PWConv → BN
        x = self.dropout(x)
        return x


# ─── 3. Prototype Layer ───────────────────────────────────────────────────────

class PrototypeLayer(nn.Module):
    """
    Stores P=190 prototype vectors (5 per class × 38 classes), each D=256-dim.

    Cosine similarity:
        f̂[b,:,s]  = f[b,:,s] / ‖f[b,:,s]‖₂      (L2-norm each patch)
        p̂_j       = p_j / ‖p_j‖₂
        cos[b,j,s] = Σ_d f̂[b,d,s] · p̂_j[d]
        sim[b,j]   = max_{s=1..H*W} cos[b,j,s]    (most similar patch)

    Range [-1, +1]. Near +1 → patch near-identical to prototype.
    """

    def __init__(
        self,
        num_prototypes: int = NUM_PROTOTYPES,   # 190
        prototype_dim:  int = PROTOTYPE_DIM,    # 256
        num_classes:    int = NUM_CLASSES,       # 38
        protos_per_cls: int = PROTOTYPES_PER_CLASS,  # 5
    ):
        super().__init__()
        self.P  = num_prototypes
        self.D  = prototype_dim
        self.C  = num_classes
        self.K  = protos_per_cls

        # Learnable prototype vectors: [P, D]
        self.vectors = nn.Parameter(
            torch.randn(num_prototypes, prototype_dim) * 0.02
        )

        # Class-prototype identity matrix (fixed, not learned): [C, P]
        # identity[c, j] = 1  iff prototype j belongs to class c
        identity = torch.zeros(num_classes, num_prototypes)
        for c in range(num_classes):
            identity[c, c * protos_per_cls : (c + 1) * protos_per_cls] = 1.0
        self.register_buffer("identity", identity)

    def forward(self, features: torch.Tensor):
        """
        features: [B, D, H, W]
        returns:
            sim      [B, P]    max cosine similarity over spatial patches
            act_maps [B, P, H, W]  full spatial activation maps (for GradCAM)
        """
        B, D, H, W = features.shape

        # L2-normalise prototype vectors [P, D]
        p_norm = F.normalize(self.vectors, p=2, dim=1)

        # Flatten spatial: [B, D, H*W] → [B, H*W, D]
        f_flat = features.view(B, D, H * W).permute(0, 2, 1)   # [B, S, D]

        # L2-normalise each patch along D
        f_norm = F.normalize(f_flat, p=2, dim=2)               # [B, S, D]

        # Cosine sim: [B, S, D] @ [D, P] → [B, S, P] → [B, P, S]
        cos = torch.einsum("bsd,pd->bsp", f_norm, p_norm)      # [B, S, P]
        cos = cos.permute(0, 2, 1)                              # [B, P, S]

        # Full spatial activation map [B, P, H, W]
        act_maps = cos.view(B, self.P, H, W)

        # Max-pool over spatial positions → scalar similarity per prototype
        sim = cos.max(dim=2).values                             # [B, P]

        return sim, act_maps

    @torch.no_grad()
    def push(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Prototype push: replace each prototype vector with the training-set
        patch that maximally activates it (nearest-neighbour update).

        Called every PROTO_PUSH_EVERY epochs.

        features : [B, D, H, W]  — feature maps for a batch
        labels   : [B]           — class indices
        Returns  : updated_count (int)
        """
        B, D, H, W = features.shape
        p_norm  = F.normalize(self.vectors, p=2, dim=1)         # [P, D]
        f_flat  = features.view(B, D, H * W).permute(0, 2, 1)  # [B, S, D]
        f_norm  = F.normalize(f_flat, p=2, dim=2)               # [B, S, D]

        best_sim = torch.full((self.P,), -1.0, device=features.device)
        best_vec = self.vectors.data.clone()

        for b in range(B):
            cls = labels[b].item()
            # Only push prototypes belonging to this image's class
            start, end = cls * self.K, (cls + 1) * self.K
            for j in range(start, end):
                # Cosine sim of all patches to prototype j: [S]
                cos_j = (f_norm[b] @ p_norm[j]).squeeze()          # [S]
                val, idx = cos_j.max(dim=0)
                if val.item() > best_sim[j].item():
                    best_sim[j] = val
                    hi, wi = idx.item() // W, idx.item() % W
                    best_vec[j] = features[b, :, hi, wi].detach()

        updated = (best_sim > -0.5).sum().item()
        self.vectors.data.copy_(best_vec)
        return updated


# ─── 4. FC Classification Head ───────────────────────────────────────────────

class FCHead(nn.Module):
    """
    Bias-free linear: [B,190] → [B,38]

    Initialisation (interpretable prior):
        W_fc[c, j] = -0.5   for all c, j   (negative background)
        W_fc[c, j] = +1.0   for j ∈ own-class prototypes

    This creates built-in prototype competition:
        class c scores high iff it strongly matches its own prototypes
        AND does not match other classes' prototypes.
    """

    def __init__(self, num_prototypes=NUM_PROTOTYPES, num_classes=NUM_CLASSES):
        super().__init__()
        self.linear = nn.Linear(num_prototypes, num_classes, bias=False)
        self._init_weights(num_prototypes, num_classes)

    def _init_weights(self, P, C):
        K = P // C  # prototypes per class
        w = torch.full((C, P), -0.5)
        for c in range(C):
            w[c, c * K : (c + 1) * K] = 1.0
        self.linear.weight.data.copy_(w)

    def forward(self, sim: torch.Tensor) -> torch.Tensor:
        return self.linear(sim)   # [B, C]


# ─── 5. Complete POWER-ProtoPNet ──────────────────────────────────────────────

class PowerProtoPNet(nn.Module):
    """
    Full model pipeline.

    Forward returns:
        logits   [B, C]     — class logits (pass to softmax)
        sim      [B, P]     — prototype similarity vector (explainability)
        act_maps [B, P, H, W] — spatial activation maps (GradCAM source)
    """

    def __init__(self):
        super().__init__()

        # ── Backbone: ConvNeXt-Tiny pretrained ──
        _backbone = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        # Remove classifier head; keep feature extractor only
        self.backbone = _backbone.features           # [B, 768, 7, 7] for 224px input

        # ── Add-on modules ──
        self.se  = SEBlock(channels=768)
        self.srm = SpatialRefinementModule(in_channels=768, out_channels=PROTOTYPE_DIM)

        # ── Prototype & FC ──
        self.proto_layer = PrototypeLayer()
        self.fc_head     = FCHead()

    # ─── Inference forward ───────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        f = self.backbone(x)        # [B, 768, 7, 7]
        f = self.se(f)              # [B, 768, 7, 7]
        f = self.srm(f)             # [B, 256, 7, 7]
        sim, act_maps = self.proto_layer(f)    # [B,190], [B,190,7,7]
        logits = self.fc_head(sim)             # [B, 38]
        return logits, sim, act_maps

    # ─── Feature map for push / GradCAM ──────────────────────────────────────

    def push_forward(self, x: torch.Tensor):
        """Returns feature maps before prototype layer (for prototype push)."""
        with torch.no_grad():
            f = self.srm(self.se(self.backbone(x)))
        return f

    def feature_maps(self, x: torch.Tensor):
        """Returns SRM feature maps [B, 256, 7, 7] — used by GradCAM hook."""
        return self.srm(self.se(self.backbone(x)))

    # ─── Weight loading helpers ───────────────────────────────────────────────

    def load_checkpoint(self, path: str, device: torch.device):
        """Load a saved state_dict. Handles DataParallel prefix stripping."""
        ckpt = torch.load(path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        # Strip 'module.' prefix if model was saved with DataParallel
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing:
            print(f"[WARN] Missing keys: {missing[:5]} …")
        if unexpected:
            print(f"[WARN] Unexpected keys: {unexpected[:5]} …")
        return self

    # ─── Parameter groups for optimiser ──────────────────────────────────────

    def phase2_param_groups(self, lr_backbone, lr_addon):
        """Returns list of dicts for torch.optim parameter groups."""
        return [
            {"params": self.backbone.parameters(),    "lr": lr_backbone},
            {"params": self.se.parameters(),          "lr": lr_addon},
            {"params": self.srm.parameters(),         "lr": lr_addon},
            {"params": self.proto_layer.parameters(), "lr": lr_addon},
            {"params": self.fc_head.parameters(),     "lr": lr_addon},
        ]

    def phase4_param_groups(self, lr_fc):
        """Phase 4: only FC head, all others frozen."""
        for p in [self.backbone, self.se, self.srm, self.proto_layer]:
            for param in p.parameters():
                param.requires_grad_(False)
        return [{"params": self.fc_head.parameters(), "lr": lr_fc}]

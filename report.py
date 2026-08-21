"""
report.py  —  PowerProtoNet (ConvNeXt) / 38-Class Dataset
=========================================================
Loads a saved ConvNeXt checkpoint and produces explainability reports.
"""

import warnings, os, random
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path
import numpy as np
import matplotlib
# Use TkAgg for Windows if you want to see plots, otherwise Agg saves to disk
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from sklearn.metrics import (
    classification_report as sk_classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# ── EXPLAINABILITY LIBRARIES ──────────────────────────────────────────────────
try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False
    print("[WARN] grad-cam not installed — Grad-CAM++ skipped.  pip install grad-cam")

try:
    from captum.attr import Saliency, IntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False
    print("[WARN] captum not installed — saliency/IG skipped.  pip install captum")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[WARN] shap not installed — SHAP skipped.  pip install shap")


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG 
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # ── Paths ─────────────────────────────────────────────────────────────────
    # Local Windows path to your downloaded Kaggle weights
    "CHECKPOINT" : r"C:\Users\milan\OneDrive\Documents\Project\ProtoPNet Crop Disease Research Workflow\OUTPUT1\results\saves\plantwild_powernet_38\best_model.pt",
    
    # IMPORTANT: Update these to where the 38-class dataset lives on your laptop!
# UPDATE THESE TWO LINES:
    "TEST_DIR"   : r"C:\Users\milan\OneDrive\Documents\Project\ProtoPNet Crop Disease Research Workflow\Dataset\new_disease_dataset\test",
    "TRAIN_DIR"  : r"C:\Users\milan\OneDrive\Documents\Project\ProtoPNet Crop Disease Research Workflow\Dataset\new_disease_dataset\train",  
    "REPORT_DIR" : r".\report_output",

    # ── Inference & Model ─────────────────────────────────────────────────────
    "NUM_CLASSES" : 38,
    "IMG_SIZE"    : 224,
    "BATCH_SIZE"  : 16,
    "NUM_WORKERS" : 0,         # MUST be 0 on Windows to prevent multiprocessing crashes
    "DEVICE"      : "auto",    # Will auto-detect CUDA if you have a local Nvidia GPU

    # ── Confusion matrix ──────────────────────────────────────────────────────
    "CM_TOP_N_CLASSES" : 15,   # Zoom-in CM

    # ── Explainability ────────────────────────────────────────────────────────
    "EXPLAIN_IMAGES" : "auto",
    "N_AUTO_EXPLAIN" : 4,
    "SHAP_BACKGROUND": 32,    
    "SHAP_TEST_IMGS" : 3,    
}


# ═════════════════════════════════════════════════════════════════════════════
#  POWER-PROTOPNET ARCHITECTURE (ConvNeXt)
# ═════════════════════════════════════════════════════════════════════════════
class AttentionBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SpatialRefinementModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3)
        )
    def forward(self, x): return self.net(x)

class PrototypeLayer(nn.Module):
    def __init__(self, n_proto, dim, n_classes, per_class):
        super().__init__()
        self.n_proto, self.dim, self.n_classes, self.per_class = n_proto, dim, n_classes, per_class
        self.vectors = nn.Parameter(torch.randn(n_proto, dim, 1, 1) * 0.02)
        identity = torch.zeros(n_proto, n_classes)
        for j in range(n_proto): identity[j, j // per_class] = 1
        self.register_buffer("identity", identity)

    def forward(self, x):
        B, D, H, W = x.shape
        P = self.vectors.shape[0]
        patches = F.normalize(x.view(B, D, -1), p=2, dim=1)
        pvecs = F.normalize(self.vectors.view(P, D), p=2, dim=1)
        cos = torch.einsum("bds,pd->bps", patches, pvecs)
        return cos.max(dim=2).values

class PowerProtoNet(nn.Module):
    def __init__(self):
        super().__init__()
        C, K, D = CONFIG["NUM_CLASSES"], 5, 256
        P = C * K
        
        base = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.backbone = base.features 
        in_ch = 768 
        
        self.attention = AttentionBlock(in_ch)
        self.refinement = SpatialRefinementModule(in_ch, D)
        self.proto_layer = PrototypeLayer(P, D, C, K)
        self.fc = nn.Linear(P, C, bias=False)
        
        self.n_proto, self.per_class, self.proto_dim = P, K, D

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.attention(feat)
        f = self.refinement(feat)
        sim = self.proto_layer(f)
        return self.fc(sim), sim


# ═════════════════════════════════════════════════════════════════════════════
#  WRAPPER FOR XAI LIBRARIES
# ═════════════════════════════════════════════════════════════════════════════
class LogitWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(x)
        return logits


# ═════════════════════════════════════════════════════════════════════════════
#  REPORT GENERATOR
# ═════════════════════════════════════════════════════════════════════════════
class Report:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() and CONFIG["DEVICE"] != "cpu" else "cpu")
        self.report_dir = Path(CONFIG["REPORT_DIR"])
        self.report_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*62}")
        print(f"  PowerProtoNet Report  |  device: {self.device}")
        print(f"  Output → {self.report_dir.resolve()}")
        print(f"{'='*62}\n")

        self.model = self._load_model()
        self.wrapper = LogitWrapper(self.model)
        
        self.transform = transforms.Compose([
            transforms.Resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        self.test_ds = datasets.ImageFolder(CONFIG["TEST_DIR"], transform=self.transform)
        self.test_loader = DataLoader(self.test_ds, CONFIG["BATCH_SIZE"], shuffle=False, num_workers=CONFIG["NUM_WORKERS"])
        self.class_names = self.test_ds.classes
        
        print(f"  Test set : {len(self.test_ds):,} images  |  {len(self.class_names)} classes\n")
        self.all_labels, self.all_preds, self.all_probs, self.all_sims = self._run_inference()

    def _load_model(self) -> PowerProtoNet:
        print(f"[setup] Loading checkpoint: {CONFIG['CHECKPOINT']}")
        model = PowerProtoNet().to(self.device)
        ckpt = torch.load(CONFIG["CHECKPOINT"], map_location=self.device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state)
        model.eval()
        return model

    @torch.no_grad()
    def _run_inference(self):
        print("[setup] Test-set inference …")
        labels, preds, probs, sims = [], [], [], []
        for imgs, lbls in tqdm(self.test_loader, desc="  Inference"):
            imgs = imgs.to(self.device)
            logits, sm = self.model(imgs)
            probs.append(torch.softmax(logits, 1).cpu().numpy())
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(lbls.numpy())
            sims.append(sm.cpu().numpy())
        print()
        return np.array(labels), np.array(preds), np.vstack(probs), np.vstack(sims)

    def _unnorm(self, tensor: torch.Tensor) -> np.ndarray:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    def _save(self, name: str, fig) -> None:
        path = self.report_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"      → {path}")

    def classification_report(self):
        print("[1/5] Classification report …")
        report = sk_classification_report(self.all_labels, self.all_preds, target_names=self.class_names, digits=4)
        (self.report_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    def confusion_matrix(self):
        print("[2/5] Confusion matrix …")
        cm = confusion_matrix(self.all_labels, self.all_preds)
        fig, ax = plt.subplots(figsize=(20, 18))
        sns.heatmap(cm, ax=ax, fmt="d", cmap="Blues", xticklabels=self.class_names, yticklabels=self.class_names)
        ax.set_title("Confusion Matrix", fontsize=14)
        plt.xticks(rotation=90, fontsize=6); plt.yticks(fontsize=6)
        plt.tight_layout()
        self._save("confusion_matrix.png", fig)

    def gradcam_plusplus(self):
        if not HAS_GRADCAM: return
        print("[3/5] Grad-CAM++ …")
        idx = random.sample(range(len(self.test_ds)), min(CONFIG["N_AUTO_EXPLAIN"], len(self.test_ds)))
        images = [self.test_ds[i][0] for i in idx]
        labels = [self.test_ds[i][1] for i in idx]

        target_layer = [self.model.backbone[-1]]
        cam_extractor = GradCAMPlusPlus(model=self.wrapper, target_layers=target_layer)

        fig = plt.figure(figsize=(12, 3.5 * len(images)))
        gs = gridspec.GridSpec(len(images), 3, figure=fig, hspace=0.4, wspace=0.1)

        for i, (img_t, lbl) in enumerate(zip(images, labels)):
            img_np = self._unnorm(img_t)
            inp = img_t.unsqueeze(0).to(self.device)
            grayscale_cam = cam_extractor(input_tensor=inp, targets=[ClassifierOutputTarget(lbl)])[0]
            overlay = show_cam_on_image(img_np.astype(np.float32), grayscale_cam, use_rgb=True)

            ax0 = fig.add_subplot(gs[i, 0]); ax0.imshow(img_np); ax0.axis("off")
            ax0.set_title(f"True: {self.class_names[lbl][:20]}", fontsize=9)
            ax1 = fig.add_subplot(gs[i, 1]); ax1.imshow(overlay); ax1.axis("off")
            ax2 = fig.add_subplot(gs[i, 2]); ax2.imshow(grayscale_cam, cmap="jet"); ax2.axis("off")

        self._save("gradcam_plusplus.png", fig)

    def saliency(self):
        if not HAS_CAPTUM: return
        print("[4/5] Saliency + IG …")
        idx = random.sample(range(len(self.test_ds)), min(CONFIG["N_AUTO_EXPLAIN"], len(self.test_ds)))
        images = [self.test_ds[i][0] for i in idx]
        labels = [self.test_ds[i][1] for i in idx]

        sal_method = Saliency(self.wrapper)
        ig_method  = IntegratedGradients(self.wrapper)

        fig = plt.figure(figsize=(16, 3.5 * len(images)))
        gs  = gridspec.GridSpec(len(images), 4, figure=fig, hspace=0.4, wspace=0.15)

        for i, (img_t, lbl) in enumerate(zip(images, labels)):
            img_np = self._unnorm(img_t)
            inp = img_t.unsqueeze(0).to(self.device)
            
            sal_attr = sal_method.attribute(inp, target=lbl)
            sal_map = (sal_attr.abs().squeeze(0).max(0).values).cpu().detach().numpy()
            sal_map = sal_map / (sal_map.max() + 1e-8)

            ig_attr = ig_method.attribute(inp, torch.zeros_like(inp), target=lbl, n_steps=15)
            ig_map = (ig_attr.abs().squeeze(0).max(0).values).cpu().detach().numpy()
            ig_map = ig_map / (ig_map.max() + 1e-8)

            fig.add_subplot(gs[i, 0]).imshow(img_np)
            fig.add_subplot(gs[i, 1]).imshow(sal_map, cmap="hot", vmin=0, vmax=1)
            ax2 = fig.add_subplot(gs[i, 2]); ax2.imshow(img_np); ax2.imshow(sal_map, cmap="hot", alpha=0.5)
            fig.add_subplot(gs[i, 3]).imshow(ig_map, cmap="viridis", vmin=0, vmax=1)
            for ax in fig.axes[-4:]: ax.axis("off")

        self._save("saliency_and_ig.png", fig)

    def shap_explain(self):
        if not HAS_SHAP: return
        print("[5/5] SHAP DeepExplainer …")
        train_ds = datasets.ImageFolder(CONFIG["TRAIN_DIR"], transform=self.transform)
        bg_imgs = torch.stack([train_ds[i][0] for i in random.sample(range(len(train_ds)), CONFIG["SHAP_BACKGROUND"])]).to(self.device)
        t_idx = random.sample(range(len(self.test_ds)), CONFIG["SHAP_TEST_IMGS"])
        t_imgs = torch.stack([self.test_ds[i][0] for i in t_idx]).to(self.device)
        t_lbls = [self.test_ds[i][1] for i in t_idx]

        explainer = shap.DeepExplainer(self.wrapper, bg_imgs)
        shap_vals = explainer.shap_values(t_imgs)

        fig = plt.figure(figsize=(14, 4 * len(t_idx)))
        gs = gridspec.GridSpec(len(t_idx), 3, figure=fig, hspace=0.45, wspace=0.1)

        for row, (img_t, lbl) in enumerate(zip(t_imgs.cpu(), t_lbls)):
            img_np = self._unnorm(img_t)
            sv_scalar = shap_vals[lbl][row].mean(0)
            sv_norm = sv_scalar / (np.abs(sv_scalar).max() + 1e-8)

            fig.add_subplot(gs[row, 0]).imshow(img_np)
            fig.add_subplot(gs[row, 1]).imshow(sv_norm, cmap="RdBu_r", vmin=-1, vmax=1)
            ax2 = fig.add_subplot(gs[row, 2]); ax2.imshow(img_np); ax2.imshow(sv_norm, cmap="RdBu_r", alpha=0.55, vmin=-1, vmax=1)
            for ax in fig.axes[-3:]: ax.axis("off")

        self._save("shap_deep.png", fig)

    def run_all(self):
        self.classification_report()
        self.confusion_matrix()
        self.gradcam_plusplus()
        self.saliency()
        self.shap_explain()
        print(f"\n[SUCCESS] All done. Reports saved to: {self.report_dir.resolve()}\n")


if __name__ == "__main__":
    r = Report()
    r.run_all()
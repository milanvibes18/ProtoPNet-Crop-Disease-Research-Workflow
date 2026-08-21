# =============================================================================
# POWER-PROTOPNET — XAI Report Generator
# Generates: Classification Reports, ROC/AUC, Confusion Matrix,
#            GradCAM++, SHAP, and side-by-side XAI comparison panels.
# Compatible with: POWER-PROTOPNET___PlantWild_v3.py
# Paper-grade libraries: pytorch-grad-cam, shap
# =============================================================================

import sys, json, logging, warnings, math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from PIL import Image

# Classification & metrics
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# Plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# GradCAM++ — pytorch-grad-cam
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# SHAP
import shap

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# REPORT CONFIG
# =============================================================================
REPORT_CONFIG = {
    # ── Paths ──────────────────────────────────────────────────────────────────
    "TRAIN_DIR"         : "/kaggle/input/datasets/entishar/disease-dataset/new_disease_dataset/train",
    "VAL_DIR"           : "/kaggle/input/datasets/entishar/disease-dataset/new_disease_dataset/val",
    "TEST_DIR"          : "/kaggle/input/datasets/entishar/disease-dataset/new_disease_dataset/test",
    "MODEL_PATH"        : "/kaggle/input/datasets/milanvibes18/result2/saves/plantwild_powernet_38/best_model.pt",
    "SAVE_DIR"          : "/kaggle/working/reports",

    # ── Dataset ────────────────────────────────────────────────────────────────
    "NUM_CLASSES"       : 38,
    "IMG_SIZE"          : 224,
    "IMAGENET_MEAN"     : [0.485, 0.456, 0.406],
    "IMAGENET_STD"      : [0.229, 0.224, 0.225],

    # ── Architecture (must match training) ─────────────────────────────────────
    "BACKBONE"          : "convnext_tiny",
    "PROTOTYPE_DIM"     : 256,
    "PROTOTYPES_PER_CLASS": 5,
    "MID_CHANNELS"      : 512,
    "DROPOUT"           : 0.3,

    # ── Report settings ────────────────────────────────────────────────────────
    "BATCH_SIZE"        : 32,
    "NUM_WORKERS"       : 0, # <--- 0 prevents Kaggle background thread locks with SHAP
    "PIN_MEMORY"        : True,

    # Number of samples per class for XAI panel (keep low for speed)
    "XAI_SAMPLES_PER_CLASS" : 2,
    "XAI_MAX_PANELS"    : 20,

    # SHAP background samples
    "SHAP_BG_SAMPLES"   : 50,
    "SHAP_EXPLAIN_SAMPLES": 5,

    # Figure DPI
    "DPI"               : 200,
    "LOG_FILE"          : "/kaggle/working/report_generation.log",
}

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(REPORT_CONFIG["LOG_FILE"], mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# =============================================================================
# ── RE-DEFINE MODEL ARCHITECTURE ─────────────────────────────────────────────
# =============================================================================
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
            nn.Dropout2d(p=REPORT_CONFIG["DROPOUT"])
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
        return self._cosine_similarity(x, self.vectors)

    @staticmethod
    def _cosine_similarity(feat, protos):
        B, D, H, W = feat.shape
        P = protos.shape[0]
        patches = F.normalize(feat.view(B, D, -1), p=2, dim=1)
        pvecs = F.normalize(protos.view(P, D), p=2, dim=1)
        cos = torch.einsum("bds,pd->bps", patches, pvecs)
        return cos.max(dim=2).values

class PowerProtoNet(nn.Module):
    def __init__(self):
        super().__init__()
        C  = REPORT_CONFIG["NUM_CLASSES"]
        K  = REPORT_CONFIG["PROTOTYPES_PER_CLASS"]
        D  = REPORT_CONFIG["PROTOTYPE_DIM"]
        P  = C * K

        base = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.backbone  = base.features
        in_ch = 768

        self.attention  = AttentionBlock(in_ch)
        self.refinement = SpatialRefinementModule(in_ch, D)
        self.proto_layer = PrototypeLayer(P, D, C, K)
        self.fc = nn.Linear(P, C, bias=False)

        with torch.no_grad():
            self.fc.weight.fill_(-0.5)
            for c in range(C):
                self.fc.weight[c, c * K: (c + 1) * K] = 1.0

        self.n_proto, self.per_class, self.proto_dim = P, K, D

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.attention(feat)
        f    = self.refinement(feat)
        sim  = self.proto_layer(f)
        return self.fc(sim), sim

# =============================================================================
# TRANSFORMS
# =============================================================================
def get_val_transform():
    sz = REPORT_CONFIG["IMG_SIZE"]
    return transforms.Compose([
        transforms.Resize((sz, sz), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(REPORT_CONFIG["IMAGENET_MEAN"], REPORT_CONFIG["IMAGENET_STD"]),
    ])

def denormalize(tensor):
    mean = np.array(REPORT_CONFIG["IMAGENET_MEAN"])
    std  = np.array(REPORT_CONFIG["IMAGENET_STD"])
    img  = tensor.cpu().permute(1, 2, 0).numpy()
    img  = img * std + mean
    return np.clip(img, 0, 1)

# =============================================================================
# REPORT CLASS
# =============================================================================
class ReportGenerator:
    def __init__(self):
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = Path(REPORT_CONFIG["SAVE_DIR"])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.tf       = get_val_transform()

        log.info(f"Device: {self.device}")
        self._load_class_names()
        self._load_model()
        self._build_test_loader()

    def _load_class_names(self):
        train_path = Path(REPORT_CONFIG["TRAIN_DIR"])
        self.class_names = sorted(d.name for d in train_path.iterdir() if d.is_dir())
        assert len(self.class_names) == REPORT_CONFIG["NUM_CLASSES"], \
            f"Expected {REPORT_CONFIG['NUM_CLASSES']} classes, found {len(self.class_names)}"

    def _load_model(self):
        self.model = PowerProtoNet().to(self.device)
        ckpt_path  = REPORT_CONFIG["MODEL_PATH"]
        ckpt       = torch.load(ckpt_path, map_location=self.device)
        state = ckpt["model"] if "model" in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.eval()

    def _build_test_loader(self):
        ds = datasets.ImageFolder(REPORT_CONFIG["TEST_DIR"], transform=self.tf)
        self.test_dataset = ds
        self.test_loader  = DataLoader(
            ds, batch_size=REPORT_CONFIG["BATCH_SIZE"], shuffle=False,
            num_workers=REPORT_CONFIG["NUM_WORKERS"], pin_memory=REPORT_CONFIG["PIN_MEMORY"],
        )

    @torch.no_grad()
    def _run_inference(self):
        if hasattr(self, "_y_true"):
            return self._y_true, self._y_pred, self._y_probs

        log.info("Running inference on test set …")
        all_labels, all_preds, all_probs = [], [], []

        for imgs, lbls in tqdm(self.test_loader, desc="Inference"):
            imgs = imgs.to(self.device)
            logits, _ = self.model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(lbls.numpy())

        self._y_true  = np.concatenate(all_labels)
        self._y_pred  = np.concatenate(all_preds)
        self._y_probs = np.concatenate(all_probs)
        return self._y_true, self._y_pred, self._y_probs

    # =========================================================================
    # 1. Classification Report
    # =========================================================================
    def generate_classification_report(self):
        log.info("── Generating Classification Report ──")
        y_true, y_pred, _ = self._run_inference()

        report_str = classification_report(
            y_true, y_pred, target_names=self.class_names, digits=4
        )
        print(report_str)

        rpt_path = self.save_dir / "classification_report.txt"
        with open(rpt_path, "w") as f:
            f.write(report_str)
            
        from sklearn.metrics import classification_report as cr_json
        report_dict = cr_json(y_true, y_pred, target_names=self.class_names, output_dict=True)
        return report_dict

    # =========================================================================
    # 2. Confusion Matrix
    # =========================================================================
    def generate_confusion_matrix(self):
        log.info("── Generating Confusion Matrix ──")
        y_true, y_pred, _ = self._run_inference()
        cm = confusion_matrix(y_true, y_pred)
        n  = REPORT_CONFIG["NUM_CLASSES"]

        fig, axes = plt.subplots(1, 2, figsize=(28, 12))
        fig.suptitle("POWER-ProtoPNet  |  Confusion Matrix", fontsize=16, fontweight="bold", y=1.01)

        sns.heatmap(cm, ax=axes[0], cmap="Blues", fmt="d", xticklabels=self.class_names, yticklabels=self.class_names, linewidths=0.4, linecolor="white", annot=(n <= 20))
        axes[0].set_title("Raw Counts", fontsize=13)

        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        sns.heatmap(cm_norm, ax=axes[1], cmap="RdYlGn", fmt=".2f", xticklabels=self.class_names, yticklabels=self.class_names, linewidths=0.4, linecolor="white", annot=(n <= 20), vmin=0, vmax=1)
        axes[1].set_title("Row-Normalised (Recall per class)", fontsize=13)

        plt.tight_layout()
        fig.savefig(self.save_dir / "confusion_matrix.png", dpi=REPORT_CONFIG["DPI"], bbox_inches="tight")
        plt.close(fig)

    # =========================================================================
    # 3. ROC / AUC (With Missing Class Safeguard)
    # =========================================================================
    def generate_roc_curves(self):
        log.info("── Generating ROC / AUC Curves ──")
        y_true, _, y_probs = self._run_inference()
        n = REPORT_CONFIG["NUM_CLASSES"]
        y_bin = label_binarize(y_true, classes=list(range(n)))

        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(n):
            if np.sum(y_bin[:, i]) == 0:
                fpr[i], tpr[i] = np.array([0., 1.]), np.array([0., 1.])
                roc_auc[i] = 0.0
            else:
                fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_probs[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n
        fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr["micro"], tpr["micro"], label=f"Micro-avg ROC (AUC = {roc_auc['micro']:.4f})", color="deeppink", linestyle=":", linewidth=2.5)
        ax.plot(fpr["macro"], tpr["macro"], label=f"Macro-avg ROC (AUC = {roc_auc['macro']:.4f})", color="navy", linestyle=":",  linewidth=2.5)

        cmap = plt.get_cmap("tab20")
        for i, idx in enumerate(np.linspace(0, n - 1, min(n, 12), dtype=int)):
            ax.plot(fpr[idx], tpr[idx], color=cmap(i / 12), alpha=0.75, linewidth=1.2, label=f"{self.class_names[idx][:22]} (AUC={roc_auc[idx]:.3f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate",  fontsize=12)
        ax.set_title("POWER-ProtoPNet  |  ROC Curves", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=7.5, ncol=2)
        fig.savefig(self.save_dir / "roc_curves.png", dpi=REPORT_CONFIG["DPI"], bbox_inches="tight")
        plt.close(fig)

    # =========================================================================
    # 4. Additional Metrics
    # =========================================================================
    def generate_summary_metrics(self):
        y_true, y_pred, y_probs = self._run_inference()
        from sklearn.metrics import classification_report as cr

        top5 = sum(1 for i, t in enumerate(y_true) if t in np.argsort(y_probs[i])[-5:])
        top1_acc = (y_true == y_pred).mean()
        top5_acc = top5 / len(y_true)

        report_dict = cr(y_true, y_pred, target_names=self.class_names, output_dict=True, zero_division=0)
        classes = self.class_names
        precision = [report_dict[c]["precision"] for c in classes]
        recall    = [report_dict[c]["recall"]    for c in classes]
        f1        = [report_dict[c]["f1-score"]  for c in classes]

        fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)
        fig.suptitle(f"POWER-ProtoPNet  |  Per-Class Metrics\nTop-1: {top1_acc:.4f}  |  Top-5: {top5_acc:.4f}", fontsize=14, fontweight="bold")
        
        x = np.arange(len(classes))
        axes[0].bar(x, precision, color="#3498db"); axes[0].set_ylabel("Precision")
        axes[1].bar(x, recall, color="#2ecc71"); axes[1].set_ylabel("Recall")
        axes[2].bar(x, f1, color="#e67e22"); axes[2].set_ylabel("F1-Score")
        axes[2].set_xticks(x); axes[2].set_xticklabels(classes, rotation=90, fontsize=7)
        fig.savefig(self.save_dir / "per_class_metrics.png", dpi=REPORT_CONFIG["DPI"], bbox_inches="tight")
        plt.close(fig)

    # =========================================================================
    # 5. XAI Implementation (GradCAM & SHAP)
    # =========================================================================
    def _build_shap_explainer(self, background_loader):
        log.info("Building SHAP GradientExplainer background …")
        bg_imgs = []
        for imgs, _ in background_loader:
            bg_imgs.append(imgs)
            if sum(x.shape[0] for x in bg_imgs) >= REPORT_CONFIG["SHAP_BG_SAMPLES"]:
                break
        background = torch.cat(bg_imgs)[:REPORT_CONFIG["SHAP_BG_SAMPLES"]].to(self.device)

        class _LogitsOnly(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x): logits, _ = self.m(x); return logits

        explainer = shap.GradientExplainer(_LogitsOnly(self.model), background)
        return explainer

    def _shap_for_image(self, tensor, true_class, explainer):
        inp = tensor.unsqueeze(0).to(self.device)
        shap_vals = explainer.shap_values(inp)       

        # Bulletproof Shape Unpacking for GradientExplainer
        if isinstance(shap_vals, list):
            sv = shap_vals[true_class][0]
        elif isinstance(shap_vals, np.ndarray):
            if shap_vals.ndim == 5: 
                if shap_vals.shape[-1] == REPORT_CONFIG["NUM_CLASSES"]:
                    # Case: (Batch, Channels, H, W, Classes)
                    sv = shap_vals[0, ..., true_class]
                else:
                    # Case: (Batch, Classes, Channels, H, W)
                    sv = shap_vals[0, true_class]
            elif shap_vals.ndim == 4:
                if shap_vals.shape[-1] == REPORT_CONFIG["NUM_CLASSES"]:
                    sv = shap_vals[..., true_class]
                else:
                    sv = shap_vals[true_class]
            else:
                sv = shap_vals[0]
        else:
            if hasattr(shap_vals, "values"):
                v = shap_vals.values
                if v.ndim == 5 and v.shape[-1] == REPORT_CONFIG["NUM_CLASSES"]:
                    sv = v[0, ..., true_class]
                elif v.ndim == 5:
                    sv = v[0, true_class]
                else:
                    sv = v[0]
            else:
                sv = shap_vals[0]

        # Ensure correct axis alignment (C, H, W)
        if sv.shape[0] != 3 and sv.shape[-1] == 3:
            sv = np.moveaxis(sv, -1, 0)

        sv_abs = np.abs(sv).sum(0)
        sv_norm = (sv_abs - sv_abs.min()) / (sv_abs.max() - sv_abs.min() + 1e-8)
        
        cmap_fn = plt.get_cmap("inferno")
        heatmap = cmap_fn(sv_norm)[:, :, :3].astype(np.float32)
        rgb_img = denormalize(tensor)
        blend   = np.clip(0.55 * heatmap + 0.45 * rgb_img, 0, 1)
        return rgb_img, sv_norm, blend

    def generate_xai_panels(self):
        log.info("── Generating XAI Comparison Panels ──")
        ds = self.test_dataset
        target_layers = [self.model.refinement.net[4]]
        cam = GradCAMPlusPlus(model=self.model, target_layers=target_layers)

        bg_loader = DataLoader(ds, batch_size=REPORT_CONFIG["SHAP_BG_SAMPLES"], shuffle=True, num_workers=0)
        explainer = self._build_shap_explainer(bg_loader)

        xai_dir = self.save_dir / "xai_panels"
        xai_dir.mkdir(exist_ok=True)

        class_indices = {c: [] for c in range(REPORT_CONFIG["NUM_CLASSES"])}
        for idx, (_, label) in enumerate(ds.samples): class_indices[label].append(idx)

        selected = []
        for c in range(REPORT_CONFIG["NUM_CLASSES"]):
            idxs = class_indices[c][:REPORT_CONFIG["XAI_SAMPLES_PER_CLASS"]]
            selected.extend([(idx, c) for idx in idxs])
            if REPORT_CONFIG["XAI_MAX_PANELS"] and len(selected) >= REPORT_CONFIG["XAI_MAX_PANELS"]: break

        for sample_idx, (ds_idx, true_cls) in enumerate(tqdm(selected, desc="XAI Panels")):
            tensor, _ = ds[ds_idx]
            
            with torch.no_grad():
                logits, _ = self.model(tensor.unsqueeze(0).to(self.device))
                pred_cls   = logits.argmax(1).item()
                conf = torch.softmax(logits, 1)[0, pred_cls].item()

            rgb_img, _, gradcam_overlay = self._gradcam_for_image(tensor, true_cls, cam)
            _, _,          shap_overlay = self._shap_for_image(tensor, true_cls, explainer)

            self._save_xai_panel(rgb_img, gradcam_overlay, shap_overlay, self.class_names[true_cls], self.class_names[pred_cls], conf, (pred_cls == true_cls), xai_dir / f"panel_{sample_idx:04d}_{self.class_names[true_cls][:20]}.png")

    def _gradcam_for_image(self, tensor, true_class, cam):
        targets  = [ClassifierOutputTarget(true_class)]
        grayscale_cam = cam(input_tensor=tensor.unsqueeze(0).to(self.device), targets=targets)[0]
        rgb_img  = denormalize(tensor)
        overlay  = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, colormap=2)
        return rgb_img, grayscale_cam, overlay.astype(np.float32) / 255.0

    def _save_xai_panel(self, orig, gradcam, shap_img, true_name, pred_name, confidence, correct, out_path):
        fig = plt.figure(figsize=(13, 4.8), facecolor="#1a1a2e")
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.04, left=0.01, right=0.99, top=0.82, bottom=0.02)

        panels = [(orig, "Original Image", "#ecf0f1"), (gradcam, "GradCAM++", "#3498db"), (shap_img,"SHAP GradientExplainer", "#e67e22")]
        for col, (img_np, title, color) in enumerate(panels):
            ax = fig.add_subplot(gs[col]); ax.imshow(img_np); ax.set_title(title, color=color, fontsize=11, fontweight="bold", pad=5); ax.axis("off")
            for spine in ax.spines.values(): spine.set_edgecolor(color); spine.set_linewidth(1.5); spine.set_visible(True)

        fig.text(0.5, 0.93, f"True: {true_name}", ha="center", fontsize=12, fontweight="bold", color="#ecf0f1")
        fig.text(0.5, 0.88, f"Predicted: {pred_name}  [{confidence*100:.1f}%]", ha="center", fontsize=10, color="#27ae60" if correct else "#e74c3c")
        fig.savefig(out_path, dpi=REPORT_CONFIG["DPI"], bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    # =========================================================================
    # 6. SHAP Summary Plot
    # =========================================================================
    def generate_shap_summary(self):
        log.info("── Generating SHAP Summary (Global) ──")
        ds = self.test_dataset
        n  = REPORT_CONFIG["SHAP_EXPLAIN_SAMPLES"]

        indices = np.random.choice(len(ds), size=min(n * REPORT_CONFIG["NUM_CLASSES"], len(ds)), replace=False)
        sample_loader = DataLoader(Subset(ds, indices), batch_size=n, shuffle=False, num_workers=0)
        explainer = self._build_shap_explainer(DataLoader(ds, batch_size=REPORT_CONFIG["SHAP_BG_SAMPLES"], shuffle=True, num_workers=0))

        all_shap = []
        for imgs, _ in tqdm(sample_loader, desc="SHAP Summary"):
            sv = explainer.shap_values(imgs.to(self.device))
            
            # Robust extraction matching _shap_for_image logic
            if isinstance(sv, list): 
                sv_stack = np.stack(sv, axis=1)
            elif isinstance(sv, np.ndarray):
                if sv.ndim == 5:
                    if sv.shape[-1] == REPORT_CONFIG["NUM_CLASSES"]:
                        sv_stack = np.moveaxis(sv, -1, 1)
                    else:
                        sv_stack = sv
                else:
                    continue
            elif hasattr(sv, "values"):
                v = sv.values
                if v.ndim == 5:
                    if v.shape[-1] == REPORT_CONFIG["NUM_CLASSES"]:
                        sv_stack = np.moveaxis(v, -1, 1)
                    else:
                        sv_stack = v
                else:
                    continue
            else: 
                continue
            all_shap.append(sv_stack)

        if len(all_shap) == 0: return
        all_shap = np.concatenate(all_shap, axis=0)
        shap_mean_spatial = np.abs(all_shap).mean(axis=(0, 1)).mean(0)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(shap_mean_spatial, cmap="inferno")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("SHAP Mean |φ|  –  Spatial Importance", fontsize=11, fontweight="bold")
        ax.axis("off")
        fig.savefig(self.save_dir / "shap_spatial_importance.png", dpi=REPORT_CONFIG["DPI"], bbox_inches="tight")
        plt.close(fig)

    # =========================================================================
    # 9. Master Runner
    # =========================================================================
    def run_all(self):
        log.info(f"{'='*60}\n  POWER-ProtoPNet  XAI Report\n{'='*60}")
        self.generate_classification_report()
        self.generate_confusion_matrix()
        self.generate_roc_curves()
        self.generate_summary_metrics()
        self.generate_xai_panels()
        self.generate_shap_summary()
        log.info(f"{'='*60}\n  All reports saved to: {self.save_dir}\n{'='*60}")

if __name__ == "__main__":
    reporter = ReportGenerator()
    reporter.run_all()
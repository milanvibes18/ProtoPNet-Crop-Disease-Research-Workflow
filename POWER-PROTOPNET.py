# =============================================================================
# POWER-PROTOPNET — PlantWild v3 (ConvNeXt + SRM)
# =============================================================================

import sys, json, logging, warnings, math, random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# CONFIG - OPTIMIZED FOR 38-CLASS KAGGLE DATASET
# =============================================================================
CONFIG = {
    # ── Paths ──────────────────────────────────────────────────────────────────
    "TRAIN_DIR"             : "/kaggle/input/datasets/entishar/disease-dataset/new_disease_dataset/train",
    "VAL_DIR"               : "/kaggle/input/datasets/entishar/disease-dataset/new_disease_dataset/val",
    "TEST_DIR"              : "/kaggle/input/datasets/entishar/disease-dataset/new_disease_dataset/test",
    "SAVE_DIR"              : "/kaggle/working/saves/plantwild_powernet_38",

    # ── Dataset ────────────────────────────────────────────────────────────────
    "NUM_CLASSES"           : 38,        
    "IMG_SIZE"              : 224,       # Native ConvNeXt size (prevents Kaggle OOM crash)
    "IMAGENET_MEAN"         : [0.485, 0.456, 0.406],
    "IMAGENET_STD"          : [0.229, 0.224, 0.225],

    # ── Architecture ───────────────────────────────────────────────────────────
    "BACKBONE"              : "convnext_tiny",
    "PROTOTYPE_DIM"         : 256,       
    "PROTOTYPES_PER_CLASS"  : 5,         
    "MID_CHANNELS"          : 512,

    # ── Loss weights ───────────────────────────────────────────────────────────
    "CE_WEIGHT"             : 1.0,
    "CLUSTER_WEIGHT"        : 1.0,
    "SEPARATION_WEIGHT"     : 0.2,       
    "L1_WEIGHT"             : 1e-5,

    # ── Training phases ────────────────────────────────────────────────────────
    "PHASE1_EPOCHS"         : 10,        
    "PHASE1_LR"             : 1e-4,      # Stable warmup LR

    "PHASE2_EPOCHS"         : 60,        
    "PHASE2_LR_BACKBONE"    : 2e-5,      
    "PHASE2_LR_PROTO"       : 1e-4,      # Stable phase 2 LR
    "PHASE2_LR_FC"          : 1e-4,      # Stable phase 2 LR
    "PROTO_PUSH_EVERY"      : 2,         

    "PHASE4_EPOCHS"         : 10,        
    "PHASE4_LR_FC"          : 1e-5,

    # ── Anti-Overfitting ───────────────────────────────────────────────────────
    "ONLINE_AUG"            : True,
    "MIXUP_ALPHA"           : 0.0,       # Disabled to preserve XAI visual prototypes
    "CUTMIX_ALPHA"          : 0.0,       # Disabled to preserve XAI visual prototypes
    "DROPOUT"               : 0.3,
    "LABEL_SMOOTHING"       : 0.1,
    "WEIGHT_DECAY_BACKBONE" : 5e-4,
    "WEIGHT_DECAY_ADDON"    : 1e-4,
    "GRAD_CLIP"             : 1.0,       # Gradient clipping to prevent math explosions

    # ── Data loading ───────────────────────────────────────────────────────────
    "BATCH_SIZE"            : 32,        
    "NUM_WORKERS"           : 2,         # MUST BE 0: Prevents Kaggle silent crashes between epochs
    "SAVE_EVERY"            : 5,
    "RANDOM_SEED"           : 42,
    "USE_AMP"               : False,     # Disabled: Prevents NaN precision errors in cosine similarity
    "PIN_MEMORY"            : True,

    # ── Early stopping ─────────────────────────────────────────────────────────
    "EARLY_STOP_PATIENCE"   : 12,
    "EARLY_STOP_MIN_DELTA"  : 1e-4,

    # ── LR schedule ────────────────────────────────────────────────────────────
    "WARMUP_EPOCHS"         : 5,
    "COSINE_ETA_MIN_FRAC"   : 0.01,

    "LOG_FILE"              : "/kaggle/working/power_training.log",
}

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(CONFIG["LOG_FILE"], mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# =============================================================================
# UTILITY & AUGMENTATION
# =============================================================================
class Utility:
    @staticmethod
    def get_train_transform():
        sz = CONFIG["IMG_SIZE"]
        ops = [
            transforms.Resize((sz, sz), interpolation=transforms.InterpolationMode.BICUBIC),
        ]
        if CONFIG["ONLINE_AUG"]:
            ops += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(0.2, 0.2, 0.2),
            ]
        ops += [
            transforms.ToTensor(),
            transforms.Normalize(CONFIG["IMAGENET_MEAN"], CONFIG["IMAGENET_STD"]),
        ]
        return transforms.Compose(ops)

    @staticmethod
    def get_val_transform():
        sz = CONFIG["IMG_SIZE"]
        return transforms.Compose([
            transforms.Resize((sz, sz), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(CONFIG["IMAGENET_MEAN"], CONFIG["IMAGENET_STD"]),
        ])

    @staticmethod
    def build_loader(split: str, shuffle: bool = False) -> DataLoader:
        path = {"train": CONFIG["TRAIN_DIR"], "val": CONFIG["VAL_DIR"], "test": CONFIG["TEST_DIR"]}[split]
        tf = Utility.get_train_transform() if split == "train" else Utility.get_val_transform()
        ds = datasets.ImageFolder(path, transform=tf)
        return DataLoader(ds, CONFIG["BATCH_SIZE"], shuffle=shuffle,
                          num_workers=CONFIG["NUM_WORKERS"], pin_memory=CONFIG["PIN_MEMORY"],
                          drop_last=(split == "train"))

    @staticmethod
    def class_weights(train_dir: str, device: torch.device) -> torch.Tensor:
        ds = datasets.ImageFolder(train_dir)
        counts = defaultdict(int)
        for _, lbl in ds.samples: counts[lbl] += 1
        total = sum(counts.values())
        n = len(ds.classes)
        w = torch.tensor([total / (n * counts[c]) if counts[c] else 0. for c in range(n)])
        return w.to(device)

    @staticmethod
    def compute_loss(logits, sims, labels, model, criterion, use_l1=False):
        ce = criterion(logits, labels)
        iden = model.proto_layer.identity
        B = logits.size(0)
        clust = sep = torch.tensor(0., device=logits.device)
        for i in range(B):
            lbl = labels[i].item()
            s = sims[i]
            own = iden[:, lbl].bool()
            if own.any(): clust -= s[own].max()
            if (~own).any(): sep += s[~own].max()
        clust /= B
        sep /= B
        l1 = model.fc.weight.norm(1) if use_l1 else torch.tensor(0., device=logits.device)
        total = (CONFIG["CE_WEIGHT"] * ce + CONFIG["CLUSTER_WEIGHT"] * clust + 
                 CONFIG["SEPARATION_WEIGHT"] * sep + CONFIG["L1_WEIGHT"] * l1)
        return total, ce, clust, sep, l1

    @staticmethod
    @torch.no_grad()
    def push_prototypes(model, loader, device):
        log.info("Updating Prototypes (Patch Matching)...")
        model.eval()
        P, D = model.n_proto, model.proto_dim
        best_sim = torch.full((P,), -float("inf"))
        best_vec = torch.zeros(P, D)
        for imgs, lbls in tqdm(loader, desc="Push", leave=False):
            imgs, lbls = imgs.to(device), lbls.to(device)
            feats, _ = model.push_forward(imgs)
            B, _, H, W = feats.shape
            patches_norm = F.normalize(feats.view(B, D, -1).permute(0, 2, 1), p=2, dim=2)
            for i in range(B):
                lbl = lbls[i].item()
                for j in range(lbl * model.per_class, (lbl + 1) * model.per_class):
                    p_norm = F.normalize(model.proto_layer.vectors[j].view(1, D), p=2, dim=1)
                    cos_sim = (patches_norm[i] @ p_norm.T).squeeze(1)
                    val, idx = cos_sim.max(0)
                    if val.item() > best_sim[j]:
                        best_sim[j] = val.item()
                        best_vec[j] = feats[i, :, idx // W, idx % W].cpu()
        with torch.no_grad():
            for j in range(P):
                if best_sim[j] > -float("inf"):
                    model.proto_layer.vectors[j].copy_(best_vec[j].view(D, 1, 1).to(device))

    @staticmethod
    @torch.no_grad()
    def evaluate(model, loader, criterion, device, split="val") -> dict:
        model.eval()
        loss_sum = c1 = c5 = total = 0
        for imgs, lbls in tqdm(loader, desc=f"[{split}]", leave=False):
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits, _ = model(imgs)
            loss = criterion(logits, lbls)
            loss_sum += loss.item() * imgs.size(0)
            preds = logits.argmax(1)
            c1 += (preds == lbls).sum().item()
            top5 = logits.topk(min(5, logits.size(1)), 1).indices
            for i, lbl in enumerate(lbls):
                if lbl in top5[i]: c5 += 1
            total += imgs.size(0)
        return {"loss": loss_sum / total, "top1_acc": c1 / total, "top5_acc": c5 / total}

    @staticmethod
    def save_checkpoint(path, model, opt=None, metrics=None):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict() if opt else None, "metrics": metrics}, path)

    @staticmethod
    def load_checkpoint(path, model, opt=None, device="cpu"):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if opt and "opt" in ckpt: opt.load_state_dict(ckpt["opt"])

# =============================================================================
# POWER ARCHITECTURE MODULES
# =============================================================================
class AttentionBlock(nn.Module):
    """Squeeze-and-Excitation to focus on lesion areas."""
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
    """Depthwise-Separable Conv to extract textural motifs of diseases."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=CONFIG["DROPOUT"])
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
        C, K, D = CONFIG["NUM_CLASSES"], CONFIG["PROTOTYPES_PER_CLASS"], CONFIG["PROTOTYPE_DIM"]
        P = C * K
        
        # ConvNeXt Tiny is superior for fine-grained plant disease
        base = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.backbone = base.features 
        in_ch = 768 # ConvNeXt Tiny output channels
        
        self.attention = AttentionBlock(in_ch)
        self.refinement = SpatialRefinementModule(in_ch, D)
        self.proto_layer = PrototypeLayer(P, D, C, K)
        self.fc = nn.Linear(P, C, bias=False)
        
        with torch.no_grad():
            self.fc.weight.fill_(-0.5)
            for c in range(C): self.fc.weight[c, c*K : (c+1)*K] = 1.0
            
        self.n_proto, self.per_class, self.proto_dim = P, K, D

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.attention(feat)
        f = self.refinement(feat)
        sim = self.proto_layer(f)
        return self.fc(sim), sim

    def push_forward(self, x):
        feat = self.backbone(x)
        feat = self.attention(feat)
        f = self.refinement(feat)
        sim = PrototypeLayer._cosine_similarity(f, self.proto_layer.vectors)
        return f, sim

# =============================================================================
# MIXUP & CUTMIX UTILS
# =============================================================================
def mixup_data(x, y, alpha=1.0):
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y, y[index], lam

def cutmix_data(x, y, alpha=1.0):
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
    index = torch.randperm(x.size(0)).to(x.device)
    W, H = x.size()[2], x.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W - cut_w), np.random.randint(H - cut_h)
    x_mixed = x.clone()
    x_mixed[:, :, cy:cy+cut_h, cx:cx+cut_w] = x[index, :, cy:cy+cut_h, cx:cx+cut_w]
    return x_mixed, y, y[index], lam

# =============================================================================
# TRAINER
# =============================================================================
class Trainer:
    def __init__(self, device):
        self.device = device
        self.train_loader = Utility.build_loader("train", shuffle=True)
        self.val_loader = Utility.build_loader("val")
        self.test_loader = Utility.build_loader("test")
        self.criterion = nn.CrossEntropyLoss(
            weight=Utility.class_weights(CONFIG["TRAIN_DIR"], device),
            label_smoothing=CONFIG["LABEL_SMOOTHING"]
        )
        self.model = PowerProtoNet().to(device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=CONFIG["USE_AMP"])
        self.save_dir = Path(CONFIG["SAVE_DIR"])
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _epoch(self, opt, use_l1=False, ce_only=False, apply_mixup=False):
        self.model.train()
        loss_sum = c1 = n = 0
        bar = tqdm(self.train_loader, desc="train", leave=False)
        
        for imgs, lbls in bar:
            imgs, lbls = imgs.to(self.device), lbls.to(self.device)
            
            # --- Apply Mixup/CutMix ---
            if apply_mixup and (CONFIG["MIXUP_ALPHA"] > 0 or CONFIG["CUTMIX_ALPHA"] > 0):
                if random.random() > 0.5 and CONFIG["MIXUP_ALPHA"] > 0:
                    imgs, y_a, y_b, lam = mixup_data(imgs, lbls, CONFIG["MIXUP_ALPHA"])
                elif CONFIG["CUTMIX_ALPHA"] > 0:
                    imgs, y_a, y_b, lam = cutmix_data(imgs, lbls, CONFIG["CUTMIX_ALPHA"])
                else:
                    y_a = y_b = lbls
                    lam = 1.0
            else:
                y_a = y_b = lbls
                lam = 1.0

            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=CONFIG["USE_AMP"]):
                logits, dists = self.model(imgs)
                
                if ce_only:
                    total_l = lam * self.criterion(logits, y_a) + (1 - lam) * self.criterion(logits, y_b)
                else:
                    # Primary Loss (y_a)
                    total_l, ce, clust, sep, l1 = Utility.compute_loss(
                        logits, dists, y_a, self.model, self.criterion, use_l1)
                    # Secondary Loss (y_b) for Mixup
                    if lam < 1.0:
                        total_lb, _, _, _, _ = Utility.compute_loss(
                            logits, dists, y_b, self.model, self.criterion, use_l1)
                        total_l = lam * total_l + (1 - lam) * total_lb

            if torch.isnan(total_l): continue
            
            # Conditionally use AMP scaler based on USE_AMP config
            if CONFIG["USE_AMP"]:
                self.scaler.scale(total_l).backward()
                self.scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), CONFIG["GRAD_CLIP"])
                self.scaler.step(opt)
                self.scaler.update()
            else:
                total_l.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), CONFIG["GRAD_CLIP"])
                opt.step()

            preds = logits.argmax(1)
            loss_sum += total_l.item() * imgs.size(0)
            c1 += (preds == lbls).sum().item() 
            n += imgs.size(0)
            bar.set_postfix(loss=f"{total_l.item():.3f}", acc=f"{c1/n:.3f}" if n > 0 else "0.000")

        # Safe return to prevent ZeroDivisionError
        return {
            "loss": loss_sum / n if n > 0 else float('nan'), 
            "top1_acc": c1 / n if n > 0 else 0.0
        }

    def run(self):
        # Phase 1: Warmup
        log.info("── Phase 1: Warmup ──")
        opt1 = optim.AdamW([
            {"params": self.model.attention.parameters(), "lr": CONFIG["PHASE1_LR"]}, # <--- FIXED
            {"params": self.model.refinement.parameters(), "lr": CONFIG["PHASE1_LR"]},
            {"params": self.model.fc.parameters(), "lr": CONFIG["PHASE1_LR"]}
        ])
        for ep in range(1, CONFIG["PHASE1_EPOCHS"] + 1):
            self._epoch(opt1, ce_only=True)
            log.info(f"Ep {ep} Warmup Complete")

        # Phase 2: Joint Training
        log.info("── Phase 2: Joint Training (ConvNeXt + SRM) ──")
        opt2 = optim.AdamW([
            {"params": self.model.backbone.parameters(), "lr": CONFIG["PHASE2_LR_BACKBONE"], "weight_decay": CONFIG["WEIGHT_DECAY_BACKBONE"]},
            {"params": self.model.attention.parameters(), "lr": CONFIG["PHASE2_LR_PROTO"], "weight_decay": CONFIG["WEIGHT_DECAY_ADDON"]}, # <--- FIXED
            {"params": self.model.refinement.parameters(), "lr": CONFIG["PHASE2_LR_PROTO"], "weight_decay": CONFIG["WEIGHT_DECAY_ADDON"]},
            {"params": self.model.proto_layer.parameters(), "lr": CONFIG["PHASE2_LR_PROTO"]},
            {"params": self.model.fc.parameters(), "lr": CONFIG["PHASE2_LR_FC"]}
        ])
        
        sched = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=CONFIG["PHASE2_EPOCHS"])
        best_acc = 0
        patience_cnt = 0

        for ep in range(1, CONFIG["PHASE2_EPOCHS"] + 1):
            tr = self._epoch(opt2, apply_mixup=(CONFIG["MIXUP_ALPHA"] > 0 or CONFIG["CUTMIX_ALPHA"] > 0))
            val = Utility.evaluate(self.model, self.val_loader, self.criterion, self.device)
            sched.step()
            
            if ep % CONFIG["PROTO_PUSH_EVERY"] == 0:
                Utility.push_prototypes(self.model, self.train_loader, self.device)
            
            log.info(f"Ep {ep:03d} | Train Acc: {tr['top1_acc']:.4f} | Val Acc: {val['top1_acc']:.4f} | Val Loss: {val['loss']:.4f}")
            
            if val['top1_acc'] > best_acc + CONFIG["EARLY_STOP_MIN_DELTA"]:
                best_acc = val['top1_acc']
                Utility.save_checkpoint(self.save_dir / "best_model.pt", self.model, opt2, val)
                patience_cnt = 0
            else:
                patience_cnt += 1
            
            if patience_cnt >= CONFIG["EARLY_STOP_PATIENCE"]:
                log.info("Early Stopping Triggered")
                break

        # Phase 4: Final FC Sparsity Fine-tune
        log.info("── Phase 4: FC Sparsity Fine-tune ──")
        Utility.load_checkpoint(str(self.save_dir / "best_model.pt"), self.model, device=self.device)
        # Freeze all but FC
        for p in self.model.parameters(): p.requires_grad = False
        for p in self.model.fc.parameters(): p.requires_grad = True
        
        opt4 = optim.AdamW(self.model.fc.parameters(), lr=CONFIG["PHASE4_LR_FC"])
        for ep in range(1, CONFIG["PHASE4_EPOCHS"] + 1):
            self._epoch(opt4, use_l1=True)
            log.info(f"Ep {ep} FC Tuning Complete")

        return self.model

# =============================================================================
# MODEL API
# =============================================================================
class Model:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PowerProtoNet().to(self.device)
        self.names = sorted(d.name for d in Path(CONFIG["TRAIN_DIR"]).iterdir() if d.is_dir())
        self.criterion = nn.CrossEntropyLoss(
            weight=Utility.class_weights(CONFIG["TRAIN_DIR"], self.device),
            label_smoothing=CONFIG["LABEL_SMOOTHING"]
        )
        self._tf = Utility.get_val_transform()

    def train(self):
        trainer = Trainer(self.device)
        self.model = trainer.run()

    def load(self, path=None):
        path = path or str(Path(CONFIG["SAVE_DIR"]) / "best_model.pt")
        Utility.load_checkpoint(path, self.model, device=self.device)
        self.model.eval()

    def predict(self, img_path):
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        t = self._tf(img).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(t)
            prob = torch.softmax(logits, 1)[0]
        idx = prob.argmax().item()
        return {"class": self.names[idx], "confidence": prob[idx].item()}

    def evaluate(self, split="test"):
        loader = Utility.build_loader(split)
        return Utility.evaluate(self.model, loader, self.criterion, self.device, split)

# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
    M = Model()
    M.train()
    res = M.evaluate("test")
    log.info(f"FINAL TEST ACCURACY: {res['top1_acc']:.4f}")
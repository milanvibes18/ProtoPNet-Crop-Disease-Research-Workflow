# =============================================================================
# PROTOPNET — PlantWild v2 (EfficientNet-B0 Version)
# =============================================================================

import sys, json, logging, warnings
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
# CONFIG
# =============================================================================
CONFIG = {
    "TRAIN_DIR"             : "./Dataset/plantwild/train",
    "VAL_DIR"               : "./Dataset/plantwild/val",
    "TEST_DIR"              : "./Dataset/plantwild/test",
    "SAVE_DIR"              : "./saves/plantwild_protopnet",
    "NUM_CLASSES"           : 98,
    "IMG_SIZE"              : 224,
    "IMAGENET_MEAN"         : [0.485, 0.456, 0.406],
    "IMAGENET_STD"          : [0.229, 0.224, 0.225],
    "BACKBONE"              : "efficientnet_b0",   # Changed to EfficientNet-B0
    "PROTOTYPE_DIM"         : 128,
    "PROTOTYPES_PER_CLASS"  : 10,
    "ADDON_CHANNELS"        : 256,
    "CE_WEIGHT"             : 1.0,
    "CLUSTER_WEIGHT"        : 0.8,
    "SEPARATION_WEIGHT"     : 0.08,
    "L1_WEIGHT"             : 1e-4,
    "PHASE1_EPOCHS"         : 5,
    "PHASE1_LR_ADDON"       : 3e-3,
    "PHASE1_LR_FC"          : 3e-3,
    "PHASE2_EPOCHS"         : 30,
    "PHASE2_LR_BACKBONE"    : 1e-4,
    "PHASE2_LR_ADDON"       : 3e-3,
    "PHASE2_LR_PROTO"       : 3e-3,
    "PHASE2_LR_FC"          : 3e-3,
    "PROTO_PUSH_EVERY"      : 5,
    "PHASE4_EPOCHS"         : 20,
    "PHASE4_LR_FC"          : 1e-4,
    "BATCH_SIZE"            : 32,
    "NUM_WORKERS"           : 4,
    "SAVE_EVERY"            : 5,
    "RANDOM_SEED"           : 42,
    "USE_AMP"               : True,
    "PIN_MEMORY"            : True,
    "POSITIVE_WEIGHT_INIT"  :  1.0,
    "NEGATIVE_WEIGHT_INIT"  : -0.5,
    "LOG_FILE"              : "training.log",
    "SIMILARITY_MODE"       : "cosine",
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
        logging.StreamHandler(
            open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
        ),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# Utility
# =============================================================================
class Utility:
    @staticmethod
    def get_transform():
        return transforms.Compose([
            transforms.Resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"])),
            transforms.ToTensor(),
            transforms.Normalize(CONFIG["IMAGENET_MEAN"], CONFIG["IMAGENET_STD"]),
        ])

    @staticmethod
    def build_loader(split: str, shuffle: bool = False) -> DataLoader:
        path = {"train": CONFIG["TRAIN_DIR"], "val": CONFIG["VAL_DIR"], "test": CONFIG["TEST_DIR"]}[split]
        ds = datasets.ImageFolder(path, transform=Utility.get_transform())
        log.info(f"[{split:5s}] {len(ds)} images | {len(ds.classes)} classes")
        return DataLoader(ds, CONFIG["BATCH_SIZE"], shuffle=shuffle, num_workers=CONFIG["NUM_WORKERS"], pin_memory=CONFIG["PIN_MEMORY"], drop_last=(split == "train"))

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
    def class_names() -> list:
        return sorted(d.name for d in Path(CONFIG["TRAIN_DIR"]).iterdir() if d.is_dir())

    @staticmethod
    def compute_loss(logits, dists, labels, model, criterion, use_l1=False):
        ce = criterion(logits, labels)
        iden = model.proto_layer.identity
        B = logits.size(0)
        clust = sep = torch.tensor(0., device=logits.device)
        for i in range(B):
            lbl = labels[i].item()
            d = dists[i]
            own = iden[:, lbl].bool()
            if own.any(): clust += d[own].min()
            if (~own).any(): sep -= d[~own].min()
        clust /= B
        sep /= B
        l1 = (model.fc.weight.norm(1) if use_l1 else torch.tensor(0., device=logits.device))
        total = (CONFIG["CE_WEIGHT"] * ce + CONFIG["CLUSTER_WEIGHT"] * clust + CONFIG["SEPARATION_WEIGHT"] * sep + CONFIG["L1_WEIGHT"] * l1)
        return total, ce, clust, sep, l1

    @staticmethod
    @torch.no_grad()
    def push_prototypes(model, loader, device):
        log.info("Prototype push...")
        model.eval()
        P, D = model.n_proto, model.proto_dim
        best_dist = torch.full((P,), float("inf"))
        best_vec = torch.zeros(P, D)
        for imgs, lbls in tqdm(loader, desc="Push", leave=False):
            imgs, lbls = imgs.to(device), lbls.to(device)
            feats, _ = model.push_forward(imgs)
            H, W = feats.size(2), feats.size(3)
            for i in range(imgs.size(0)):
                lbl, feat = lbls[i].item(), feats[i]
                for j in range(lbl * model.per_class, (lbl + 1) * model.per_class):
                    p = model.proto_layer.vectors[j].expand(-1, H, W)
                    sdist = ((feat - p) ** 2).sum(0).flatten()
                    val, idx = sdist.min(0)
                    if val.item() < best_dist[j]:
                        best_dist[j] = val.item()
                        best_vec[j] = feat[:, idx // W, idx % W].cpu()
        with torch.no_grad():
            for j in range(P):
                if best_dist[j] < float("inf"):
                    model.proto_layer.vectors[j].copy_(best_vec[j].view(D, 1, 1).to(device))

    @staticmethod
    def enforce_weight_constraint(model):
        iden = model.proto_layer.identity
        with torch.no_grad():
            for c in range(model.num_classes):
                for j in range(model.n_proto):
                    if iden[j, c] == 1: model.fc.weight[c, j].clamp_(min=0)
                    else: model.fc.weight[c, j].clamp_(max=0)

    @staticmethod
    @torch.no_grad()
    def evaluate(model, loader, criterion, device, split="val") -> dict:
        model.eval()
        loss_sum = c1 = c5 = total = 0
        cls_c, cls_t = defaultdict(int), defaultdict(int)
        for imgs, lbls in tqdm(loader, desc=f"[{split}]", leave=False):
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits, _ = model(imgs)
            loss_sum += criterion(logits, lbls).item() * imgs.size(0)
            preds = logits.argmax(1)
            c1 += (preds == lbls).sum().item()
            top5 = logits.topk(min(5, logits.size(1)), 1).indices
            for i, lbl in enumerate(lbls):
                if lbl in top5[i]: c5 += 1
                cls_t[lbl.item()] += 1
                cls_c[lbl.item()] += int(preds[i] == lbl)
            total += imgs.size(0)
        return {"loss": loss_sum / total, "top1_acc": c1 / total, "top5_acc": c5 / total, "per_class_acc": {c: cls_c[c] / cls_t[c] for c in cls_t}, "total": total}

    @staticmethod
    def save_checkpoint(path, model, opt=None, phase=None, epoch=None, metrics=None):
        payload = {"model": model.state_dict(), "config": CONFIG, "phase": phase, "epoch": epoch, "metrics": metrics or {}, "saved_at": datetime.now().isoformat()}
        if opt is not None: payload["opt"] = opt.state_dict()
        torch.save(payload, path)

    @staticmethod
    def load_checkpoint(path, model, opt=None, device="cpu"):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if opt and "opt" in ckpt: opt.load_state_dict(ckpt["opt"])
        return ckpt.get("metrics", {})

    @staticmethod
    def set_seed(seed=CONFIG["RANDOM_SEED"]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def get_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def freeze(*modules):
        for m in modules:
            for p in m.parameters(): p.requires_grad = False

    @staticmethod
    def unfreeze(*modules):
        for m in modules:
            for p in m.parameters(): p.requires_grad = True


# =============================================================================
# AddOnLayers
# =============================================================================
class AddOnLayers(nn.Module):
    def __init__(self, in_ch: int, proto_dim: int, mid_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU(),
            nn.Conv2d(mid_ch, proto_dim, 1, bias=False), nn.BatchNorm2d(proto_dim), nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x)


# =============================================================================
# PrototypeLayer
# =============================================================================
class PrototypeLayer(nn.Module):
    def __init__(self, n_proto: int, dim: int, n_classes: int, per_class: int):
        super().__init__()
        self.n_proto, self.dim, self.n_classes, self.per_class = n_proto, dim, n_classes, per_class
        self.mode = CONFIG["SIMILARITY_MODE"]
        self.vectors = nn.Parameter(torch.rand(n_proto, dim, 1, 1))
        identity = torch.zeros(n_proto, n_classes)
        for j in range(n_proto): identity[j, j // per_class] = 1
        self.register_buffer("identity", identity)

    def forward(self, x):
        if self.mode == "cosine": return self._cosine_similarity(x, self.vectors)
        return self._l2_distances(x, self.vectors)

    @staticmethod
    def _cosine_similarity(feat, protos):
        B, D, H, W = feat.shape
        P = protos.shape[0]
        patches = F.normalize(feat.view(B, D, -1), p=2, dim=1)
        pvecs = F.normalize(protos.view(P, D), p=2, dim=1)
        cos = torch.einsum("bds,pd->bps", patches, pvecs)
        return cos.max(dim=2).values

    @staticmethod
    def _l2_distances(feat, protos):
        f2 = (feat ** 2).sum(1, keepdim=True)
        p2 = (protos ** 2).sum(1, keepdim=True).permute(1, 0, 2, 3)
        dot = F.conv2d(feat, protos)
        return F.relu(f2 + p2 - 2 * dot).flatten(2).min(2).values

    @staticmethod
    def dist_to_sim(dist):
        return torch.log((dist + 1) / (dist + 1e-4))


# =============================================================================
# ProtoPNet (Updated for EfficientNet-B0)
# =============================================================================
class ProtoPNet(nn.Module):
    def __init__(self):
        super().__init__()
        C, K, D, mid = (CONFIG["NUM_CLASSES"], CONFIG["PROTOTYPES_PER_CLASS"],
                        CONFIG["PROTOTYPE_DIM"], CONFIG["ADDON_CHANNELS"])
        P = C * K

        # --- CHANGED: Load EfficientNet-B0 instead of DenseNet ---
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # EfficientNet's classifier is a Sequential(Dropout, Linear). 
        # We need the input features of that Linear layer.
        in_ch = base.classifier[1].in_features
        
        # We take only the feature extraction part (all conv blocks)
        self.backbone = base.features 
        # --------------------------------------------------------

        self.addon = AddOnLayers(in_ch, D, mid)
        self.proto_layer = PrototypeLayer(P, D, C, K)
        self.fc = nn.Linear(P, C, bias=False)
        
        with torch.no_grad():
            self.fc.weight.fill_(CONFIG["NEGATIVE_WEIGHT_INIT"])
            for c in range(C):
                self.fc.weight[c, c * K:(c + 1) * K] = CONFIG["POSITIVE_WEIGHT_INIT"]

        self.num_classes, self.n_proto, self.per_class, self.proto_dim = C, P, K, D

    def _features(self, x):
        return self.backbone(x) # EfficientNet features already end in an activation

    def forward(self, x):
        f = self.addon(self._features(x))
        sim = self.proto_layer(f)
        if CONFIG["SIMILARITY_MODE"] == "l2":
            sim = PrototypeLayer.dist_to_sim(sim)
        return self.fc(sim), sim

    def push_forward(self, x):
        f = self.addon(self._features(x))
        dists = PrototypeLayer._l2_distances(f, self.proto_layer.vectors)
        return f, dists


# =============================================================================
# HistoryTracker, Trainer, and Model classes (same as original)
# =============================================================================
class HistoryTracker:
    def __init__(self, save_dir):
        self.path = Path(save_dir) / "training_history.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = {"config": CONFIG, "phases": {}, "best": {"epoch": -1, "val_acc": 0., "val_loss": float("inf")}, "started_at": datetime.now().isoformat()}
    def start_phase(self, phase): self.data["phases"][phase] = {"epochs": [], "started_at": datetime.now().isoformat()}
    def record(self, phase, epoch, metrics): self.data["phases"][phase]["epochs"].append({"epoch": epoch, **metrics, "ts": datetime.now().isoformat()})
    def update_best(self, epoch, val_acc, val_loss):
        if val_acc > self.data["best"]["val_acc"]:
            self.data["best"] = {"epoch": epoch, "val_acc": val_acc, "val_loss": val_loss}
            return True
        return False
    def save(self):
        self.data["saved_at"] = datetime.now().isoformat()
        with open(self.path, "w", encoding="utf-8") as f: json.dump(self.data, f, indent=2)
    def log_epoch(self, tag, epoch, m):
        log.info(f"[{tag}] ep={epoch:03d}  tr_loss={m.get('train_loss',0):.4f}  tr_acc={m.get('train_acc',0):.4f}  val_loss={m.get('val_loss',0):.4f}  val_acc={m.get('val_acc',0):.4f}")

class Trainer:
    def __init__(self, device):
        self.device = device
        self.train_loader = Utility.build_loader("train", shuffle=True)
        self.val_loader = Utility.build_loader("val")
        self.test_loader = Utility.build_loader("test")
        self.criterion = nn.CrossEntropyLoss(weight=Utility.class_weights(CONFIG["TRAIN_DIR"], device))
        self.model = ProtoPNet().to(device)
        self.history = HistoryTracker(CONFIG["SAVE_DIR"])
        self.save_dir = Path(CONFIG["SAVE_DIR"])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = torch.cuda.amp.GradScaler(enabled=CONFIG["USE_AMP"])

    def _epoch(self, opt, use_l1=False) -> dict:
        self.model.train()
        loss_sum = c1 = c5 = n = 0
        bar = tqdm(self.train_loader, desc="train", leave=False)
        for imgs, lbls in bar:
            imgs, lbls = imgs.to(self.device), lbls.to(self.device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=CONFIG["USE_AMP"]):
                logits, dists = self.model(imgs)
                total_l, ce, clust, sep, l1 = Utility.compute_loss(logits, dists, lbls, self.model, self.criterion, use_l1)
            self.scaler.scale(total_l).backward()
            self.scaler.step(opt)
            self.scaler.update()
            preds = logits.argmax(1)
            loss_sum += total_l.item() * imgs.size(0)
            c1 += (preds == lbls).sum().item()
            top5 = logits.topk(min(5, logits.size(1)), 1).indices
            c5 += sum(lbls[i] in top5[i] for i in range(len(lbls)))
            n += imgs.size(0)
            bar.set_postfix(loss=f"{total_l.item():.3f}", acc=f"{c1/n:.3f}")
        return {"loss": loss_sum/n, "top1_acc": c1/n, "top5_acc": c5/n}

    def _step(self, tag, phase, epoch, opt, tr):
        val = Utility.evaluate(self.model, self.val_loader, self.criterion, self.device)
        flat = {**{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in val.items()}}
        self.history.record(phase, epoch, flat)
        self.history.log_epoch(tag, epoch, {"train_loss": tr["loss"], "train_acc": tr["top1_acc"], "val_loss": val["loss"], "val_acc": val["top1_acc"]})
        if self.history.update_best(epoch, val["top1_acc"], val["loss"]):
            Utility.save_checkpoint(self.save_dir / "best_model.pt", self.model, metrics={"val_acc": val["top1_acc"], "val_loss": val["loss"]})
        if epoch % CONFIG["SAVE_EVERY"] == 0:
            Utility.save_checkpoint(self.save_dir / f"ckpt_{phase}_ep{epoch:03d}.pt", self.model, opt, phase, epoch, flat)
        self.history.save()

    def phase1(self):
        log.info("── Phase 1: Warm-up ──")
        self.history.start_phase("phase1")
        Utility.freeze(self.model.backbone, self.model.proto_layer)
        opt = optim.Adam([{"params": self.model.addon.parameters(), "lr": CONFIG["PHASE1_LR_ADDON"]}, {"params": self.model.fc.parameters(), "lr": CONFIG["PHASE1_LR_FC"]}])
        for ep in range(1, CONFIG["PHASE1_EPOCHS"] + 1): self._step("P1", "phase1", ep, opt, self._epoch(opt))

    def phase2(self):
        log.info("── Phase 2: Joint ──")
        self.history.start_phase("phase2")
        Utility.unfreeze(self.model.backbone, self.model.proto_layer)
        opt = optim.Adam([{"params": self.model.backbone.parameters(), "lr": CONFIG["PHASE2_LR_BACKBONE"]}, {"params": self.model.addon.parameters(), "lr": CONFIG["PHASE2_LR_ADDON"]}, {"params": self.model.proto_layer.parameters(), "lr": CONFIG["PHASE2_LR_PROTO"]}, {"params": self.model.fc.parameters(), "lr": CONFIG["PHASE2_LR_FC"]}])
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, CONFIG["PHASE2_EPOCHS"], eta_min=1e-6)
        for ep in range(1, CONFIG["PHASE2_EPOCHS"] + 1):
            tr = self._epoch(opt)
            if ep % CONFIG["PROTO_PUSH_EVERY"] == 0 or ep == CONFIG["PHASE2_EPOCHS"]: Utility.push_prototypes(self.model, self.train_loader, self.device)
            self._step("P2", "phase2", ep, opt, tr)
            sched.step()

    def phase3(self):
        log.info("── Phase 3: Final push ──")
        self.history.start_phase("phase3")
        Utility.push_prototypes(self.model, self.train_loader, self.device)
        self.history.record("phase3", 1, {"push_complete": True})
        self.history.save()

    def phase4(self):
        log.info("── Phase 4: FC fine-tune ──")
        self.history.start_phase("phase4")
        Utility.freeze(self.model.backbone, self.model.addon, self.model.proto_layer)
        Utility.unfreeze(self.model.fc)
        opt = optim.Adam([{"params": self.model.fc.parameters(), "lr": CONFIG["PHASE4_LR_FC"]}])
        for ep in range(1, CONFIG["PHASE4_EPOCHS"] + 1):
            tr = self._epoch(opt, use_l1=True)
            Utility.enforce_weight_constraint(self.model)
            self._step("P4", "phase4", ep, opt, tr)

    def final_test(self) -> dict:
        log.info("── Final Test ──")
        m = Utility.evaluate(self.model, Utility.build_loader("test"), self.criterion, self.device, split="test")
        self.history.data["final_test"] = m
        Utility.save_checkpoint(self.save_dir / "final_model.pt", self.model, metrics=m)
        self.history.save()
        return m

    def run(self) -> "ProtoPNet":
        self.phase1(); self.phase2(); self.phase3(); self.phase4()
        self.final_test()
        return self.model

class Model:
    def __init__(self, config_overrides: dict = None):
        if config_overrides: CONFIG.update(config_overrides)
        Utility.set_seed()
        self.device = Utility.get_device()
        self.model = ProtoPNet().to(self.device)
        self.names = Utility.class_names()
        self.criterion = nn.CrossEntropyLoss(weight=Utility.class_weights(CONFIG["TRAIN_DIR"], self.device))
        self._tf = Utility.get_transform()
        self._trained = False

    def train(self):
        trainer = Trainer(self.device)
        self.model = trainer.run()
        self._trained = True

    def load(self, path: str = None):
        path = path or str(Path(CONFIG["SAVE_DIR"]) / "best_model.pt")
        Utility.load_checkpoint(path, self.model, device=self.device)
        self.model.eval()
        self._trained = True

    def save(self, path: str = None):
        path = path or str(Path(CONFIG["SAVE_DIR"]) / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
        Utility.save_checkpoint(path, self.model, metrics={"class_names": self.names})

    def predict(self, img, top_k: int = 5) -> dict:
        if not self._trained: raise RuntimeError("Model not trained. Call train() or load() first.")
        from PIL import Image as PIL_Image
        if isinstance(img, str): img = PIL_Image.open(img).convert("RGB")
        if isinstance(img, PIL_Image.Image): t = self._tf(img).unsqueeze(0).to(self.device)
        elif isinstance(img, torch.Tensor): t = (img.unsqueeze(0) if img.dim() == 3 else img).to(self.device)
        else: raise TypeError(f"Unsupported type: {type(img)}")
        self.model.eval()
        with torch.no_grad():
            logits, scores = self.model(t)
            probs = torch.softmax(logits, 1)[0]
        k = min(top_k, len(self.names))
        p, i = probs.topk(k)
        p, i = p.cpu().tolist(), i.cpu().tolist()
        return {"predicted_class": self.names[i[0]], "predicted_index": i[0], "confidence": round(p[0], 6), "top_k_classes": [self.names[x] for x in i], "top_k_probs": [round(x, 6) for x in p], "prototype_scores": scores[0].cpu().tolist()}

    def evaluate(self, split: str = "test") -> dict:
        if not self._trained: raise RuntimeError("Model not trained. Call train() or load() first.")
        return Utility.evaluate(self.model, Utility.build_loader(split), self.criterion, self.device, split)

    def summary(self) -> dict:
        total = sum(p.numel() for p in self.model.parameters())
        info = {"backbone": CONFIG["BACKBONE"], "num_classes": self.model.num_classes, "total_params": total, "device": str(self.device)}
        for k, v in info.items(): log.info(f"  {k:<22}: {v}")
        return info

# =============================================================================
# EXECUTION
# =============================================================================

# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
    M = Model()
    M.summary()
    M.train()
    results = M.evaluate("test")
    print(f"Final Test Accuracy: {results['top1_acc']:.4f}")
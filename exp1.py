
import os, json, time, random, math
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import timm

try:
    from torchinfo import summary as ti_summary
    HAS_TORCHINFO = True
except ImportError:
    HAS_TORCHINFO = False

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TRAIN_DIR = '/kaggle/input/datasets/entishar/disease-dataset/new_disease_dataset/train'
VAL_DIR   = '/kaggle/input/datasets/entishar/disease-dataset/new_disease_dataset/val'
TEST_DIR  = '/kaggle/input/datasets/entishar/disease-dataset/new_disease_dataset/test'
CHECKPOINT_DIR  = "/kaggle/working/ablation_ckpts"
RESULTS_FILE    = "/kaggle/working/ablation_results.csv"

NUM_CLASSES     = 38
PROTO_PER_CLS   = 5
PROTO_DIM       = 256
IMG_SIZE        = 224
BATCH_SIZE      = 32
BASE_SEED       = 42
N_ABLATION_SEEDS = 1       # set to 1 for speed, 3 for stats

PHASE1_EPOCHS   = 10
PHASE2_EPOCHS   = 60
PHASE4_EPOCHS   = 10
PUSH_EVERY      = 2
PATIENCE        = 12

LR_BACKBONE     = 2e-5
LR_ADDON        = 1e-3
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTH    = 0.1
MIXUP_ALPHA     = 0.2
CUTMIX_ALPHA    = 1.0
MIX_PROB        = 0.5

START_VARIANT   = 3
END_VARIANT     = 5

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ─── SEED ─────────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ─── DATA ─────────────────────────────────────────────────────────────────────
def get_loaders():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    
    # ── FIX IS HERE: Using the specific directory variables instead of DATASET_ROOT ──
    train_ds = datasets.ImageFolder(TRAIN_DIR, train_tf)
    val_ds   = datasets.ImageFolder(VAL_DIR,   eval_tf)
    test_ds  = datasets.ImageFolder(TEST_DIR,  eval_tf)
    
    kw = dict(num_workers=4, pin_memory=True)
    return (DataLoader(train_ds, BATCH_SIZE, shuffle=True,  **kw),
            DataLoader(val_ds,   BATCH_SIZE, shuffle=False, **kw),
            DataLoader(test_ds,  BATCH_SIZE, shuffle=False, **kw),
            train_ds)

# ─── COMPONENTS ───────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(ch, ch//r, bias=False), nn.ReLU(inplace=True),
            nn.Linear(ch//r, ch, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c = x.shape[:2]
        y = self.pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1, 1)


class SRM(nn.Module):
    def __init__(self, in_ch=768, out_ch=256, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.Dropout2d(dropout))
    def forward(self, x): return self.block(x)


class SimpleProj(nn.Module):
    """1×1 projection for variants without SRM."""
    def __init__(self, in_ch=768, out_ch=256):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.block(x)


class PrototypeLayer(nn.Module):
    """Supports 'cosine' and 'l2' (ProtoPNet log-transform) similarity."""
    def __init__(self, P, D, similarity='cosine'):
        super().__init__()
        self.P, self.D, self.similarity = P, D, similarity
        self.vectors = nn.Parameter(torch.randn(P, D) * 0.02)

    def forward(self, f):
        B, D, H, W = f.shape
        patches = f.view(B, D, -1).permute(0, 2, 1)          # B×S×D
        if self.similarity == 'cosine':
            fn = F.normalize(patches, p=2, dim=2)             # B×S×D
            pn = F.normalize(self.vectors, p=2, dim=1)        # P×D
            sim = torch.einsum('bsd,pd->bsp', fn, pn)         # B×S×P
            out, _ = sim.max(dim=1)
        else:   # l2
            f_sq  = (patches**2).sum(2, keepdim=True)          # B×S×1
            p_sq  = (self.vectors**2).sum(1).unsqueeze(0)       # 1×P
            cross = torch.einsum('bsd,pd->bsp', patches, self.vectors)
            d2    = (f_sq + p_sq - 2*cross).clamp(min=0)       # B×S×P
            # FIX: ProtoPNet log similarity — higher is more similar
            sim_l2 = torch.log((d2 + 1.0) / (d2 + 1e-4))
            out, _ = sim_l2.max(dim=1)                          # B×P
        return out


class AblationModel(nn.Module):
    def __init__(self, num_classes=38, K=5, D=256,
                 backbone='convnext_tiny',
                 use_se=True, use_srm=True, similarity='cosine'):
        super().__init__()
        self.num_classes, self.K, self.D = num_classes, K, D
        P = num_classes * K

        if backbone == 'resnet50':
            base   = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.backbone = nn.Sequential(*list(base.children())[:-2])
            bb_ch  = 2048
        else:
            self.backbone = timm.create_model(
                'convnext_tiny', pretrained=True, features_only=False)
            self.backbone.head = nn.Identity()
            bb_ch = 768

        self.use_se = use_se
        if use_se: self.se = SEBlock(bb_ch)

        self.use_srm = use_srm
        self.proj = SRM(bb_ch, D) if use_srm else SimpleProj(bb_ch, D)

        self.proto = PrototypeLayer(P, D, similarity)

        self.fc = nn.Linear(P, num_classes, bias=False)
        with torch.no_grad():
            self.fc.weight.fill_(-0.5)
            for c in range(num_classes):
                self.fc.weight[c, c*K:(c+1)*K] = 1.0

    def features(self, x):
        if hasattr(self.backbone, 'forward_features'):
            f = self.backbone.forward_features(x)
        else:
            f = self.backbone(x)
        return f   # B×C×H×W

    def forward(self, x):
        f = self.features(x)
        if self.use_se: f = self.se(f)
        f = self.proj(f)
        sim = self.proto(f)
        return self.fc(sim), sim

    def push_forward(self, x):
        with torch.no_grad():
            f = self.features(x)
            if self.use_se: f = self.se(f)
            f = self.proj(f)
        return f

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ─── MIXUP / CUTMIX ───────────────────────────────────────────────────────────
def apply_mixup_cutmix(imgs, targets, num_classes, device,
                       use_mixup=True):
    """
    FIX: Was defined in original but never called.
    Returns mixed imgs and soft targets (B×C floats).
    """
    if not use_mixup or random.random() > MIX_PROB:
        # one-hot hard targets
        soft = torch.zeros(imgs.size(0), num_classes, device=device)
        soft.scatter_(1, targets.unsqueeze(1), 1.0)
        return imgs, soft

    B = imgs.size(0)
    use_cutmix = random.random() > 0.5
    alpha = CUTMIX_ALPHA if use_cutmix else MIXUP_ALPHA
    lam   = float(np.random.beta(alpha, alpha))
    idx   = torch.randperm(B, device=device)

    imgs_b, targets_b = imgs[idx], targets[idx]

    if use_cutmix:
        cut_rat = math.sqrt(1.0 - lam)
        cut_w   = int(IMG_SIZE * cut_rat)
        cut_h   = int(IMG_SIZE * cut_rat)
        cx, cy  = random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE)
        x1 = max(cx - cut_w//2, 0); x2 = min(cx + cut_w//2, IMG_SIZE)
        y1 = max(cy - cut_h//2, 0); y2 = min(cy + cut_h//2, IMG_SIZE)
        imgs = imgs.clone()
        imgs[:, :, y1:y2, x1:x2] = imgs_b[:, :, y1:y2, x1:x2]
        lam = 1 - (x2-x1)*(y2-y1)/(IMG_SIZE**2)
    else:
        imgs = lam * imgs + (1-lam) * imgs_b

    # Soft targets (B×C)
    soft_a = torch.zeros(B, num_classes, device=device).scatter_(
        1, targets.unsqueeze(1), 1.0)
    soft_b = torch.zeros(B, num_classes, device=device).scatter_(
        1, targets_b.unsqueeze(1), 1.0)
    soft = lam * soft_a + (1-lam) * soft_b
    return imgs, soft

# ─── LOSS ─────────────────────────────────────────────────────────────────────
def ce_soft(logits, soft_targets, smoothing=0.1, num_classes=38):
    """Cross-entropy with soft labels + label smoothing."""
    smooth = (1 - smoothing) * soft_targets + smoothing / num_classes
    return -(smooth * F.log_softmax(logits, dim=1)).sum(1).mean()


def cluster_loss(sim, hard_targets, K):
    """
    PERFORMANCE FIX: the original looped over every sample in the batch
    in Python (`for i in range(sim.size(0))`), each iteration doing a
    GPU->Python sync via indexing with a Python int. Across the full
    pipeline (9 ablation variants x 3 seeds, 5 multiseed runs, PIPNet
    baselines) this loop executes at the per-batch level for the entire
    duration of every training run — tens of millions of iterations in
    aggregate. Vectorized via gather: build the own-class prototype
    index for every sample in one shot, gather, then max+mean.
    Numerically identical result, no Python-level looping.
    """
    B, P = sim.shape
    proto_offsets = torch.arange(K, device=sim.device).unsqueeze(0)   # 1×K
    idx = hard_targets.unsqueeze(1) * K + proto_offsets                # B×K
    own_sims = torch.gather(sim, 1, idx)                                # B×K
    return -own_sims.max(dim=1).values.mean()


def sep_loss(sim, hard_targets, num_classes, K):
    """
    PERFORMANCE FIX: same issue as cluster_loss above — vectorized via
    a one-hot own-class mask expanded to prototype width, instead of a
    per-sample Python loop building a fresh boolean mask every iteration.
    """
    B, P = sim.shape
    own_mask = F.one_hot(hard_targets, num_classes).repeat_interleave(
        K, dim=1).bool()                                                # B×P
    masked_sim = sim.masked_fill(own_mask, float('-inf'))
    return masked_sim.max(dim=1).values.mean()

# ─── PROTOTYPE PUSH ───────────────────────────────────────────────────────────
@torch.no_grad()
def push_prototypes(model, loader, device):
    model.eval()
    P, D = model.proto.P, model.D
    best_sim = torch.full((P,), -1e9)
    best_vec = torch.zeros(P, D)

    for imgs, _ in loader:
        feats = model.push_forward(imgs.to(device))    # B×D×H×W
        B, _, H, W = feats.shape
        patches = feats.view(B, D, -1).permute(0,2,1)  # B×S×D
        fn = F.normalize(patches, p=2, dim=2)
        pn = F.normalize(model.proto.vectors, p=2, dim=1)
        sim = torch.einsum('bsd,pd->bsp', fn, pn)
        # max over (B, S) jointly
        sim_flat = sim.view(-1, P)
        max_sim, max_idx = sim_flat.max(dim=0)
        for j in range(P):
            if max_sim[j] > best_sim[j]:
                best_sim[j] = max_sim[j]
                bi  = max_idx[j].item() // (H*W)
                s   = max_idx[j].item() % (H*W)
                best_vec[j] = feats[bi, :, s//W, s%W].cpu()

    model.proto.vectors.data.copy_(best_vec.to(device))
    model.train()

# ─── TRAIN / EVAL ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, opt, device, use_mixup=True, l1=0.0):
    model.train()
    tot_loss = correct = total = 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)

        # FIX: actually apply MixUp/CutMix
        imgs_mixed, soft = apply_mixup_cutmix(
            imgs, targets, model.num_classes, device, use_mixup)

        opt.zero_grad()
        logits, sim = model(imgs_mixed)
        loss = ce_soft(logits, soft, LABEL_SMOOTH, model.num_classes)
        # Cluster/sep use hard targets (not mixed)
        loss += cluster_loss(sim, targets, model.K)
        loss += 0.2 * sep_loss(sim, targets, model.num_classes, model.K)
        if l1 > 0:
            loss += l1 * model.fc.weight.abs().sum()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        tot_loss += loss.item() * imgs.size(0)
        pred = logits.argmax(1)
        # Accuracy measured against hard targets
        correct += (pred == targets).sum().item()
        total   += imgs.size(0)
    return tot_loss/total, correct/total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    c = t = 0
    for imgs, targets in loader:
        logits, _ = model(imgs.to(device))
        c += (logits.argmax(1) == targets.to(device)).sum().item()
        t += imgs.size(0)
    return c/t

# ─── FULL TRAINING PIPELINE ───────────────────────────────────────────────────
def train_variant(name, cfg, train_ld, val_ld, test_ld, device,
                  seed=42, skip_phase4=False, use_mixup=True):
    print(f"\n{'='*60}\n  {name}  [seed={seed}]\n{'='*60}")
    set_seed(seed)

    model = AblationModel(
        backbone   = cfg.get('backbone','convnext_tiny'),
        use_se     = cfg['use_se'],
        use_srm    = cfg['use_srm'],
        similarity = cfg['similarity'],
    ).to(device)

    n_params = model.count_params()

    # Phase 1 — Warmup
    for p in model.backbone.parameters(): p.requires_grad_(False)
    opt1 = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR_ADDON, weight_decay=WEIGHT_DECAY)
    for ep in range(1, PHASE1_EPOCHS+1):
        train_epoch(model, train_ld, opt1, device, use_mixup=use_mixup)
    print("  Phase 1 complete")

    # Phase 2 — Joint Training
    for p in model.backbone.parameters(): p.requires_grad_(True)
    bb_params  = list(model.backbone.parameters())
    add_params = [p for n,p in model.named_parameters() if 'backbone' not in n]
    opt2 = optim.AdamW([
        {'params': bb_params,  'lr': LR_BACKBONE, 'weight_decay': 5e-4},
        {'params': add_params, 'lr': LR_ADDON,    'weight_decay': WEIGHT_DECAY},
    ])
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=60, eta_min=LR_ADDON*0.01)

    best_val, best_ep, best_state, no_imp = 0.0, 0, None, 0
    for ep in range(1, PHASE2_EPOCHS+1):
        if ep % PUSH_EVERY == 0:
            push_prototypes(model, train_ld, device)
        tr_loss, tr_acc = train_epoch(
            model, train_ld, opt2, device, use_mixup=use_mixup)
        val_acc = evaluate(model, val_ld, device)
        sched.step()
        print(f"  P2 ep={ep:2d} train={tr_acc:.4f} val={val_acc:.4f}")
        if val_acc > best_val + 1e-4:
            best_val, best_ep, no_imp = val_acc, ep, 0
            best_state = {k: v.clone() for k,v in model.state_dict().items()}
        else:
            no_imp += 1
        if no_imp >= PATIENCE:
            print(f"  Early stop ep={ep}  best={best_ep}")
            break

    if best_state: model.load_state_dict(best_state)
    gen_gap_before_p4 = evaluate(model, val_ld, device) - \
                        evaluate(model, train_ld, device)

    # Phase 4 — FC Sparsity
    if not skip_phase4:
        for p in model.parameters(): p.requires_grad_(False)
        model.fc.weight.requires_grad_(True)
        opt4 = optim.AdamW([model.fc.weight], lr=1e-5,
                           weight_decay=WEIGHT_DECAY)
        for ep in range(1, PHASE4_EPOCHS+1):
            train_epoch(model, train_ld, opt4, device,
                        use_mixup=False, l1=1e-5)

    test_acc = evaluate(model, test_ld, device)
    val_acc_final = evaluate(model, val_ld, device)
    print(f"\n  ✓ {name} [seed={seed}]  test={test_acc*100:.2f}%")

    ckpt = f"{name.replace(' ','_').replace(':','')}_s{seed}.pth"
    torch.save({'model': model.state_dict(), 'test_acc': test_acc,
                'config': cfg}, os.path.join(CHECKPOINT_DIR, ckpt))

    return {
        'variant':      name,
        'seed':         seed,
        'backbone':     cfg.get('backbone','convnext_tiny'),
        'use_se':       cfg['use_se'],
        'use_srm':      cfg['use_srm'],
        'similarity':   cfg['similarity'],
        'phase4':       not skip_phase4,
        'use_mixup':    use_mixup,
        'n_params_M':   round(n_params/1e6, 2),
        'best_val':     round(best_val*100, 2),
        'test_acc':     round(test_acc*100, 2),
        'gen_gap':      round(gen_gap_before_p4*100, 3),
        'delta_vs_full': round((test_acc - 0.9965)*100, 2),
    }

# ─── VARIANT DEFINITIONS ──────────────────────────────────────────────────────
VARIANTS = [
    # (name, config, skip_phase4, use_mixup)
    ("V0: ResNet-50 L2 (ProtoPNet baseline)",
     {'backbone':'resnet50','use_se':False,'use_srm':False,'similarity':'l2'},
     False, True),

    ("V1: ConvNeXt L2 (no SE, no SRM)",
     {'use_se':False,'use_srm':False,'similarity':'l2'},
     False, True),

    ("V2: ConvNeXt Cosine (no SE, no SRM)",
     {'use_se':False,'use_srm':False,'similarity':'cosine'},
     False, True),

    ("V3: ConvNeXt SE+Cosine (no SRM)",
     {'use_se':True,'use_srm':False,'similarity':'cosine'},
     False, True),

    ("V4: ConvNeXt SRM+Cosine (no SE)",
     {'use_se':False,'use_srm':True,'similarity':'cosine'},
     False, True),

    # NEW: isolates cosine vs L2 when SE is present
    ("V5: ConvNeXt SE+L2 (no SRM)",
     {'use_se':True,'use_srm':False,'similarity':'l2'},
     False, True),

    ("V6: Full model, no Phase 4",
     {'use_se':True,'use_srm':True,'similarity':'cosine'},
     True, True),

    # NEW: ablates MixUp/CutMix contribution
    ("V7: Full model, no MixUp/CutMix",
     {'use_se':True,'use_srm':True,'similarity':'cosine'},
     False, False),

    # Full POWER-ProtoPNet — known result, but re-run for multi-seed
    ("V8: Full POWER-ProtoPNet",
     {'use_se':True,'use_srm':True,'similarity':'cosine'},
     False, True),
]

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  Seeds per variant: {N_ABLATION_SEEDS}")

    train_ld, val_ld, test_ld, _ = get_loaders()
    all_results = []

    for i, (name, cfg, skip_p4, use_mix) in enumerate(VARIANTS):
        if i < START_VARIANT or i > END_VARIANT:
            print(f"Skipping V{i}: {name}")
            continue

        seed_results = []
        for s in range(N_ABLATION_SEEDS):
            seed = BASE_SEED + s
            ckpt_key = f"{name.replace(' ','_').replace(':','')}_s{seed}.pth"
            ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_key)
            if os.path.exists(ckpt_path):
                ck = torch.load(ckpt_path, map_location='cpu')
                r = {'variant': name, 'seed': seed,
                     'test_acc': round(ck['test_acc']*100, 2)}
                print(f"  {name} seed={seed} cached: {r['test_acc']}%")
                seed_results.append(r)
                all_results.append(r)
                continue

            t0  = time.time()
            res = train_variant(name, cfg, train_ld, val_ld,
                                test_ld, device, seed, skip_p4, use_mix)
            res['runtime_h'] = round((time.time()-t0)/3600, 2)
            seed_results.append(res)
            all_results.append(res)

        # Summary across seeds
        accs = [r['test_acc'] for r in seed_results]
        std_val = np.std(accs, ddof=1) if len(accs) > 1 else 0.0
        print(f"\n  {name}: mean={np.mean(accs):.2f} "
              f"std={std_val:.2f}  n={len(accs)}")

    # Aggregate: mean ± std per variant
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_FILE, index=False)

    agg = (df.groupby('variant')['test_acc']
             .agg(['mean','std','count'])
             .round(3)
             .reset_index())
    agg.columns = ['Variant','Mean(%)','Std(%)','n']

    print("\n" + "="*70)
    print("ABLATION SUMMARY (mean ± std)")
    print("="*70)
    for _, row in agg.iterrows():
        print(f"  {row['Variant']:<45}  "
              f"{row['Mean(%)']:.2f} ± {row['Std(%)']:.2f}%  "
              f"(n={int(row['n'])})")

    agg.to_csv(RESULTS_FILE.replace('.csv','_summary.csv'), index=False)
    print(f"\nFull results: {RESULTS_FILE}")
    print(f"Summary:      {RESULTS_FILE.replace('.csv','_summary.csv')}")


if __name__ == "__main__":
    main()
"""
preprocess.py  –  PlantWildV2  →  ProtoP Net (EfficientNetB3 backbone)
=======================================================================
Steps
-----
1. Scan dataset root, keep TOP-80 classes by image count.
2. Split each class so val and test each have EXACTLY 50 images
   (or as many as available when the class is small).
3. Augment train set so every class reaches ~1000 images.
4. Write a detailed report to  details.txt.

Expected dataset layout (ImageFolder-style):
    <dataset_root>/
        <class_name>/
            img1.jpg
            img2.png
            ...

Edit the CONFIG dict below, then run:
    python preprocess.py
"""

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG  ← only section you need to edit
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # Path to the raw PlantWildV2 dataset (ImageFolder layout)
    "src": "/path/to/PlantWildV2",

    # Where to write train/ val/ test/ and details.txt
    "dst": "/path/to/output",

    # Number of top classes to keep (ranked by image count)
    "top_k": 80,

    # Target images per class in val and test (hard cap)
    "val_size": 50,
    "test_size": 50,

    # Target images per class in train AFTER augmentation
    "train_target": 1000,

    # Reproducibility
    "seed": 42,
}
# ─────────────────────────────────────────────────────────────────────────────

import os
import random
import shutil
import textwrap
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T


# ─────────────────────────────────────────────────────────────────────────────
# Image discovery
# ─────────────────────────────────────────────────────────────────────────────
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def discover_classes(src: Path) -> dict[str, list[Path]]:
    """Return {class_name: [image_paths]} for every sub-directory."""
    class_map: dict[str, list[Path]] = {}
    for cls_dir in sorted(src.iterdir()):
        if not cls_dir.is_dir():
            continue
        imgs = [
            f for f in cls_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMG_EXTS
        ]
        if imgs:
            class_map[cls_dir.name] = imgs
    return class_map


def select_top_k(class_map: dict, k: int) -> tuple[dict, dict]:
    """Return (kept, removed) sorted by descending image count."""
    ranked  = sorted(class_map.items(), key=lambda x: len(x[1]), reverse=True)
    kept    = dict(ranked[:k])
    removed = dict(ranked[k:])
    return kept, removed


# ─────────────────────────────────────────────────────────────────────────────
# Splitting
# ─────────────────────────────────────────────────────────────────────────────
def split_class(
    images: list[Path],
    val_size: int,
    test_size: int,
    rng: random.Random,
) -> tuple[list[Path], list[Path], list[Path]]:
    """
    Returns (train, val, test).

    Rules
    -----
    * val  gets exactly min(val_size,  n) images
    * test gets exactly min(test_size, n - val) images
    * train gets everything that remains (≥ 1 guaranteed)
    * If total is too small, val/test are shrunk to leave ≥ 1 train image.
    """
    imgs = images.copy()
    rng.shuffle(imgs)

    n           = len(imgs)
    actual_val  = min(val_size,  n)
    actual_test = min(test_size, n - actual_val)

    # Guarantee at least 1 train image
    while actual_val > 0 and (n - actual_val - actual_test < 1):
        actual_val -= 1
    while actual_test > 0 and (n - actual_val - actual_test < 1):
        actual_test -= 1

    test  = imgs[:actual_test]
    val   = imgs[actual_test: actual_test + actual_val]
    train = imgs[actual_test + actual_val:]
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation pipeline  (PlantWild / ProtoP Net / EfficientNetB3)
# ─────────────────────────────────────────────────────────────────────────────
#
# EfficientNetB3 native input  : 300 × 300
# PlantWild domain             : outdoor plant photos — colour, lighting, angle
# ProtoP Net constraint        : keep spatial distortions mild so patch
#                                prototypes remain coherent
#
AUG_TRANSFORM = T.Compose([
    T.RandomResizedCrop(300, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.2),
    T.RandomRotation(degrees=30),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.05),
    T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
    T.RandomPerspective(distortion_scale=0.2, p=0.3),
])


def augment_image(src_path: Path, dst_path: Path) -> None:
    img = Image.open(src_path).convert("RGB")
    AUG_TRANSFORM(img).save(dst_path, quality=95)


def augment_class_to_target(
    train_images: list[Path],
    dst_cls_dir: Path,
    target: int,
    cls_name: str,
    rng: random.Random,
) -> int:
    """
    Copy originals into dst_cls_dir, then synthesise augmented images
    until the folder reaches `target`.  Returns the final image count.
    """
    dst_cls_dir.mkdir(parents=True, exist_ok=True)

    # ── copy originals ──────────────────────────────────────────────────────
    for img_path in train_images:
        shutil.copy2(img_path, dst_cls_dir / img_path.name)

    n_orig = len(train_images)
    if n_orig >= target:
        return n_orig

    needed = target - n_orig
    pool   = train_images.copy()

    label = f"  Aug [{cls_name[:28]:<28}]"
    for i in tqdm(range(needed), desc=label, leave=False, unit="img"):
        src  = rng.choice(pool)
        dst  = dst_cls_dir / f"{src.stem}_aug{i:05d}{src.suffix}"
        # resolve rare name collisions
        while dst.exists():
            dst = dst_cls_dir / f"{src.stem}_aug{i:05d}_{rng.randint(0, 9999)}{src.suffix}"
        try:
            augment_image(src, dst)
        except Exception as exc:
            tqdm.write(f"    [WARN] {src.name}: {exc}")

    return len(list(dst_cls_dir.iterdir()))


# ─────────────────────────────────────────────────────────────────────────────
# Copy helper
# ─────────────────────────────────────────────────────────────────────────────
def copy_split(images: list[Path], dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for img in images:
        shutil.copy2(img, dst_dir / img.name)


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────
def write_report(
    dst: Path,
    cfg: dict,
    kept: dict,
    removed: dict,
    split_stats: list[dict],
) -> None:
    report_path = dst / "details.txt"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    L = []
    L += [
        "=" * 78,
        "  PlantWildV2 Preprocessing Report",
        f"  Generated : {now}",
        "=" * 78,
        "",
        "── CONFIG ──────────────────────────────────────────────────────────────────",
        f"  src              : {cfg['src']}",
        f"  dst              : {cfg['dst']}",
        f"  top_k            : {cfg['top_k']}",
        f"  val_size  (max)  : {cfg['val_size']}",
        f"  test_size (max)  : {cfg['test_size']}",
        f"  train_target     : {cfg['train_target']}  (after augmentation)",
        f"  seed             : {cfg['seed']}",
        "",
    ]

    total_train = sum(s["aug_train"] for s in split_stats)
    total_val   = sum(s["val"]       for s in split_stats)
    total_test  = sum(s["test"]      for s in split_stats)
    L += [
        "── Dataset Summary ─────────────────────────────────────────────────────────",
        f"  Classes kept     : {len(kept)}",
        f"  Classes removed  : {len(removed)}",
        f"  Total train(aug) : {total_train:,}",
        f"  Total val        : {total_val:,}",
        f"  Total test       : {total_test:,}",
        f"  Grand total      : {total_train + total_val + total_test:,}",
        "",
    ]

    # per-class table
    L.append("── Kept Classes ────────────────────────────────────────────────────────────")
    L.append(f"  {'#':>4}  {'Class':<45} {'Raw':>6} {'Tr(orig)':>8} {'Tr(aug)':>8} {'Val':>5} {'Test':>5}")
    L.append("  " + "-" * 74)
    for i, s in enumerate(split_stats, 1):
        L.append(
            f"  {i:>4}  {s['class']:<45} {s['raw']:>6} "
            f"{s['orig_train']:>8} {s['aug_train']:>8} "
            f"{s['val']:>5} {s['test']:>5}"
        )
    L.append("")

    # removed classes
    L.append("── Removed Classes ─────────────────────────────────────────────────────────")
    L.append(f"  {'#':>4}  {'Class':<45} {'Images':>6}")
    L.append("  " + "-" * 58)
    for i, (cls, imgs) in enumerate(
        sorted(removed.items(), key=lambda x: len(x[1]), reverse=True), 1
    ):
        L.append(f"  {i:>4}  {cls:<45} {len(imgs):>6}")
    L.append("")

    L.append("── Augmentation Pipeline ───────────────────────────────────────────────────")
    for line in [
        "RandomResizedCrop(300, scale=(0.60-1.00), ratio=(0.75-1.33))",
        "RandomHorizontalFlip(p=0.50)",
        "RandomVerticalFlip(p=0.20)",
        "RandomRotation(30°)",
        "ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)",
        "RandomGrayscale(p=0.05)",
        "GaussianBlur(kernel=5, sigma=(0.1-2.0), p=0.30)",
        "RandomPerspective(distortion=0.20, p=0.30)",
    ]:
        L.append("    " + line)
    L += ["", "=" * 78]

    report_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\n  ✓  details.txt → {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    cfg = CONFIG
    rng = random.Random(cfg["seed"])
    src = Path(cfg["src"])
    dst = Path(cfg["dst"])
    dst.mkdir(parents=True, exist_ok=True)

    # ── 1. Discover ───────────────────────────────────────────────────────────
    print("\n[1/4] Scanning dataset …")
    class_map = discover_classes(src)
    print(f"      Found {len(class_map)} classes total.")

    # ── 2. Select top-K ───────────────────────────────────────────────────────
    print(f"\n[2/4] Selecting top-{cfg['top_k']} classes by image count …")
    kept, removed = select_top_k(class_map, cfg["top_k"])
    counts = sorted([len(v) for v in kept.values()], reverse=True)
    print(f"      Kept    : {len(kept)}  classes  (range {counts[-1]}–{counts[0]} images)")
    print(f"      Removed : {len(removed)} classes")

    # ── 3. Split + Copy + Augment ─────────────────────────────────────────────
    print(f"\n[3/4] Splitting → copying → augmenting …")
    split_stats: list[dict] = []

    for cls_name, imgs in tqdm(kept.items(), desc="Classes", unit="cls", ncols=80):
        train_imgs, val_imgs, test_imgs = split_class(
            imgs, cfg["val_size"], cfg["test_size"], rng
        )

        copy_split(val_imgs,  dst / "val"  / cls_name)
        copy_split(test_imgs, dst / "test" / cls_name)

        aug_count = augment_class_to_target(
            train_imgs,
            dst / "train" / cls_name,
            cfg["train_target"],
            cls_name,
            rng,
        )

        split_stats.append({
            "class":      cls_name,
            "raw":        len(imgs),
            "orig_train": len(train_imgs),
            "aug_train":  aug_count,
            "val":        len(val_imgs),
            "test":       len(test_imgs),
        })

    # ── 4. Report ─────────────────────────────────────────────────────────────
    print("\n[4/4] Writing details.txt …")
    write_report(dst, cfg, kept, removed, split_stats)

    total_train = sum(s["aug_train"] for s in split_stats)
    total_val   = sum(s["val"]       for s in split_stats)
    total_test  = sum(s["test"]      for s in split_stats)
    print("\n" + "=" * 60)
    print("  Done!")
    print(f"  Output : {dst}")
    print(f"  train/ : {total_train:,} images  ({len(kept)} classes)")
    print(f"  val/   : {total_val:,} images  ({len(kept)} classes)")
    print(f"  test/  : {total_test:,} images  ({len(kept)} classes)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

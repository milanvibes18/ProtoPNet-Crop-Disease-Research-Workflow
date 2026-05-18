"""
Val / Test Set Balancing Script
--------------------------------
Finds the largest class in each split and tops up every other class
to match it using very mild augmentation (no strong transforms).

Rules:
  - Originals are NEVER deleted or modified.
  - Only augmented files (prefixed with AUG_PREFIX) are added.
  - If a class already has enough total images it is skipped.
  - The target is AUTO-DETECTED as the max class size in that split.

Mild augmentation only:
  - Small random crop          (scale 0.85–1.0)
  - Random horizontal flip     (p=0.5)
  - Tiny rotation              (±10°)
  - Very slight colour jitter
  No blur, no erasing, no vertical flip, no aggressive crops.

Install:
  pip install Pillow tqdm torchvision torch
"""

import logging
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    # Paths to val and test split folders.
    # Each must follow the ImageFolder structure: <split>/<class_name>/<images>
    "VAL_DIR":  "./Dataset/plantwild/val",
    "TEST_DIR": "./Dataset/plantwild/test",

    # Set to False to skip that split entirely.
    "PROCESS_VAL":  True,
    "PROCESS_TEST": True,

    # Mild augmentation parameters — intentionally conservative
    "CROP_SCALE_MIN":   0.85,   # very small crop variation
    "CROP_SCALE_MAX":   1.0,
    "FLIP_H_PROB":      0.5,
    "ROTATION_DEGREES": 10,     # tiny rotation only
    "COLOR_BRIGHTNESS": 0.1,
    "COLOR_CONTRAST":   0.1,
    "COLOR_SATURATION": 0.05,
    "COLOR_HUE":        0.02,
    "OUTPUT_IMG_SIZE":  (224, 224),

    "SAVE_FORMAT":   "JPEG",
    "SAVE_QUALITY":  95,
    "AUG_PREFIX":    "aug",     # generated files start with this

    "RANDOM_SEED":   42,
    "LOG_FILE":      "augment_val_test.log",
    "SUPPORTED_EXT": {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"},
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(CONFIG["LOG_FILE"], mode="w", encoding="utf-8"),
        logging.StreamHandler(
            stream=open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
        ),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------

def build_mild_pipeline() -> T.Compose:
    cfg = CONFIG
    pipeline = T.Compose([
        T.RandomResizedCrop(
            size=cfg["OUTPUT_IMG_SIZE"],
            scale=(cfg["CROP_SCALE_MIN"], cfg["CROP_SCALE_MAX"]),
            ratio=(0.9, 1.1),                   # nearly square — very conservative
            interpolation=T.InterpolationMode.LANCZOS,
        ),
        T.RandomHorizontalFlip(p=cfg["FLIP_H_PROB"]),
        T.RandomRotation(
            degrees=cfg["ROTATION_DEGREES"],
            interpolation=T.InterpolationMode.BILINEAR,
            fill=0,
        ),
        T.ColorJitter(
            brightness=cfg["COLOR_BRIGHTNESS"],
            contrast=cfg["COLOR_CONTRAST"],
            saturation=cfg["COLOR_SATURATION"],
            hue=cfg["COLOR_HUE"],
        ),
    ])
    log.info(
        f"Mild pipeline: crop={cfg['CROP_SCALE_MIN']}–{cfg['CROP_SCALE_MAX']}, "
        f"flip=p{cfg['FLIP_H_PROB']}, rotation=±{cfg['ROTATION_DEGREES']}°, "
        f"jitter=brightness{cfg['COLOR_BRIGHTNESS']}/contrast{cfg['COLOR_CONTRAST']}"
    )
    return pipeline


# ---------------------------------------------------------------------------
# Scan a split directory
# ---------------------------------------------------------------------------

def scan_split(split_dir: Path) -> dict[str, dict]:
    """
    Returns {class_name: {"originals": [...], "augmented": [...]}}.
    """
    if not split_dir.exists():
        log.error(f"Directory not found: {split_dir}")
        sys.exit(1)

    class_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
    if not class_dirs:
        log.error(f"No class subfolders found in {split_dir}")
        sys.exit(1)

    valid_ext  = CONFIG["SUPPORTED_EXT"]
    aug_prefix = CONFIG["AUG_PREFIX"]
    result: dict[str, dict] = {}

    for class_dir in class_dirs:
        originals, augmented = [], []
        for f in class_dir.iterdir():
            if f.suffix.lower() not in valid_ext:
                continue
            (augmented if f.stem.startswith(aug_prefix) else originals).append(f)
        result[class_dir.name] = {"originals": originals, "augmented": augmented}

    counts = {cls: len(d["originals"]) + len(d["augmented"]) for cls, d in result.items()}
    max_cls   = max(counts, key=counts.get)
    max_count = counts[max_cls]

    log.info(f"  {len(result)} classes found.")
    log.info(f"  Largest class: '{max_cls}' with {max_count} images  ← this is the target.")

    already_ok = sum(1 for c in counts.values() if c >= max_count)
    needs_topup = len(counts) - already_ok
    log.info(f"  {needs_topup} class(es) need topping up.")

    return result, max_count


# ---------------------------------------------------------------------------
# Balance one split
# ---------------------------------------------------------------------------

def balance_split(
    split_name: str,
    split_dir: Path,
    pipeline: T.Compose,
) -> dict[str, dict]:
    """
    Top up every class to max_count. Returns per-class stats.
    """
    log.info("")
    log.info(f"{'─'*60}")
    log.info(f"  Processing split: {split_name.upper()}  ({split_dir})")
    log.info(f"{'─'*60}")

    class_data, target = scan_split(split_dir)

    cfg         = CONFIG
    save_format = cfg["SAVE_FORMAT"]
    save_quality = cfg["SAVE_QUALITY"]
    prefix      = cfg["AUG_PREFIX"]
    stats: dict[str, dict] = {}
    n_errors = 0

    for cls, data in tqdm(class_data.items(), desc=f"Balancing {split_name}"):
        originals = data["originals"]
        augmented = data["augmented"]
        n_orig    = len(originals)
        current   = n_orig + len(augmented)

        if current >= target:
            stats[cls] = {"original": n_orig, "generated": 0, "total": current, "status": "ok"}
            continue

        if n_orig == 0:
            log.warning(f"  [{cls}] No originals — cannot augment, skipping.")
            stats[cls] = {"original": 0, "generated": 0, "total": 0, "status": "no originals"}
            continue

        n_needed = target - current
        out_dir  = split_dir / cls          # always write into same folder

        # Determine next safe index to avoid filename collisions
        existing_indices = set()
        for f in out_dir.iterdir():
            if f.stem.startswith(prefix):
                parts = f.stem.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    existing_indices.add(int(parts[1]))
        next_idx = (max(existing_indices) + 1) if existing_indices else 0

        # Shuffle originals for even cycling
        pool = originals.copy()
        random.shuffle(pool)

        n_generated = 0
        for i in range(n_needed):
            src_path = pool[i % len(pool)]
            if i > 0 and i % len(pool) == 0:
                random.shuffle(pool)

            try:
                with Image.open(src_path) as img:
                    img = img.convert("RGB")
                aug_img  = pipeline(img)
                out_name = f"{prefix}_{src_path.stem}_{next_idx:05d}.jpg"
                aug_img.save(out_dir / out_name, save_format, quality=save_quality)
                next_idx    += 1
                n_generated += 1
            except Exception as exc:
                n_errors += 1
                log.warning(f"  Skipped {src_path.name}: {exc}")

        stats[cls] = {
            "original":  n_orig,
            "generated": n_generated,
            "total":     current + n_generated,
            "status":    "topped up",
        }

    if n_errors:
        log.warning(f"  {n_errors} error(s) during {split_name} augmentation.")

    return stats, target


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(split_name: str, stats: dict, target: int, split_dir: Path) -> None:
    total_orig = sum(v["original"]  for v in stats.values())
    total_gen  = sum(v["generated"] for v in stats.values())
    total_fin  = sum(v["total"]     for v in stats.values())

    lines = [
        "=" * 65,
        f"SUMMARY — {split_name.upper()}",
        "=" * 65,
        f"Target per class  : {target}",
        f"Total classes     : {len(stats)}",
        f"Original images   : {total_orig:>7}",
        f"Generated images  : {total_gen:>7}",
        f"Final total       : {total_fin:>7}",
        "",
        f"{'Class':<45} {'Orig':>5} {'Gen':>5} {'Total':>6}  Status",
        "-" * 65,
    ]
    for cls, v in sorted(stats.items(), key=lambda x: x[1]["total"]):
        lines.append(
            f"  {cls:<43} {v['original']:>5} {v['generated']:>5} "
            f"{v['total']:>6}  {v['status']}"
        )
    lines.append("=" * 65)
    summary = "\n".join(lines)

    out_path = split_dir.parent / f"augmentation_summary_{split_name}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summary)

    log.info(f"Summary written to: {out_path}")
    print("\n" + summary)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(CONFIG["RANDOM_SEED"])
    np.random.seed(CONFIG["RANDOM_SEED"])

    log.info("=" * 65)
    log.info("  Val / Test Balancing Script  (mild augmentation)")
    log.info("=" * 65)

    pipeline = build_mild_pipeline()

    for split_name, dir_key, enabled_key in [
        ("val",  "VAL_DIR",  "PROCESS_VAL"),
        ("test", "TEST_DIR", "PROCESS_TEST"),
    ]:
        if not CONFIG[enabled_key]:
            log.info(f"Skipping {split_name} (PROCESS_{split_name.upper()}=False).")
            continue

        split_dir = Path(CONFIG[dir_key])
        stats, target = balance_split(split_name, split_dir, pipeline)
        write_summary(split_name, stats, target, split_dir)

    log.info("")
    log.info("Done.")


if __name__ == "__main__":
    main()

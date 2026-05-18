"""
ProtoPNet Dataset Augmentation Pipeline  (v2 — Target-Count Mode)
------------------------------------------------------------------
Balances every class to exactly TARGET_IMAGES_PER_CLASS images.

Logic per class
---------------
  current total  > TARGET  → delete excess augmented files (originals never touched)
  current total == TARGET  → nothing to do
  current total  < TARGET  → generate exactly (TARGET − current) new augmented images,
                             cycling through originals and using stronger augmentation
                             when the gap is large relative to the number of originals.

Val and test sets are never touched.

Augmentations applied (matching the ProtoPNet research workflow):
  - Random resized crop    (scale 0.7–1.0, output 224×224)
  - Random horizontal flip (p=0.5)
  - Random vertical flip   (p=0.3)
  - Random rotation        (±30 degrees)
  - Colour jitter          (brightness, contrast, saturation, hue)
  - Gaussian blur          (p=0.2)
  - Extra (strong mode):   elastic distortion + random erasing

Install dependencies:
  pip install Pillow tqdm numpy torchvision torch
"""

import logging
import math
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Configuration — edit these values before running
# ---------------------------------------------------------------------------

CONFIG = {
    # Input: the train/ folder produced by preprocess.py
    # Expected structure: TRAIN_DIR/<class_name>/<image_files>
    "TRAIN_DIR": "./Dataset/plantwild/train",

    # Where to write augmented images.
    # "same" → saves augmented files into the same class folders as the originals
    #          (recommended — PyTorch ImageFolder sees originals + augmented together)
    # any path → copies originals there and writes augmented files alongside them
    "OUTPUT_MODE": "same",

    # Only used when OUTPUT_MODE is not "same"
    "OUTPUT_DIR": "./Dataset/plantwild_augmented/train",

    # ── TARGET-COUNT MODE ──────────────────────────────────────────────────
    # Set this to the exact number of images you want in every class.
    #   • Classes with MORE images → excess augmented files are deleted.
    #   • Classes with FEWER images → augmented files are generated to top up.
    #   • Originals are NEVER deleted regardless of this setting.
    # Set to None to disable and fall back to the legacy COPIES_PER_IMAGE mode.
    "TARGET_IMAGES_PER_CLASS": 700,

    # Ratio threshold: if a class needs more than this many copies per original
    # to hit the target, the STRONG augmentation pipeline is used instead of
    # the standard one.  e.g. 5 means "need > 5× the originals → go strong".
    "STRONG_AUG_MULTIPLIER_THRESHOLD": 5,

    # ── LEGACY MODE (used only when TARGET_IMAGES_PER_CLASS is None) ───────
    "COPIES_PER_IMAGE": 3,
    "ENABLE_BOOST": True,
    "MIN_SAMPLES_FOR_BOOST": 50,
    "BOOST_COPIES_PER_IMAGE": 7,

    # ── Augmentation parameters ────────────────────────────────────────────
    "FLIP_H_PROB":      0.5,
    "FLIP_V_PROB":      0.3,
    "ROTATION_DEGREES": 30,
    "COLOR_BRIGHTNESS": 0.3,
    "COLOR_CONTRAST":   0.3,
    "COLOR_SATURATION": 0.2,
    "COLOR_HUE":        0.05,
    "CROP_SCALE_MIN":   0.7,
    "CROP_SCALE_MAX":   1.0,
    "BLUR_PROB":        0.2,
    "BLUR_KERNEL_MAX":  5,          # must be odd
    "OUTPUT_IMG_SIZE":  (224, 224),

    # Strong augmentation extras (activated when gap/originals > threshold)
    "STRONG_ROTATION_DEGREES": 45,
    "STRONG_COLOR_BRIGHTNESS": 0.5,
    "STRONG_COLOR_CONTRAST":   0.5,
    "STRONG_COLOR_SATURATION": 0.4,
    "STRONG_COLOR_HUE":        0.1,
    "STRONG_CROP_SCALE_MIN":   0.5,
    "STRONG_BLUR_PROB":        0.4,
    "STRONG_ERASE_PROB":       0.3,   # random erasing

    # Output file settings
    "SAVE_FORMAT":  "JPEG",
    "SAVE_QUALITY": 95,
    "AUG_PREFIX":   "aug",

    "RANDOM_SEED":   42,
    "LOG_FILE":      "augmentation.log",
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
# Step 1 — Scan the training directory
# ---------------------------------------------------------------------------

def scan_train_dir(train_dir: Path) -> dict[str, dict]:
    """
    Collect image paths grouped by class, separating originals from
    previously augmented files (identified by AUG_PREFIX).

    Returns:
        {class_name: {"originals": [...], "augmented": [...]}}
    """
    log.info(f"Scanning training directory: {train_dir}")

    if not train_dir.exists():
        log.error(f"TRAIN_DIR not found: {train_dir}")
        sys.exit(1)

    class_dirs = sorted(d for d in train_dir.iterdir() if d.is_dir())
    if not class_dirs:
        log.error("No class subfolders found in TRAIN_DIR.")
        sys.exit(1)

    valid_ext  = CONFIG["SUPPORTED_EXT"]
    aug_prefix = CONFIG["AUG_PREFIX"]
    result: dict[str, dict] = {}

    for class_dir in tqdm(class_dirs, desc="Scanning classes"):
        originals  = []
        augmented  = []
        for f in class_dir.iterdir():
            if f.suffix.lower() not in valid_ext:
                continue
            if f.stem.startswith(aug_prefix):
                augmented.append(f)
            else:
                originals.append(f)

        if originals or augmented:
            result[class_dir.name] = {
                "originals": originals,
                "augmented": augmented,
            }

    total_orig = sum(len(v["originals"]) for v in result.values())
    total_aug  = sum(len(v["augmented"]) for v in result.values())
    log.info(
        f"Found {total_orig} original + {total_aug} existing augmented images "
        f"across {len(result)} classes."
    )
    return result


# ---------------------------------------------------------------------------
# Step 2 — Build augmentation pipelines
# ---------------------------------------------------------------------------

def build_pipelines() -> tuple[T.Compose, T.Compose]:
    """
    Returns (standard_pipeline, strong_pipeline).

    Standard: used when relatively few copies are needed.
    Strong:   used when the class has very few originals and needs many copies
              — adds stronger colour/crop variation and random erasing to
              prevent the model seeing near-identical images.
    """
    cfg = CONFIG

    standard = T.Compose([
        T.RandomResizedCrop(
            size=cfg["OUTPUT_IMG_SIZE"],
            scale=(cfg["CROP_SCALE_MIN"], cfg["CROP_SCALE_MAX"]),
            ratio=(0.75, 1.333),
            interpolation=T.InterpolationMode.LANCZOS,
        ),
        T.RandomHorizontalFlip(p=cfg["FLIP_H_PROB"]),
        T.RandomVerticalFlip(p=cfg["FLIP_V_PROB"]),
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
        T.RandomApply([
            T.GaussianBlur(kernel_size=cfg["BLUR_KERNEL_MAX"], sigma=(0.1, 2.0))
        ], p=cfg["BLUR_PROB"]),
    ])

    strong = T.Compose([
        T.RandomResizedCrop(
            size=cfg["OUTPUT_IMG_SIZE"],
            scale=(cfg["STRONG_CROP_SCALE_MIN"], cfg["CROP_SCALE_MAX"]),
            ratio=(0.65, 1.50),
            interpolation=T.InterpolationMode.LANCZOS,
        ),
        T.RandomHorizontalFlip(p=cfg["FLIP_H_PROB"]),
        T.RandomVerticalFlip(p=cfg["FLIP_V_PROB"]),
        T.RandomRotation(
            degrees=cfg["STRONG_ROTATION_DEGREES"],
            interpolation=T.InterpolationMode.BILINEAR,
            fill=0,
        ),
        T.ColorJitter(
            brightness=cfg["STRONG_COLOR_BRIGHTNESS"],
            contrast=cfg["STRONG_COLOR_CONTRAST"],
            saturation=cfg["STRONG_COLOR_SATURATION"],
            hue=cfg["STRONG_COLOR_HUE"],
        ),
        T.RandomApply([
            T.GaussianBlur(kernel_size=cfg["BLUR_KERNEL_MAX"], sigma=(0.1, 2.5))
        ], p=cfg["STRONG_BLUR_PROB"]),
        # Random erasing: simulates occlusion / missing patches
        T.ToTensor(),
        T.RandomErasing(p=cfg["STRONG_ERASE_PROB"], scale=(0.02, 0.20), ratio=(0.3, 3.3)),
        T.ToPILImage(),
    ])

    log.info("Pipelines built:")
    log.info(f"  Standard: rotation=±{cfg['ROTATION_DEGREES']}°, crop={cfg['CROP_SCALE_MIN']}–{cfg['CROP_SCALE_MAX']}")
    log.info(f"  Strong  : rotation=±{cfg['STRONG_ROTATION_DEGREES']}°, crop={cfg['STRONG_CROP_SCALE_MIN']}–{cfg['CROP_SCALE_MAX']}, erasing=p{cfg['STRONG_ERASE_PROB']}")
    log.info(f"  Strong pipeline activates when gap/originals > {cfg['STRONG_AUG_MULTIPLIER_THRESHOLD']}×")

    return standard, strong


# ---------------------------------------------------------------------------
# Step 3 — Target-count mode: delete or generate to hit TARGET_IMAGES_PER_CLASS
# ---------------------------------------------------------------------------

def _delete_excess_augmented(cls: str, augmented: list[Path], current_total: int, target: int) -> int:
    """
    Delete the oldest augmented files until the class total equals target.
    Returns the number of files deleted.
    """
    n_to_delete = current_total - target
    # Sort so deletion is deterministic (youngest files, by name, are kept)
    to_delete = sorted(augmented)[:n_to_delete]
    deleted = 0
    for f in to_delete:
        try:
            f.unlink()
            deleted += 1
        except Exception as exc:
            log.warning(f"Could not delete {f}: {exc}")
    if deleted:
        log.info(f"  [{cls}] Deleted {deleted} excess augmented file(s).")
    return deleted


def run_target_count_mode(
    class_data: dict[str, dict],
    standard_pipeline: T.Compose,
    strong_pipeline: T.Compose,
    train_dir: Path,
    target: int,
) -> dict[str, dict]:
    """
    For each class:
      - If total > target  → delete excess augmented files.
      - If total < target  → generate exactly (target − total) new images,
                             cycling through originals in random order.
      - If total == target → skip.

    Returns per-class stats dict.
    """
    cfg         = CONFIG
    save_format = cfg["SAVE_FORMAT"]
    save_quality = cfg["SAVE_QUALITY"]
    prefix      = cfg["AUG_PREFIX"]
    threshold   = cfg["STRONG_AUG_MULTIPLIER_THRESHOLD"]

    stats: dict[str, dict] = {}
    n_errors = 0

    for cls, data in class_data.items():
        originals = data["originals"]
        augmented = data["augmented"]
        n_orig    = len(originals)
        n_aug     = len(augmented)
        total     = n_orig + n_aug

        if n_orig == 0:
            log.warning(f"  [{cls}] No original images — skipping.")
            stats[cls] = {"original": 0, "deleted": 0, "generated": 0, "total": 0, "status": "skipped"}
            continue

        # ── OVER TARGET: delete excess augmented images ──────────────────
        if total > target:
            if n_aug == 0:
                log.warning(
                    f"  [{cls}] {total} originals exceed target {target} but there are "
                    f"no augmented files to delete. Originals are never removed."
                )
                stats[cls] = {
                    "original": n_orig, "deleted": 0, "generated": 0,
                    "total": total, "status": "over-target (originals only)",
                }
                continue
            deleted = _delete_excess_augmented(cls, augmented, total, target)
            stats[cls] = {
                "original": n_orig, "deleted": deleted, "generated": 0,
                "total": total - deleted, "status": "trimmed",
            }
            continue

        # ── AT TARGET: nothing to do ──────────────────────────────────────
        if total == target:
            stats[cls] = {
                "original": n_orig, "deleted": 0, "generated": 0,
                "total": total, "status": "already at target",
            }
            continue

        # ── UNDER TARGET: generate exactly (target − total) images ────────
        n_needed  = target - total
        gap_ratio = n_needed / n_orig          # how many copies per original we need
        pipeline  = strong_pipeline if gap_ratio > threshold else standard_pipeline
        mode_tag  = "STRONG" if gap_ratio > threshold else "standard"

        log.info(
            f"  [{cls:<45}] orig={n_orig:>4}  existing_aug={n_aug:>5}  "
            f"need={n_needed:>5}  ratio={gap_ratio:.1f}×  pipeline={mode_tag}"
        )

        out_dir = (
            train_dir / cls
            if cfg["OUTPUT_MODE"] == "same"
            else Path(cfg["OUTPUT_DIR"]) / cls
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        # Shuffle originals so we cycle through them evenly
        shuffled_originals = originals.copy()
        random.shuffle(shuffled_originals)

        # Determine the next available index for naming (avoid collisions)
        existing_indices = set()
        for f in out_dir.iterdir():
            if f.stem.startswith(prefix):
                # filename format: aug_<stem>_<index>.jpg
                parts = f.stem.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    existing_indices.add(int(parts[1]))
        next_index = (max(existing_indices) + 1) if existing_indices else 0

        n_generated = 0
        with tqdm(total=n_needed, desc=f"{cls[:38]:<38} [{mode_tag}]", leave=False) as pbar:
            while n_generated < n_needed:
                src_path = shuffled_originals[n_generated % len(shuffled_originals)]
                # Re-shuffle every full cycle to maximise variety
                if n_generated > 0 and n_generated % len(shuffled_originals) == 0:
                    random.shuffle(shuffled_originals)

                try:
                    with Image.open(src_path) as img:
                        img = img.convert("RGB")
                    aug_img  = pipeline(img)
                    out_name = f"{prefix}_{src_path.stem}_{next_index:05d}.jpg"
                    aug_img.save(out_dir / out_name, save_format, quality=save_quality)
                    next_index  += 1
                    n_generated += 1
                    pbar.update(1)

                except Exception as exc:
                    n_errors += 1
                    log.warning(f"Skipped (error): {src_path} — {exc}")
                    # Still advance so we don't loop forever on a bad image
                    n_generated += 1
                    pbar.update(1)

        stats[cls] = {
            "original":  n_orig,
            "deleted":   0,
            "generated": n_generated,
            "total":     total + n_generated,
            "status":    f"topped up ({mode_tag})",
        }

    if n_errors:
        log.warning(f"{n_errors} image(s) skipped due to errors during generation.")

    return stats


# ---------------------------------------------------------------------------
# Step 4 — Legacy mode (COPIES_PER_IMAGE, kept for backward compatibility)
# ---------------------------------------------------------------------------

def compute_copies_per_class(class_data: dict) -> dict[str, int]:
    copies_map: dict[str, int] = {}
    boosted: list[tuple[str, int]] = []

    for cls, data in class_data.items():
        n = len(data["originals"])
        if CONFIG["ENABLE_BOOST"] and n < CONFIG["MIN_SAMPLES_FOR_BOOST"]:
            copies_map[cls] = CONFIG["BOOST_COPIES_PER_IMAGE"]
            boosted.append((cls, n))
        else:
            copies_map[cls] = CONFIG["COPIES_PER_IMAGE"]

    if boosted:
        log.info(
            f"Boost applied to {len(boosted)} minority classes "
            f"(< {CONFIG['MIN_SAMPLES_FOR_BOOST']} images):"
        )
        for cls, n in sorted(boosted, key=lambda x: x[1]):
            log.info(f"  {cls:<45} original={n:>4}  copies={CONFIG['BOOST_COPIES_PER_IMAGE']}")

    return copies_map


def run_legacy_mode(
    class_data: dict[str, dict],
    pipeline: T.Compose,
    copies_map: dict[str, int],
    train_dir: Path,
) -> dict[str, dict]:
    cfg         = CONFIG
    save_format = cfg["SAVE_FORMAT"]
    save_quality = cfg["SAVE_QUALITY"]
    prefix      = cfg["AUG_PREFIX"]
    stats: dict[str, dict] = {}
    n_errors = 0

    for cls, data in class_data.items():
        originals = data["originals"]
        n_copies  = copies_map[cls]
        out_dir   = (
            train_dir / cls
            if cfg["OUTPUT_MODE"] == "same"
            else Path(cfg["OUTPUT_DIR"]) / cls
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        n_generated = 0

        for src_path in tqdm(originals, desc=f"{cls[:40]:<40} (×{n_copies})", leave=False):
            try:
                with Image.open(src_path) as img:
                    img = img.convert("RGB")
                for i in range(n_copies):
                    aug_img  = pipeline(img)
                    out_name = f"{prefix}_{src_path.stem}_{i:03d}.jpg"
                    aug_img.save(out_dir / out_name, save_format, quality=save_quality)
                    n_generated += 1
            except Exception as exc:
                n_errors += 1
                log.warning(f"Skipped (error): {src_path} — {exc}")

        stats[cls] = {
            "original":  len(originals),
            "deleted":   0,
            "generated": n_generated,
            "total":     len(originals) + n_generated,
            "status":    "legacy",
        }

    if n_errors:
        log.warning(f"{n_errors} image(s) skipped due to errors.")

    return stats


# ---------------------------------------------------------------------------
# Step 5 — Copy originals when using a separate output directory
# ---------------------------------------------------------------------------

def copy_originals_if_needed(class_data: dict, train_dir: Path) -> None:
    if CONFIG["OUTPUT_MODE"] == "same":
        return
    log.info("Copying originals to output directory...")
    for cls, data in tqdm(class_data.items(), desc="Copying originals"):
        out_dir = Path(CONFIG["OUTPUT_DIR"]) / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in data["originals"]:
            dest = out_dir / f.name
            if not dest.exists():
                shutil.copy2(f, dest)


# ---------------------------------------------------------------------------
# Step 6 — Summary report
# ---------------------------------------------------------------------------

def write_summary(stats: dict, train_dir: Path, target: int | None) -> None:
    total_original  = sum(v["original"]  for v in stats.values())
    total_deleted   = sum(v["deleted"]   for v in stats.values())
    total_generated = sum(v["generated"] for v in stats.values())
    total_final     = sum(v["total"]     for v in stats.values())

    lines = [
        "=" * 72,
        "AUGMENTATION SUMMARY",
        "=" * 72,
        f"Train directory          : {CONFIG['TRAIN_DIR']}",
        f"Output mode              : {CONFIG['OUTPUT_MODE']}",
        f"Mode                     : {'Target-count (' + str(target) + '/class)' if target else 'Legacy (copies-per-image)'}",
        "",
        f"Total classes            : {len(stats)}",
        f"Original images          : {total_original:>8}",
        f"Augmented images deleted : {total_deleted:>8}",
        f"Augmented images created : {total_generated:>8}",
        f"Final train total        : {total_final:>8}",
        f"Effective multiplier     : {total_final / max(total_original, 1):.2f}×",
        "",
        f"{'Class':<45} {'Orig':>5} {'Del':>5} {'Gen':>6} {'Total':>6}  Status",
        "-" * 72,
    ]

    for cls, v in sorted(stats.items(), key=lambda x: x[1]["total"], reverse=True):
        lines.append(
            f"  {cls:<43} {v['original']:>5} {v['deleted']:>5} "
            f"{v['generated']:>6} {v['total']:>6}  {v['status']}"
        )

    lines.append("=" * 72)
    summary = "\n".join(lines)

    if CONFIG["OUTPUT_MODE"] == "same":
        summary_path = train_dir.parent / "augmentation_summary.txt"
    else:
        summary_path = Path(CONFIG["OUTPUT_DIR"]).parent / "augmentation_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    log.info(f"Summary written to: {summary_path}")
    print("\n" + summary)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(CONFIG["RANDOM_SEED"])
    np.random.seed(CONFIG["RANDOM_SEED"])

    train_dir = Path(CONFIG["TRAIN_DIR"])
    target    = CONFIG["TARGET_IMAGES_PER_CLASS"]

    log.info("=" * 72)
    log.info("  ProtoPNet Augmentation Pipeline  (v2 — Target-Count Mode)")
    log.info("=" * 72)
    log.info(f"Train dir   : {train_dir}")

    if target is not None:
        log.info(f"Mode        : TARGET-COUNT — {target} images per class")
        log.info(f"Strong aug  : activates when gap/originals > {CONFIG['STRONG_AUG_MULTIPLIER_THRESHOLD']}×")
    else:
        log.info(f"Mode        : LEGACY — {CONFIG['COPIES_PER_IMAGE']} copies per image")

    class_data = scan_train_dir(train_dir)
    copy_originals_if_needed(class_data, train_dir)

    if target is not None:
        standard_pipeline, strong_pipeline = build_pipelines()
        log.info("Starting target-count augmentation...")
        stats = run_target_count_mode(
            class_data, standard_pipeline, strong_pipeline, train_dir, target
        )
    else:
        standard_pipeline, _ = build_pipelines()
        copies_map = compute_copies_per_class(class_data)
        log.info("Starting legacy augmentation...")
        stats = run_legacy_mode(class_data, standard_pipeline, copies_map, train_dir)

    write_summary(stats, train_dir, target)
    log.info("Augmentation complete.")


if __name__ == "__main__":
    main()

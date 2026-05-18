"""
ProtoPNet Dataset Preprocessing Pipeline
-----------------------------------------
Cleans and prepares an image classification dataset for training.

Pipeline steps:
  1. Scan raw dataset folders
  2. Remove duplicate images
  3. Filter blurry / corrupt images
  4. Check class balance
  5. Stratified train / val / test split
  6. Resize and copy images to output structure
  7. Write a summary report

Install dependencies:
  pip install Pillow tqdm numpy scikit-learn opencv-python matplotlib
"""

import csv
import hashlib
import logging
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration — edit these values before running
# ---------------------------------------------------------------------------

CONFIG = {
    # Input: Points to your nested raw downloaded dataset
    "RAW_DATA_DIR": "./Dataset/plantwild_v2/plantwild_v2",

    # Output: Where the clean, stratified train/val/test splits will be created
    "OUTPUT_DIR": "./Dataset/plantwild",

    # Resize all images to this (width, height)
    "IMG_SIZE": (224, 224),

    # ImageNet stats — logged for reference; actual normalisation happens in the DataLoader
    "IMAGENET_MEAN": [0.485, 0.456, 0.406],
    "IMAGENET_STD":  [0.229, 0.224, 0.225],

    # Images with Laplacian variance below this are considered blurry and removed.
    # Raise toward 200 if too many images are slipping through.
    "BLUR_THRESHOLD": 100.0,

    # Images smaller than this (in KB) are likely corrupt
    "MIN_FILE_SIZE_KB": 1,

    # Split ratios — must sum to 1.0
    "TRAIN_RATIO": 0.70,
    "VAL_RATIO":   0.15,
    "TEST_RATIO":  0.15,

    # Warn when the largest class has this many more images than the smallest
    "IMBALANCE_RATIO_WARN": 10,

    "IMG_EXTENSIONS": {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"},
    "RANDOM_SEED": 42,
    "LOG_FILE": "preprocessing.log",

    # Save a bar chart of class sizes after cleaning
    "PLOT_CLASS_DIST": True,
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(CONFIG["LOG_FILE"], mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Scan the raw dataset
# ---------------------------------------------------------------------------

def scan_dataset(raw_dir: Path) -> dict[str, list[Path]]:
    """
    Walk raw_dir and collect image paths grouped by class name.

    Expects one subfolder per class:
        raw_dir/
            Tomato_Early_Blight/img001.jpg
            Corn_Healthy/img002.jpg
            ...

    Returns a dict mapping class name → list of image Paths.
    """
    log.info(f"Scanning dataset at: {raw_dir}")

    class_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir())
    if not class_dirs:
        log.error("No class subfolders found. Check RAW_DATA_DIR and its structure.")
        sys.exit(1)

    valid_extensions = CONFIG["IMG_EXTENSIONS"]
    class_to_files: dict[str, list[Path]] = defaultdict(list)

    for class_dir in tqdm(class_dirs, desc="Scanning classes"):
        for filepath in class_dir.rglob("*"):
            if filepath.suffix.lower() in valid_extensions:
                class_to_files[class_dir.name].append(filepath)

    total = sum(len(v) for v in class_to_files.values())
    log.info(f"Found {total} images across {len(class_to_files)} classes.")
    return dict(class_to_files)


# ---------------------------------------------------------------------------
# Step 2 — Remove duplicates
# ---------------------------------------------------------------------------

def _md5(filepath: Path) -> str:
    """Return the MD5 hash of a file's raw bytes."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def remove_duplicates(class_to_files: dict) -> dict:
    """
    Drop exact-duplicate images (identical byte content) across the whole dataset.
    The first occurrence of each file is kept.
    """
    log.info("Checking for duplicates via MD5 hash...")

    seen: dict[str, Path] = {}
    cleaned: dict[str, list[Path]] = defaultdict(list)
    n_duplicates = 0

    all_files = [(cls, f) for cls, files in class_to_files.items() for f in files]

    for cls, filepath in tqdm(all_files, desc="Deduplicating"):
        file_hash = _md5(filepath)
        if file_hash in seen:
            n_duplicates += 1
            log.debug(f"Duplicate: {filepath}  ↔  {seen[file_hash]}")
        else:
            seen[file_hash] = filepath
            cleaned[cls].append(filepath)

    log.info(f"Duplicates removed: {n_duplicates}")
    return dict(cleaned)


# ---------------------------------------------------------------------------
# Step 3 — Filter blurry and corrupt images
# ---------------------------------------------------------------------------

def _blur_score(filepath: Path) -> float:
    """
    Compute the variance of the Laplacian — a standard sharpness metric.
    Low variance means the image is blurry.
    Returns 0.0 if the image can't be read.
    """
    img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def filter_blurry(class_to_files: dict) -> dict:
    """Remove images that are blurry or suspiciously small (likely corrupt)."""
    blur_threshold = CONFIG["BLUR_THRESHOLD"]
    min_bytes = CONFIG["MIN_FILE_SIZE_KB"] * 1024

    log.info(
        f"Filtering blurry images (sharpness < {blur_threshold}) "
        f"and tiny files (< {CONFIG['MIN_FILE_SIZE_KB']} KB)..."
    )

    cleaned: dict[str, list[Path]] = defaultdict(list)
    n_blurry = 0
    n_tiny = 0

    all_files = [(cls, f) for cls, files in class_to_files.items() for f in files]

    for cls, filepath in tqdm(all_files, desc="Blur filtering"):
        if filepath.stat().st_size < min_bytes:
            n_tiny += 1
            log.debug(f"Removed (too small): {filepath}")
            continue

        score = _blur_score(filepath)
        if score < blur_threshold:
            n_blurry += 1
            log.debug(f"Removed (blurry, score={score:.1f}): {filepath}")
        else:
            cleaned[cls].append(filepath)

    log.info(f"Blurry images removed: {n_blurry} | Tiny files removed: {n_tiny}")
    return dict(cleaned)


# ---------------------------------------------------------------------------
# Step 4 — Check class balance
# ---------------------------------------------------------------------------

def check_class_balance(class_to_files: dict) -> None:
    """
    Log the per-class image counts and warn if the dataset is severely imbalanced.
    Optionally saves a bar chart.
    """
    counts = {cls: len(files) for cls, files in class_to_files.items()}
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    max_count = sorted_counts[0][1]
    min_count = sorted_counts[-1][1]
    imbalance_ratio = max_count / max(min_count, 1)

    log.info("Class distribution after cleaning:")
    for cls, count in sorted_counts:
        bar = "#" * min(count // 20, 60)
        log.info(f"  {cls:<45} {count:>5}  {bar}")
    log.info(f"Largest: {max_count}  |  Smallest: {min_count}  |  Ratio: {imbalance_ratio:.1f}x")

    if imbalance_ratio > CONFIG["IMBALANCE_RATIO_WARN"]:
        log.warning(
            f"Severe class imbalance detected ({imbalance_ratio:.1f}x). "
            "Consider weighted loss or oversampling."
        )

    if CONFIG["PLOT_CLASS_DIST"]:
        fig, ax = plt.subplots(figsize=(max(12, len(counts) // 3), 6))
        classes, values = zip(*sorted_counts)
        ax.bar(range(len(classes)), values, color="steelblue", edgecolor="white")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=90, fontsize=6)
        ax.set_ylabel("Image count")
        ax.set_title("Class distribution after cleaning")
        plt.tight_layout()

        chart_path = Path(CONFIG["OUTPUT_DIR"]) / "class_distribution.png"
        chart_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(chart_path, dpi=150)
        plt.close()
        log.info(f"Class distribution chart saved to: {chart_path}")


# ---------------------------------------------------------------------------
# Step 5 — Stratified train / val / test split
# ---------------------------------------------------------------------------

def stratified_split(
    class_to_files: dict,
) -> tuple[list[tuple], list[tuple], list[tuple]]:
    """
    Split the dataset into train, val, and test sets while preserving
    the class distribution in each split.

    Returns three lists of (image_path, class_name) tuples.
    """
    train_r = CONFIG["TRAIN_RATIO"]
    val_r   = CONFIG["VAL_RATIO"]
    test_r  = CONFIG["TEST_RATIO"]

    assert abs(train_r + val_r + test_r - 1.0) < 1e-6, \
        "TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0"

    log.info(f"Splitting: train={train_r:.0%}  val={val_r:.0%}  test={test_r:.0%}  (stratified)")

    all_paths = []
    all_labels = []
    for cls, files in class_to_files.items():
        all_paths.extend(files)
        all_labels.extend([cls] * len(files))

    all_paths  = np.array(all_paths,  dtype=object)
    all_labels = np.array(all_labels, dtype=object)

    # Split 1: train vs (val + test)
    val_test_size = val_r + test_r
    splitter1 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_test_size, random_state=CONFIG["RANDOM_SEED"]
    )
    train_idx, valtest_idx = next(splitter1.split(all_paths, all_labels))

    # Split 2: val vs test (within the held-out portion)
    relative_test_size = test_r / val_test_size
    splitter2 = StratifiedShuffleSplit(
        n_splits=1, test_size=relative_test_size, random_state=CONFIG["RANDOM_SEED"]
    )
    val_idx_rel, test_idx_rel = next(
        splitter2.split(all_paths[valtest_idx], all_labels[valtest_idx])
    )

    val_idx  = valtest_idx[val_idx_rel]
    test_idx = valtest_idx[test_idx_rel]

    train_set = list(zip(all_paths[train_idx],  all_labels[train_idx]))
    val_set   = list(zip(all_paths[val_idx],    all_labels[val_idx]))
    test_set  = list(zip(all_paths[test_idx],   all_labels[test_idx]))

    log.info(f"Split sizes — train: {len(train_set)}  val: {len(val_set)}  test: {len(test_set)}")
    return train_set, val_set, test_set


# ---------------------------------------------------------------------------
# Step 6 — Resize and copy to output directory
# ---------------------------------------------------------------------------

def resize_and_copy(split_data: list, split_name: str, output_dir: Path) -> None:
    """
    Resize each image to IMG_SIZE and save it under:
        output_dir/<split_name>/<class_name>/<filename>.jpg

    Images that can't be opened are skipped and logged as warnings.
    """
    target_size = CONFIG["IMG_SIZE"]
    n_skipped = 0

    for src_path, class_name in tqdm(split_data, desc=f"Copying {split_name}"):
        dest_dir = output_dir / split_name / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / (Path(src_path).stem + ".jpg")

        try:
            with Image.open(src_path) as img:
                img = img.convert("RGB")  # normalise to 3 channels (handles RGBA, grayscale)
                img = img.resize(target_size, Image.LANCZOS)
                img.save(dest_path, "JPEG", quality=95)
        except Exception as exc:
            n_skipped += 1
            log.warning(f"Skipped unreadable image: {src_path} — {exc}")

    if n_skipped:
        log.warning(f"{split_name}: {n_skipped} image(s) skipped due to read errors.")


# ---------------------------------------------------------------------------
# Step 7 — Write summary report
# ---------------------------------------------------------------------------

def write_summary(
    class_to_files: dict,
    train: list,
    val: list,
    test: list,
    output_dir: Path,
) -> None:
    """Save a plain-text summary of the preprocessing run."""
    summary_path = output_dir / "preprocessing_summary.txt"
    total_clean = sum(len(v) for v in class_to_files.values())

    train_counts = Counter(cls for _, cls in train)
    val_counts   = Counter(cls for _, cls in val)
    test_counts  = Counter(cls for _, cls in test)

    lines = [
        "=" * 60,
        "PREPROCESSING SUMMARY",
        "=" * 60,
        f"Dataset root   : {CONFIG['RAW_DATA_DIR']}",
        f"Output dir     : {CONFIG['OUTPUT_DIR']}",
        f"Image size     : {CONFIG['IMG_SIZE']}",
        f"Blur threshold : {CONFIG['BLUR_THRESHOLD']}",
        f"Random seed    : {CONFIG['RANDOM_SEED']}",
        "",
        f"Total classes  : {len(class_to_files)}",
        f"Total clean    : {total_clean}",
        f"  Train        : {len(train)}  ({len(train) / total_clean:.1%})",
        f"  Val          : {len(val)}   ({len(val) / total_clean:.1%})",
        f"  Test         : {len(test)}  ({len(test) / total_clean:.1%})",
        "",
        "Per-class counts (train | val | test):",
    ]

    for cls in sorted(class_to_files):
        lines.append(
            f"  {cls:<45}  {train_counts[cls]:>4} | {val_counts[cls]:>3} | {test_counts[cls]:>3}"
        )

    lines.append("=" * 60)
    summary_text = "\n".join(lines)

    with open(summary_path, "w") as f:
        f.write(summary_text)

    log.info(f"Summary written to: {summary_path}")
    print("\n" + summary_text)


# ---------------------------------------------------------------------------
# Optional helper — reorganise a flat dataset from a CSV label file
# ---------------------------------------------------------------------------

def reorganise_from_csv(
    csv_path: str,
    image_root: str,
    output_root: str,
    img_col: str = "filename",
    label_col: str = "label",
) -> None:
    """
    If your raw dataset has a flat image folder + CSV labels instead of
    class subfolders, run this first to create the expected structure.

    Args:
        csv_path:    Path to a CSV with at least two columns: filename and label.
        image_root:  Directory containing all raw images (flat).
        output_root: Where to create the class-organised subfolders.
        img_col:     CSV column that holds the image filename.
        label_col:   CSV column that holds the class label.

    Example:
        reorganise_from_csv("labels.csv", "./raw_images", "./plantwild_raw")
    """
    csv_path_    = Path(csv_path)
    image_root_  = Path(image_root)
    output_root_ = Path(output_root)

    log.info(f"Reorganising images from CSV: {csv_path_}")

    with open(csv_path_, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in tqdm(rows, desc="Organising by label"):
        src = image_root_ / row[img_col]
        dest_dir = output_root_ / row[label_col]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / row[img_col]
        if src.exists() and not dest.exists():
            shutil.copy2(src, dest)

    log.info(f"Done. Point RAW_DATA_DIR to: {output_root_}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    raw_dir    = Path(CONFIG["RAW_DATA_DIR"])
    output_dir = Path(CONFIG["OUTPUT_DIR"])

    if not raw_dir.exists():
        log.error(f"RAW_DATA_DIR not found: {raw_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("  ProtoPNet Preprocessing Pipeline")
    log.info("=" * 60)

    class_to_files = scan_dataset(raw_dir)
    class_to_files = remove_duplicates(class_to_files)
    class_to_files = filter_blurry(class_to_files)
    check_class_balance(class_to_files)

    train_set, val_set, test_set = stratified_split(class_to_files)

    resize_and_copy(train_set, "train", output_dir)
    resize_and_copy(val_set,   "val",   output_dir)
    resize_and_copy(test_set,  "test",  output_dir)

    write_summary(class_to_files, train_set, val_set, test_set, output_dir)

    log.info("Preprocessing complete.")
    log.info(f"Output ready at: {output_dir}")


if __name__ == "__main__":
    main()
"""
Dataset Summary Script
Scans an image classification dataset (ImageFolder-style layout) and writes
a detailed summary.txt with per-class statistics and overall dataset info.

Expected dataset layout:
    dataset_root/
        class_a/
            img1.jpg
            img2.png
            ...
        class_b/
            ...
        ...
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
#  CONFIG  –  edit these values before running
# ─────────────────────────────────────────────────────────────
CONFIG = {
    # Root folder that contains one sub-folder per class
    "dataset_path": r"C:/path/to/your/dataset",

    # Where to write the output summary file
    "output_path": r"./summary.txt",

    # Image extensions to recognise (case-insensitive)
    "image_extensions": {".jpg", ".jpeg", ".png", ".bmp", ".gif",
                         ".tiff", ".tif", ".webp", ".ppm", ".pgm"},

    # Scan sub-directories inside each class folder (True = recursive)
    "recursive": False,

    # Minimum images a class must have to be listed individually.
    # Classes below this threshold are grouped under "small classes".
    "min_images_threshold": 1,

    # If True, also record file sizes and report total dataset size
    "collect_file_sizes": True,
}
# ─────────────────────────────────────────────────────────────


def human_readable_size(num_bytes: int) -> str:
    """Convert bytes to a human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} PB"


def scan_dataset(cfg: dict) -> dict:
    """
    Walk the dataset directory and collect per-class image statistics.
    Returns a results dict consumed by write_summary().
    """
    root = Path(cfg["dataset_path"])
    if not root.exists():
        sys.exit(f"[ERROR] Dataset path does not exist: {root}")
    if not root.is_dir():
        sys.exit(f"[ERROR] Dataset path is not a directory: {root}")

    # Discover class folders (direct children that are directories)
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not class_dirs:
        sys.exit(f"[ERROR] No sub-directories (classes) found in: {root}")

    print(f"\n📂 Dataset root  : {root.resolve()}")
    print(f"📁 Classes found : {len(class_dirs)}\n")

    ext_set = cfg["image_extensions"]
    collect_sizes = cfg["collect_file_sizes"]

    class_stats: dict[str, dict] = {}
    all_extensions: dict[str, int] = defaultdict(int)

    for cls_dir in tqdm(class_dirs, desc="Scanning classes", unit="class"):
        class_name = cls_dir.name
        image_count = 0
        total_bytes = 0
        ext_counts: dict[str, int] = defaultdict(int)

        # Choose flat vs recursive walk
        if cfg["recursive"]:
            file_iter = cls_dir.rglob("*")
        else:
            file_iter = cls_dir.iterdir()

        for fpath in file_iter:
            if not fpath.is_file():
                continue
            suffix = fpath.suffix.lower()
            if suffix not in ext_set:
                continue
            image_count += 1
            ext_counts[suffix] += 1
            all_extensions[suffix] += 1
            if collect_sizes:
                try:
                    total_bytes += fpath.stat().st_size
                except OSError:
                    pass

        class_stats[class_name] = {
            "count": image_count,
            "total_bytes": total_bytes,
            "ext_counts": dict(ext_counts),
        }

    return {
        "root": root,
        "class_stats": class_stats,
        "all_extensions": dict(all_extensions),
        "collect_sizes": collect_sizes,
    }


def write_summary(results: dict, cfg: dict) -> None:
    """Format scan results and write summary.txt."""
    root         = results["root"]
    class_stats  = results["class_stats"]
    all_exts     = results["all_extensions"]
    collect_sizes = results["collect_sizes"]
    output_path  = Path(cfg["output_path"])
    threshold    = cfg["min_images_threshold"]

    total_classes = len(class_stats)
    total_images  = sum(v["count"] for v in class_stats.values())
    total_bytes   = sum(v["total_bytes"] for v in class_stats.values())

    # Sort classes by image count descending for the report
    sorted_classes = sorted(class_stats.items(),
                            key=lambda x: x[1]["count"], reverse=True)

    lines: list[str] = []
    sep  = "=" * 60
    dash = "-" * 60

    lines.append(sep)
    lines.append("  DATASET SUMMARY REPORT")
    lines.append(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)
    lines.append("")

    # ── Overview ──────────────────────────────────────────────
    lines.append("OVERVIEW")
    lines.append(dash)
    lines.append(f"  Dataset path      : {root.resolve()}")
    lines.append(f"  Total classes     : {total_classes}")
    lines.append(f"  Total images      : {total_images:,}")
    if collect_sizes:
        lines.append(f"  Total size        : {human_readable_size(total_bytes)}")
    if total_classes > 0:
        avg = total_images / total_classes
        counts = [v["count"] for v in class_stats.values()]
        lines.append(f"  Avg images/class  : {avg:.1f}")
        lines.append(f"  Max images/class  : {max(counts):,}  ({sorted_classes[0][0]})")
        lines.append(f"  Min images/class  : {min(counts):,}  ({sorted_classes[-1][0]})")
    lines.append("")

    # ── Extension breakdown ───────────────────────────────────
    lines.append("FILE EXTENSION BREAKDOWN")
    lines.append(dash)
    if all_exts:
        for ext, cnt in sorted(all_exts.items(), key=lambda x: x[1], reverse=True):
            pct = cnt / total_images * 100 if total_images else 0
            lines.append(f"  {ext:<10} {cnt:>8,}  ({pct:.1f}%)")
    else:
        lines.append("  (no images found)")
    lines.append("")

    # ── Per-class table ───────────────────────────────────────
    lines.append("PER-CLASS STATISTICS")
    lines.append(dash)
    header = f"  {'#':<5}  {'Class':<35}  {'Images':>8}"
    if collect_sizes:
        header += f"  {'Size':>12}"
    header += f"  {'Extensions'}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    small_classes = []
    for idx, (cls_name, stats) in enumerate(sorted_classes, start=1):
        count = stats["count"]
        if count < threshold:
            small_classes.append((cls_name, stats))
            continue
        pct   = count / total_images * 100 if total_images else 0
        exts  = ", ".join(
            f"{e}×{n}" for e, n in
            sorted(stats["ext_counts"].items(), key=lambda x: x[1], reverse=True)
        ) or "—"
        row = f"  {idx:<5}  {cls_name:<35}  {count:>8,}  ({pct:5.1f}%)"
        if collect_sizes:
            row += f"  {human_readable_size(stats['total_bytes']):>12}"
        row += f"  {exts}"
        lines.append(row)

    if small_classes:
        lines.append("")
        lines.append(f"  ── {len(small_classes)} class(es) with fewer than "
                     f"{threshold} image(s) ──")
        for cls_name, stats in small_classes:
            lines.append(f"     {cls_name}  ({stats['count']} images)")

    lines.append("")

    # ── Class balance info ────────────────────────────────────
    if total_classes > 1:
        counts = [v["count"] for v in class_stats.values()]
        max_c, min_c = max(counts), min(counts)
        imbalance = max_c / min_c if min_c > 0 else float("inf")
        lines.append("BALANCE INFO")
        lines.append(dash)
        lines.append(f"  Imbalance ratio (max/min) : {imbalance:.2f}x")
        if imbalance > 10:
            lines.append("  ⚠  High class imbalance detected (ratio > 10).")
        elif imbalance > 3:
            lines.append("  ⚠  Moderate class imbalance detected (ratio > 3).")
        else:
            lines.append("  ✓  Dataset appears reasonably balanced.")
        lines.append("")

    lines.append(sep)
    lines.append("  END OF REPORT")
    lines.append(sep)

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n✅ Summary written to: {output_path.resolve()}")


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = scan_dataset(CONFIG)
    write_summary(results, CONFIG)

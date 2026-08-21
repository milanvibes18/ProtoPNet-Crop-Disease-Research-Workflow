"""
assemble_grid.py
────────────────
Assembles individual per-class panel images (each containing
[Original | GradCAM++ | SHAP] side-by-side) into a single
paper-style grid figure for the top-5 classes.

Usage:
    python assemble_grid.py

Edit the PANELS list below with your actual file paths and class names.
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

# ─────────────────────────────────────────────
# CONFIGURATION — edit these
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# CONFIGURATION — edit these
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# CONFIGURATION — edit these
# ─────────────────────────────────────────────

# Notice the 'r' before the quote! This makes it a raw string.
BASE_DIR = r"C:\Users\milan\OneDrive\Documents\Project\ProtoPNet Crop Disease Research Workflow\Result\result shap\reports\xai_panels/"

PANELS = [
    # (image_path,                                                class_label,              confidence)
    (BASE_DIR + "panel_0000_Apple_scab(Apple).png",               "Apple Scab",             98.2),
    
    # ⚠️ IMPORTANT: Update the filenames below to match EXACTLY what is in your VS Code sidebar!
    (BASE_DIR + "panel_0001_Apple_scab(Apple).png",                 "Apple Black Rot",        97.4),
    (BASE_DIR + "panel_0002_Bacterial_spot(Peach.png",            "Bacterial Spot (Peach)", 96.1),
    (BASE_DIR + "panel_0003_Bacterial_spot(Peach.png",       "Northern Leaf Blight",   95.8),
    (BASE_DIR + "panel_0004_Bacterial_spot(Peppe.png",              "Tomato Late Blight",     94.3),
]

COLUMN_HEADERS = ["Original Image", "GradCAM++", "SHAP GradientExplainer"]

OUTPUT_PATH = "top5_grid.png"

# ── Panel crop settings ──────────────────────
# Each panel image has a title bar at the top.
# Set TITLE_CROP_PX to the approximate pixel height of that title bar.
# If your panels have NO title bar (pure image), set to 0.
TITLE_CROP_PX = 60        # pixels to remove from top of each panel

# If there's a thin caption strip at the bottom, set this:
BOTTOM_CROP_PX = 0

# ── Layout ───────────────────────────────────
CELL_W = 220              # width of each sub-image cell (pixels)
CELL_H = 180              # height of each sub-image cell (pixels)

LABEL_COL_W  = 170        # width of the left row-label column
HEADER_ROW_H = 40         # height of the top column-header row
GAP          = 4          # gap between cells (pixels)

BG_COLOR     = (15, 15, 15)     # dark background (like the paper)
HEADER_COLOR = (30, 30, 30)
LABEL_COLOR  = (20, 20, 20)

TEXT_COLOR   = (255, 255, 255)
CONF_COLOR   = (100, 220, 100)  # green for confidence

FONT_SIZE_HEADER = 14
FONT_SIZE_LABEL  = 12
FONT_SIZE_CONF   = 10

# ─────────────────────────────────────────────


def load_font(size):
    """Try to load a clean font, fall back to default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def split_panel(img: Image.Image, title_crop: int, bottom_crop: int):
    """
    Given a panel image [Original | GradCAM++ | SHAP] side-by-side,
    crop the title bar and split into 3 equal sub-images.
    Returns list of 3 PIL Images.
    """
    w, h = img.size

    # Crop title / bottom
    top = title_crop
    bot = h - bottom_crop if bottom_crop > 0 else h
    img = img.crop((0, top, w, bot))
    w, h = img.size

    # Split into 3 equal vertical slices
    slice_w = w // 3
    sub = []
    for i in range(3):
        x0 = i * slice_w
        x1 = (i + 1) * slice_w if i < 2 else w   # last slice gets remainder
        sub.append(img.crop((x0, 0, x1, h)))

    return sub


def draw_centered_text(draw, text, bbox, font, color=TEXT_COLOR):
    """Draw text centered within a bounding box (x0,y0,x1,y1)."""
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    tw, th = draw.textlength(text, font=font), font.size
    tx = x0 + (bw - tw) / 2
    ty = y0 + (bh - th) / 2
    draw.text((tx, ty), text, font=font, fill=color)


def assemble_grid(panels, output_path):
    n_rows = len(panels)
    n_cols = 3  # Original | GradCAM++ | SHAP

    # ── Canvas dimensions ────────────────────
    total_w = LABEL_COL_W + n_cols * (CELL_W + GAP)
    total_h = HEADER_ROW_H + n_rows * (CELL_H + GAP)

    canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)
    draw   = ImageDraw.Draw(canvas)

    font_header = load_font(FONT_SIZE_HEADER)
    font_label  = load_font(FONT_SIZE_LABEL)
    font_conf   = load_font(FONT_SIZE_CONF)

    # ── Column headers ───────────────────────
    for ci, header in enumerate(COLUMN_HEADERS):
        x0 = LABEL_COL_W + ci * (CELL_W + GAP)
        x1 = x0 + CELL_W
        # Header background
        draw.rectangle([x0, 0, x1, HEADER_ROW_H], fill=HEADER_COLOR)
        draw_centered_text(draw, header,
                           (x0, 0, x1, HEADER_ROW_H),
                           font_header, color=(200, 200, 200))

    # ── Row label column header ──────────────
    draw.rectangle([0, 0, LABEL_COL_W, HEADER_ROW_H], fill=HEADER_COLOR)
    draw_centered_text(draw, "Class",
                       (0, 0, LABEL_COL_W, HEADER_ROW_H),
                       font_header, color=(200, 200, 200))

    # ── Rows ─────────────────────────────────
    for ri, (img_path, label, conf) in enumerate(panels):
        y0 = HEADER_ROW_H + ri * (CELL_H + GAP)
        y1 = y0 + CELL_H

        # ── Row label ────────────────────────
        draw.rectangle([0, y0, LABEL_COL_W, y1], fill=LABEL_COLOR)
        # Class name
        text_x = 8
        text_y = y0 + 10
        draw.text((text_x, text_y), label, font=font_label, fill=TEXT_COLOR)
        # Confidence
        conf_text = f"{conf:.1f}%"
        draw.text((text_x, text_y + FONT_SIZE_LABEL + 6),
                  conf_text, font=font_conf, fill=CONF_COLOR)

        # ── Load & split panel ───────────────
        if not os.path.exists(img_path):
            print(f"  [WARN] File not found: {img_path} — inserting placeholder")
            subs = [
                Image.new("RGB", (CELL_W, CELL_H), (40, 40, 40))
                for _ in range(3)
            ]
            for s in subs:
                d = ImageDraw.Draw(s)
                d.text((10, CELL_H // 2 - 8), "not found",
                       font=font_conf, fill=(180, 60, 60))
        else:
            try:
                panel_img = Image.open(img_path).convert("RGB")
                subs = split_panel(panel_img, TITLE_CROP_PX, BOTTOM_CROP_PX)
            except Exception as e:
                print(f"  [ERROR] {img_path}: {e}")
                subs = [Image.new("RGB", (CELL_W, CELL_H), (50, 20, 20))
                        for _ in range(3)]

        # ── Paste sub-images ─────────────────
        for ci, sub in enumerate(subs):
            sub_resized = sub.resize((CELL_W, CELL_H), Image.LANCZOS)
            x_paste = LABEL_COL_W + ci * (CELL_W + GAP)
            canvas.paste(sub_resized, (x_paste, y0))

    # ── Thin grid lines ──────────────────────
    line_color = (50, 50, 50)
    for ri in range(n_rows + 1):
        y = HEADER_ROW_H + ri * (CELL_H + GAP)
        draw.line([(0, y), (total_w, y)], fill=line_color, width=1)
    for ci in range(n_cols + 1):
        x = LABEL_COL_W + ci * (CELL_W + GAP)
        draw.line([(x, 0), (x, total_h)], fill=line_color, width=1)
    draw.line([(LABEL_COL_W, 0), (LABEL_COL_W, total_h)],
              fill=line_color, width=2)

    canvas.save(output_path, dpi=(300, 300))
    print(f"\n✓ Saved → {output_path}  ({total_w}×{total_h} px)")


if __name__ == "__main__":
    assemble_grid(PANELS, OUTPUT_PATH)
